"""公开基准检索评测：C-MTEB CmedqaRetrieval（中文在线医疗问诊垂类）。

数据（hf-mirror 下载，无需登录）：
- corpus: 100,001 段落（~60MB parquet）
- queries: 3,999 条查询
- qrels(dev): query -> 相关段落 id 标注（C-MTEB/CmedqaRetrieval-qrels）

评测方式：抽样 N 条查询（默认 200），corpus 建独立临时索引（不混入生产知识库），
用生产同款混合检索器（向量+BM25+关键词/RRF/可选rerank）检索 top-k，
按 qrels 计算 recall@5 / recall@10 / MRR@10 —— 与自制 14 条标注集同口径、可对外引用。

运行：cd backend && python -m app.eval.bench_eval [--n 200] [--skip-index]
首次运行需建索引（10 万段落约 4,000 批 embedding，1 小时级；--skip-index 复用已建索引）。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from loguru import logger

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "database" / "bench_cmedqa"
EVAL_SEED = 42


def _download_all():
    from huggingface_hub import hf_hub_download

    files = {}
    for repo, filename in [
        ("C-MTEB/CmedqaRetrieval", "data/corpus-00000-of-00001-a3949861f65a3226.parquet"),
        ("C-MTEB/CmedqaRetrieval", "data/queries-00000-of-00001-daeedab899d3c839.parquet"),
        ("C-MTEB/CmedqaRetrieval-qrels", "data/dev-00000-of-00001-57fb84a4aceaa695.parquet"),
    ]:
        files[filename.split("/")[-1]] = hf_hub_download(
            repo_id=repo, filename=filename, repo_type="dataset"
        )
        logger.info(f"已下载 {repo}/{filename}")
    return files


def _load_parquet(path: str):
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _resolve_eval_embedding(embedding_backend: str):
    """评测 embedding 后端：dashscope(text-embedding-v3，默认) / bge-m3(硅基流动)。

    返回 (client, 指纹标签)。指纹用于隔离向量库目录，不同后端不串库。
    """
    if embedding_backend == "bge-m3":
        import hashlib
        import os

        from dotenv import load_dotenv
        from langchain_openai import OpenAIEmbeddings

        load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise SystemExit("bge-m3 后端需在 .env 配置 SILICONFLOW_API_KEY")
        client = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            api_key=api_key,
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        )
        fingerprint = hashlib.sha1(b"openai|BAAI/bge-m3|siliconflow").hexdigest()[:12]
        return client, fingerprint
    # 默认：生产同款 DashScope text-embedding-v3
    from app.rag.model_clients import get_embedding_client, get_retrieval_runtime_status

    return get_embedding_client(), get_retrieval_runtime_status()["embedding"]["fingerprint"]


def build_bench_retriever(force: bool = False, corpus_sample: int | None = None, n_queries: int = 200,
                           use_rerank: bool = False, embedding_backend: str = "dashscope"):
    """corpus parquet -> Documents（每段落一个检索单元，与 qrels 的 pid 对齐）。

    注意：不走 MyCustomLoader 的文本切分——C-MTEB 的检索单元就是原段落，
    重切分反而会破坏 pid 对齐（一个段落裂成多块，pid 前缀只落在首块），
    且超大单文件的递归切分本身病态慢。

    corpus_sample：抽样语料建库（评测口径为「在 N 段子库上检索」）。默认 None
    用全量 10 万段；配额受限时可抽样（如 10000）——注意抽样会把落库外的
    qrels 正例变成不可命中，recall 上限因此下降，报告需注明语料规模。
    抽样保留全部「被抽样查询命中的段落」优先（oracle 子集思路的反向：
    这里简单地均匀抽样，指标只作相对比较用）。
    """
    from langchain_core.documents import Document

    from app.rag.knowledge_manager import get_md5, normalize_documents
    from app.rag.model_clients import get_embedding_client, get_retrieval_runtime_status
    from app.rag.retriever import KeywordPriorityRetriever, build_bm25_retriever
    from app.rag.vector_store import SqliteVectorStore

    status = get_retrieval_runtime_status()
    embedding_client, embedding_fp = _resolve_eval_embedding(embedding_backend)
    collection = get_md5(f"cmedqa_bench_s{corpus_sample or 'full'}_{embedding_backend}")
    store = SqliteVectorStore(
        collection_name=collection,
        db_path=Path(__file__).resolve().parent.parent.parent
        / "database" / "vector_store" / embedding_fp / f"{collection}.sqlite",
        embedding_function=embedding_client,
    )

    if store.count() == 0 or force:
        if force:
            store.clear()
        files = _download_all()
        rows = _load_parquet(files["corpus-00000-of-00001-a3949861f65a3226.parquet"])
        logger.info(f"corpus 加载完成: {len(rows)} 段落")
        if corpus_sample and corpus_sample < len(rows):
            # 构造性抽样：先抽定查询（同种子），再保证其全部正例段落入子库，
            # 其余名额均匀随机填充——查询可评性 100%，语料规模可控。
            queries_rows = _load_parquet(files["queries-00000-of-00001-daeedab899d3c839.parquet"])
            qrels_rows_ = _load_parquet(files["dev-00000-of-00001-57fb84a4aceaa695.parquet"])
            qrels_: dict[str, set[str]] = {}
            for row in qrels_rows_:
                qrels_.setdefault(str(row["qid"]), set()).add(str(row["pid"]))
            query_map_ = {str(q["id"]): (q.get("text") or "").strip() for q in queries_rows}
            labeled = [qid for qid in qrels_ if query_map_.get(qid)]
            random.seed(EVAL_SEED)
            picked_queries = set(random.sample(labeled, min(n_queries, len(labeled))))
            must_pids = set()
            for qid in picked_queries:
                must_pids |= qrels_[qid]

            by_pid = {str(r["id"]): r for r in rows}
            chosen = [by_pid[p] for p in must_pids if p in by_pid]
            remaining = [r for r in rows if str(r["id"]) not in must_pids]
            fill = max(0, corpus_sample - len(chosen))
            chosen += random.sample(remaining, min(fill, len(remaining)))
            rows = chosen
            logger.info(
                f"构造性抽样: {len(picked_queries)} 查询的 {len(must_pids)} 个正例段落必收，"
                f"子库共 {len(rows)} 段"
            )
        documents = normalize_documents([
            Document(
                page_content=(row.get("text") or "").strip(),
                metadata={"source": "cmedqa_corpus", "pid": str(row["id"])},
            )
            for row in rows
            if (row.get("text") or "").strip()
        ])
        store.add_documents(documents, ids=[d.metadata["doc_id"] for d in documents])
    logger.info(f"基准向量库就绪: {store.count()} 段落")

    # BM25 + 混合检索器（10 万段落 > 3000 自动走大库轻量 keyword 路径）
    all_rows = store._conn.execute("SELECT doc_id, content, metadata FROM vectors").fetchall()
    import json as _json

    corpus_documents = [
        Document(page_content=content, metadata={**_json.loads(meta), "doc_id": doc_id})
        for doc_id, content, meta in all_rows
    ]
    bm25 = build_bm25_retriever(corpus_documents, k=10)
    reranker = None
    if use_rerank:
        from app.rag.model_clients import get_rerank_client

        try:
            # 候选取 top_k*2 进 rerank，返回 top_k——与生产链路同构
            reranker = get_rerank_client(top_n=10)
            logger.info("基准评测开启 rerank（gte-rerank-v2，与生产同款）")
        except Exception as exc:
            logger.warning(f"rerank 客户端不可用，回退纯混合检索口径: {exc}")
    return KeywordPriorityRetriever(
        documents=corpus_documents,
        vector_retriever=store.as_retriever(search_kwargs={"k": 10}),
        bm25_retriever=bm25,
        reranker=reranker,
        top_k=10,
    )


def run_eval(n_queries: int = 200, top_ks=(5, 10), corpus_sample: int | None = None,
             use_rerank: bool = False, embedding_backend: str = "dashscope"):
    files = _download_all()
    queries = _load_parquet(files["queries-00000-of-00001-daeedab899d3c839.parquet"])
    qrels_rows = _load_parquet(files["dev-00000-of-00001-57fb84a4aceaa695.parquet"])

    # qrels 字段（实测）：qid / pid / score
    sample_row = qrels_rows[0]
    logger.info(f"qrels 字段: {list(sample_row.keys())}")
    qid_key = next((k for k in sample_row if k.lower() in {"qid", "query_id", "query-id"}), "qid")
    pid_key = next((k for k in sample_row if k.lower() in {"pid", "corpus-id", "docid", "doc_id"}), "pid")
    score_key = next((k for k in sample_row if "score" in k.lower()), None)
    logger.info(f"qrels 映射: qid={qid_key}, pid={pid_key}, score={score_key}")

    qrels: dict[str, set[str]] = {}
    for row in qrels_rows:
        if score_key and float(row[score_key]) <= 0:
            continue
        qid = str(row[qid_key])
        qrels.setdefault(qid, set()).add(str(row[pid_key]))

    # 只抽有标注的查询
    query_map = {str(q["id"]): (q.get("text") or "").strip() for q in queries}
    labeled_qids = [qid for qid in qrels if query_map.get(qid)]
    random.seed(EVAL_SEED)
    sampled = random.sample(labeled_qids, min(n_queries, len(labeled_qids)))
    logger.info(f"抽样 {len(sampled)} / {len(labeled_qids)} 条有标注查询")

    retriever = build_bench_retriever(corpus_sample=corpus_sample, n_queries=n_queries,
                                      use_rerank=use_rerank, embedding_backend=embedding_backend)
    retriever.top_k = max(top_ks)

    # 语料抽样时只评「正例段落全部在子库内」的查询（构造性抽样下应全量保留，
    # 此处为防御性校验）
    if corpus_sample:
        in_store_pids = {d.metadata.get("pid") for d in retriever.documents}
        eligible = [qid for qid in sampled if qrels[qid] <= in_store_pids]
        dropped = len(sampled) - len(eligible)
        sampled = eligible[:n_queries]
        logger.info(
            f"语料抽样模式: {dropped} 条查询因正例不在子库内被剔除，"
            f"实际评测 {len(sampled)} 条"
        )
        if not sampled:
            raise SystemExit("抽样后无可用查询，请增大 --corpus-sample 或用全量")

    def chunk_pid(document) -> str | None:
        return str(document.metadata.get("pid") or "") or None

    recalls = {k: [] for k in top_ks}
    mrr10_list = []
    no_hit = 0
    t_start = time.time()

    for index, qid in enumerate(sampled, 1):
        query_text = query_map[qid]
        relevant = qrels[qid]
        try:
            documents = retriever.invoke(query_text)
        except Exception as exc:
            logger.warning(f"查询失败 {qid}: {exc}")
            documents = []

        ranks = []
        for rank, doc in enumerate(documents, 1):
            pid = chunk_pid(doc)
            if pid and pid in relevant:
                ranks.append(rank)

        for k in top_ks:
            recalls[k].append(1.0 if any(r <= k for r in ranks) else 0.0)
        mrr10_list.append(1.0 / ranks[0] if ranks and ranks[0] <= 10 else 0.0)
        if not ranks:
            no_hit += 1
        if index % 20 == 0:
            elapsed = time.time() - t_start
            logger.info(f"进度 {index}/{len(sampled)}，平均 {elapsed / index:.2f}s/查询")

    elapsed_total = time.time() - t_start
    result = {
        "benchmark": "C-MTEB/CmedqaRetrieval (dev qrels)",
        "queries": len(sampled),
        "corpus_chunks": retriever.vector_retriever and None,
        "recall@5": sum(recalls[5]) / len(recalls[5]),
        "recall@10": sum(recalls[10]) / len(recalls[10]),
        "mrr@10": sum(mrr10_list) / len(mrr10_list),
        "no_hit_rate": no_hit / len(sampled),
        "avg_latency_s": elapsed_total / len(sampled),
        "embedding": embedding_backend,
        "rerank": "on（gte-rerank-v2，生产同款）" if use_rerank else "off（纯混合检索口径）",
    }
    print("\n" + "=" * 60)
    print("CmedqaRetrieval 基准评测（生产同款混合检索器）")
    print("=" * 60)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="抽样查询数")
    parser.add_argument("--corpus-sample", type=int, default=None,
                        help="语料抽样建库段数（默认全量 10 万；配额受限时可用如 10000）")
    parser.add_argument("--force-index", action="store_true", help="强制重建基准索引")
    parser.add_argument("--rerank", action="store_true",
                        help="开启 rerank 精排口径（默认纯混合检索）")
    parser.add_argument("--embedding", choices=["dashscope", "bge-m3"], default="dashscope",
                        help="embedding 后端：dashscope(text-embedding-v3) / bge-m3(硅基流动)")
    args = parser.parse_args()
    if args.force_index:
        build_bench_retriever(force=True, corpus_sample=args.corpus_sample, n_queries=args.n,
                              embedding_backend=args.embedding)
    run_eval(args.n, corpus_sample=args.corpus_sample, use_rerank=args.rerank,
             embedding_backend=args.embedding)


if __name__ == "__main__":
    main()
