"""SQLite + numpy 向量存储（本机替代 Chroma 的自实现方案）。

背景：chromadb / chroma-hnswlib 的二进制 wheel 在当前机器（Win11 + Py3.12）
上 add_items 段错误（faulthandler 定位到 rust/hnsw 原生层，与 numpy 版本无关），
故用纯 numpy + SQLite 实现同等能力的嵌入式向量库：
- 持久化：每个 collection 一个 SQLite 文件（向量 BLOB + 文本 + metadata JSON）
- 检索：暴力余弦相似度 top-k（万级 chunk 内延迟毫秒级，医学知识库规模完全适用）
- 零原生依赖，接口对齐 LangChain VectorStore 的最小使用面（add_documents / as_retriever）

embedding 指纹目录隔离的设计保留：换 embedding 模型自动使用新目录，不串库。
"""

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

BATCH_SIZE = 25  # DashScope embed_documents 单批上限
# DashScope text-embedding-v3 输入上限 8192 token；中文约 1 字≈1 token，按字符截断留余量
MAX_EMBED_CHARS = 6000


def _clip_for_embedding(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_EMBED_CHARS else text[:MAX_EMBED_CHARS]


def _is_throttling(exc: Exception) -> bool:
    """识别 DashScope 限流类错误（速率/配额），可退避重试；其余视为硬错误。"""
    message = str(exc)
    return (
        "Throttling" in message
        or "429" in message
        or "RateLimit" in message
        or "AllocationQuota" in message
    )


class SqliteVectorStore:
    """嵌入式向量库：SQLite 存储 + numpy 余弦检索。

    大库（10万+ chunk）时启用内存向量矩阵缓存（VECTOR_MATRIX_CACHE=on，默认开）：
    首次查询把全部向量读入一个 float32 矩阵常驻内存，此后每次检索是一次
    矩阵乘（20万×1024 约 100-200ms），消除逐查询全表读 SQLite 的 IO 瓶颈。
    """

    def __init__(self, collection_name: str, db_path: Path, embedding_function: Embeddings):
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_function = embedding_function
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS vectors (
                doc_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                vec BLOB NOT NULL
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON vectors(doc_id)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self._cache_ids: set[str] | None = None
        # 内存矩阵缓存（懒加载，写操作置脏）
        self._matrix: np.ndarray | None = None
        self._matrix_doc_ids: list[str] = None
        self._matrix_rows: list[tuple[str, str, str]] | None = None  # (doc_id, content, metadata_json)

    # ---------- 写入 ----------
    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        """向量化并写入（增量：已存在的 doc_id 跳过）。"""
        if not documents:
            return
        ids = ids or [d.metadata.get("doc_id") or str(i) for i, d in enumerate(documents)]
        existing = self._existing_ids()

        pending = [
            (doc_id, doc)
            for doc_id, doc in zip(ids, documents)
            if doc_id not in existing
        ]
        if not pending:
            logger.debug(f"[{self.collection_name}] 全部 {len(documents)} chunks 已索引，跳过")
            return

        texts = [doc.page_content for _, doc in pending]
        vectors = self._embed_batch(texts)

        with self._lock:
            for (doc_id, doc), vec in zip(pending, vectors):
                self._conn.execute(
                    "INSERT OR REPLACE INTO vectors (doc_id, content, metadata, vec) VALUES (?, ?, ?, ?)",
                    (
                        doc_id,
                        doc.page_content,
                        json.dumps(doc.metadata, ensure_ascii=False),
                        np.asarray(vec, dtype=np.float32).tobytes(),
                    ),
                )
            self._conn.commit()
            self._cache_ids = None
            self._invalidate_matrix()
        logger.info(f"[{self.collection_name}] 新增向量 {len(pending)} 条（共 {self.count()} 条）")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化：切批后线程池并发调用 embedding API，限流自适应退避。

        串行时代 3.7 万 chunk 需约 1,500 次顺序 API 往返（每次 0.5-1s），
        是建索引慢的主因；并发后吞吐提升约等于并发度。
        撞 429/Throttling 时：未完成批次指数退避重试并把并发减半（至 1 转串行）；
        非限流异常直接抛出（add_documents 失败后可断点重跑，增量不重复计费）。
        并发度由 EMBED_CONCURRENCY 控制（默认 4，夹紧到 [1,8]）。
        """
        batches = [
            [_clip_for_embedding(t) for t in texts[start : start + BATCH_SIZE]]
            for start in range(0, len(texts), BATCH_SIZE)
        ]
        if len(batches) <= 1:
            return [v for b in batches for v in self.embedding_function.embed_documents(b)]

        import os

        try:
            workers = max(1, min(8, int(os.getenv("EMBED_CONCURRENCY", "4"))))
        except ValueError:
            workers = 4
        workers = min(workers, len(batches))

        results: list[list[list[float]] | None] = [None] * len(batches)
        pending = list(range(len(batches)))
        backoff = 4.0

        while pending:
            throttled: list[int] = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_map = {
                    pool.submit(self.embedding_function.embed_documents, batches[i]): i
                    for i in pending
                }
                for future, index in future_map.items():
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        if _is_throttling(exc):
                            throttled.append(index)
                        else:
                            raise RuntimeError(f"embedding 批次 {index} 失败: {exc}") from exc

            if throttled:
                pending = throttled
                workers = max(1, workers // 2)
                logger.warning(
                    f"[{self.collection_name}] embedding 限流，退避 {backoff:.0f}s，"
                    f"并发降至 {workers}，待重试 {len(pending)} 批"
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 120.0)
            else:
                pending = []

        vectors: list[list[float]] = []
        for result in results:
            vectors.extend(result)
        return vectors

    def _existing_ids(self) -> set[str]:
        if self._cache_ids is None:
            rows = self._conn.execute("SELECT doc_id FROM vectors").fetchall()
            self._cache_ids = {row[0] for row in rows}
        return self._cache_ids

    # ---------- 查询 ----------
    def similarity_search(self, query: str, k: int = 5, query_vector=None) -> list[Document]:
        """余弦 top-k。可传入预算好的 query_vector 避免重复调 embedding API。"""
        if query_vector is None:
            query_vector = self.embedding_function.embed_query(query)
        query_vec = np.asarray(query_vector, dtype=np.float32)

        rows, matrix = self._load_rows_and_matrix()
        if rows is None or not len(rows):
            return []

        dim = len(query_vec)
        scores = self._cosine_scores(matrix, query_vec)

        top_indices = np.argsort(-scores)[:k]
        results = []
        for index in top_indices:
            doc_id, content, metadata_json = rows[int(index)]
            metadata = json.loads(metadata_json)
            metadata["doc_id"] = doc_id
            results.append(Document(page_content=content, metadata=metadata))
        return results

    @staticmethod
    def _cosine_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vec)
        norms = np.linalg.norm(matrix, axis=1)
        denom = norms * max(query_norm, 1e-10)
        denom[denom == 0] = 1e-10
        return matrix @ query_vec / denom

    # ---------- 内存矩阵缓存 ----------
    def _matrix_cache_enabled(self) -> bool:
        import os

        return (os.getenv("VECTOR_MATRIX_CACHE", "on").strip().lower()
                not in {"off", "0", "false", "disabled"})

    def _invalidate_matrix(self) -> None:
        """清空矩阵缓存。调用方（clear/add_documents）已持有 _lock，这里只清引用。

        注意不能用 with self._lock：clear() 持锁调用本方法会造成非重入锁自我死锁。
        """
        self._matrix = None
        self._matrix_rows = None
        self._matrix_doc_ids = None

    def _load_rows_and_matrix(self):
        """返回 (rows, matrix)。缓存开启且已加载时直接命中内存；否则流式读 SQLite。"""
        if self._matrix_cache_enabled():
            with self._lock:
                if self._matrix is not None:
                    return self._matrix_rows, self._matrix

            rows = self._conn.execute(
                "SELECT doc_id, content, metadata, vec FROM vectors"
            ).fetchall()
            if not rows:
                return [], None

            dim = len(np.frombuffer(rows[0][3], dtype=np.float32))
            # frombuffer 零拷贝拼批，避免逐行 frombuffer 的循环开销
            matrix = np.frombuffer(b"".join(row[3] for row in rows), dtype=np.float32)
            matrix = matrix.reshape(len(rows), dim)

            with self._lock:
                self._matrix_rows = [(r[0], r[1], r[2]) for r in rows]
                self._matrix_doc_ids = [r[0] for r in rows]
                self._matrix = matrix
                return self._matrix_rows, self._matrix

        # 缓存关闭：每次流式读（省内存，慢）
        rows = self._conn.execute(
            "SELECT doc_id, content, metadata, vec FROM vectors"
        ).fetchall()
        if not rows:
            return [], None
        dim = len(np.frombuffer(rows[0][3], dtype=np.float32))
        matrix = np.frombuffer(b"".join(row[3] for row in rows), dtype=np.float32).reshape(
            len(rows), dim
        )
        return [(r[0], r[1], r[2]) for r in rows], matrix

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()
            self._cache_ids = None
            self._invalidate_matrix()

    def close(self) -> None:
        self._conn.close()

    # ---------- 检索器适配 ----------
    def as_retriever(self, search_kwargs: dict | None = None):
        """返回最小化检索器（duck-typing 对齐 BaseRetriever.invoke）。"""
        kwargs = search_kwargs or {}
        store = self

        class _Retriever:
            def invoke(self, query: str, config=None, query_vector=None, **kwargs2) -> list[Document]:
                return store.similarity_search(query, k=kwargs.get("k", 5), query_vector=query_vector)

        return _Retriever()


def rebuild_if_changed(store: SqliteVectorStore, documents: list[Document], file_fingerprint: str) -> None:
    """文件指纹变化时全量重建（指纹存 meta 表；doc_id 保持内容哈希不变，
    以维持向量/BM25/关键词三路检索的跨路对齐）。"""
    row = store._conn.execute(
        "SELECT value FROM meta WHERE key = 'fingerprint'"
    ).fetchone()
    stored_fingerprint = row[0] if row else ""

    if stored_fingerprint != file_fingerprint:
        store.clear()
        with store._lock:
            store._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('fingerprint', ?)",
                (file_fingerprint,),
            )
            store._conn.commit()
        logger.info(f"[{store.collection_name}] 索引按指纹 {file_fingerprint[:12]} 全量重建")

    store.add_documents(documents, ids=[d.metadata["doc_id"] for d in documents])
