"""向量存储层：可切换后端（sqlite 自实现 / Qdrant，预留 Milvus 插槽）。

背景：chromadb / chroma-hnswlib 的二进制 wheel 在当前机器（Win11 + Py3.12）
上 add_items 段错误（faulthandler 定位到 rust/hnsw 原生层，与 numpy 版本无关），
最初用纯 numpy + SQLite 自实现嵌入式向量库顶替。

现抽象出 VectorStoreBase 统一契约，提供两个实现：
- SqliteVectorStore：SQLite BLOB + numpy 暴力余弦（零依赖默认值，万级 chunk 毫秒级）
- QdrantVectorStore：成熟开源向量库。local 嵌入式模式（纯 Python，Windows 可用，
  数据落 vector_store_dir/<指纹>/qdrant_local/）；生产填 QDRANT_URL 即切 Docker
  服务端，代码零改动。当前 3.7 万 chunk 规模 local 模式完全够用。

embedding 指纹目录隔离的设计保留：换 embedding 模型自动使用新目录，不串库。

升级 Milvus 的路径：实现 VectorStoreBase 的一个子类（MilvusVectorStore），
在 create_vector_store 工厂注册 "milvus" 分支即可，上层（knowledge_manager/
检索器/迁移脚本）全部面向基类契约编程，不需要任何改动。
"""

import json
import sqlite3
import threading
import time
import uuid
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


class VectorStoreBase:
    """向量库后端统一契约（上层只依赖这些方法，不摸任何 _conn 之类内部）。

    构造签名统一为 (collection_name, db_path, embedding_function)：
    db_path 为「指纹目录/<collection>.sqlite」形态的占位路径——sqlite 后端直接用它
    做库文件；qdrant 后端取 db_path.parent 作为该指纹目录的存储根。
    """

    def __init__(self, collection_name: str, db_path: Path, embedding_function: Embeddings):
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        self.embedding_function = embedding_function

    # ---- 写入 ----
    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        """向量化并写入（增量：已存在的 doc_id 跳过）。"""
        raise NotImplementedError

    def add_embeddings(self, doc_ids: list[str], texts: list[str], metadatas: list[dict],
                       vectors: list[list[float]]) -> None:
        """跳过 embedding 环节直接写入现成向量（迁移脚本/测试用）。"""
        raise NotImplementedError

    # ---- 查询 ----
    def similarity_search(self, query: str, k: int = 5, query_vector=None) -> list[Document]:
        """余弦 top-k。可传入预算好的 query_vector 避免重复调 embedding API。"""
        raise NotImplementedError

    def as_retriever(self, search_kwargs: dict | None = None):
        """最小化检索器（duck-typing 对齐 BaseRetriever.invoke）。"""
        kwargs = search_kwargs or {}
        store = self

        class _Retriever:
            def invoke(self, query: str, config=None, query_vector=None, **kwargs2) -> list[Document]:
                return store.similarity_search(query, k=kwargs.get("k", 5), query_vector=query_vector)

        return _Retriever()

    # ---- 元数据/生命周期 ----
    def get_meta(self, key: str) -> str:
        """取 collection 级元数据（文件指纹等），无则空串。"""
        raise NotImplementedError

    def set_meta(self, key: str, value: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ---- 共享：批量向量化（与后端无关）----
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


# =====================================================================
# 后端一：SQLite + numpy（零依赖默认值）
# =====================================================================
class SqliteVectorStore(VectorStoreBase):
    """嵌入式向量库：SQLite 存储 + numpy 余弦检索。

    大库（10万+ chunk）时启用内存向量矩阵缓存（VECTOR_MATRIX_CACHE=on，默认开）：
    首次查询把全部向量读入一个 float32 矩阵常驻内存，此后每次检索是一次
    矩阵乘（20万×1024 约 100-200ms），消除逐查询全表读 SQLite 的 IO 瓶颈。
    """

    def __init__(self, collection_name: str, db_path: Path, embedding_function: Embeddings):
        super().__init__(collection_name, db_path, embedding_function)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.add_embeddings(
            [doc_id for doc_id, _ in pending],
            texts,
            [doc.metadata for _, doc in pending],
            vectors,
        )
        logger.info(f"[{self.collection_name}] 新增向量 {len(pending)} 条（共 {self.count()} 条）")

    def add_embeddings(self, doc_ids, texts, metadatas, vectors) -> None:
        with self._lock:
            for doc_id, text, metadata, vec in zip(doc_ids, texts, metadatas, vectors):
                self._conn.execute(
                    "INSERT OR REPLACE INTO vectors (doc_id, content, metadata, vec) VALUES (?, ?, ?, ?)",
                    (
                        doc_id,
                        text,
                        json.dumps(metadata, ensure_ascii=False),
                        np.asarray(vec, dtype=np.float32).tobytes(),
                    ),
                )
            self._conn.commit()
            self._cache_ids = None
            self._invalidate_matrix()

    def _existing_ids(self) -> set[str]:
        if self._cache_ids is None:
            rows = self._conn.execute("SELECT doc_id FROM vectors").fetchall()
            self._cache_ids = {row[0] for row in rows}
        return self._cache_ids

    # ---------- 查询 ----------
    def similarity_search(self, query: str, k: int = 5, query_vector=None) -> list[Document]:
        if query_vector is None:
            query_vector = self.embedding_function.embed_query(query)
        query_vec = np.asarray(query_vector, dtype=np.float32)

        rows, matrix = self._load_rows_and_matrix()
        if rows is None or not len(rows):
            return []

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

    # ---------- 元数据/生命周期 ----------
    def get_meta(self, key: str) -> str:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else ""

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value)
            )
            self._conn.commit()

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


# =====================================================================
# 后端二：Qdrant（local 嵌入式 / 服务端二合一）
# =====================================================================
class QdrantVectorStore(VectorStoreBase):
    """Qdrant 向量库适配器。

    - local 模式（QDRANT_URL 留空）：QdrantClient(path=指纹目录/qdrant_local)，
      纯 Python 依赖、无服务进程，Windows 可用；同一 path 的客户端全进程共享
      （local 模式对同一路径只允许一个实例，QdrantClient 自己按 path 锁目录）。
    - 服务端模式（QDRANT_URL=http://...）：同一套客户端 API 切 Docker 服务，
      建索引/检索代码零改动，适合生产或多 worker 部署。

    点 ID 用 uuid5(doc_id) 确定性派生：内容哈希 doc_id 不变则点 ID 不变，
    upsert 天然幂等，与 sqlite 后端的增量语义对齐。
    """

    _META_COLLECTION = "__meta__"  # collection 级元数据（文件指纹）的公共存储
    _UPSERT_BATCH = 256
    _clients: dict[str, "object"] = {}  # path/url -> QdrantClient（进程级共享）
    _clients_lock = threading.Lock()

    def __init__(self, collection_name: str, db_path: Path, embedding_function: Embeddings):
        super().__init__(collection_name, db_path, embedding_function)
        from app.core.config import settings

        if settings.qdrant_url:
            cache_key = settings.qdrant_url
            client_kwargs = {"url": settings.qdrant_url}
            self._local_root = None
        else:
            self._local_root = self.db_path.parent / "qdrant_local"
            self._local_root.mkdir(parents=True, exist_ok=True)
            cache_key = str(self._local_root.resolve())
            client_kwargs = {"path": str(self._local_root)}

        self._client = self._shared_client(cache_key, client_kwargs)

    @classmethod
    def _shared_client(cls, cache_key: str, client_kwargs: dict):
        with cls._clients_lock:
            client = cls._clients.get(cache_key)
            if client is None:
                from qdrant_client import QdrantClient

                client = QdrantClient(**client_kwargs)
                cls._clients[cache_key] = client
            return client

    # ---------- 内部工具 ----------
    @staticmethod
    def _point_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    def _collection_exists(self) -> bool:
        from qdrant_client import models  # noqa: F401

        return self._client.collection_exists(self.collection_name)

    def _ensure_collection(self, dim: int) -> None:
        from qdrant_client import models

        if not self._collection_exists():
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            logger.info(
                f"[{self.collection_name}] 创建 Qdrant collection（dim={dim}，"
                f"{'local' if self._local_root else self._client.url}）"
            )

    # ---------- 写入 ----------
    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
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
        self.add_embeddings(
            [doc_id for doc_id, _ in pending],
            texts,
            [doc.metadata for _, doc in pending],
            vectors,
        )
        logger.info(f"[{self.collection_name}] 新增向量 {len(pending)} 条（共 {self.count()} 条）")

    def add_embeddings(self, doc_ids, texts, metadatas, vectors) -> None:
        if not vectors:
            return
        from qdrant_client import models

        self._ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=self._point_id(doc_id),
                vector=[float(x) for x in vec],
                payload={"doc_id": doc_id, "content": text, "metadata": metadata},
            )
            for doc_id, text, metadata, vec in zip(doc_ids, texts, metadatas, vectors)
        ]
        for start in range(0, len(points), self._UPSERT_BATCH):
            self._client.upsert(
                collection_name=self.collection_name,
                points=points[start : start + self._UPSERT_BATCH],
                wait=True,
            )

    def _existing_ids(self) -> set[str]:
        """已索引 doc_id 集合（scroll 全量点 payload，建索引增量判断用一次）。"""
        if not self._collection_exists():
            return set()
        from qdrant_client import models

        ids: set[str] = set()
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=1024,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            ids.update(r.payload.get("doc_id", "") for r in records)
            if offset is None:
                break
        return ids

    # ---------- 查询 ----------
    def similarity_search(self, query: str, k: int = 5, query_vector=None) -> list[Document]:
        if not self._collection_exists():
            return []
        if query_vector is None:
            query_vector = self.embedding_function.embed_query(query)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=[float(x) for x in query_vector],
            limit=k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["doc_id"] = payload.get("doc_id", "")
            results.append(Document(page_content=payload.get("content", ""), metadata=metadata))
        return results

    # ---------- 元数据/生命周期 ----------
    def get_meta(self, key: str) -> str:
        if not self._client.collection_exists(self._META_COLLECTION):
            return ""
        from qdrant_client import models

        try:
            response = self._client.retrieve(
                collection_name=self._META_COLLECTION,
                ids=[self._point_id(f"{self.collection_name}:{key}")],
                with_payload=True,
            )
            return response[0].payload.get("value", "") if response else ""
        except Exception:
            return ""

    def set_meta(self, key: str, value: str) -> None:
        from qdrant_client import models

        if not self._client.collection_exists(self._META_COLLECTION):
            # dim=1 占位向量（meta 点只关心 payload）
            self._client.create_collection(
                collection_name=self._META_COLLECTION,
                vectors_config=models.VectorParams(size=1, distance=models.Distance.COSINE),
            )
        self._client.upsert(
            collection_name=self._META_COLLECTION,
            points=[
                models.PointStruct(
                    id=self._point_id(f"{self.collection_name}:{key}"),
                    vector=[0.0],
                    payload={"collection": self.collection_name, "key": key, "value": value},
                )
            ],
            wait=True,
        )

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        return self._client.count(collection_name=self.collection_name, exact=True).count

    def clear(self) -> None:
        if self._collection_exists():
            self._client.delete_collection(self.collection_name)
            logger.info(f"[{self.collection_name}] Qdrant collection 已删除（全量重建前置）")

    def close(self) -> None:
        # local 客户端进程级共享（多 collection 复用），不在此关闭
        pass


# =====================================================================
# 工厂与跨后端工具
# =====================================================================
def create_vector_store(
    collection_name: str, db_path: Path, embedding_function: Embeddings
) -> VectorStoreBase:
    """按 settings.vector_backend 构造向量库（上层唯一入口）。

    - sqlite（默认）：零依赖兜底，行为与历史版本完全一致
    - qdrant：需 pip install qdrant-client；local 模式无服务进程
    - milvus：预留插槽——实现 MilvusVectorStore(VectorStoreBase) 后在此注册即可，
      上层代码（knowledge_manager / 检索器 / 迁移脚本）零改动
    """
    from app.core.config import settings

    backend = (settings.vector_backend or "sqlite").strip().lower()
    if backend == "sqlite":
        return SqliteVectorStore(collection_name, db_path, embedding_function)
    if backend == "qdrant":
        try:
            import qdrant_client  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "VECTOR_BACKEND=qdrant 需要 qdrant-client：pip install qdrant-client"
            ) from exc
        return QdrantVectorStore(collection_name, db_path, embedding_function)
    if backend == "milvus":
        raise NotImplementedError(
            "Milvus 后端尚未接入：实现 app/rag/vector_store.py 中的 "
            "MilvusVectorStore(VectorStoreBase)（add_documents/add_embeddings/"
            "similarity_search/get_meta/set_meta/count/clear），并在本工厂注册 "
            "'milvus' 分支即可，上层零改动。pymilvus 的 collection/upsert/search "
            "与 Qdrant 概念一一对应，可参照 QdrantVectorStore 实现。"
        )
    raise ValueError(f"未知的向量库后端: {backend}（可选 sqlite/qdrant/milvus）")


def drop_collection_storage(collection_name: str) -> None:
    """删除某个 collection 在所有 embedding 指纹目录下的物理存储（文件删除联动）。"""
    from app.core.config import settings

    backend = (settings.vector_backend or "sqlite").strip().lower()
    if backend == "qdrant":
        # qdrant：local 模式删 collection（逐指纹目录的共享客户端）；服务端模式删一次
        if settings.qdrant_url:
            QdrantVectorStore._shared_client(
                settings.qdrant_url, {"url": settings.qdrant_url}
            ).delete_collection(collection_name, wait=True)
        else:
            for fingerprint_dir in (_ for _ in settings.vector_store_dir.iterdir() if _.is_dir()):
                root = fingerprint_dir / "qdrant_local"
                if root.exists():
                    client = QdrantVectorStore._shared_client(str(root.resolve()), {"path": str(root)})
                    if client.collection_exists(collection_name):
                        client.delete_collection(collection_name)
        return

    # sqlite 默认：删各指纹目录下的库文件（历史行为）
    if settings.vector_store_dir.exists():
        for fingerprint_dir in settings.vector_store_dir.iterdir():
            target = fingerprint_dir / f"{collection_name}.sqlite"
            if target.exists():
                target.unlink()


def rebuild_if_changed(store: VectorStoreBase, documents: list[Document], file_fingerprint: str) -> None:
    """文件指纹变化时全量重建（指纹存 collection 元数据；doc_id 保持内容哈希不变，
    以维持向量/BM25/关键词三路检索的跨路对齐）。面向基类契约，与后端无关。"""
    stored_fingerprint = store.get_meta("fingerprint")

    if stored_fingerprint != file_fingerprint:
        store.clear()
        store.set_meta("fingerprint", file_fingerprint)
        logger.info(f"[{store.collection_name}] 索引按指纹 {file_fingerprint[:12]} 全量重建")

    store.add_documents(documents, ids=[d.metadata["doc_id"] for d in documents])
