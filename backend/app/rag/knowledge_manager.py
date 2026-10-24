"""知识库管理器：文件管理、索引构建、跨文件聚合检索。

迁移并增强自 general-doc-qa-assistant 的 MyKnowledge：
- 去除 Gradio 依赖，改为纯服务层
- 向量库为自实现 SqliteVectorStore（SQLite + numpy 余弦）：
  chromadb 的原生 wheel 在当前机器段错误，纯 Python 实现零原生依赖且同等持久化
- delete_file：联动清理文件、向量库文件、检索器缓存（原项目缺失删除能力）
- search()：跨全部知识文件的聚合检索（医疗 Agent 检索入口），
  各文件先按混合检索召回，再全局去重 + 可选全局 rerank
"""

import hashlib
import shutil
import threading
from copy import deepcopy
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from loguru import logger

from app.core.config import settings
from app.rag.evidence import document_to_citation
from app.rag.loader import SUPPORTED_KNOWLEDGE_SUFFIXES, MyCustomLoader
from app.rag.model_clients import (
    embedding_fingerprint,
    get_embedding_client,
    get_retrieval_runtime_status,
    get_rerank_client,
)
from app.rag.retriever import KeywordPriorityRetriever, build_bm25_retriever
from app.rag.vector_store import create_vector_store, drop_collection_storage, rebuild_if_changed
from app.schemas.medical import Citation


def get_md5(input_string: str) -> str:
    hash_md5 = hashlib.md5()
    hash_md5.update(input_string.encode("utf-8"))
    return hash_md5.hexdigest()


def build_document_id(document: Document) -> str:
    """chunk 稳定唯一 id：来源 + 起始位置 + 内容的哈希（跨检索路对齐用）。"""
    source = document.metadata.get("source", "")
    start_index = document.metadata.get("start_index")
    page_content = document.page_content or ""
    raw = "::".join([str(source), "" if start_index is None else str(start_index), page_content])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_documents(documents: list[Document]) -> list[Document]:
    """metadata 规范化 + 补充稳定 doc_id（写入 Chroma 后检索时可原样取回）。"""
    normalized_documents = []
    for document in documents:
        normalized_document = deepcopy(document)
        normalized_document.metadata = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in normalized_document.metadata.items()
        }
        normalized_document.metadata["doc_id"] = build_document_id(normalized_document)
        normalized_documents.append(normalized_document)
    return normalized_documents


def create_indexes(
    collection_name: str,
    loader: BaseLoader,
    embedding_function: Embeddings,
    embedding_fingerprint_value: str,
    reranker=None,
) -> KeywordPriorityRetriever:
    """单文件建索引：向量库（可切后端）+ BM25 -> 混合检索器。

    文件指纹（路径+mtime+大小）变化时全量重建，否则增量补录。
    """
    store = create_vector_store(
        collection_name=collection_name,
        db_path=settings.vector_store_dir / embedding_fingerprint_value / f"{collection_name}.sqlite",
        embedding_function=embedding_function,
    )

    documents = normalize_documents(loader.load())
    logger.info(f"知识文档切分完成: {len(documents)} chunks")

    source = str(getattr(loader, "file_path", collection_name))
    try:
        stat = Path(source).stat()
        file_fingerprint = hashlib.sha1(
            f"{source}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
        ).hexdigest()
    except OSError:
        file_fingerprint = "unknown"

    rebuild_if_changed(store, documents, file_fingerprint)

    vector_retriever = store.as_retriever(search_kwargs={"k": 8})
    keyword_retriever = build_bm25_retriever(documents, k=8)

    return KeywordPriorityRetriever(
        documents=documents,
        vector_retriever=vector_retriever,
        bm25_retriever=keyword_retriever,
        reranker=reranker,
        top_k=5,
    )


class KnowledgeManager:
    """知识库单例：检索器缓存 + 文件生命周期 + 跨文件聚合检索。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._retrievers: dict[str, KeywordPriorityRetriever] = {}
        self._embeddings = None
        self._reranker = None
        self._runtime_signature = None
        self._retriever_lock = threading.Lock()

    # ---------- 检索运行时 ----------
    def _ensure_runtime(self):
        settings.ensure_runtime_directories()
        status = get_retrieval_runtime_status()
        signature = (
            status["embedding"]["fingerprint"],
            status["rerank"]["provider"],
            status["rerank"]["model"],
            status["rerank"]["enabled"],
            status["rerank"]["available"],
        )
        if self._runtime_signature == signature:
            return status

        self._runtime_signature = signature
        self._retrievers = {}

        if not status["knowledge_ready"]:
            self._embeddings = None
            self._reranker = None
            logger.warning(status["embedding"]["status_message"])
            return status

        self._embeddings = get_embedding_client()
        if status["rerank"]["enabled"] and status["rerank"]["available"]:
            try:
                self._reranker = get_rerank_client(top_n=5)
            except Exception as exc:
                logger.warning(f"rerank 客户端初始化失败，降级关闭: {exc}")
                self._reranker = None
        else:
            self._reranker = None

        logger.info(
            f"检索运行时就绪: embedding={status['embedding']['provider']}::{status['embedding']['model']}, "
            f"rerank={'on' if self._reranker else 'off'}"
        )
        return status

    def runtime_status(self) -> dict:
        return get_retrieval_runtime_status()

    # ---------- 文件管理 ----------
    def list_files(self) -> list[str]:
        """列出全部知识文件（含子目录），返回相对 knowledge_dir 的 POSIX 路径。

        子目录用于语料分组（guidelines/ 指南、huatuo/ 百科QA），collection 名
        取 md5(相对路径)，天然与根目录文件名空间隔离。
        """
        settings.ensure_runtime_directories()
        results = []
        for path in settings.knowledge_dir.rglob("*"):
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in SUPPORTED_KNOWLEDGE_SUFFIXES
            ):
                results.append(path.relative_to(settings.knowledge_dir).as_posix())
        return sorted(results)

    def save_uploaded_file(self, file_name: str, data: bytes) -> Path:
        """保存上传文件到知识目录根（同名覆盖），并失效对应检索器缓存。"""
        settings.ensure_runtime_directories()
        safe_name = Path(file_name).name  # 防路径穿越
        file_path = settings.knowledge_dir / safe_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        with self._retriever_lock:
            self._retrievers.pop(get_md5(safe_name.as_posix()), None)
        logger.info(f"知识文件已保存: {safe_name} ({len(data)} bytes)")
        return file_path

    def delete_file(self, file_name: str) -> bool:
        """删除知识文件（支持子目录相对路径），联动清理向量库文件与检索器缓存。"""
        rel_path = Path(file_name).as_posix()
        file_path = (settings.knowledge_dir / rel_path).resolve()
        # 防路径穿越：解析后的目标必须仍在知识目录内（与 save_file 的 Path.name 防护同级）
        if not file_path.is_relative_to(settings.knowledge_dir.resolve()):
            logger.warning(f"拒绝删除知识目录外的路径: {file_name}")
            return False
        if not file_path.exists():
            return False

        collection_name = get_md5(rel_path)
        file_path.unlink()
        # 清掉空掉的父目录（如子分组被清空）
        try:
            parent = file_path.parent
            if parent != settings.knowledge_dir and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
        with self._retriever_lock:
            self._retrievers.pop(collection_name, None)

        # 清理所有 embedding 指纹目录下该 collection 的向量存储（按后端分发：
        # sqlite 删库文件 / qdrant 删 collection）
        drop_collection_storage(collection_name)

        logger.info(f"知识文件已删除: {rel_path}")
        return True

    # ---------- 检索 ----------
    def get_retriever(self, file_name: str) -> KeywordPriorityRetriever | None:
        """按文件名（支持子目录相对路径）取（或惰性构建）混合检索器。"""
        status = self._ensure_runtime()
        if self._embeddings is None:
            raise ValueError(status["embedding"]["status_message"])

        rel_path = Path(file_name).as_posix()
        collection_name = get_md5(rel_path)
        with self._retriever_lock:
            retriever = self._retrievers.get(collection_name)
            if retriever is not None:
                return retriever

            file_path = settings.knowledge_dir / file_name
            if not file_path.exists():
                return None

            loader = MyCustomLoader(file_path)
            retriever = create_indexes(
                collection_name,
                loader,
                self._embeddings,
                embedding_fingerprint_value=status["embedding"]["fingerprint"],
                reranker=self._reranker,
            )
            self._retrievers[collection_name] = retriever
            return retriever

    def search(self, query: str, top_k: int | None = None) -> list[Document]:
        """跨全部知识文件聚合检索（医疗 Agent 的统一入口）。

        各文件独立混合检索召回 -> doc_id 全局去重 -> 可选全局 rerank -> 截取 top_k。
        embed-once：查询向量只算一次，透传给每个文件的向量检索器，
        避免 N 个文件 × 各自调用 embedding API。
        """
        top_k = top_k or settings.rag_top_k
        merged: list[Document] = []
        seen: set[str] = set()

        query_vector = None
        try:
            query_vector = self._embeddings.embed_query(query)
        except Exception as exc:
            logger.warning(f"查询向量化失败，各文件回退自行 embed: {exc}")

        # 文件间并发检索：各 collection 检索器相互独立（矩阵缓存/倒排懒加载均线程安全，
        # 矩阵加载有锁保护，加载一次后只读）。首次触碰未建好的检索器仍会串行建索引。
        from concurrent.futures import ThreadPoolExecutor

        file_names = self.list_files()

        def _search_one(file_name: str) -> list[Document]:
            try:
                retriever = self.get_retriever(file_name)
            except ValueError as exc:
                logger.warning(f"知识文件 {file_name} 检索器不可用: {exc}")
                return []
            if retriever is None:
                return []
            retriever.top_k = top_k
            try:
                if query_vector is not None:
                    return retriever.invoke_with_vector(query, query_vector)
                return retriever.invoke(query)
            except Exception as exc:
                logger.warning(f"知识文件 {file_name} 检索失败（跳过）: {exc}")
                return []

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(file_names)))) as pool:
            for documents in pool.map(_search_one, file_names):
                for document in documents:
                    doc_id = document.metadata.get("doc_id") or build_document_id(document)
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    merged.append(document)

        if self._reranker and len(merged) > top_k:
            try:
                reranked = self._reranker.compress_documents(merged, query)
                if reranked:
                    merged = list(reranked)
            except Exception as exc:
                logger.warning(f"全局 rerank 失败，保持聚合顺序: {exc}")

        return merged[:top_k]

    def search_citations(self, query: str, top_k: int | None = None) -> list[Citation]:
        """检索并直接转换为证据引用列表（Agent 注入用）。"""
        documents = self.search(query, top_k=top_k)
        citations = []
        seen: set[str] = set()
        for document in documents:
            citation = document_to_citation(document, query)
            key = (citation.source, citation.location, citation.snippet[:80])
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
        return citations


knowledge_manager = KnowledgeManager()
