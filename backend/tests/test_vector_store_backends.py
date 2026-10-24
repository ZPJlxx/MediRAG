"""向量库后端抽象与 Qdrant 适配器测试（假向量，不触 embedding API）。

local 嵌入式 Qdrant 落在 tmp_path 下，测试间完全隔离；
sqlite 后端同时复测一遍保证两实现行为一致（契约测试）。
"""

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """确定性假向量：文档内容哈希派生，query/doc 同源可复现。"""

    def _vec(self, text: str) -> list[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=8).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _make_docs(n=5, prefix="doc"):
    return [
        Document(
            page_content=f"{prefix}-content-{i} 关于高血压的诊断标准内容",
            metadata={"doc_id": f"{prefix}-id-{i}", "source": f"{prefix}.md", "start_index": i * 100},
        )
        for i in range(n)
    ]


@pytest.fixture()
def sqlite_store(tmp_path):
    from app.rag.vector_store import SqliteVectorStore

    return SqliteVectorStore("t_col", tmp_path / "fp1" / "t_col.sqlite", FakeEmbeddings())


@pytest.fixture()
def qdrant_store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.core.config.settings.qdrant_url", "", raising=False)
    from app.rag.vector_store import QdrantVectorStore

    return QdrantVectorStore("t_col", tmp_path / "fp1" / "t_col.sqlite", FakeEmbeddings())


@pytest.fixture(params=["sqlite", "qdrant"])
def store(request, sqlite_store, qdrant_store):
    """契约测试：两后端跑同一套用例。"""
    return {"sqlite": sqlite_store, "qdrant": qdrant_store}[request.param]


class TestContract:
    def test_add_and_search(self, store):
        docs = _make_docs(5)
        store.add_documents(docs)
        assert store.count() == 5

        target = docs[2]
        results = store.similarity_search(target.page_content, k=3)
        assert len(results) == 3
        assert results[0].metadata["doc_id"] == target.metadata["doc_id"]
        # 元数据完整往返
        assert results[0].metadata["source"] == "doc.md"
        assert results[0].page_content == target.page_content

    def test_incremental_skip_existing(self, store):
        docs = _make_docs(5)
        store.add_documents(docs)
        # 重复 add 同一批（doc_id 相同）不重复计费不重复写
        store.add_documents(_make_docs(5))
        assert store.count() == 5

    def test_add_documents_with_partial_new(self, store):
        store.add_documents(_make_docs(3))
        store.add_documents(_make_docs(5))  # 前 3 个已存在，只补 2 个
        assert store.count() == 5

    def test_add_embeddings_direct(self, store):
        """迁移脚本路径：现成向量直写，不调 embedding。"""
        vectors = [FakeEmbeddings()._vec(f"t{i}") for i in range(3)]
        store.add_embeddings(
            ["m-1", "m-2", "m-3"],
            [f"migrated text {i}" for i in range(3)],
            [{"doc_id": f"m-{i+1}", "source": "m.pdf"} for i in range(3)],
            vectors,
        )
        assert store.count() == 3
        results = store.similarity_search("migrated text 1", k=1, query_vector=vectors[0])
        assert results[0].metadata["doc_id"] == "m-1"

    def test_meta_roundtrip(self, store):
        assert store.get_meta("fingerprint") == ""
        store.set_meta("fingerprint", "abc123")
        assert store.get_meta("fingerprint") == "abc123"
        store.set_meta("fingerprint", "def456")  # 覆盖
        assert store.get_meta("fingerprint") == "def456"

    def test_clear(self, store):
        store.add_documents(_make_docs(3))
        store.set_meta("fingerprint", "abc")
        store.clear()
        assert store.count() == 0
        # clear 只清向量不清指纹（rebuild_if_changed 依赖该语义）
        assert store.get_meta("fingerprint") == "abc"

    def test_rebuild_if_changed(self, store):
        from app.rag.vector_store import rebuild_if_changed

        docs = _make_docs(3)
        rebuild_if_changed(store, docs, "fp-aaa")
        assert store.count() == 3
        rebuild_if_changed(store, docs, "fp-aaa")  # 指纹不变 -> 增量（无新）
        assert store.count() == 3
        rebuild_if_changed(store, _make_docs(5), "fp-bbb")  # 指纹变 -> 全量重建
        assert store.count() == 5

    def test_empty_search_before_index(self, store):
        assert store.similarity_search("任何查询", k=3) == []

    def test_as_retriever(self, store):
        docs = _make_docs(3)
        store.add_documents(docs)
        retriever = store.as_retriever(search_kwargs={"k": 2})
        results = retriever.invoke(docs[1].page_content)
        assert len(results) == 2
        assert results[0].metadata["doc_id"] == docs[1].metadata["doc_id"]


class TestFactory:
    def test_default_sqlite(self, tmp_path, monkeypatch):
        from app.rag.vector_store import SqliteVectorStore, create_vector_store

        monkeypatch.setattr("app.core.config.settings.vector_backend", "sqlite", raising=False)
        store = create_vector_store("c", tmp_path / "c.sqlite", FakeEmbeddings())
        assert isinstance(store, SqliteVectorStore)

    def test_qdrant_backend(self, tmp_path, monkeypatch):
        from app.rag.vector_store import QdrantVectorStore, create_vector_store

        monkeypatch.setattr("app.core.config.settings.vector_backend", "qdrant", raising=False)
        monkeypatch.setattr("app.core.config.settings.qdrant_url", "", raising=False)
        store = create_vector_store("c", tmp_path / "fp" / "c.sqlite", FakeEmbeddings())
        assert isinstance(store, QdrantVectorStore)
        store.add_documents(_make_docs(2))
        assert store.count() == 2

    def test_milvus_slot_not_implemented(self, tmp_path, monkeypatch):
        from app.rag.vector_store import create_vector_store

        monkeypatch.setattr("app.core.config.settings.vector_backend", "milvus", raising=False)
        with pytest.raises(NotImplementedError, match="MilvusVectorStore"):
            create_vector_store("c", tmp_path / "c.sqlite", FakeEmbeddings())

    def test_unknown_backend(self, tmp_path, monkeypatch):
        from app.rag.vector_store import create_vector_store

        monkeypatch.setattr("app.core.config.settings.vector_backend", "redis", raising=False)
        with pytest.raises(ValueError, match="未知的向量库后端"):
            create_vector_store("c", tmp_path / "c.sqlite", FakeEmbeddings())


class TestQdrantPointIdDeterminism:
    def test_point_id_stable(self):
        from app.rag.vector_store import QdrantVectorStore

        a = QdrantVectorStore._point_id("doc-abc")
        b = QdrantVectorStore._point_id("doc-abc")
        c = QdrantVectorStore._point_id("doc-xyz")
        assert a == b and a != c
