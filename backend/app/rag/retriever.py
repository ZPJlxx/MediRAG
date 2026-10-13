"""混合检索器（迁移自 general-doc-qa-assistant 的 KeywordPriorityRetriever）。

三路信号融合，这是本项目 RAG 质量的核心：
1. 向量检索（Chroma，语义相似）
2. BM25（关键词精确命中；本次迁移加入 jieba 分词预处理，修复原实现对中文
   按“空格分词”导致 BM25 几乎退化为全词匹配的问题）
3. 本地关键词打分（倒排索引缩小候选，避免全量扫描）
融合方式：RRF 只看排名不看分数，天然规避分数量纲问题。
"""

import hashlib
import re

import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import ConfigDict, PrivateAttr


def normalize_retrieval_text(text: str) -> str:
    """问题与 chunk 的统一归一化，提升中文文本命中稳定性。"""
    normalized_text = re.sub(r"\s+", "", str(text or "").lower())
    filler_patterns = [
        "请问", "一下", "一下子", "是什么", "是啥", "多少", "怎么", "如何",
        "有没有", "是否", "呢", "吗", "呀", "啊", "吧", "的",
    ]
    for pattern in filler_patterns:
        normalized_text = normalized_text.replace(pattern, "")
    return re.sub(r"[^\w\u4e00-\u9fff]", "", normalized_text)


def build_query_keywords(question: str):
    """从问题里提取适合本地打分的短语（多粒度滑窗）。"""
    normalized_question = normalize_retrieval_text(question)
    keywords = set()
    if len(normalized_question) >= 2:
        keywords.add(normalized_question)
    for size in (8, 6, 5, 4, 3, 2):
        if len(normalized_question) < size:
            continue
        for index in range(len(normalized_question) - size + 1):
            keywords.add(normalized_question[index : index + size])
    return sorted(keywords, key=len, reverse=True)


def score_document_by_keywords(question: str, document: Document, normalized_document: str | None = None) -> int:
    """把真正包含问句关键词的 chunk 顶上来。"""
    normalized_question = normalize_retrieval_text(question)
    if normalized_document is None:
        normalized_document = normalize_retrieval_text(getattr(document, "page_content", ""))
    if not normalized_question or not normalized_document:
        return 0

    score = 0
    if normalized_question in normalized_document:
        score += 20
    for keyword in build_query_keywords(question):
        if keyword not in normalized_document:
            continue
        if len(keyword) >= 6:
            score += 8
        elif len(keyword) >= 4:
            score += 5
        elif len(keyword) == 3:
            score += 3
        else:
            score += 1
    return score


def jieba_preprocess(text: str) -> list[str]:
    """BM25 中文分词预处理（本迁移新增：原实现用默认按空格切分，中文失效）。"""
    return [token for token in jieba.lcut(text.lower().strip()) if token.strip()]


def build_bm25_retriever(documents: list[Document], k: int = 8) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(documents, preprocess_func=jieba_preprocess)
    retriever.k = k
    return retriever


class KeywordPriorityRetriever(BaseRetriever):
    """混合检索器：向量 + BM25(jieba) + 本地关键词打分，RRF 融合，可选 rerank。

    大库优化：chunk 数超过 KEYWORD_INDEX_MAX_DOCS（默认 3,000）时跳过
    多粒度滑窗倒排索引的构建（其构建与查询复杂度在大库上不可接受），
    keyword 路退化为「查询短语直接包含匹配 + 召回集重打分」，
    精度略降但延迟从分钟级回到亚秒级；小库保持完整倒排。
    """

    documents: list[Document]
    vector_retriever: object
    bm25_retriever: object = None
    reranker: object = None
    top_k: int = 5
    rrf_k: int = 60
    keyword_index_max_docs: int = 3000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _normalized_contents: list = PrivateAttr(default=None)
    _keyword_index: dict = PrivateAttr(default=None)
    _doc_id_to_index: dict = PrivateAttr(default=None)

    @staticmethod
    def _doc_key(document) -> str:
        metadata = getattr(document, "metadata", {}) or {}
        doc_id = metadata.get("doc_id")
        if doc_id:
            return doc_id
        return hashlib.sha1(
            (getattr(document, "page_content", "") or "").encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _rrf_fusion(*rank_lists, k=60) -> dict:
        """RRF：score(doc) = sum(1/(k+rank))，只看排名不看分数。"""
        doc_scores = {}
        for rank_list in rank_lists:
            for rank, doc in enumerate(rank_list):
                key = KeywordPriorityRetriever._doc_key(doc)
                doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank + 1)
        return doc_scores

    def _ensure_keyword_index(self):
        if self._keyword_index is not None:
            return
        # 大库跳过滑窗倒排（构建 O(N·L·5) 且索引巨大），keyword 走轻量包含匹配
        if len(self.documents) > self.keyword_index_max_docs:
            self._normalized_contents = [
                normalize_retrieval_text(getattr(d, "page_content", ""))
                for d in self.documents
            ]
            self._keyword_index = {}  # 空索引：_candidate_indices 自然返回空集
            return
        normalized_contents = []
        keyword_index = {}
        for index, document in enumerate(self.documents):
            content = normalize_retrieval_text(getattr(document, "page_content", ""))
            normalized_contents.append(content)
            if not content:
                continue
            seen = set()
            length = len(content)
            for size in (6, 5, 4, 3, 2):
                if length < size:
                    continue
                for start in range(length - size + 1):
                    token = content[start : start + size]
                    if token in seen:
                        continue
                    seen.add(token)
                    keyword_index.setdefault(token, []).append(index)
        self._normalized_contents = normalized_contents
        self._keyword_index = keyword_index

    def _ensure_doc_id_to_index(self):
        if self._doc_id_to_index is not None:
            return self._doc_id_to_index
        mapping = {}
        for index, document in enumerate(self.documents):
            key = self._doc_key(document)
            if key not in mapping:
                mapping[key] = index
        self._doc_id_to_index = mapping
        return mapping

    def _candidate_indices(self, query):
        self._ensure_keyword_index()
        candidates = set()
        for keyword in build_query_keywords(query):
            matched = self._keyword_index.get(keyword)
            if matched:
                candidates.update(matched)
        return candidates

    def invoke_with_vector(self, query: str, query_vector) -> list[Document]:
        """外部入口：透传预算好的查询向量，绕过 BaseRetriever.invoke 的签名限制。"""
        return self._get_relevant_documents(query, query_vector=query_vector)

    def _get_relevant_documents(self, query: str, *, run_manager=None, query_vector=None):
        # 第1路：向量检索（失败降级为空，不阻断）。query_vector 由聚合层预算传入，
        # 避免多文件检索时对同一查询重复调用 embedding API。
        try:
            vector_documents = self.vector_retriever.invoke(query, query_vector=query_vector)
        except TypeError:
            # 兼容不接受 query_vector 的检索器实现
            try:
                vector_documents = self.vector_retriever.invoke(query)
            except Exception as exc:
                logger.warning(f"向量检索失败: {exc}")
                vector_documents = []
        except Exception as exc:
            logger.warning(f"向量检索失败: {exc}")
            vector_documents = []

        # 第2路：BM25（jieba 分词）
        bm25_documents = []
        if self.bm25_retriever:
            try:
                bm25_documents = self.bm25_retriever.invoke(query)
            except Exception as exc:
                logger.warning(f"BM25检索失败: {exc}")

        recalled_documents = list(vector_documents) + list(bm25_documents)
        rrf_scores = self._rrf_fusion(vector_documents, bm25_documents, k=self.rrf_k)

        # 第3路：本地关键词打分（倒排候选 ∪ 召回结果，避免全量扫描）
        self._ensure_keyword_index()
        doc_id_to_index = self._ensure_doc_id_to_index()
        large_corpus = len(self.documents) > self.keyword_index_max_docs
        if large_corpus:
            # 大库：跳过全量滑窗候选，只对「两路召回集 + 查询短语直接包含」打分
            candidate_indices = set()
            probe = normalize_retrieval_text(query)
            for index, content in enumerate(self._normalized_contents):
                if probe and probe in content:
                    candidate_indices.add(index)
        else:
            candidate_indices = self._candidate_indices(query)
        for document in recalled_documents:
            mapped_index = doc_id_to_index.get(self._doc_key(document))
            if mapped_index is not None:
                candidate_indices.add(mapped_index)

        ranked_documents = []
        for index in candidate_indices:
            document = self.documents[index]
            keyword_score = score_document_by_keywords(
                query, document, normalized_document=self._normalized_contents[index]
            )
            rrf_score = rrf_scores.get(self._doc_key(document), 0) * 100
            merged_score = keyword_score + rrf_score
            if merged_score > 0:
                ranked_documents.append((merged_score, index, document))

        if not ranked_documents:
            ranked_documents = [
                (rrf_scores.get(self._doc_key(doc), 0) * 100, index, doc)
                for index, doc in enumerate(recalled_documents)
                if rrf_scores.get(self._doc_key(doc), 0) > 0
            ]

        ranked_documents.sort(key=lambda item: (-item[0], item[1]))
        selected_documents = [
            doc for _score, _index, doc in ranked_documents[: max(self.top_k * 2, self.top_k)]
        ]

        # 可选 rerank 精排，失败回退
        if self.reranker and selected_documents:
            try:
                reranked_documents = self.reranker.compress_documents(selected_documents, query)
                if reranked_documents:
                    return list(reranked_documents[: self.top_k])
            except Exception as exc:
                logger.warning(f"重排序失败，回退混合检索结果: {exc}")

        return selected_documents[: self.top_k]
