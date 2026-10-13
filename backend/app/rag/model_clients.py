"""检索侧模型客户端（迁移自 general-doc-qa-assistant 的 models.py 检索部分）。

保留：embedding 双通道（dashscope/openai）+ 指纹隔离 + 自封装 gte-rerank-v2。
聊天模型接入收敛到 core/config.py 的统一 LLM 网关，不再单独维护。
"""

import hashlib

from dashscope import TextReRank
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.core.config import settings


def _resolve_embedding_config() -> dict:
    provider = (settings.retrieval_embedding_provider or "dashscope").strip().lower()
    if provider == "dashscope":
        api_key = settings.retrieval_embedding_api_key_resolved()
        return {
            "provider": "dashscope",
            "model": settings.retrieval_embedding_model or "text-embedding-v3",
            "api_key": api_key,
            "available": bool(api_key),
            "status_message": (
                f"embedding 使用 DashScope {settings.retrieval_embedding_model}"
                if api_key
                else "embedding 配置为 DashScope，但缺少 API Key（RETRIEVAL_EMBEDDING_API_KEY / DASHSCOPE_API_KEY）"
            ),
        }
    if provider == "openai":
        api_key = settings.retrieval_embedding_api_key_resolved()
        return {
            "provider": "openai",
            "model": settings.retrieval_embedding_model or "text-embedding-3-small",
            "base_url": settings.retrieval_embedding_base_url or None,
            "api_key": api_key,
            "available": bool(api_key),
            "status_message": (
                f"embedding 使用 OpenAI 兼容 {settings.retrieval_embedding_model}"
                if api_key
                else "embedding 配置为 OpenAI 兼容，但缺少 API Key"
            ),
        }
    return {
        "provider": provider,
        "available": False,
        "status_message": f"不支持的 embedding provider：{provider}（仅支持 dashscope / openai）",
    }


def _resolve_rerank_config() -> dict:
    provider = (settings.retrieval_rerank_provider or "dashscope").strip().lower()
    if provider == "disabled":
        return {"provider": "disabled", "enabled": False, "available": False,
                "model": None, "api_key": None,
                "status_message": "rerank 已关闭"}
    if provider == "dashscope":
        api_key = settings.retrieval_rerank_api_key_resolved()
        return {
            "provider": "dashscope",
            "enabled": True,
            "model": settings.retrieval_rerank_model or "gte-rerank-v2",
            "api_key": api_key,
            "available": bool(api_key),
            "status_message": (
                f"rerank 使用 DashScope {settings.retrieval_rerank_model}"
                if api_key
                else "rerank 配置为 DashScope，但缺少 API Key（可设 RETRIEVAL_RERANK_PROVIDER=disabled 关闭）"
            ),
        }
    return {"provider": provider, "enabled": True, "available": False, "model": None,
            "api_key": None, "status_message": f"不支持的 rerank provider：{provider}"}


def get_retrieval_runtime_status() -> dict:
    embedding = _resolve_embedding_config()
    rerank = _resolve_rerank_config()
    return {
        "embedding": {
            "provider": embedding["provider"],
            "model": embedding.get("model"),
            "available": embedding["available"],
            "status_message": embedding["status_message"],
            "fingerprint": embedding_fingerprint(),
        },
        "rerank": {
            "provider": rerank["provider"],
            "model": rerank.get("model"),
            "enabled": rerank["enabled"],
            "available": rerank["available"],
            "status_message": rerank["status_message"],
        },
        "knowledge_ready": embedding["available"],
    }


def embedding_fingerprint() -> str:
    """与 embedding 配置绑定的稳定指纹：换模型自动隔离到新的 Chroma 目录，不串库。"""
    config = _resolve_embedding_config()
    raw = "|".join(
        [config["provider"], config.get("model") or "", config.get("base_url") or ""]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def get_embedding_client():
    config = _resolve_embedding_config()
    if not config["available"]:
        raise ValueError(config["status_message"])
    if config["provider"] == "dashscope":
        return DashScopeEmbeddings(model=config["model"], dashscope_api_key=config["api_key"])
    return OpenAIEmbeddings(
        model=config["model"], api_key=config["api_key"], base_url=config.get("base_url")
    )


class DashScopeRerankV2:
    """绕过 langchain_community 已失效的 DashScopeRerank，直连 gte-rerank-v2。

    失败时回退原顺序，保证检索链路始终可用。
    """

    def __init__(self, model: str = "gte-rerank-v2", top_n: int = 5, api_key: str | None = None):
        self.model = model
        self.top_n = top_n
        self.api_key = api_key

    def compress_documents(self, documents, query):
        if not documents:
            return []
        docs = [getattr(doc, "page_content", "") for doc in documents]
        try:
            result = TextReRank.call(
                model=self.model,
                query=query,
                documents=docs,
                top_n=min(self.top_n, len(docs)),
                return_documents=False,
                api_key=self.api_key,
            )
        except Exception:
            return documents

        if result is None or result.status_code != 200 or result.output is None:
            return documents

        reranked = []
        for res in result.output.results:
            if res.index < len(documents):
                reranked.append(documents[res.index])
        return reranked


def get_rerank_client(top_n: int = 5):
    config = _resolve_rerank_config()
    if not config["enabled"]:
        return None
    if not config["available"]:
        raise ValueError(config["status_message"])
    return DashScopeRerankV2(model=config["model"], api_key=config["api_key"], top_n=top_n)
