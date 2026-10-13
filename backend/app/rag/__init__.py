"""RAG 包公开接口。"""

from app.rag.evidence import (
    Citation as CitationDTO,  # noqa: F401  便捷重导出
    document_to_citation,
    format_reference_block,
    map_llm_citations,
)
from app.rag.knowledge_manager import knowledge_manager  # noqa: F401

__all__ = [
    "knowledge_manager",
    "document_to_citation",
    "format_reference_block",
    "map_llm_citations",
    "CitationDTO",
]
