"""证据溯源：把检索命中转换为结构化 Citation，并构建注入 Prompt 的参考资料块。

复用 general-doc-qa-assistant 的位置标签（location_label）与命中窗口截取思路，
这是医疗场景「每条建议给得出处」的关键。
"""

from pathlib import Path

from langchain_core.documents import Document

from app.schemas.medical import Citation

SNIPPET_MAX_CHARS = 320


def _extract_snippet(content: str, query: str | None = None) -> str:
    """截取命中窗口：优先围绕查询关键词，否则取片段开头。"""
    content = (content or "").strip()
    if not content:
        return ""
    if not query:
        return content[:SNIPPET_MAX_CHARS]

    # 用查询中最长的若干字符子串定位命中位置
    probe = query[: min(len(query), 12)]
    position = content.find(probe)
    if position < 0:
        probe = query[: min(len(query), 6)]
        position = content.find(probe)
    if position < 0:
        return content[:SNIPPET_MAX_CHARS]

    window_start = max(0, position - 60)
    window_end = min(len(content), position + SNIPPET_MAX_CHARS - 60)
    snippet = content[window_start:window_end]
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"


def document_to_citation(document: Document, query: str | None = None) -> Citation:
    """检索命中的 chunk -> Citation（来源文件名 + 位置标签 + 命中窗口）。"""
    metadata = getattr(document, "metadata", {}) or {}
    source = Path(str(metadata.get("source", ""))).name
    location = str(metadata.get("location_label", "") or "")
    snippet = _extract_snippet(getattr(document, "page_content", ""), query)
    return Citation(source=source, location=location, snippet=snippet)


def format_reference_block(citations: list[Citation]) -> str:
    """把证据列表格式化为 Prompt 注入用的编号参考资料块。

    LLM 被要求以 [编号] 形式引用；编号与列表顺序一一对应。
    """
    if not citations:
        return "（知识库中未检索到直接相关的参考资料）"
    lines = ["【参考资料（来自院内医学知识库，回答时必须以 [编号] 标注引用依据）】"]
    for index, citation in enumerate(citations, 1):
        location = f"（{citation.location}）" if citation.location else ""
        snippet = citation.snippet.replace("\n", " ").strip()
        lines.append(f"[{index}] 《{citation.source}》{location}：{snippet}")
    return "\n".join(lines)


def map_llm_citations(
    raw_citations: list, evidence: list[Citation]
) -> list[Citation]:
    """把 LLM 返回的引用声明（编号/quote）映射回真实证据对象。

    防幻觉关键：LLM 只能“选择”提供了的证据编号，不能凭空编造出处；
    非法编号一律丢弃。
    """
    mapped = []
    seen = set()
    for item in raw_citations or []:
        if isinstance(item, dict):
            ref = item.get("ref") or item.get("index") or item.get("source_index")
            quote = str(item.get("quote", "") or "")
        elif isinstance(item, int):
            ref, quote = item, ""
        else:
            continue
        try:
            ref_index = int(ref) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= ref_index < len(evidence) and ref_index not in seen:
            citation = evidence[ref_index].model_copy(deep=True)
            if quote:
                citation.snippet = quote[:SNIPPET_MAX_CHARS]
            mapped.append(citation)
            seen.add(ref_index)
    return mapped
