"""患者端「智慧医疗诊疗聊天」编排服务。

链路：会话历史 -> 意图识别（规则短路 + LLM 分类）-> 按意图分流 -> 落库审计。

分流策略（合规优先）：
- chitchat  友好回应，介绍能力，不检索（省配额）
- consult   RAG 检索 -> 参考资料注入 -> 带引用的科普解答
- inquiry   RAG 检索 -> 科普说明 + 就医引导（推荐科室/红旗症状），严禁诊断结论
- emergency 固定警示文案（不经过 LLM 生成，内容可控、零延迟、零配额）

引用防编造复用医生端同一套机制：search_citations -> format_reference_block
（编号参考资料）-> LLM 声明引用编号 -> map_llm_citations 映射回真实证据，
编号越界一律丢弃，患者端展示的每条出处都来自真实检索结果。
"""

import uuid

from loguru import logger

from app.dao.medical import PatientChatDAO
from app.core.database import SessionLocal
from app.medical.intent import intent_classifier
from app.medical.llm_client import llm_client
from app.rag.evidence import format_reference_block, map_llm_citations
from app.rag.knowledge_manager import knowledge_manager

# 急症警示固定文案（不经过 LLM，内容完全可控）
EMERGENCY_REPLY = (
    "⚠️ 您描述的情况可能属于急症征象，请立即行动：\n\n"
    "1. 马上拨打 120 急救电话，或让身边人陪同前往最近医院的急诊科；\n"
    "2. 不要自行开车前往，途中保持镇静，解开领口保持呼吸通畅；\n"
    "3. 如有随身药物（如硝酸甘油、哮喘吸入剂）按既往医嘱使用；\n"
    "4. 昏迷或呕吖的患者应侧卧，防止窒息。\n\n"
    "本助手无法处理急症，请以线下急救为准。"
)

# 免责声明（所有回复必带，合规）
DISCLAIMER = "本回复由 AI 助手生成，仅供健康科普参考，不能替代执业医师的诊断和治疗建议。"


class PatientChatService:
    """患者端聊天编排：一次 handle_chat 处理一条患者消息。"""

    def handle_chat(self, session_key: str, message: str, client_ip: str = "") -> dict:
        # 会话定位/创建 + 历史拉取（多轮上下文）
        session = SessionLocal()
        try:
            chat_session = PatientChatDAO.get_or_create_session(
                session, session_key, client_ip
            )
            history_rows = PatientChatDAO.list_messages(session, chat_session.id, limit=12)
            history = [{"role": r.role, "content": r.content} for r in history_rows]
        finally:
            session.close()

        # 意图识别（规则快路径可零 LLM 成本短路）
        intent_result = intent_classifier.classify(message, history)
        intent = intent_result["intent"]
        logger.info(
            f"[PATIENT_CHAT] session={session_key[:8]}… intent={intent}"
            f"{'(rule)' if intent_result['by_rule'] else '(llm)'}"
        )

        # 按意图分流生成回复
        if intent == "emergency":
            reply, citations = EMERGENCY_REPLY, []
        else:
            reply, citations = self._generate_reply(message, history, intent)

        # 落库审计：患者消息（带意图标签）+ 助手回复（带引用）
        session = SessionLocal()
        try:
            PatientChatDAO.add_message(
                session, chat_session.id, "user", message, intent=intent
            )
            import json as _json

            PatientChatDAO.add_message(
                session, chat_session.id, "assistant", reply,
                citations_json=_json.dumps(
                    [c.model_dump() for c in citations], ensure_ascii=False
                ),
            )
        finally:
            session.close()

        return {
            "session_id": session_key,
            "intent": intent,
            "intent_label": intent_result["intent_label"],
            "confidence": intent_result["confidence"],
            "by_rule": intent_result["by_rule"],
            "reply": reply,
            "citations": [c.model_dump() for c in citations],
            "disclaimer": DISCLAIMER,
        }

    def _generate_reply(self, message: str, history: list[dict], intent: str) -> tuple[str, list]:
        """chitchat 轻量回应；consult/inquiry 走 RAG 答疑（引用防编造）。"""
        reference_block = ""
        evidence = []
        if intent in ("consult", "inquiry"):
            evidence = self._safe_search(message)
            reference_block = format_reference_block(evidence)

        messages = history + [{"role": "user", "content": message}]
        try:
            result = llm_client.patient_chat_reply(messages, intent, reference_block)
        except Exception as exc:
            logger.error(f"患者聊天回复生成失败: {exc}")
            return (
                "抱歉，助手暂时开小差了，请稍后再试。如果症状持续或加重，建议直接前往医院就诊。",
                [],
            )

        raw_citations = result.get("citations") or []
        citations = map_llm_citations(raw_citations, evidence)
        return str(result.get("reply", "")).strip(), citations
    @staticmethod
    def _safe_search(query: str) -> list:
        """检索失败降级为无参考（回复仍可生成，仅失去引用支撑）。"""
        try:
            return knowledge_manager.search_citations(query)
        except Exception as exc:
            logger.warning(f"患者聊天检索失败（降级无参考）: {exc}")
            return []


patient_chat_service = PatientChatService()


def new_session_key() -> str:
    """签发新会话标识（首次进入聊天页由后端生成，前端存 localStorage）。"""
    return f"pchat_{uuid.uuid4().hex[:16]}"
