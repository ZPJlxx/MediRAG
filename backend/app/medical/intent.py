"""患者消息意图识别器（患者端「智慧医疗诊疗聊天」的分流中枢）。

意图识别的标准构建范式 = 规则快路径 + LLM 结构化分类 + 失败兜底：

1. 规则快路径（零成本）：急症红旗词直接短路为 emergency。急症警示对延迟最敏感、
   误判代价最高（LLM 分类抖动不能耽误「立即就医」的提示），且模式高度固定，
   关键词匹配的准确率不低于 LLM，还能省一次调用配额。
2. LLM 结构化分类：覆盖规则无法穷举的自然语言变体（「头有点晕」是闲聊还是问诊？
   依赖上下文），低温（0.1）+ 固定枚举值 + JSON Schema 约束，输出可校验。
3. 失败兜底：LLM 不可用/输出不合法时降级 consult（保守侧），医疗场景宁可信其严肃，
   绝不把求助消息静默当闲聊丢弃。

四类意图：
- chitchat  闲聊寒暄        -> 轻量友好回应，不检索不烧配额
- consult   医学知识咨询    -> RAG 检索知识库，带引用的科普解答
- inquiry   症状问诊        -> RAG 答疑 + 就医引导（推荐科室/红旗症状），严禁诊断
- emergency 急症征象        -> 立即就医红色警示，跳过一切 LLM 生成
"""

from loguru import logger

from app.medical.llm_client import llm_client

# 意图标签（前端徽章直接消费）
INTENT_LABELS = {
    "chitchat": "闲聊",
    "consult": "健康咨询",
    "inquiry": "症状问诊",
    "emergency": "急症警示",
}

VALID_INTENTS = set(INTENT_LABELS)

# 急症红旗词：命中即短路（大出血/咯血、意识不清、抽搐、剧烈胸痛胸闷、呼吸困难等）
EMERGENCY_KEYWORDS = (
    "大出血", "吐血", "咯血", "便血", "血不止", "止不住血",
    "昏迷", "晕厥", "意识不清", "叫不醒", "不省人事", "翻白眼",
    "抽搐", "惊厥", "痉挛",
    "胸痛", "胸闷", "心绞痛", "压榨感",
    "呼吸困难", "喘不上气", "喘不过气", "气短", "窒息",
    "剧烈头痛", "炸裂样头痛", "雷击样头痛",
    "服药过量", "吃错药", "误食", "吞了", "自杀", "轻生",
    "中风", "偏瘫", "口角歪斜", "面瘫", "说不出话", "肢体无力",
    "高烧不退", "烧到抽", "药物过敏", "过敏性休克",
)

# 问候/闲聊快路径：纯寒暄消息也无需 LLM 分类（「你好」「在吗」「谢谢」）
CHITCHAT_EXACT = {"你好", "您好", "hi", "hello", "在吗", "谢谢", "多谢", "thanks", "thank you"}


def _normalize(message: str) -> str:
    return (message or "").strip().lower()


class IntentClassifier:
    """规则短路 + LLM 结构化分类的两级意图识别。"""

    def classify(self, message: str, history: list[dict] | None = None) -> dict:
        """返回 {"intent", "confidence", "reasoning", "by_rule": bool}。"""
        text = _normalize(message)
        if not text:
            return self._result("chitchat", 1.0, "空消息", by_rule=True)

        # 规则快路径 1：急症红旗词（子串匹配，消息含「你好我爸爸突然胸痛」也必须命中）
        if any(kw in text for kw in EMERGENCY_KEYWORDS):
            return self._result("emergency", 1.0, "命中急症红旗关键词", by_rule=True)

        # 规则快路径 2：纯寒暄（整句完全等于问候语，避免「你好我头晕」被误判闲聊）
        if text in CHITCHAT_EXACT:
            return self._result("chitchat", 1.0, "纯问候语", by_rule=True)

        # LLM 结构化分类（携带最近历史消歧）
        try:
            history_text = self._history_text(history)
            result = llm_client.classify_intent(message, history_text)
            intent = str(result.get("intent", "")).strip().lower()
            if intent not in VALID_INTENTS:
                raise ValueError(f"非法意图值: {intent}")
            confidence = result.get("confidence", 0.6)
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                confidence = 0.6
            return self._result(
                intent, float(confidence), str(result.get("reasoning", "")), by_rule=False
            )
        except Exception as exc:
            # 兜底 consult：解析失败/LLM 不可用时按知识咨询处理（会走 RAG 答疑，
            # 宁可多检索一次也不能把求助消息当闲聊轻飘飘带过）
            logger.warning(f"意图分类失败，兜底 consult: {exc}")
            return self._result("consult", 0.3, "分类失败，保守按健康咨询处理", by_rule=False)

    @staticmethod
    def _history_text(history: list[dict] | None) -> str:
        if not history:
            return ""
        lines = []
        for msg in history[-6:]:  # 最近 3 轮足够消歧
            role = "患者" if msg.get("role") == "user" else "助手"
            lines.append(f"{role}: {str(msg.get('content', ''))[:80]}")
        return "\n".join(lines)

    @staticmethod
    def _result(intent: str, confidence: float, reasoning: str, by_rule: bool) -> dict:
        return {
            "intent": intent,
            "intent_label": INTENT_LABELS[intent],
            "confidence": confidence,
            "reasoning": reasoning,
            "by_rule": by_rule,
        }


intent_classifier = IntentClassifier()
