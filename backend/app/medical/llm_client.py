"""医疗 LLM 客户端（升级自 MediAgent-LLM 的 tools/llm_client.py）。

保留：
- OpenAI 兼容协议接入（统一走 config.resolve_llm() 网关，可切换任意厂商）
- 三层输出防御：Prompt 约束 JSON -> 容错提取 -> Pydantic 校验
- 推理模型 reasoning_content 兼容

升级（本项目核心创新）：
- 三个分析方法的 Prompt 均注入「编号参考资料块」（RAG 检索结果）
- 要求 LLM 输出 citations 数组声明引用了哪些编号，供上层映射回真实证据
"""

import json
import re
import time

from loguru import logger
from openai import OpenAI

from app.core.config import settings
from app.rag.evidence import format_reference_block

# LLM 调用韧性参数：单次超时 / 重试次数 / 重试退避基数
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF_BASE = 2.0


class LLMClient:
    """医疗 Agent 统一 LLM 调用入口。"""

    def __init__(self):
        llm_config = settings.resolve_llm()
        if not llm_config["api_key"]:
            raise ValueError(
                "LLM 未配置：请在 .env 中设置 LLM_API_KEY/LLM_BASE_URL，"
                "或配置 DASHSCOPE_API_KEY / DEEPSEEK_API_KEY"
            )
        self.model = llm_config["model"]
        self.provider = llm_config["provider"]
        self.client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=0,  # 重试策略由 _chat 统一管理（带退避与日志），关闭 SDK 内部静默重试
        )
        logger.info(f"医疗 LLM 网关就绪: provider={self.provider}, model={self.model}")

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """带超时与指数退避重试的对话调用。

        医疗工作流单次问诊串多次 LLM 调用，网关抖动若直接抛错会打断六分钟的
        诊疗流程；可重试的瞬时错误（限流/超时/5xx）在此消化，重试耗尽才上抛。
        """
        last_exc: Exception | None = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=4000,
                )
                break
            except Exception as exc:  # openai 的异常类型跨版本不稳定，宽捕获后按需重试
                last_exc = exc
                if attempt == LLM_MAX_RETRIES:
                    logger.error(f"LLM 调用失败（第 {attempt} 次重试后放弃）: {exc}")
                    raise
                backoff = LLM_RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"LLM 调用异常（第 {attempt}/{LLM_MAX_RETRIES} 次），{backoff:.0f}s 后重试: {exc}"
                )
                time.sleep(backoff)
        else:  # for-break 不会走到这，防御性兜底
            raise last_exc or RuntimeError("LLM 调用失败")

        msg = response.choices[0].message
        content = msg.content or ""
        if not content.strip():
            reasoning = getattr(msg, "reasoning_content", None) or ""
            if reasoning:
                logger.warning(
                    f"模型返回空 content，reasoning_content 长度 {len(reasoning)}，可能 max_tokens 不足"
                )
            raise ValueError("模型返回空内容，请检查 max_tokens 或模型配置")
        return content

    def _chat_messages(self, messages: list[dict], temperature: float = 0.3) -> str:
        """多轮 messages 对话（患者聊天需要携带历史）；重试骨架与 _chat 一致。"""
        last_exc: Exception | None = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=4000,
                )
                break
            except Exception as exc:
                last_exc = exc
                if attempt == LLM_MAX_RETRIES:
                    logger.error(f"LLM 调用失败（第 {attempt} 次重试后放弃）: {exc}")
                    raise
                backoff = LLM_RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"LLM 调用异常（第 {attempt}/{LLM_MAX_RETRIES} 次），{backoff:.0f}s 后重试: {exc}"
                )
                time.sleep(backoff)
        else:
            raise last_exc or RuntimeError("LLM 调用失败")

        msg = response.choices[0].message
        content = msg.content or ""
        if not content.strip():
            raise ValueError("模型返回空内容，请检查 max_tokens 或模型配置")
        return content

    # ---------- 三个业务方法 ----------
    def analyze_symptoms(
        self,
        symptom_description: str,
        patient_history: list[dict],
        reference_block: str = "",
    ) -> dict:
        prompt = self._build_symptom_prompt(symptom_description, patient_history, reference_block)
        result = self._chat("你是一个专业的医疗AI助手，专门进行症状分析。回答必须基于参考资料并标注引用。", prompt)
        return self._parse_json_result(result, "症状分析")

    def analyze_diagnosis(
        self,
        symptom_analysis: str,
        lab_results: list[dict],
        patient_history: list[dict],
        reference_block: str = "",
        verification_block: str = "",
    ) -> dict:
        prompt = self._build_diagnosis_prompt(
            symptom_analysis, lab_results, patient_history, reference_block, verification_block
        )
        result = self._chat("你是一个专业的医疗AI助手，专门进行疾病诊断分析。回答必须基于参考资料并标注引用。", prompt)
        return self._parse_json_result(result, "诊断分析")

    def verify_hypothesis(
        self,
        disease: str,
        probability,
        symptom_description: str,
        key_symptoms: list[str],
        lab_results: list[dict],
        reference_block: str = "",
    ) -> dict:
        """单个鉴别假设的核验（Send fan-out 并行分支调用）。"""
        prompt = self._build_hypothesis_prompt(
            disease, probability, symptom_description, key_symptoms, lab_results, reference_block
        )
        result = self._chat("你是鉴别诊断核验专家，负责独立核验单个疾病假设，回答必须基于参考资料并标注引用。", prompt)
        return self._parse_json_result(result, "假设核验")

    def suggest_treatment(
        self,
        diagnosis_result: str,
        allergy_history: list[str],
        reference_block: str = "",
    ) -> dict:
        prompt = self._build_treatment_prompt(diagnosis_result, allergy_history, reference_block)
        result = self._chat("你是一个专业的医疗AI助手，专门提供治疗建议。回答必须基于参考资料并标注引用。", prompt)
        return self._parse_json_result(result, "治疗建议")

    def check_symptom_sufficiency(
        self,
        symptom_description: str,
        patient_age: int | None = None,
        patient_gender: str = "",
    ) -> dict:
        """症状描述充分性检查（医生端追问澄清的第一步）。

        输入越模糊（如「喉咙不舒服」「头有点晕」），后续症状分析的质量越差且无任何
        预警。此方法按五个关键维度判定描述是否足以支撑分析，不足时产出针对性追问，
        由工作流 interrupt 暂停等待医生补答（与「等待化验」同一人机协同模式）。
        """
        prompt = self._build_sufficiency_prompt(symptom_description, patient_age, patient_gender)
        result = self._chat(
            "你是资深的门急诊问诊质控专家，负责判断症状主诉信息是否足够进入分析流程。",
            prompt,
            temperature=0.1,
        )
        parsed = self._parse_json_result(result, "症状充分性检查")
        # 归一化：LLM 偶发把 clarify_questions 略写成字符串数组，统一为 {dimension, question}
        questions = []
        for q in parsed.get("clarify_questions", []):
            if isinstance(q, dict) and str(q.get("question", "")).strip():
                questions.append({
                    "dimension": str(q.get("dimension", "其他")).strip(),
                    "question": str(q["question"]).strip(),
                })
            elif isinstance(q, str) and q.strip():
                questions.append({"dimension": "其他", "question": q.strip()})
        parsed["clarify_questions"] = questions[:5]
        parsed["sufficient"] = bool(parsed.get("sufficient")) and not questions
        return parsed

    def classify_intent(self, message: str, history_text: str = "") -> dict:
        """患者消息意图分类：chitchat / consult / inquiry / emergency。"""
        prompt = self._build_intent_prompt(message, history_text)
        result = self._chat(
            "你是医疗对话系统的意图识别模块，只输出JSON，不输出任何其他内容。",
            prompt,
            temperature=0.1,
        )
        return self._parse_json_result(result, "意图识别")

    def patient_chat_reply(
        self,
        messages: list[dict],
        intent: str,
        reference_block: str = "",
    ) -> dict:
        """患者端聊天回复（自然语言 + 可选引用声明），携带多轮历史。

        宽松解析：聊天场景模型偶尔无视 JSON 约束直接输出纯文本（尤其闲聊
        短回复），此时把整段文本当 reply 降级返回，不视为错误——聊天体验
        优先于结构化（引用缺失只是少卡片，回复失败是整个气泡变兜底文案）。
        """
        prompt = self._build_patient_reply_prompt(intent, reference_block)
        chat_messages = [{"role": "system", "content": prompt}] + messages[-10:]
        result = self._chat_messages(chat_messages, temperature=0.5)
        try:
            return self._parse_json_result(result, "患者聊天回复")
        except ValueError:
            if reference_block:
                # 带参考资料时仍要 JSON（引用映射依赖编号）；纯文本视为格式错误上抛
                raise
            logger.warning("患者聊天回复未按 JSON 返回，降级为纯文本 reply")
            return {"reply": result.strip(), "citations": []}

    # ---------- Prompt 构建 ----------
    @staticmethod
    def _history_text(patient_history: list[dict]) -> str:
        if not patient_history:
            return ""
        lines = ["\n患者历史症状和诊断："]
        for record in patient_history:
            lines.append(f"- 症状：{record.get('symptoms', '')}")
            lines.append(f"- 诊断：{record.get('diagnosis', '')}")
            lines.append(f"- 时间：{record.get('date', '')}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _lab_text(lab_results: list[dict]) -> str:
        if not lab_results:
            return ""
        lines = ["\n化验结果："]
        for result in lab_results:
            lines.append(
                f"- {result.get('test_name', '')}: {result.get('result_value', '')} "
                f"{result.get('unit', '')} (参考范围: {result.get('reference_range', '')}"
                f"{'，异常' if result.get('is_abnormal') else ''})"
            )
        return "\n".join(lines) + "\n"

    def _build_symptom_prompt(self, symptom_description, patient_history, reference_block) -> str:
        return f"""请根据以下信息进行症状分析：

当前症状描述：{symptom_description}
{self._history_text(patient_history)}
{reference_block}

请严格按照以下JSON格式返回分析结果（不要输出JSON以外的任何内容）：
{{
    "key_symptoms": ["症状1", "症状2", "症状3"],
    "possible_causes": ["可能原因1", "可能原因2"],
    "possible_diseases": [
        {{"disease": "疾病名称", "probability": 85}},
        {{"disease": "疾病名称", "probability": 70}},
        {{"disease": "疾病名称", "probability": 60}},
        {{"disease": "疾病名称", "probability": 45}},
        {{"disease": "疾病名称", "probability": 30}}
    ],
    "further_information_needed": ["需要确认的信息1"],
    "lab_tests_needed": ["血常规", "尿常规"],
    "citations": [{{"ref": 1, "quote": "参考资料中支持该结论的原文短语"}}]
}}

要求：
1. 尽量列出5种可能疾病，按可能性从高到低排序，probability 为0-100的整数
2. 症状明确时 lab_tests_needed 可以为空数组；症状不明确时列出需要的检查项目
3. 【引用要求】如参考资料中有支持你判断的内容，必须在 citations 中以 {{"ref": 编号}} 声明，
   quote 摘录参考资料中的原文短语（不超过50字）；参考资料未覆盖的部分基于医学常识给出，此时 citations 可为空数组
4. 严禁编造参考资料中不存在的编号或内容
"""

    def _build_diagnosis_prompt(
        self, symptom_analysis, lab_results, patient_history, reference_block, verification_block=""
    ) -> str:
        return f"""请根据以下信息进行诊断分析：

症状分析结果：{symptom_analysis}
{self._lab_text(lab_results)}
{verification_block}
{self._history_text(patient_history)}
{reference_block}

请严格按照以下JSON格式返回诊断结果（不要输出JSON以外的任何内容）：
{{
    "main_diagnosis": "最可能的诊断结果",
    "differential_diagnoses": ["鉴别诊断1", "鉴别诊断2", "鉴别诊断3"],
    "diagnosis_basis": ["诊断依据1", "诊断依据2"],
    "certainty": 85.5,
    "citations": [{{"ref": 2, "quote": "参考资料中支持该结论的原文短语"}}]
}}

要求：
1. 尽量列出至少3种鉴别诊断
2. certainty 为0-100之间的数值，表示诊断确定性
3. 诊断依据要具体明确，并优先引用参考资料中的标准/指南内容（如诊断阈值、分级标准）
4. 如给出了「鉴别假设并行核验结果」，请综合各假设的支持度与缺口做判断：
   支持度高且有证据的假设优先，缺口信息可作为进一步检查建议
5. 【引用要求】诊断依据如来自参考资料，必须在 citations 中声明对应编号并摘录原文短语（不超过50字）；
   参考资料未覆盖时基于医学常识给出，citations 可为空数组
6. 严禁编造参考资料中不存在的编号或内容
"""

    def _build_hypothesis_prompt(
        self, disease, probability, symptom_description, key_symptoms, lab_results, reference_block
    ) -> str:
        symptoms_text = "、".join(key_symptoms[:6]) if key_symptoms else "无"
        lab_text = self._lab_text(lab_results).strip() or "（暂无化验结果）"

        return f"""请独立核验以下鉴别诊断假设：

待核验假设：{disease}（症状分析阶段给出的可能性：{probability if probability is not None else "未知"}%）
患者症状描述：{symptom_description}
关键症状：{symptoms_text}
{lab_text}

{reference_block}

请严格按照以下JSON格式返回核验结果（不要输出JSON以外的任何内容）：
{{
    "disease": "{disease}",
    "support_score": 65,
    "matched_criteria": ["符合的标准/依据1", "符合的标准/依据2"],
    "missing_information": ["尚缺的支撑信息1"],
    "rationale": "一句话核验结论",
    "citations": [{{"ref": 1, "quote": "参考资料中支持该判断的原文短语"}}]
}}

要求：
1. support_score 为0-100的整数：患者症状+化验结果+参考资料对该假设的综合支持度
2. matched_criteria 优先对照参考资料中的诊断标准/阈值（如血压分级、血糖界值），并在 citations 声明编号与原文短语（不超过50字）
3. missing_information 列出确认该假设还缺少的关键信息（可作为进一步检查建议）
4. 只核验这一个假设，不要展开分析其他疾病
5. 严禁编造参考资料中不存在的编号或内容；参考资料未覆盖时基于医学常识评估，citations 可为空数组
"""

    def _build_treatment_prompt(self, diagnosis_result, allergy_history, reference_block) -> str:
        allergy_text = f"\n患者过敏史：{', '.join(allergy_history)}" if allergy_history else ""
        return f"""请根据以下信息提供治疗建议：

诊断结果：{diagnosis_result}{allergy_text}

{reference_block}

请严格按照以下JSON格式返回治疗建议（不要输出JSON以外的任何内容）：
{{
    "treatment_plan": {{
        "medication": ["药物1", "药物2"],
        "non_medication": ["非药物治疗1"]
    }},
    "medication_advice": [
        {{
            "drug_name": "药物名称",
            "dosage": "用法用量",
            "duration": "用药时长",
            "precautions": "注意事项"
        }}
    ],
    "lifestyle_advice": ["生活方式建议1"],
    "follow_up_advice": ["随访建议1"],
    "warning_signals": ["警示信号1"],
    "citations": [{{"ref": 1, "quote": "参考资料中支持该建议的原文短语"}}]
}}

要求：
1. 严格考虑患者的过敏史，禁止推荐过敏药物；如资料中提示禁忌请务必遵守
2. 用药建议（剂量/疗程）优先采用参考资料中的标准方案并在 citations 中声明编号与原文短语（不超过50字）
3. 给出实用的生活方式、随访建议和需要警惕的警示信号
4. 严禁编造参考资料中不存在的编号或内容
"""

    def _build_sufficiency_prompt(self, symptom_description, patient_age, patient_gender) -> str:
        profile = f"{patient_gender or '未知'}性，{patient_age if patient_age is not None else '未知'}岁"
        return f"""请判断以下症状描述是否足够支撑症状分析（患者画像：{profile}）：

症状描述：{symptom_description}

按五个关键维度逐一评估：部位、持续时间、性质与程度、诱因或缓解因素、伴随症状。

请严格按照以下JSON格式返回（不要输出JSON以外的任何内容）：
{{
    "sufficient": false,
    "missing_dimensions": ["持续时间", "伴随症状"],
    "clarify_questions": [
        {{"dimension": "持续时间", "question": "喉咙不舒服持续多久了？是持续性还是间断性的？"}},
        {{"dimension": "性质与程度", "question": "不舒服具体是什么感觉：疼痛、干痒、异物感，还是吞咽困难？"}}
    ],
    "reasoning": "一句话判定理由"
}}

要求：
1. 描述完整覆盖五个维度中与主诉相关的关键信息时 sufficient=true；像「喉咙不舒服」「头有点晕」
   这类抽象模糊的主诉，关键维度缺失时 sufficient=false
2. sufficient=false 时给出3-5个针对性追问，每个问题口语化、医生问诊口吻、一次只问一个点，
   dimension 填五维度之一（部位/持续时间/性质与程度/诱因或缓解因素/伴随症状）
3. sufficient=true 时 clarify_questions 为空数组
4. 不做诊断判断，只评估信息是否充分
"""

    def _build_intent_prompt(self, message: str, history_text: str) -> str:
        history_block = f"\n最近对话记录（用于消歧）：\n{history_text}\n" if history_text else ""
        return f"""请对以下患者消息进行意图分类：

患者消息：{message}{history_block}
意图定义：
- chitchat：寒暄闲聊、问候、与医疗无关的话题（如「你好」「今天天气怎么样」）
- consult：医学知识咨询，询问疾病/药物/检查等一般知识（如「二甲双胍是什么药」「高血压的诊断标准」）
- inquiry：症状问诊，描述自己或他人的身体不适、寻求病因或就诊建议（如「我最近老是头晕」「孩子发烧三天了怎么办」）
- emergency：急症征象，含胸痛、呼吸困难、大出血、意识不清、抽搐、剧烈头痛、服药过量等需立即就医的情况

请严格按照以下JSON格式返回（不要输出JSON以外的任何内容）：
{{
    "intent": "inquiry",
    "confidence": 0.9,
    "reasoning": "消息描述了身体不适并寻求建议"
}}

要求：
1. intent 只能取 chitchat / consult / inquiry / emergency 四个值之一
2. confidence 为0-1的小数；不确定时取概率最高者并降低 confidence
3. 宁可错分为 inquiry/consult 也不要把求助类消息判为 chitchat
"""

    def _build_patient_reply_prompt(self, intent: str, reference_block: str) -> str:
        intent_rules = {
            "chitchat": "当前是闲聊意图：友好简短地回应，可以自然地介绍你能提供的医疗帮助（知识解答、症状咨询、就医建议引导），不要展开长篇医疗内容。",
            "consult": "当前是医学知识咨询意图：用患者能听懂的语言解答，专业术语要解释；回答须基于参考资料，未覆盖的部分明确说明「知识库暂未覆盖，以下为一般性常识仅供参考」。",
            "inquiry": (
                "当前是症状问诊意图，你只做知识答疑与就医引导，【严禁给出诊断结论或具体治疗方案】。回复必须包含："
                "1) 对症状可能涉及的方向做通俗的科普性说明（基于参考资料）；"
                "2) 就医引导：建议就诊科室 + 出现哪些情况应立即就医；"
                "3) 结尾提醒：本回答仅供参考，不能替代执业医师的诊断。"
            ),
        }.get(intent, "当前意图未识别：按医学知识咨询处理，保守、通俗地回应并引导就医。")
        return f"""你是「智慧医疗诊疗助手」的患者端客服，面向普通患者，语气亲切、通俗易懂、有同理心。
{intent_rules}
{reference_block}

请严格按照以下JSON格式返回（不要输出JSON以外的任何内容）：
{{
    "reply": "给患者看的回复正文（可分段，用\\n换行）",
    "citations": [{{"ref": 1, "quote": "参考资料中支持该内容的原文短语"}}]
}}

要求：
1. reply 控制在300字以内，分段清晰，不使用 Markdown 标记
2. 【引用要求】如回复内容有参考资料支持，必须在 citations 中以 {{"ref": 编号}} 声明并摘录原文短语（不超过50字）；
   无参考资料支持时 citations 为空数组，严禁编造参考资料中不存在的编号或内容
"""

    # ---------- 输出解析 ----------
    @staticmethod
    def _parse_json_result(result: str, context: str) -> dict:
        """通用 JSON 解析：直接解析 -> 剥离代码围栏 -> 大括号截取，三级容错。"""
        candidates = [result]

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", result)
        if fenced:
            candidates.insert(0, fenced.group(1))

        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            candidates.append(result[start:end])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        logger.error(f"无法解析{context}结果: {result[:500]}")
        raise ValueError(f"{context}结果格式错误")


# 全局 LLM 客户端单例（配置缺失时延迟报错，不阻断其他模块导入）
try:
    llm_client = LLMClient()
except ValueError as exc:
    from loguru import logger as _logger

    _logger.warning(f"LLM 客户端暂不可用: {exc}")
    llm_client = None
