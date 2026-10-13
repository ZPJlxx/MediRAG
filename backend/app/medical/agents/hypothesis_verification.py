"""鉴别假设核验智能体（Send fan-out 并行分支的工作单元）。

每个实例调用核验一个假设疾病：检索该病的诊断标准 -> LLM 对照患者
症状/化验评估支持度 -> 输出带证据引用的核验结果。
结果不单独建表（经 merge_verifications 节点聚合进工作流状态），
但调用审计仍完整落入 agent_session_his（agent_type=4）。
"""

from sqlalchemy.orm import Session

from app.medical.agents.base import BaseMedicalAgent
from app.medical.llm_client import llm_client
from app.schemas.medical import (
    AgentType,
    HypothesisVerificationResult,
)


class _VerificationRequest:
    """核验请求（由 workflow 的 Send payload 构造）。"""

    def __init__(self, hypothesis: dict, symptom_description: str,
                 key_symptoms: list[str], lab_results: list[dict], use_rag: bool):
        self.disease = str(hypothesis.get("disease", "")).strip()
        self.probability = hypothesis.get("probability")
        self.symptom_description = symptom_description
        self.key_symptoms = key_symptoms
        self.lab_results = lab_results
        self.use_rag = use_rag


class HypothesisVerificationAgent(BaseMedicalAgent):
    agent_type = AgentType.HYPOTHESIS_VERIFICATION
    prompt_version = "v2.0-rag"

    def build_retrieval_query(self, request) -> str:
        """检索查询 = 假设疾病 + 诊断标准关键词（命中指南/速查文档）。"""
        return f"{request.disease} 诊断标准 鉴别诊断 诊断依据"[:200]

    def call_llm(self, request, reference_block: str, patient_history: list[dict]) -> dict:
        return llm_client.verify_hypothesis(
            disease=request.disease,
            probability=request.probability,
            symptom_description=request.symptom_description,
            key_symptoms=request.key_symptoms,
            lab_results=request.lab_results,
            reference_block=reference_block,
        )

    def validate(self, llm_result: dict):
        return HypothesisVerificationResult(**llm_result)

    def persist(self, session: Session, inquiry_record_id: int, result_json: str, citations_json: str) -> None:
        """无独立结果表：结果由 merge_verifications 聚合进工作流状态，
        审计（request/response）已由基类写入 agent_session_his。"""
        return None


hypothesis_verification_agent = HypothesisVerificationAgent()
