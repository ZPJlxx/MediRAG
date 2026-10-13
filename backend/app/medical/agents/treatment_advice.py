"""治疗建议智能体（升级自 MediAgent-LLM，接入 RAG 药品说明书核对 + 真实过敏史）。"""

import json

from loguru import logger
from sqlalchemy.orm import Session

from app.dao.medical import ResultDAO
from app.medical.agents.base import BaseMedicalAgent
from app.medical.llm_client import llm_client
from app.models.entities import TreatmentSuggestionResult
from app.schemas.medical import AgentType, TreatmentAdviceResult as TreatmentAdviceResultSchema


class TreatmentAdviceAgent(BaseMedicalAgent):
    agent_type = AgentType.TREATMENT_ADVICE
    prompt_version = "v2.0-rag"

    def build_retrieval_query(self, request) -> str:
        """检索查询 = 主诊断 + 鉴别诊断（检索治疗指南/药品用法用量）。"""
        try:
            diagnosis = json.loads(request.diagnosis_result)
        except (TypeError, ValueError):
            diagnosis = {}
        terms = [diagnosis.get("main_diagnosis", "")]
        terms += diagnosis.get("differential_diagnoses", [])[:2]
        return " ".join(t for t in terms if t)[:200]

    def call_llm(self, request, reference_block: str, patient_history: list[dict]) -> dict:
        return llm_client.suggest_treatment(
            diagnosis_result=request.diagnosis_result,
            allergy_history=request.allergy_history,
            reference_block=reference_block,
        )

    def validate(self, llm_result: dict):
        return TreatmentAdviceResultSchema(**llm_result)

    def persist(self, session: Session, inquiry_record_id: int, result_json: str, citations_json: str) -> None:
        ResultDAO.create_record(
            session, TreatmentSuggestionResult, inquiry_record_id, result_json, citations_json
        )

    def post_validate(self, validated, request) -> None:
        """落库前的程序化过敏禁忌核对（不依赖 LLM 自觉，结果随 to_json 持久化）。"""
        if request.allergy_history:
            self.check_allergy_contraindications(validated, request.allergy_history)

    @staticmethod
    def check_allergy_contraindications(validated, allergy_history: list[str]) -> None:
        """过敏药物核对：双向子串匹配（修复原项目过敏史写死 [] 不生效的问题）。

        直接修改已校验的 TreatmentAdviceResult 模型（medication_advice 为 list[dict]、
        warning_signals 为 list[str]，原地修改即随 to_json 持久化）。
        """
        if not allergy_history:
            return
        for medication in validated.medication_advice:
            drug_name = str(medication.get("drug_name", "")).lower()
            for allergy in allergy_history:
                allergy_lower = str(allergy).lower()
                if allergy_lower and (allergy_lower in drug_name or drug_name in allergy_lower):
                    warning = f"【禁忌提醒】患者对「{allergy}」过敏，请人工复核该用药！"
                    medication["precautions"] = f"{medication.get('precautions', '')} {warning}".strip()
                    if warning not in validated.warning_signals:
                        validated.warning_signals.append(warning)
                    logger.warning(f"检测到疑似过敏药物: {drug_name}，过敏史: {allergy}")


treatment_advice_agent = TreatmentAdviceAgent()
