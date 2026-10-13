"""诊断分析智能体（升级自 MediAgent-LLM，接入 RAG 证据引用）。"""

import json

from sqlalchemy.orm import Session

from app.dao.medical import ResultDAO, safe_json_loads
from app.medical.agents.base import BaseMedicalAgent, PatientKey
from app.medical.agents.symptom_analysis import _past_inquiry_ids
from app.medical.llm_client import llm_client
from app.models.entities import DiseaseDiagnosisResult
from app.schemas.medical import AgentType, DiagnosisAnalysisResult as DiagnosisAnalysisResultSchema


class DiagnosisAnalysisAgent(BaseMedicalAgent):
    agent_type = AgentType.DIAGNOSIS_ANALYSIS
    prompt_version = "v2.0-rag"

    def build_retrieval_query(self, request) -> str:
        """检索查询 = 症状分析得到的高概率疾病 + 关键症状（检索诊断标准/指南）。"""
        symptom_result = safe_json_loads(request.symptom_analysis_result, {})
        diseases = [
            d.get("disease", "")
            for d in symptom_result.get("possible_diseases", [])[:3]
            if d.get("disease")
        ]
        symptoms = [s for s in symptom_result.get("key_symptoms", [])[:4] if s]
        return " ".join(diseases + symptoms)[:200]

    def call_llm(self, request, reference_block: str, patient_history: list[dict]) -> dict:
        return llm_client.analyze_diagnosis(
            symptom_analysis=request.symptom_analysis_result,
            lab_results=request.lab_results,
            patient_history=patient_history,
            reference_block=reference_block,
            verification_block=getattr(request, "verification_summary", "") or "",
        )

    def validate(self, llm_result: dict):
        return DiagnosisAnalysisResultSchema(**llm_result)

    def persist(self, session: Session, inquiry_record_id: int, result_json: str, citations_json: str) -> None:
        ResultDAO.create_record(
            session, DiseaseDiagnosisResult, inquiry_record_id, result_json, citations_json
        )

    def fetch_patient_history(self, session: Session, patient_key) -> list[dict]:
        """既往诊断历史（同患者过往问诊的主诊断）。"""
        past_ids = _past_inquiry_ids(session, patient_key) if isinstance(patient_key, PatientKey) else []
        if not past_ids:
            return []
        records = (
            session.query(DiseaseDiagnosisResult)
            .filter(DiseaseDiagnosisResult.inquiry_record_id.in_(past_ids))
            .order_by(DiseaseDiagnosisResult.created_at.desc())
            .limit(5)
            .all()
        )
        history = []
        for record in records:
            parsed = safe_json_loads(record.agent_result, {})
            history.append({
                "diagnosis": parsed.get("main_diagnosis", ""),
                "certainty": parsed.get("certainty", 0),
                "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
            })
        return history


diagnosis_analysis_agent = DiagnosisAnalysisAgent()
