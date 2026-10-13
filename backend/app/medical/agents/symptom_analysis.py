"""症状分析智能体（升级自 MediAgent-LLM，接入 RAG 证据引用）。"""

from sqlalchemy.orm import Session

from app.dao.medical import ResultDAO, safe_json_loads
from app.medical.agents.base import BaseMedicalAgent, PatientKey
from app.medical.llm_client import llm_client
from app.models.entities import InquiryRecord, SymptomAnalysisResult
from app.schemas.medical import AgentType, InquiryStatus, SymptomAnalysisResult as SymptomAnalysisResultSchema


def _past_inquiry_ids(session: Session, patient_key: PatientKey) -> list[int]:
    """同患者（按姓名）已完成的既往问诊 ID 列表。"""
    if not patient_key or not patient_key.patient_name:
        return []
    rows = (
        session.query(InquiryRecord.id)
        .filter(
            InquiryRecord.patient_name == patient_key.patient_name,
            InquiryRecord.id != patient_key.current_inquiry_id,
            InquiryRecord.main_status == InquiryStatus.COMPLETED.value,
        )
        .order_by(InquiryRecord.created_at.desc())
        .limit(5)
        .all()
    )
    return [row[0] for row in rows]


class SymptomAnalysisAgent(BaseMedicalAgent):
    agent_type = AgentType.SYMPTOM_ANALYSIS
    prompt_version = "v2.0-rag"

    def build_retrieval_query(self, request) -> str:
        """症状描述本身即最直接的检索查询。"""
        return request.symptom_description[:200]

    def call_llm(self, request, reference_block: str, patient_history: list[dict]) -> dict:
        return llm_client.analyze_symptoms(
            symptom_description=request.symptom_description,
            patient_history=patient_history,
            reference_block=reference_block,
        )

    def validate(self, llm_result: dict):
        return SymptomAnalysisResultSchema(**llm_result)

    def persist(self, session: Session, inquiry_record_id: int, result_json: str, citations_json: str) -> None:
        ResultDAO.create_record(
            session, SymptomAnalysisResult, inquiry_record_id, result_json, citations_json
        )

    def fetch_patient_history(self, session: Session, patient_key) -> list[dict]:
        past_ids = _past_inquiry_ids(session, patient_key)
        if not past_ids:
            return []
        records = (
            session.query(SymptomAnalysisResult)
            .filter(SymptomAnalysisResult.inquiry_record_id.in_(past_ids))
            .order_by(SymptomAnalysisResult.created_at.desc())
            .limit(5)
            .all()
        )
        history = []
        for record in records:
            parsed = safe_json_loads(record.agent_result, {})
            history.append({
                "symptoms": ", ".join(parsed.get("key_symptoms", [])),
                "diagnosis": ", ".join(
                    d.get("disease", "") for d in parsed.get("possible_diseases", [])[:3]
                ),
                "date": record.created_at.strftime("%Y-%m-%d") if record.created_at else "",
            })
        return history


symptom_analysis_agent = SymptomAnalysisAgent()
