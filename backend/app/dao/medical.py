"""DAO 层：从 MediAgent-LLM 的 db_operations.py 迁移并适配统一模型。"""

import json

from sqlalchemy.orm import Session

from app.models.entities import (
    AgentSessionHis,
    DiseaseDiagnosisResult,
    InquiryRecord,
    PatientChatMessage,
    PatientChatSession,
    SymptomAnalysisResult,
    TreatmentSuggestionResult,
    WorkflowEvent,
)
from app.schemas.medical import AgentStatus, AgentType, InquiryStatus


class InquiryRecordDAO:
    @staticmethod
    def create(session: Session, **fields) -> InquiryRecord:
        record = InquiryRecord(**fields)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    @staticmethod
    def get(session: Session, inquiry_id: int) -> InquiryRecord | None:
        return session.get(InquiryRecord, inquiry_id)

    @staticmethod
    def update_status(
        session: Session,
        inquiry_id: int,
        status: InquiryStatus,
        failed_reason: str = "",
    ) -> bool:
        record = session.get(InquiryRecord, inquiry_id)
        if not record:
            return False
        record.main_status = status.value
        record.failed_reason = failed_reason[:500]
        session.commit()
        return True

    @staticmethod
    def set_thread_id(session: Session, inquiry_id: int, thread_id: str) -> bool:
        record = session.get(InquiryRecord, inquiry_id)
        if not record:
            return False
        record.thread_id = thread_id
        session.commit()
        return True

    @staticmethod
    def list_by_doctor(session: Session, doctor_id: int, limit: int = 50) -> list[InquiryRecord]:
        return (
            session.query(InquiryRecord)
            .filter(InquiryRecord.doctor_id == doctor_id)
            .order_by(InquiryRecord.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def page_by_doctor(
        session: Session,
        doctor_id: int,
        page: int = 1,
        page_size: int = 20,
        patient_keyword: str = "",
    ) -> tuple[list[InquiryRecord], int]:
        """分页 + 患者名模糊搜索。返回 (records, total)。"""
        query = session.query(InquiryRecord).filter(InquiryRecord.doctor_id == doctor_id)
        if patient_keyword.strip():
            query = query.filter(
                InquiryRecord.patient_name.like(f"%{patient_keyword.strip()}%")
            )
        total = query.count()
        records = (
            query.order_by(InquiryRecord.created_at.desc())
            .offset((max(page, 1) - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return records, total


class ResultDAO:
    """三张结果表的通用 CRUD（结构一致，合并实现）。"""

    @staticmethod
    def create_record(session: Session, model, inquiry_record_id: int, agent_result: str, citations_json: str):
        record = model(
            inquiry_record_id=inquiry_record_id,
            agent_result=agent_result,
            citations=citations_json,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    @staticmethod
    def get_record(session: Session, model, inquiry_record_id: int):
        return (
            session.query(model)
            .filter(model.inquiry_record_id == inquiry_record_id)
            .order_by(model.created_at.desc())
            .first()
        )

    # 便捷方法
    @classmethod
    def get_symptom(cls, session: Session, inquiry_record_id: int):
        return cls.get_record(session, SymptomAnalysisResult, inquiry_record_id)

    @classmethod
    def get_diagnosis(cls, session: Session, inquiry_record_id: int):
        return cls.get_record(session, DiseaseDiagnosisResult, inquiry_record_id)

    @classmethod
    def get_treatment(cls, session: Session, inquiry_record_id: int):
        return cls.get_record(session, TreatmentSuggestionResult, inquiry_record_id)


class AgentSessionDAO:
    """全链路审计（新增 rag_enabled / retrieval_query 字段记录检索行为）。"""

    @staticmethod
    def create_session(
        session: Session,
        inquiry_record_id: int,
        agent_type: AgentType,
        prompt_version_code: str,
        request_body: str,
        rag_enabled: bool = False,
        retrieval_query: str = "",
    ) -> AgentSessionHis:
        record = AgentSessionHis(
            inquiry_record_id=inquiry_record_id,
            agent_type=agent_type.value,
            prompt_version_code=prompt_version_code,
            agent_status=AgentStatus.ANALYZING.value,
            request_body=(request_body or "")[:8000],
            rag_enabled=rag_enabled,
            retrieval_query=(retrieval_query or "")[:2000],
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    @staticmethod
    def update_result(
        session: Session,
        session_id: int,
        response_body: str,
        agent_status: AgentStatus,
        failed_reason: str = "",
    ) -> bool:
        record = session.get(AgentSessionHis, session_id)
        if not record:
            return False
        record.response_body = (response_body or "")[:16000]
        record.agent_status = agent_status.value
        record.failed_reason = failed_reason[:500]
        session.commit()
        return True


class WorkflowEventDAO:
    @staticmethod
    def log(session: Session, inquiry_record_id: int, stage: str, status: str, detail: str = ""):
        event = WorkflowEvent(
            inquiry_record_id=inquiry_record_id,
            stage=stage,
            status=status,
            detail=detail[:500],
        )
        session.add(event)
        session.commit()

    @staticmethod
    def list_events(session: Session, inquiry_record_id: int) -> list[WorkflowEvent]:
        return (
            session.query(WorkflowEvent)
            .filter(WorkflowEvent.inquiry_record_id == inquiry_record_id)
            .order_by(WorkflowEvent.created_at.asc(), WorkflowEvent.id.asc())
            .all()
        )

    @staticmethod
    def last_event_id(session: Session, inquiry_record_id: int) -> int:
        """最新事件 ID（自增），作为状态变更令牌的一部分；无事件返回 0。"""
        row = (
            session.query(WorkflowEvent.id)
            .filter(WorkflowEvent.inquiry_record_id == inquiry_record_id)
            .order_by(WorkflowEvent.id.desc())
            .first()
        )
        return row[0] if row else 0


class PatientChatDAO:
    """患者端聊天会话/消息持久化（照 WorkflowEventDAO 静态方法风格）。"""

    @staticmethod
    def get_or_create_session(session: Session, session_key: str, client_ip: str = "") -> PatientChatSession:
        record = (
            session.query(PatientChatSession)
            .filter(PatientChatSession.session_key == session_key)
            .first()
        )
        if record:
            return record
        record = PatientChatSession(session_key=session_key, client_ip=client_ip[:50])
        session.add(record)
        session.commit()
        session.refresh(record)
        return record

    @staticmethod
    def add_message(
        session: Session,
        session_id: int,
        role: str,
        content: str,
        intent: str = "",
        citations_json: str = "[]",
    ) -> PatientChatMessage:
        message = PatientChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            intent=intent[:20],
            citations=citations_json,
        )
        session.add(message)
        session.query(PatientChatSession).filter(
            PatientChatSession.id == session_id
        ).update({"message_count": PatientChatSession.message_count + 1})
        session.commit()
        session.refresh(message)
        return message

    @staticmethod
    def list_messages(session: Session, session_id: int, limit: int = 20) -> list[PatientChatMessage]:
        """最近消息（时间正序返回，供 LLM 做多轮上下文）。"""
        rows = (
            session.query(PatientChatMessage)
            .filter(PatientChatMessage.session_id == session_id)
            .order_by(PatientChatMessage.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))


def safe_json_loads(text: str, default=None):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return default
