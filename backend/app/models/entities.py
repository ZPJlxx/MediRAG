"""统一 ORM 模型：合并三个项目的表结构。

来源与改动：
- sys_user            <- Enterprise-RAG 的 user 表 + MediAgent 的 med_user（合一，加 role: admin/doctor）
- knowledge_file      <- Enterprise-RAG 的 user_file（上传台账，向量库改为 Chroma 无需 MySQL 父块表）
- inquiry_record      <- MediAgent 的 inquiry_record（患者信息内嵌、过敏史字段打通、thread_id 持久化）
- *_result 三张表     <- MediAgent（新增 citations 列存证据引用 JSON）
- agent_session_his   <- MediAgent（全链路审计）
- workflow_event      <- 新增：阶段流转事件，供前端进度时间线
"""

import datetime as dt

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now(), comment="更新时间"
    )


class SysUser(Base, TimestampMixin):
    """系统用户（admin 管理员 / doctor 医生）。密码只存 bcrypt 哈希。"""

    __tablename__ = "sys_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(200), nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="doctor", comment="admin|doctor")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class AlgorithmApiKey(Base, TimestampMixin):
    """算法接口调用方密钥（面向外部系统的服务间鉴权，与用户 JWT 分离）。"""

    __tablename__ = "algorithm_api_key"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), nullable=False, comment="调用方名称")
    key_hash: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True, comment="sha256(api_key)")
    rate_limit_qps: Mapped[int] = mapped_column(Integer, nullable=False, default=5, comment="每秒请求上限")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class KnowledgeFile(Base, TimestampMixin):
    """医学知识库文件台账（向量库为 Chroma，按 collection=md5(文件名) 隔离）。"""

    __tablename__ = "knowledge_file"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    collection_name: Mapped[str] = mapped_column(String(64), nullable=False, comment="md5(文件名)")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uploader_id: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class InquiryRecord(Base, TimestampMixin):
    """问诊记录（主状态机）。

    main_status: 0初始 1症状分析 2待化验 3诊断分析 4治疗建议 5完成 99失败
    """

    __tablename__ = "inquiry_record"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doctor_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="接诊医生=sys_user.id")
    patient_name: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    patient_age: Mapped[int] = mapped_column(Integer, nullable=True)
    patient_gender: Mapped[str] = mapped_column(String(10), nullable=False, default="")
    allergy_history: Mapped[str] = mapped_column(
        Text, nullable=False, default="[]", comment="过敏史 JSON 数组"
    )
    symptom_description: Mapped[str] = mapped_column(Text, nullable=False)
    use_rag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="本次诊疗是否启用 RAG")
    main_status: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_reason: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    thread_id: Mapped[str] = mapped_column(
        String(100), nullable=False, default="", comment="LangGraph 线程 ID（检查点恢复用）"
    )


class SymptomAnalysisResult(Base, TimestampMixin):
    __tablename__ = "symptom_analysis_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inquiry_record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_result: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="证据引用 JSON")
    is_agreed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    disagree_reason: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    doctor_result: Mapped[str] = mapped_column(Text, nullable=False, default="")


class DiseaseDiagnosisResult(Base, TimestampMixin):
    __tablename__ = "disease_diagnosis_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inquiry_record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_result: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    is_agreed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    disagree_reason: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    doctor_result: Mapped[str] = mapped_column(Text, nullable=False, default="")


class TreatmentSuggestionResult(Base, TimestampMixin):
    __tablename__ = "treatment_suggestion_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inquiry_record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_result: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    is_agreed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    disagree_reason: Mapped[str] = mapped_column(String(400), nullable=False, default="")
    doctor_result: Mapped[str] = mapped_column(Text, nullable=False, default="")


class AgentSessionHis(Base, TimestampMixin):
    """Agent 调用全链路审计（医疗可追溯）。"""

    __tablename__ = "agent_session_his"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inquiry_record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    agent_type: Mapped[int] = mapped_column(Integer, nullable=False, comment="1症状 2诊断 3治疗")
    prompt_version_code: Mapped[str] = mapped_column(String(20), nullable=False, default="v2.0")
    agent_status: Mapped[int] = mapped_column(Integer, nullable=False, comment="1分析中 2成功 3失败")
    failed_reason: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    request_body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    response_body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    rag_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    retrieval_query: Mapped[str] = mapped_column(Text, nullable=False, default="", comment="本次检索使用的查询")


class WorkflowEvent(Base, TimestampMixin):
    """工作流阶段事件，供前端时间线展示与审计。"""

    __tablename__ = "workflow_event"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inquiry_record_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, comment="start|done|failed|waiting")
    detail: Mapped[str] = mapped_column(String(500), nullable=False, default="")


class PatientChatSession(Base, TimestampMixin):
    """患者端聊天会话（免登录：以 session_key 关联浏览器 localStorage）。"""

    __tablename__ = "patient_chat_session"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True, comment="客户端生成/后端签发的会话标识"
    )
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="累计消息数")
    client_ip: Mapped[str] = mapped_column(String(50), nullable=False, default="", comment="最近来源 IP（限流审计）")


class PatientChatMessage(Base, TimestampMixin):
    """患者端聊天消息（含意图标签，全链路落库审计）。"""

    __tablename__ = "patient_chat_message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True, comment="会话 ID")
    role: Mapped[str] = mapped_column(String(10), nullable=False, comment="user|assistant")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(
        String(20), nullable=False, default="", comment="chitchat|consult|inquiry|emergency（仅 user 消息）"
    )
    citations: Mapped[str] = mapped_column(Text, nullable=False, default="[]", comment="证据引用 JSON（仅 assistant 消息）")
