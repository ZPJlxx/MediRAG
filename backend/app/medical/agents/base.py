"""医疗 Agent 公共基类。

统一原 MediAgent-LLM 三个 Agent 中重复的骨架：
审计会话创建 -> 患者历史 -> RAG 检索 -> LLM 调用（注入参考资料）
-> Pydantic 校验 -> 引用映射回真实证据 -> 结果落库 -> 审计更新

子类只需实现 retrieve_query / call_llm / validate / persist 四个钩子。
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.dao.medical import AgentSessionDAO
from app.rag.evidence import format_reference_block, map_llm_citations
from app.rag.knowledge_manager import knowledge_manager
from app.schemas.medical import AgentStatus, AgentType, Citation


@dataclass
class PatientKey:
    """患者标识（本项目患者信息内嵌于问诊记录，用姓名+当前问诊ID定位历史）。"""

    patient_name: str = ""
    current_inquiry_id: int = 0


class BaseMedicalAgent(ABC):
    agent_type: AgentType
    prompt_version: str = "v2.0-rag"

    # ---------- 子类钩子 ----------
    @abstractmethod
    def build_retrieval_query(self, request) -> str:
        """构造本阶段的检索查询。"""

    @abstractmethod
    def call_llm(self, request, reference_block: str, patient_history: list[dict]) -> dict:
        """调用 LLM（注入参考资料块），返回原始 dict。"""

    @abstractmethod
    def validate(self, llm_result: dict) -> object:
        """Pydantic 校验，返回带 citations 字段的结果模型。"""

    @abstractmethod
    def persist(self, session: Session, inquiry_record_id: int, result_json: str, citations_json: str) -> None:
        """结果落库。"""

    def fetch_patient_history(self, session: Session, patient_key) -> list[dict]:
        """患者历史（默认无，子类按需覆写）。"""
        return []

    # ---------- 公共执行骨架 ----------
    def execute(self, request, inquiry_record_id: int, patient_key: PatientKey | None = None) -> dict:
        session = SessionLocal()
        session_record = None
        rag_enabled = False
        retrieval_query = ""
        try:
            retrieval_query = self.build_retrieval_query(request)
            # RAG 开关：问诊记录级 use_rag（A/B 评估用）优先，全局配置兜底
            request_use_rag = getattr(request, "use_rag", None)
            rag_enabled = (
                (request_use_rag if request_use_rag is not None else settings.medical_rag_enabled)
                and bool(knowledge_manager.list_files())
            )

            # 1. 审计：记录本次 Agent 调用（含检索行为）
            session_record = AgentSessionDAO.create_session(
                session=session,
                inquiry_record_id=inquiry_record_id,
                agent_type=self.agent_type,
                prompt_version_code=self.prompt_version,
                request_body=retrieval_query
                or json.dumps(getattr(request, "model_dump", lambda: {})(), default=str, ensure_ascii=False),
                rag_enabled=rag_enabled,
                retrieval_query=retrieval_query,
            )

            # 2. RAG 检索 -> 证据列表 -> 参考资料块
            evidence: list[Citation] = []
            if rag_enabled:
                try:
                    evidence = knowledge_manager.search_citations(retrieval_query)
                    logger.info(
                        f"[{self.agent_type.name}] RAG 检索到 {len(evidence)} 条证据: "
                        + "; ".join(f"《{c.source}》{c.location}" for c in evidence[:3])
                    )
                except Exception as exc:
                    logger.warning(f"[{self.agent_type.name}] RAG 检索失败，降级为无参考模式: {exc}")
                    evidence = []
            reference_block = format_reference_block(evidence)

            # 3. LLM 调用
            patient_history = self.fetch_patient_history(session, patient_key)
            llm_result = self.call_llm(request, reference_block, patient_history)
            raw_citations = llm_result.get("citations") or []

            # 4. 校验 + 引用映射（LLM 只能引用已提供的证据编号，防编造出处）。
            #    LLM 返回的 citations 是 {ref, quote} 声明，不是完整 Citation，
            #    先剥离再校验，校验通过后由 map_llm_citations 映射回真实证据。
            llm_result.pop("citations", None)
            validated = self.validate(llm_result)
            validated.citations = map_llm_citations(raw_citations, evidence)

            result_json = validated.to_json()
            citations_json = json.dumps(
                [c.model_dump() for c in validated.citations], ensure_ascii=False
            )

            # 5. 落库 + 审计更新
            self.persist(session, inquiry_record_id, result_json, citations_json)
            AgentSessionDAO.update_result(
                session=session,
                session_id=session_record.id,
                response_body=result_json,
                agent_status=AgentStatus.SUCCESS,
            )
            logger.info(
                f"[{self.agent_type.name}] 完成，证据引用 {len(validated.citations)} 条，"
                f"问询记录 {inquiry_record_id}"
            )
            return {"success": True, "session_id": session_record.id, "result": validated.model_dump()}

        except Exception as exc:
            logger.error(f"[{self.agent_type.name}] 失败: {exc}")
            if session_record is not None:
                AgentSessionDAO.update_result(
                    session=session,
                    session_id=session_record.id,
                    response_body="",
                    agent_status=AgentStatus.FAILED,
                    failed_reason=str(exc),
                )
            return {"success": False, "error": str(exc)}
        finally:
            session.close()
