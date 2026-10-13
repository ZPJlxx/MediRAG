"""医疗诊疗工作流（升级自 MediAgent-LLM 的 medical_workflow.py）。

保留原项目最有含金量的设计：
- LangGraph StateGraph 状态机 + interrupt()/Command(resume=...) 人机协同：
  「等待化验结果数小时」建模为工作流断点而非轮询
- 条件边统一错误路由

修复原项目已知缺陷：
1. MemorySaver -> SqliteSaver：检查点持久化，进程重启后 thread_id 仍可恢复
2. _thread_map 进程内字典 -> inquiry_record.thread_id 落库
3. get_workflow_status() 写死 "running" -> 从数据库真实状态机 + 检查点快照合成
4. 过敏史写死 [] -> 从问诊记录读取真实过敏史并传入治疗 Agent
5. 同步 graph.invoke 在 async 路由中阻塞事件循环 -> 线程池执行
"""

import json
import operator
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.types import Command, Send, interrupt
from loguru import logger

from app.core.config import settings
from app.core.database import SessionLocal
from app.dao.medical import InquiryRecordDAO, ResultDAO, WorkflowEventDAO, safe_json_loads
from app.medical.agents.base import PatientKey
from app.medical.agents.diagnosis_analysis import diagnosis_analysis_agent
from app.medical.agents.hypothesis_verification import (
    _VerificationRequest,
    hypothesis_verification_agent,
)
from app.medical.agents.symptom_analysis import symptom_analysis_agent
from app.medical.agents.treatment_advice import treatment_advice_agent
from app.schemas.medical import InquiryStatus

STATUS_LABELS = {
    0: "已创建", 1: "症状分析", 2: "等待化验", 3: "诊断分析", 4: "治疗建议", 5: "已完成", 99: "失败",
}


class MedicalWorkflowState(TypedDict, total=False):
    inquiry_record_id: int
    doctor_id: int
    patient_name: str
    patient_age: int | None
    patient_gender: str
    allergy_history: list[str]
    symptom_description: str
    use_rag: bool
    symptom_analysis_result: dict | None
    lab_tests_needed: list[str]
    lab_results: list[dict]
    # ---- 鉴别假设并行核验（Send fan-out / map-reduce）----
    hypotheses: list[dict]
    # 并行分支同时写同一键，必须声明 reducer（列表累加），否则并发写冲突
    hypothesis_verifications: Annotated[list, operator.add]
    verification_summary: str
    verification_ranking: list[dict]
    diagnosis_result: dict | None
    treatment_advice: dict | None
    current_status: str
    error_message: str | None
    workflow_completed: bool


class MedicalWorkflow:
    """症状分析 -> [中断等化验] -> 诊断分析 -> 治疗建议 的多 Agent 编排。"""

    def __init__(self):
        settings.ensure_runtime_directories()
        # SqliteSaver 需要显式 sqlite3 连接（from_conn_string 返回上下文管理器）
        self._checkpoint_conn = sqlite3.connect(
            str(settings.checkpoint_db_path), check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_conn)
        self.graph = self._build_workflow_graph()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mediworkflow")
        self._run_lock = threading.Lock()  # SQLite 检查点写并发保护

    # ---------- 图构建 ----------
    def _build_workflow_graph(self):
        workflow = StateGraph(MedicalWorkflowState)

        workflow.add_node("create_inquiry", self._create_inquiry_record)
        workflow.add_node("symptom_analysis", self._perform_symptom_analysis)
        workflow.add_node("check_lab_needed", self._check_lab_tests_needed)
        workflow.add_node("wait_for_lab", self._wait_for_lab_results)
        workflow.add_node("prepare_hypotheses", self._prepare_hypotheses)
        workflow.add_node("verify_hypothesis", self._verify_hypothesis)
        workflow.add_node("merge_verifications", self._merge_verifications)
        workflow.add_node("diagnosis_analysis", self._perform_diagnosis_analysis)
        workflow.add_node("treatment_advice", self._perform_treatment_advice)
        workflow.add_node("complete_workflow", self._complete_workflow)
        workflow.add_node("handle_error", self._handle_error)

        workflow.set_entry_point("create_inquiry")

        workflow.add_conditional_edges(
            "create_inquiry", self._route_after_node,
            {"continue": "symptom_analysis", "error": "handle_error"},
        )
        workflow.add_conditional_edges(
            "symptom_analysis", self._route_after_node,
            {"continue": "check_lab_needed", "error": "handle_error"},
        )
        workflow.add_conditional_edges(
            "check_lab_needed", self._should_wait_for_lab,
            {"wait": "wait_for_lab", "continue": "prepare_hypotheses"},
        )
        workflow.add_edge("wait_for_lab", "prepare_hypotheses")
        # fan-out：一个假设一个 Send 分支（同一超步并行），全部完成后才进入 merge
        workflow.add_conditional_edges(
            "prepare_hypotheses", self._fan_out_or_skip,
            ["verify_hypothesis", "diagnosis_analysis"],
        )
        workflow.add_edge("verify_hypothesis", "merge_verifications")
        workflow.add_conditional_edges(
            "merge_verifications", self._route_after_node,
            {"continue": "diagnosis_analysis", "error": "handle_error"},
        )
        workflow.add_conditional_edges(
            "diagnosis_analysis", self._route_after_node,
            {"continue": "treatment_advice", "error": "handle_error"},
        )
        workflow.add_conditional_edges(
            "treatment_advice", self._route_after_node,
            {"continue": "complete_workflow", "error": "handle_error"},
        )
        workflow.add_edge("complete_workflow", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile(checkpointer=self.checkpointer)

    # ---------- 节点 ----------
    def _create_inquiry_record(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        try:
            record = InquiryRecordDAO.create(
                session,
                doctor_id=state["doctor_id"],
                patient_name=state.get("patient_name", ""),
                patient_age=state.get("patient_age"),
                patient_gender=state.get("patient_gender", ""),
                allergy_history=json.dumps(state.get("allergy_history", []), ensure_ascii=False),
                symptom_description=state["symptom_description"],
                use_rag=state.get("use_rag", True),
                main_status=InquiryStatus.INITIAL.value,
            )
            state["inquiry_record_id"] = record.id
            InquiryRecordDAO.update_status(session, record.id, InquiryStatus.SYMPTOM_ANALYSIS)
            WorkflowEventDAO.log(session, record.id, "create_inquiry", "done", f"患者: {record.patient_name}")
            logger.info(f"创建问诊记录成功 ID={record.id}")
        except Exception as exc:
            logger.error(f"创建问诊记录失败: {exc}")
            state["error_message"] = str(exc)
            state["current_status"] = "failed"
        finally:
            session.close()
        return state

    def _perform_symptom_analysis(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        inquiry_id = state.get("inquiry_record_id")
        try:
            WorkflowEventDAO.log(session, inquiry_id, "symptom_analysis", "start")

            class _Req:  # 轻量请求对象（避免为节点单独建 Pydantic 模型）
                symptom_description = state["symptom_description"]
                use_rag = state.get("use_rag", True)

            patient_key = PatientKey(
                patient_name=state.get("patient_name", ""),
                current_inquiry_id=inquiry_id or 0,
            )
            result = symptom_analysis_agent.execute(_Req(), inquiry_id, patient_key)
            if result["success"]:
                state["symptom_analysis_result"] = result["result"]
                state["lab_tests_needed"] = result["result"].get("lab_tests_needed", [])
                WorkflowEventDAO.log(
                    session, inquiry_id, "symptom_analysis", "done",
                    f"可能疾病数: {len(result['result'].get('possible_diseases', []))}, "
                    f"证据引用: {len(result['result'].get('citations', []))}",
                )
            else:
                raise Exception(result["error"])
        except Exception as exc:
            logger.error(f"症状分析失败: {exc}")
            state["error_message"] = str(exc)
            state["current_status"] = "failed"
            try:
                WorkflowEventDAO.log(session, inquiry_id, "symptom_analysis", "failed", str(exc)[:400])
            except Exception:
                pass
        finally:
            session.close()
        return state

    def _check_lab_tests_needed(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        try:
            if state.get("lab_tests_needed"):
                InquiryRecordDAO.update_status(
                    session, state["inquiry_record_id"], InquiryStatus.WAITING_LAB
                )
                WorkflowEventDAO.log(
                    session, state["inquiry_record_id"], "wait_for_lab", "waiting",
                    f"建议化验: {', '.join(state['lab_tests_needed'])}",
                )
                logger.info(f"需要化验检查: {state['lab_tests_needed']}")
            else:
                InquiryRecordDAO.update_status(
                    session, state["inquiry_record_id"], InquiryStatus.DIAGNOSIS_ANALYSIS
                )
                logger.info("无需化验，直接进入诊断分析")
        finally:
            session.close()
        return state

    def _should_wait_for_lab(self, state: MedicalWorkflowState) -> str:
        if state.get("lab_tests_needed") and not state.get("lab_results"):
            return "wait"
        return "continue"

    def _wait_for_lab_results(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """在检查点暂停工作流，等化验结果异步回传后 Command(resume=...) 恢复。"""
        logger.info(f"等待化验结果，问诊记录 {state.get('inquiry_record_id')}")
        lab_results = interrupt({
            "type": "wait_for_lab",
            "inquiry_record_id": state.get("inquiry_record_id"),
            "lab_tests_needed": state.get("lab_tests_needed", []),
        })
        state["lab_results"] = lab_results or []
        session = SessionLocal()
        try:
            InquiryRecordDAO.update_status(
                session, state["inquiry_record_id"], InquiryStatus.DIAGNOSIS_ANALYSIS
            )
            WorkflowEventDAO.log(
                session, state["inquiry_record_id"], "wait_for_lab", "done",
                f"收到化验结果 {len(state['lab_results'])} 项",
            )
        finally:
            session.close()
        return state

    # ---------- 鉴别假设并行核验（Send fan-out / map-reduce）----------
    MAX_HYPOTHESES = 3  # 并行分支数上限：Top-N 假设

    def _prepare_hypotheses(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """map 阶段准备：从症状分析结果提取 Top-N 待核验假设。"""
        session = SessionLocal()
        inquiry_id = state.get("inquiry_record_id")
        try:
            symptom_result = state.get("symptom_analysis_result") or {}
            candidates = [
                {"disease": str(d.get("disease", "")).strip(), "probability": d.get("probability")}
                for d in symptom_result.get("possible_diseases", [])
                if str(d.get("disease", "")).strip()
            ]
            # 可能性降序，截取 Top-N；空则由诊断节点直接按原流程处理。
            # 注：hypothesis_verifications 由 start_workflow 初始化为 []，
            # 每条工作流线程独立，无需在此重置（reducer 下重赋值也只会上加空列表）。
            candidates.sort(key=lambda h: (h["probability"] if isinstance(h["probability"], (int, float)) else -1), reverse=True)
            state["hypotheses"] = candidates[: self.MAX_HYPOTHESES]
            WorkflowEventDAO.log(
                session, inquiry_id, "prepare_hypotheses", "done",
                f"待核验假设: {'、'.join(h['disease'] for h in state['hypotheses']) or '无'}",
            )
        finally:
            session.close()
        return state

    def _fan_out_or_skip(self, state: MedicalWorkflowState):
        """有假设 -> 每个假设一个 Send 分支（并行超步）；无假设 -> 跳过直接诊断。"""
        hypotheses = state.get("hypotheses") or []
        if not hypotheses:
            return "diagnosis_analysis"

        symptom_result = state.get("symptom_analysis_result") or {}
        return [
            Send(
                "verify_hypothesis",
                {
                    "inquiry_record_id": state.get("inquiry_record_id"),
                    "patient_name": state.get("patient_name", ""),
                    "use_rag": state.get("use_rag", True),
                    "symptom_description": state.get("symptom_description", ""),
                    "key_symptoms": symptom_result.get("key_symptoms", []),
                    "lab_results": state.get("lab_results", []),
                    "hypothesis": hypothesis,
                },
            )
            for hypothesis in hypotheses
        ]

    def _verify_hypothesis(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """并行分支工作单元：核验单个假设（每个分支独立执行，结果累加进 reducer 键）。"""
        inquiry_id = state.get("inquiry_record_id")
        hypothesis = state.get("hypothesis") or {}
        disease = str(hypothesis.get("disease", "")).strip() or "未知假设"
        logger.info(f"[HYPOTHESIS_VERIFICATION] 开始核验假设「{disease}」，问诊 {inquiry_id}")

        request = _VerificationRequest(
            hypothesis=hypothesis,
            symptom_description=state.get("symptom_description", ""),
            key_symptoms=state.get("key_symptoms", []),
            lab_results=state.get("lab_results", []),
            use_rag=state.get("use_rag", True),
        )
        result = hypothesis_verification_agent.execute(request, inquiry_id or 0)

        if result.get("success"):
            verification = result["result"]
        else:
            # 单分支失败不拖垮整体：记为 0 分占位，merge 阶段标注失败
            logger.warning(f"假设「{disease}」核验失败: {result.get('error')}")
            verification = {
                "disease": disease,
                "support_score": 0,
                "matched_criteria": [],
                "missing_information": [],
                "rationale": f"核验失败：{str(result.get('error', ''))[:120]}",
                "citations": [],
            }

        # 写入 reducer 键（Annotated[list, operator.add] 自动累加各分支结果）。
        # 注意：只返回要更新的 channel——Send payload 里的 hypothesis/key_symptoms
        # 不是 state channel，整包返回会触发 InvalidUpdateError。
        return {"hypothesis_verifications": [verification]}

    def _merge_verifications(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        """reduce 阶段：汇总各分支核验结果 -> 支持度榜单 + 注入诊断的摘要块。"""
        session = SessionLocal()
        inquiry_id = state.get("inquiry_record_id")
        try:
            verifications = state.get("hypothesis_verifications") or []
            ranked = sorted(
                verifications,
                key=lambda v: v.get("support_score", 0) if isinstance(v.get("support_score"), (int, float)) else 0,
                reverse=True,
            )
            state["verification_ranking"] = ranked

            lines = ["【鉴别假设并行核验结果（每个假设由独立 Agent 对照患者信息与知识库核验）】"]
            for rank, v in enumerate(ranked, 1):
                matched = "；".join(v.get("matched_criteria", [])[:2]) or "无"
                missing = "；".join(v.get("missing_information", [])[:2]) or "无"
                lines.append(
                    f"{rank}. {v.get('disease', '?')}：支持度 {v.get('support_score', 0)}/100，"
                    f"符合依据：{matched}；缺口：{missing}。{v.get('rationale', '')}"
                )
            state["verification_summary"] = "\n".join(lines)

            WorkflowEventDAO.log(
                session, inquiry_id, "merge_verifications", "done",
                f"核验 {len(ranked)} 个假设，Top1: {ranked[0]['disease']}({ranked[0].get('support_score', 0)}分)"
                if ranked else "无核验结果",
            )
        finally:
            session.close()
        return state

    def _perform_diagnosis_analysis(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        inquiry_id = state.get("inquiry_record_id")
        try:
            WorkflowEventDAO.log(session, inquiry_id, "diagnosis_analysis", "start")

            class _Req:
                symptom_analysis_result = json.dumps(
                    state.get("symptom_analysis_result") or {}, ensure_ascii=False
                )
                lab_results = state.get("lab_results", [])
                patient_name = state.get("patient_name", "")
                use_rag = state.get("use_rag", True)
                verification_summary = state.get("verification_summary", "")

            patient_key = PatientKey(
                patient_name=state.get("patient_name", ""),
                current_inquiry_id=inquiry_id or 0,
            )
            result = diagnosis_analysis_agent.execute(_Req(), inquiry_id, patient_key)
            if result["success"]:
                state["diagnosis_result"] = result["result"]
                WorkflowEventDAO.log(
                    session, inquiry_id, "diagnosis_analysis", "done",
                    f"主诊断: {result['result'].get('main_diagnosis', '')}, "
                    f"证据引用: {len(result['result'].get('citations', []))}",
                )
            else:
                raise Exception(result["error"])
        except Exception as exc:
            logger.error(f"诊断分析失败: {exc}")
            state["error_message"] = str(exc)
            state["current_status"] = "failed"
            try:
                WorkflowEventDAO.log(session, inquiry_id, "diagnosis_analysis", "failed", str(exc)[:400])
            except Exception:
                pass
        finally:
            session.close()
        return state

    def _perform_treatment_advice(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        inquiry_id = state.get("inquiry_record_id")
        try:
            WorkflowEventDAO.log(session, inquiry_id, "treatment_advice", "start")

            class _Req:
                diagnosis_result = json.dumps(
                    state.get("diagnosis_result") or {}, ensure_ascii=False
                )
                allergy_history = state.get("allergy_history", [])
                use_rag = state.get("use_rag", True)

            result = treatment_advice_agent.execute(_Req(), inquiry_id)
            if result["success"]:
                state["treatment_advice"] = result["result"]
                WorkflowEventDAO.log(
                    session, inquiry_id, "treatment_advice", "done",
                    f"用药建议: {len(result['result'].get('medication_advice', []))} 条, "
                    f"证据引用: {len(result['result'].get('citations', []))}",
                )
            else:
                raise Exception(result["error"])
        except Exception as exc:
            logger.error(f"治疗建议失败: {exc}")
            state["error_message"] = str(exc)
            state["current_status"] = "failed"
            try:
                WorkflowEventDAO.log(session, inquiry_id, "treatment_advice", "failed", str(exc)[:400])
            except Exception:
                pass
        finally:
            session.close()
        return state

    def _complete_workflow(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        try:
            InquiryRecordDAO.update_status(
                session, state["inquiry_record_id"], InquiryStatus.COMPLETED
            )
            state["workflow_completed"] = True
            WorkflowEventDAO.log(session, state["inquiry_record_id"], "complete", "done")
        except Exception as exc:
            state["error_message"] = str(exc)
        finally:
            session.close()
        return state

    def _handle_error(self, state: MedicalWorkflowState) -> MedicalWorkflowState:
        session = SessionLocal()
        try:
            InquiryRecordDAO.update_status(
                session, state.get("inquiry_record_id", 0),
                InquiryStatus.FAILED, failed_reason=state.get("error_message", "")[:480],
            )
            logger.error(
                f"工作流失败，问诊记录 {state.get('inquiry_record_id')}: {state.get('error_message')}"
            )
        finally:
            session.close()
        return state

    def _route_after_node(self, state: MedicalWorkflowState) -> str:
        if state.get("current_status") == "failed" or state.get("error_message"):
            return "error"
        return "continue"

    # ---------- 公共 API ----------
    def start_workflow(
        self,
        doctor_id: int,
        patient_name: str,
        patient_age: int | None,
        patient_gender: str,
        allergy_history: list[str],
        symptom_description: str,
        use_rag: bool = True,
    ) -> int:
        """启动工作流；若需化验则在 interrupt 处暂停并返回问诊记录 ID。"""
        initial_state: MedicalWorkflowState = {
            "doctor_id": doctor_id,
            "patient_name": patient_name,
            "patient_age": patient_age,
            "patient_gender": patient_gender,
            "allergy_history": allergy_history,
            "symptom_description": symptom_description,
            "use_rag": use_rag,
            "lab_tests_needed": [],
            "lab_results": [],
            "hypotheses": [],
            "hypothesis_verifications": [],
            "verification_summary": "",
            "verification_ranking": [],
        }
        thread_id = f"medirag_{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": thread_id}}

        inquiry_record_id = None
        with self._run_lock:
            try:
                result = self.graph.invoke(initial_state, config)
                inquiry_record_id = result.get("inquiry_record_id")
            except GraphInterrupt:
                snapshot = self.graph.get_state(config)
                if snapshot and snapshot.values:
                    inquiry_record_id = snapshot.values.get("inquiry_record_id")

        if not inquiry_record_id:
            raise RuntimeError("工作流启动失败：未取得问诊记录 ID")

        session = SessionLocal()
        try:
            InquiryRecordDAO.set_thread_id(session, inquiry_record_id, thread_id)
        finally:
            session.close()
        return inquiry_record_id

    def handle_lab_results(self, inquiry_record_id: int, lab_results: list[dict]):
        """化验结果回传：从数据库取 thread_id，恢复检查点继续诊断。"""
        session = SessionLocal()
        try:
            record = InquiryRecordDAO.get(session, inquiry_record_id)
            thread_id = record.thread_id if record else ""
        finally:
            session.close()

        if not thread_id:
            raise ValueError(
                f"问诊记录 {inquiry_record_id} 无工作流线程（可能未暂停等待化验或已恢复过）"
            )
        config = {"configurable": {"thread_id": thread_id}}
        with self._run_lock:
            result = self.graph.invoke(Command(resume=lab_results), config=config)
        logger.info(f"化验结果已回填，工作流已恢复，问诊记录 {inquiry_record_id}")
        return result

    def get_status(self, inquiry_record_id: int) -> dict:
        """真实状态合成：数据库状态机 + 各阶段结果 + 检查点是否暂停。"""
        session = SessionLocal()
        try:
            record = InquiryRecordDAO.get(session, inquiry_record_id)
            if not record:
                return {"success": False, "error": "问诊记录不存在"}

            symptom = ResultDAO.get_symptom(session, inquiry_record_id)
            diagnosis = ResultDAO.get_diagnosis(session, inquiry_record_id)
            treatment = ResultDAO.get_treatment(session, inquiry_record_id)
            events = WorkflowEventDAO.list_events(session, inquiry_record_id)

            paused = False
            lab_tests_needed = []
            if record.thread_id and record.main_status == InquiryStatus.WAITING_LAB.value:
                try:
                    snapshot = self.graph.get_state({"configurable": {"thread_id": record.thread_id}})
                    paused = bool(snapshot and snapshot.next)
                    if symptom:
                        lab_tests_needed = safe_json_loads(symptom.agent_result, {}).get(
                            "lab_tests_needed", []
                        )
                except Exception:
                    paused = record.main_status == InquiryStatus.WAITING_LAB.value

            return {
                "success": True,
                "inquiry_record_id": inquiry_record_id,
                "main_status": record.main_status,
                "status_label": STATUS_LABELS.get(record.main_status, str(record.main_status)),
                "waiting_lab": paused,
                "lab_tests_needed": lab_tests_needed,
                "patient": {
                    "name": record.patient_name,
                    "age": record.patient_age,
                    "gender": record.patient_gender,
                    "allergy_history": safe_json_loads(record.allergy_history, []),
                    "symptom_description": record.symptom_description,
                },
                "use_rag": record.use_rag,
                "symptom_analysis": safe_json_loads(symptom.agent_result) if symptom else None,
                "symptom_citations": safe_json_loads(symptom.citations, []) if symptom else [],
                "hypothesis_verifications": state_verifications if (state_verifications := self._read_verifications(record)) else [],
                "diagnosis": safe_json_loads(diagnosis.agent_result) if diagnosis else None,
                "diagnosis_citations": safe_json_loads(diagnosis.citations, []) if diagnosis else [],
                "treatment": safe_json_loads(treatment.agent_result) if treatment else None,
                "treatment_citations": safe_json_loads(treatment.citations, []) if treatment else [],
                "events": [
                    {
                        "stage": e.stage, "status": e.status, "detail": e.detail,
                        "time": e.created_at.strftime("%H:%M:%S") if e.created_at else "",
                    }
                    for e in events
                ],
            }
        finally:
            session.close()

    def _read_verifications(self, record) -> list[dict]:
        """从 LangGraph 检查点读取该问诊的假设核验榜单（无则空列表）。"""
        if not record or not record.thread_id:
            return []
        try:
            snapshot = self.graph.get_state({"configurable": {"thread_id": record.thread_id}})
            values = snapshot.values if snapshot else {}
            return values.get("verification_ranking") or []
        except Exception as exc:
            logger.warning(f"读取核验榜单失败（忽略）: {exc}")
            return []

    def run_in_executor(self, func, *args):
        """供 async 路由把同步工作流调用放到线程池，避免阻塞事件循环。"""
        return self._executor.submit(func, *args).result()


# 全局工作流实例
medical_workflow = MedicalWorkflow()
