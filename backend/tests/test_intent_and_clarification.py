"""意图识别 + 患者聊天 + 医生端追问澄清回归测试。

隔离策略与 test_algorithm_api 一致：测试专用 SQLite 引擎替换全局 engine/SessionLocal；
LLM 与检索全部 mock，不触真实网关/向量库。workflow 用 __new__ 绕过 __init__
（避免真实 SqliteSaver 检查点连接），与 test_concurrency_control 同款做法。
"""

import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.core.database as db_mod
    from app.core.database import Base
    import app.models.entities  # noqa: F401  确保模型注册

    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", TestSessionLocal)
    # 启动自检按 settings.database_url 探测，测试环境指向临时 SQLite（与 test_sse 同款）
    import app.medical.workflow as workflow_mod
    monkeypatch.setattr(workflow_mod, "SessionLocal", TestSessionLocal)
    import app.medical.patient_chat as chat_mod
    monkeypatch.setattr(chat_mod, "SessionLocal", TestSessionLocal)
    from app.core.config import settings as app_settings
    _probe_url = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    monkeypatch.setattr(type(app_settings), "database_url", property(lambda self: _probe_url), raising=False)
    monkeypatch.setattr(type(app_settings), "is_sqlite", property(lambda self: True), raising=False)
    Base.metadata.create_all(bind=test_engine)

    from app.main import app
    with TestClient(app) as c:
        yield c


# ---------- 意图识别（规则短路，不碰 LLM）----------
class TestIntentClassifierRules:
    def test_emergency_short_circuit(self, monkeypatch):
        """急症红旗词必须零 LLM 成本短路（LLM 被调用即失败）。"""
        from app.medical import intent as intent_mod

        def _no_llm(*a, **kw):
            raise AssertionError("急症不应触发 LLM 分类")

        monkeypatch.setattr(intent_mod.llm_client, "classify_intent", _no_llm)
        result = intent_mod.intent_classifier.classify("我爸爸突然胸口剧痛喘不上气")
        assert result["intent"] == "emergency"
        assert result["by_rule"] is True

    def test_chitchat_exact_greeting(self, monkeypatch):
        from app.medical import intent as intent_mod

        monkeypatch.setattr(
            intent_mod.llm_client, "classify_intent",
            lambda *a, **kw: pytest.fail("纯问候不应触发 LLM"),
        )
        assert intent_mod.intent_classifier.classify("你好")["intent"] == "chitchat"

    def test_greeting_with_symptom_is_not_chitchat_by_rule(self, monkeypatch):
        """「你好，我头晕」不能被问候快路径吞掉——必须交给 LLM 判断。"""
        from app.medical import intent as intent_mod

        called = {"n": 0}

        def _fake_llm(message, history_text=""):
            called["n"] += 1
            return {"intent": "inquiry", "confidence": 0.9, "reasoning": "t"}

        monkeypatch.setattr(intent_mod.llm_client, "classify_intent", _fake_llm)
        result = intent_mod.intent_classifier.classify("你好，我最近老头晕")
        assert result["intent"] == "inquiry"
        assert called["n"] == 1

    def test_llm_invalid_intent_falls_back_to_consult(self, monkeypatch):
        """LLM 返回非法意图值 -> 兜底 consult（保守侧），不抛错。"""
        from app.medical import intent as intent_mod

        monkeypatch.setattr(
            intent_mod.llm_client, "classify_intent",
            lambda *a, **kw: {"intent": "随便聊聊", "confidence": 0.9},
        )
        result = intent_mod.intent_classifier.classify("帮我看看这个药")
        assert result["intent"] == "consult"

    def test_llm_exception_falls_back_to_consult(self, monkeypatch):
        from app.medical import intent as intent_mod

        monkeypatch.setattr(
            intent_mod.llm_client, "classify_intent",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("gateway down")),
        )
        assert intent_mod.intent_classifier.classify("我头晕")["intent"] == "consult"


# ---------- 患者聊天端点 ----------
class TestPatientChatAPI:
    def test_chat_flow_with_citations(self, client, monkeypatch):
        """问诊消息 -> RAG 答疑 + 引用映射（防编造：越界编号被丢弃）。"""
        from app.schemas.medical import Citation
        from app.rag import evidence as evidence_mod

        evidence = [
            Citation(source="高血压指南.pdf", location="第 3 页", snippet="头晕常见于血压升高"),
            Citation(source="百科.csv", location="行 12", snippet="头晕需排查贫血"),
        ]
        monkeypatch.setattr(
            "app.medical.patient_chat.knowledge_manager.search_citations",
            lambda query, top_k=None: evidence,
            # patient_chat 模块内以 `from ... import knowledge_manager` 引用单例对象，
            # patch 其方法而非替换名字
        )
        monkeypatch.setattr(
            type(__import__("app.rag.knowledge_manager", fromlist=["knowledge_manager"]).knowledge_manager),
            "search_citations",
            lambda self, query, top_k=None: evidence,
        )
        monkeypatch.setattr(
            "app.medical.patient_chat.llm_client.classify_intent",
            lambda message, history_text="": {"intent": "inquiry", "confidence": 0.9, "reasoning": "t"},
        )
        monkeypatch.setattr(
            "app.medical.patient_chat.llm_client.patient_chat_reply",
            lambda messages, intent, reference_block="": {
                "reply": "头晕可能与血压有关，建议监测血压并就诊心内科。",
                "citations": [{"ref": 1, "quote": "头晕常见于血压升高"}, {"ref": 9, "quote": "越界"}],
            },
        )

        resp = client.post("/api/patient/chat", json={"message": "我最近老头晕"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["intent"] == "inquiry"
        assert data["intent_label"] == "症状问诊"
        assert len(data["citations"]) == 1  # ref=9 越界被丢弃
        assert data["citations"][0]["source"] == "高血压指南.pdf"
        assert data["disclaimer"]
        assert data["session_id"].startswith("pchat_")

    def test_emergency_returns_fixed_reply_without_llm(self, client, monkeypatch):
        """急症：固定警示文案，不生成不检索（内容可控、零配额）。"""
        monkeypatch.setattr(
            "app.medical.patient_chat.llm_client.patient_chat_reply",
            lambda *a, **kw: pytest.fail("急症不应生成 LLM 回复"),
        )
        resp = client.post("/api/patient/chat", json={"message": "家人突然昏迷叫不醒"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "emergency"
        assert "120" in data["reply"]

    def test_chat_rate_limit(self, client, monkeypatch):
        """免登录接口的 IP 限流闸门：超 20 次/分钟返回 429。"""
        from app.api.rate_limit import _limiter
        _limiter.reset()
        monkeypatch.setattr(
            "app.medical.patient_chat.llm_client.classify_intent",
            lambda message, history_text="": {"intent": "chitchat", "confidence": 1.0, "reasoning": "t"},
        )
        monkeypatch.setattr(
            "app.medical.patient_chat.llm_client.patient_chat_reply",
            lambda messages, intent, reference_block="": {"reply": "你好呀", "citations": []},
        )
        codes = [
            client.post("/api/patient/chat", json={"message": "你好"}).status_code
            for _ in range(22)
        ]
        assert codes[:20] == [200] * 20
        assert codes[20] == 429


# ---------- 医生端追问澄清（workflow 节点级）----------
class TestClarificationWorkflow:
    def _bare_workflow(self):
        from app.medical.workflow import MedicalWorkflow

        wf = MedicalWorkflow.__new__(MedicalWorkflow)
        wf._llm_slots = threading.Semaphore(2)
        wf._run_lock = threading.Lock()
        return wf

    def test_vague_description_triggers_clarify_state(self, monkeypatch):
        """「喉咙不舒服」-> 充分性不足 -> clarify_questions 写入 state。"""
        from app.schemas.medical import InquiryStatus
        from app.medical import workflow as wf_mod

        monkeypatch.setattr(
            wf_mod.InquiryRecordDAO, "update_status",
            lambda session, rid, status, **kw: None,
        )
        monkeypatch.setattr(
            wf_mod.WorkflowEventDAO, "log", lambda session, rid, *a, **kw: None,
        )
        monkeypatch.setattr(
            wf_mod.llm_client, "check_symptom_sufficiency",
            lambda **kw: {
                "sufficient": False,
                "missing_dimensions": ["持续时间", "性质与程度"],
                "clarify_questions": [
                    {"dimension": "持续时间", "question": "喉咙不舒服持续多久了？"},
                    {"dimension": "性质与程度", "question": "是疼痛、干痒还是异物感？"},
                ],
                "reasoning": "主诉模糊",
            },
        )

        wf = self._bare_workflow()
        state = {
            "inquiry_record_id": 1,
            "symptom_description": "喉咙不舒服",
            "patient_age": 30,
            "patient_gender": "男",
            "clarify_rounds": 0,
        }
        result = wf._check_clarification(state)
        assert len(result["clarify_questions"]) == 2
        assert wf._should_wait_for_clarification(result) == "wait"

    def test_sufficient_description_skips_clarification(self, monkeypatch):
        from app.medical import workflow as wf_mod

        monkeypatch.setattr(
            wf_mod.WorkflowEventDAO, "log", lambda session, rid, *a, **kw: None,
        )
        monkeypatch.setattr(
            wf_mod.llm_client, "check_symptom_sufficiency",
            lambda **kw: {"sufficient": True, "missing_dimensions": [], "clarify_questions": [], "reasoning": "信息完整"},
        )

        wf = self._bare_workflow()
        state = {"inquiry_record_id": 1, "symptom_description": "右下腹转移性疼痛 6 小时，伴发热 38.5℃、恶心", "clarify_rounds": 0}
        result = wf._check_clarification(state)
        assert result["clarify_questions"] == []
        assert wf._should_wait_for_clarification(result) == "continue"

    def test_clarify_failure_degrades_to_continue(self, monkeypatch):
        """充分性检查异常 -> 降级直接分析，不能挡住问诊链路。"""
        from app.medical import workflow as wf_mod

        monkeypatch.setattr(
            wf_mod.WorkflowEventDAO, "log", lambda session, rid, *a, **kw: None,
        )

        def _boom(**kw):
            raise RuntimeError("gateway down")

        monkeypatch.setattr(wf_mod.llm_client, "check_symptom_sufficiency", _boom)

        wf = self._bare_workflow()
        result = wf._check_clarification({"inquiry_record_id": 1, "symptom_description": "头晕", "clarify_rounds": 0})
        assert result["clarify_questions"] == []
        assert wf._should_wait_for_clarification(result) == "continue"

    def test_second_round_never_rechecks(self, monkeypatch):
        """已追问过 1 轮后不再复查（防无限追问）。"""
        from app.medical import workflow as wf_mod

        monkeypatch.setattr(
            wf_mod.llm_client, "check_symptom_sufficiency",
            lambda **kw: pytest.fail("第二轮不应再调充分性检查"),
        )
        wf = self._bare_workflow()
        result = wf._check_clarification({"inquiry_record_id": 1, "symptom_description": "x", "clarify_rounds": 1})
        assert wf._should_wait_for_clarification(result) == "continue"

    def test_wait_node_merges_answers_into_description(self, monkeypatch):
        """resume 后补答合并进 symptom_description（结构化追加）。"""
        from langgraph.types import Command

        from app.medical import workflow as wf_mod

        monkeypatch.setattr(
            wf_mod.InquiryRecordDAO, "update_status", lambda session, rid, status, **kw: None,
        )
        monkeypatch.setattr(
            wf_mod.WorkflowEventDAO, "log", lambda session, rid, *a, **kw: None,
        )
        # 模拟 interrupt 暂停后医生回填恢复
        monkeypatch.setattr(
            wf_mod, "interrupt",
            lambda payload: [
                {"question": "持续多久了？", "answer": "三天了，断断续续"},
                {"question": "什么感觉？", "answer": "干痒，早上明显"},
            ],
        )

        wf = self._bare_workflow()
        state = {
            "inquiry_record_id": 1,
            "symptom_description": "喉咙不舒服",
            "clarify_questions": [{"dimension": "持续时间", "question": "持续多久了？"}],
            "clarify_rounds": 0,
        }
        result = wf._wait_for_clarification(state)
        assert result["clarify_rounds"] == 1
        assert result["clarify_questions"] == []
        assert "【医生补充问诊】" in result["symptom_description"]
        assert "三天了，断断续续" in result["symptom_description"]
        assert result["symptom_description"].startswith("喉咙不舒服")


# ---------- Schema 校验 ----------
class TestClarifySchema:
    def test_clarify_request_requires_answers(self):
        from pydantic import ValidationError

        from app.schemas.medical import ClarifyRequest

        with pytest.raises(ValidationError):
            ClarifyRequest(answers=[])
        req = ClarifyRequest(answers=[{"question": "持续多久？", "answer": "三天"}])
        assert req.answers[0].answer == "三天"

    def test_waiting_clarification_status_value(self):
        from app.schemas.medical import InquiryStatus, STAGE_LABELS

        assert InquiryStatus.WAITING_CLARIFICATION.value == 6
        assert STAGE_LABELS[InquiryStatus.WAITING_CLARIFICATION] == "等待补充信息"
