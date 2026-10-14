"""SSE 状态推送端点回归测试：鉴权（query token）、事件格式、终态收流。"""

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.core.database as db_mod
    from app.core.database import Base
    import app.models.entities  # noqa: F401

    test_engine = __import__("sqlalchemy").create_engine(
        f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestSessionLocal = __import__("sqlalchemy.orm", fromlist=["sessionmaker"]).sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", TestSessionLocal)
    # workflow.py 以 `from ... import SessionLocal` 导入时绑定了引用，
    # 必须同样 patch 它模块内的名字，否则 get_status 会连到真实库
    import app.medical.workflow as workflow_mod
    monkeypatch.setattr(workflow_mod, "SessionLocal", TestSessionLocal)
    Base.metadata.create_all(bind=test_engine)

    from app.core.security import hash_password
    from app.models.entities import InquiryRecord, SysUser
    session = TestSessionLocal()
    session.add(SysUser(username="doctor", password_hash=hash_password("doctor123"), nickname="d", role="doctor"))
    session.commit()
    doctor = session.query(SysUser).filter_by(username="doctor").first()
    # 一条进行中 + 一条已完成的问诊
    session.add(InquiryRecord(doctor_id=doctor.id, patient_name="甲", main_status=1,
                              symptom_description="头痛", allergy_history="[]", use_rag=True))
    session.add(InquiryRecord(doctor_id=doctor.id, patient_name="乙", main_status=5,
                              symptom_description="发热", allergy_history="[]", use_rag=True))
    # 另一名医生的记录（越权测试）
    session.add(SysUser(username="other", password_hash=hash_password("other123"), nickname="o", role="doctor"))
    session.commit()
    other = session.query(SysUser).filter_by(username="other").first()
    session.add(InquiryRecord(doctor_id=other.id, patient_name="丙", main_status=1,
                              symptom_description="咳嗽", allergy_history="[]", use_rag=True))
    session.commit()
    session.close()

    from app.main import app
    with TestClient(app) as c:
        yield c


def _token(client):
    login = client.post("/api/auth/login", json={"username": "doctor", "password": "doctor123"})
    return login.json()["token"]


def test_sse_requires_token(client):
    resp = client.get("/api/workflow/1/events")
    assert resp.status_code == 401


def test_sse_rejects_foreign_record(client):
    token = _token(client)
    # 记录 3 属于 other 医生，doctor 访问应 404（不暴露存在性）
    resp = client.get(f"/api/workflow/3/events?token={token}")
    assert resp.status_code == 404


def test_sse_terminal_stream_pushes_once_and_closes(client):
    """终态记录：推送一份快照后收流。"""
    token = _token(client)
    with client.stream("GET", f"/api/workflow/2/events?token={token}") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_lines())
    # 至少推送了一帧 status 事件，且包含完成态
    assert "event: status" in body
    frame = body.split("data: ", 1)[1]
    status = json.loads(frame)
    assert status["main_status"] == 5
    assert status["patient"]["name"] == "乙"
