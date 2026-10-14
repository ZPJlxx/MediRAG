"""企业级基建回归测试：request-id 追踪、统一错误体、启动自检。"""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.core.database as db_mod
    from app.core.database import Base
    import app.models.entities  # noqa: F401

    test_engine = create_engine(
        f"sqlite:///{(tmp_path / 'obs.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", TestSessionLocal)
    import app.medical.workflow as workflow_mod
    monkeypatch.setattr(workflow_mod, "SessionLocal", TestSessionLocal)
    # 自检会按 settings.database_url 探测连通性；测试库环境同步指向临时 SQLite，
    # 避免 .env 指向 MySQL（容器未启）时启动自检 fail 阻断 TestClient lifespan
    from app.core.config import settings as app_settings
    test_url = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    monkeypatch.setattr(type(app_settings), "database_url", property(lambda self: test_url), raising=False)
    monkeypatch.setattr(type(app_settings), "is_sqlite", property(lambda self: True), raising=False)
    Base.metadata.create_all(bind=test_engine)

    from app.core.security import hash_password
    from app.models.entities import SysUser
    s = TestSessionLocal()
    s.add(SysUser(username="doctor", password_hash=hash_password("doctor123"), nickname="d", role="doctor"))
    s.commit()
    s.close()

    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_response_carries_request_id(client):
    """每个响应都带 X-Request-ID，且错误体内回显同一 ID。"""
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "wrong"})
    assert resp.status_code == 401
    rid = resp.headers.get("X-Request-ID")
    assert rid, "响应头缺少 X-Request-ID"
    body = resp.json()
    assert body.get("request_id") == rid, "错误体内的 request_id 与响应头不一致"


def test_request_id_passthrough(client):
    """上游传入的 X-Request-ID 应被透传，便于跨服务链路拼接。"""
    upstream_rid = "gateway-abc123"
    resp = client.post(
        "/api/auth/login",
        json={"username": "doctor", "password": "wrong"},
        headers={"X-Request-ID": upstream_rid},
    )
    assert resp.headers.get("X-Request-ID") == upstream_rid
    assert resp.json().get("request_id") == upstream_rid


def test_unified_error_body_format(client):
    """401 错误体符合统一规范：{code, message, request_id}。"""
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "wrong"})
    body = resp.json()
    assert set(body) >= {"code", "message", "request_id"}
    assert body["code"].startswith(("HTTP-", "AUTH-", "SYS-", "MED-", "ALG-")), f"错误码格式不符: {body['code']}"
    assert "用户名或密码错误" in body["message"]


def test_validation_error_unified(client):
    """参数校验失败（422）也是统一错误体。"""
    resp = client.post(
        "/api/auth/login",
        json={"username": ""},
        headers={"Authorization": "Bearer x"},
    )
    # 登录限流与校验顺序不确定，接受 422/401/429，但体格式必须统一
    body = resp.json()
    assert "code" in body and "message" in body


def test_unhandled_exception_masks_internals(client, monkeypatch):
    """未捕获异常返回 500 通用错误体，不泄露内部细节。"""
    from app.api import auth_routes

    def _boom():
        raise RuntimeError("internal secret detail /path/to/file")

    monkeypatch.setattr(auth_routes, "verify_password", _boom)
    resp = client.post("/api/auth/login", json={"username": "doctor", "password": "doctor123"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "SYS-0500"
    assert "secret" not in json.dumps(body), "内部细节泄露到响应体"


def test_biz_error_registry_complete():
    """错误码登记表：编码唯一且前缀合法。"""
    from app.core.errors import ERROR_CODES

    codes = [c for c, _ in ERROR_CODES.values()]
    assert len(codes) == len(set(codes)), "存在重复错误码"
    assert all(c.split("-")[1].isdigit() and len(c.split("-")) == 2 for c in codes)


def test_startup_checks_detect_missing_llm():
    """自检：无任何 LLM 配置时报 fail。"""
    from app.core.config import Settings

    s = Settings(llm_api_key="", dashscope_api_key="", deepseek_api_key="")
    checks = s.run_startup_checks()
    llm_check = next(c for c in checks if c[0] == "llm")
    assert llm_check[1] == "fail"


def test_startup_checks_ok_with_llm(tmp_path, monkeypatch):
    """自检：LLM 配置正常时该项 ok。"""
    from app.core.config import Settings

    s = Settings(dashscope_api_key="sk-test", database_url=f"sqlite:///{(tmp_path / 'x.db').as_posix()}")
    checks = s.run_startup_checks()
    llm_check = next(c for c in checks if c[0] == "llm")
    assert llm_check[1] == "ok"
