"""算法接口（v1）回归测试：API Key 鉴权、限流、任务状态语义。

隔离策略：测试专用 SQLite 引擎直接替换 app.core.database 的 engine/SessionLocal
（get_db 运行时查模块全局，补丁生效）；检索底层调用 mock，不触真实向量库/API。
"""

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
    Base.metadata.create_all(bind=test_engine)

    # 种子管理员（签发 key 用）
    from app.core.security import hash_password
    from app.models.entities import SysUser
    session = TestSessionLocal()
    session.add(SysUser(username="admin", password_hash=hash_password("admin123"), nickname="t", role="admin"))
    session.commit()
    session.close()

    # 检索底层 mock：算法接口测试不依赖真实向量库与 embedding API
    from app.rag.knowledge_manager import knowledge_manager
    monkeypatch.setattr(
        knowledge_manager, "search_citations",
        lambda query, top_k=None: [],
    )

    # 限流器是模块级单例，清掉上个测试的窗口残留（新库 key id 重新从 1 计）
    from app.api.algorithm_deps import _limiter
    _limiter._hits.clear()

    # 启动自检按 settings.database_url 探测，测试环境指向临时 SQLite
    from app.core.config import settings as app_settings
    _probe_url = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    monkeypatch.setattr(type(app_settings), "database_url", property(lambda self: _probe_url), raising=False)
    monkeypatch.setattr(type(app_settings), "is_sqlite", property(lambda self: True), raising=False)

    from app.main import app
    with TestClient(app) as c:
        yield c


def _issue_key(client: TestClient, qps: int = 2) -> str:
    """admin 登录 -> 签发算法 key -> 返回明文。"""
    login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    assert login.status_code == 200
    token = login.json()["token"]
    resp = client.post(
        "/api/v1/algorithm/keys",
        json={"name": "test-caller", "rate_limit_qps": qps},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["api_key"].startswith("mdrg_")
    return body["api_key"]


def test_retrieval_requires_api_key(client):
    resp = client.post("/api/v1/algorithm/retrieval", json={"query": "高血压"})
    assert resp.status_code == 401


def test_retrieval_rejects_invalid_key(client):
    resp = client.post(
        "/api/v1/algorithm/retrieval",
        json={"query": "高血压"},
        headers={"X-Api-Key": "mdrg_invalid000000000000000000000000000"},
    )
    assert resp.status_code == 401
    assert "无效" in resp.json()["message"]


def test_api_key_management_requires_admin(client):
    resp = client.post("/api/v1/algorithm/keys", json={"name": "x"})
    assert resp.status_code == 401


def test_retrieval_ok_with_key(client):
    api_key = _issue_key(client, qps=10)
    resp = client.post(
        "/api/v1/algorithm/retrieval",
        json={"query": "高血压诊断标准", "top_k": 3},
        headers={"X-Api-Key": api_key},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["latency_ms"] >= 0
    assert "辅助参考" in body["disclaimer"]


def test_rate_limit_enforced(client):
    api_key = _issue_key(client, qps=2)
    codes = [
        client.post(
            "/api/v1/algorithm/retrieval",
            json={"query": "高血压诊断标准"},
            headers={"X-Api-Key": api_key},
        ).status_code
        for _ in range(3)
    ]
    assert codes[:2] == [200, 200]
    assert codes[2] == 429, f"限流未生效: {codes}"


def test_revoked_key_rejected(client):
    api_key = _issue_key(client)
    login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    token = login.json()["token"]
    keys = client.get("/api/v1/algorithm/keys", headers={"Authorization": f"Bearer {token}"}).json()["keys"]
    key_id = keys[0]["id"]
    assert client.delete(f"/api/v1/algorithm/keys/{key_id}", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    resp = client.post(
        "/api/v1/algorithm/retrieval",
        json={"query": "高血压"},
        headers={"X-Api-Key": api_key},
    )
    assert resp.status_code == 401


def test_key_list_hides_hash(client):
    """密钥列表绝不回传哈希/明文。"""
    _issue_key(client)
    login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
    keys = client.get(
        "/api/v1/algorithm/keys", headers={"Authorization": f"Bearer {login.json()['token']}"}
    ).json()["keys"]
    assert len(keys) == 1
    assert "key_hash" not in keys[0] and "api_key" not in keys[0]


def test_task_status_mapping():
    """状态机 -> 算法任务三态语义映射。"""
    from app.api.algorithm_routes import _to_task_response

    assert _to_task_response(1, {"main_status": 0}).status == "queued"
    assert _to_task_response(1, {"main_status": 2, "waiting_lab": True}).status == "running"
    assert _to_task_response(1, {"main_status": 2, "waiting_lab": True}).waiting_lab is True
    assert _to_task_response(1, {"main_status": 5}).status == "succeeded"
    succeeded = _to_task_response(1, {"main_status": 5, "diagnosis": {}})
    assert succeeded.result is not None and "辅助参考" in succeeded.disclaimer
    assert _to_task_response(1, {"main_status": 99}).status == "failed"
    running = _to_task_response(1, {"main_status": 3})
    assert running.result is None
