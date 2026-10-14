"""并发控制回归测试：LLM 并发闸门（排队/超时）+ 用户端限流（登录/工作流）。"""

import threading
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def _bare_workflow(slots: int):
    """构造不走 __init__ 的工作流实例（避免真实检查点连接）。"""
    from app.medical.workflow import MedicalWorkflow

    wf = MedicalWorkflow.__new__(MedicalWorkflow)
    wf._llm_slots = threading.Semaphore(slots)
    wf._run_lock = threading.Lock()
    return wf


def test_workflow_semaphore_limits_concurrency(monkeypatch):
    """并发闸门：槽位=2 时 4 个并发启动的峰值执行数不超过 2。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "workflow_queue_timeout_seconds", 30, raising=False)
    wf = _bare_workflow(2)

    active = 0
    peak = 0
    lock = threading.Lock()

    class _SlowGraph:
        def invoke(self, state, config):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.15)
            with lock:
                active -= 1
            return {"inquiry_record_id": 1}

    wf.graph = _SlowGraph()
    monkeypatch.setattr(
        "app.medical.workflow.InquiryRecordDAO.set_thread_id", lambda *a, **k: None
    )

    results = []
    def run():
        try:
            wf.start_workflow(1, "p", None, "男", [], "症状描述够长", True)
            results.append("ok")
        except Exception as e:
            results.append(f"err:{e}")

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert all(r == "ok" for r in results), f"有任务异常: {results}"
    assert peak <= 2, f"并发峰值 {peak} 超过槽位 2，闸门失效"


def test_workflow_semaphore_timeout(monkeypatch):
    """槽位被占满且排队超时：应抛 RuntimeError 而非无限等待。"""
    from app.core.config import settings

    monkeypatch.setattr(settings, "workflow_queue_timeout_seconds", 1, raising=False)
    wf = _bare_workflow(1)
    wf._llm_slots.acquire()  # 占住唯一槽位

    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="排队超时"):
        wf.start_workflow(1, "p", None, "男", [], "症状描述够长", True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 3, f"超时等待 {elapsed:.1f}s，明显超过配置的 1s"
    wf._llm_slots.release()


def test_login_rate_limit(tmp_path):
    """登录限流：同 IP 连续第 11 次登录被 429 拒绝。"""
    import app.core.database as dm
    from app.core.database import Base
    import app.models.entities  # noqa: F401

    # 注意不能用 :memory:——每个连接各自建新库，表会消失；用临时文件库
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'rl.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(bind=engine)
    orig_engine, orig_local = dm.engine, dm.SessionLocal
    dm.engine, dm.SessionLocal = engine, TestSession
    # 启动自检按 settings.database_url 探测，测试环境指向临时 SQLite
    from app.core.config import settings as app_settings
    dm_probe = f"sqlite:///{(tmp_path / 'probe.db').as_posix()}"
    app_settings.__class__.database_url = property(lambda self: dm_probe)
    app_settings.__class__.is_sqlite = property(lambda self: True)
    Base.metadata.create_all(bind=engine)

    from app.core.security import hash_password
    from app.models.entities import SysUser
    s = TestSession()
    s.add(SysUser(username="doctor", password_hash=hash_password("doctor123"), nickname="d", role="doctor"))
    s.commit()
    s.close()

    from app.api import rate_limit
    rate_limit._limiter.reset()

    from app.main import app
    try:
        with TestClient(app) as c:
            codes = [
                c.post("/api/auth/login", json={"username": "doctor", "password": "wrong"}).status_code
                for _ in range(12)
            ]
    finally:
        dm.engine, dm.SessionLocal = orig_engine, orig_local
        rate_limit._limiter.reset()
        # 还原自检探测配置（直接改了类属性，需手动恢复读取 .env 的行为）
        from app.core.config import Settings
        del Settings.database_url
        del Settings.is_sqlite

    assert codes[0] == 401, "首次应是密码错误 401"
    assert codes[10] == 429 and codes[11] == 429, f"第 11/12 次未限流: {codes}"
