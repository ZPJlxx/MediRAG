"""P0 安全/正确性修复的回归测试。

覆盖：
- 越权防护（IDOR）路由依赖
- 知识文件删除的路径穿越拒绝
- 过敏禁忌核对的持久化路径
- JWT 默认密钥 fail-fast
"""

import threading
import warnings

import pytest


def _isolate_dirs(monkeypatch, tmp_path):
    """把 settings 的派生路径属性替换为临时目录（property 只读，直接替换类属性）。"""
    from app.core.config import settings

    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(
        type(settings), "knowledge_dir",
        property(lambda self: knowledge_dir),
        raising=False,
    )
    monkeypatch.setattr(
        type(settings), "vector_store_dir",
        property(lambda self: tmp_path / "vector"),
        raising=False,
    )
    return knowledge_dir


def test_allergy_check_persists_through_to_json():
    """P0-3：过敏核对修改的是已校验模型，必须随 to_json 序列化（get_status 可见）。"""
    from app.medical.agents.treatment_advice import treatment_advice_agent
    from app.schemas.medical import TreatmentAdviceResult

    validated = TreatmentAdviceResult(
        medication_advice=[
            {"drug_name": "青霉素V钾片", "dosage": "500mg tid", "duration": "7天", "precautions": "餐后服用"}
        ],
        warning_signals=[],
    )
    treatment_advice_agent.check_allergy_contraindications(validated, ["青霉素"])

    assert "【禁忌提醒】" in validated.medication_advice[0]["precautions"]
    assert any("【禁忌提醒】" in w for w in validated.warning_signals)
    assert "【禁忌提醒】" in validated.to_json()


def test_allergy_check_no_history_noop():
    """无过敏史时不做任何修改。"""
    from app.medical.agents.treatment_advice import treatment_advice_agent
    from app.schemas.medical import TreatmentAdviceResult

    validated = TreatmentAdviceResult(
        medication_advice=[{"drug_name": "阿莫西林", "dosage": "x", "duration": "y", "precautions": "z"}],
        warning_signals=[],
    )
    treatment_advice_agent.check_allergy_contraindications(validated, [])
    assert validated.medication_advice[0]["precautions"] == "z"
    assert validated.warning_signals == []


def test_delete_file_rejects_path_traversal(tmp_path, monkeypatch):
    """P0-2：删除接口拒绝知识目录外的路径。"""
    from app.rag.knowledge_manager import KnowledgeManager

    knowledge_dir = _isolate_dirs(monkeypatch, tmp_path)
    (knowledge_dir / "real.md").write_text("x", encoding="utf-8")

    manager = KnowledgeManager.__new__(KnowledgeManager)  # 绕过 __init__ 的重初始化
    manager._retriever_lock = threading.Lock()
    manager._retrievers = {}

    assert manager.delete_file("../../etc/passwd") is False
    assert manager.delete_file("../backend/.env") is False
    assert manager.delete_file("real.md") is True
    assert not (knowledge_dir / "real.md").exists()


def test_delete_file_rejects_absolute_escape(tmp_path, monkeypatch):
    """绝对路径形式的越界同样拒绝。"""
    from app.rag.knowledge_manager import KnowledgeManager

    _isolate_dirs(monkeypatch, tmp_path)

    manager = KnowledgeManager.__new__(KnowledgeManager)
    manager._retriever_lock = threading.Lock()
    manager._retrievers = {}

    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    assert manager.delete_file(str(outside)) is False
    assert outside.exists()


def test_jwt_default_secret_dev_warns(monkeypatch):
    """P0-4：dev 环境默认密钥仅告警。"""
    from app.core.config import Settings

    s = Settings(jwt_secret="dev-secret-change-me", app_env="dev")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s.validate_security()
    assert any("JWT_SECRET" in str(w.message) for w in caught)


def test_jwt_default_secret_prod_raises(monkeypatch):
    """P0-4：prod 环境默认密钥拒绝启动。"""
    from app.core.config import Settings

    for insecure in ("dev-secret-change-me", "change-me-in-production-9f8e7d6c5b4a"):
        s = Settings(jwt_secret=insecure, app_env="prod")
        with pytest.raises(RuntimeError, match="安全拦截"):
            s.validate_security()


def test_jwt_strong_secret_passes():
    """强密钥不触发任何告警。"""
    from app.core.config import Settings

    s = Settings(jwt_secret="a" * 64, app_env="prod")
    s.validate_security()  # 不抛异常即通过


def test_workflow_routes_require_ownership_check():
    """P0-1：路由模块必须存在归属校验入口且被两个端点引用。"""
    import inspect

    import app.api.workflow_routes as wr

    assert hasattr(wr, "_ensure_owned_record")
    status_src = inspect.getsource(wr.get_status)
    lab_src = inspect.getsource(wr.submit_lab_results)
    assert "_ensure_owned_record" in status_src
    assert "_ensure_owned_record" in lab_src
