"""算法服务接口（v1）：把系统能力包装为可独立调用的算法 API。

面向外部系统的服务间调用（X-Api-Key 鉴权），与面向登录用户的业务路由分离：
- A 检索服务：POST /api/v1/algorithm/retrieval —— 医学文献三路混合检索
- C 诊疗任务：POST /api/v1/algorithm/diagnosis/tasks + GET .../tasks/{id}
  —— 异步任务语义（提交 -> 轮询），需化验时挂起等待外部回填

合规要求：所有响应携带 disclaimer，声明仅供辅助参考。
"""

import asyncio
import datetime as dt
import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.algorithm_deps import get_api_key_client, hash_api_key
from app.api.deps import require_admin
from app.core.database import get_db
from app.medical.workflow import medical_workflow
from app.models.entities import AlgorithmApiKey, InquiryRecord, SysUser
from app.rag.knowledge_manager import knowledge_manager
from app.schemas.auth import LabResultRequest, StartWorkflowRequest

router = APIRouter(prefix="/api/v1/algorithm", tags=["algorithm"])

DISCLAIMER = "本接口仅供演示与辅助参考，不能替代执业医师的诊断与治疗决策；医疗决策请以临床规范和医师判断为准。"

# 任务状态机 -> 算法任务三态语义（queued/running/terminal）
_TERMINAL_STATUSES = {5, 99}


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="检索查询文本")
    top_k: int = Field(default=5, ge=1, le=20, description="返回证据条数上限")


class RetrievalEvidence(BaseModel):
    rank: int
    source: str
    location: str
    snippet: str


class RetrievalResponse(BaseModel):
    query: str
    total: int
    evidence: list[RetrievalEvidence]
    latency_ms: int
    disclaimer: str


class DiagnosisTaskResponse(BaseModel):
    task_id: int
    status: str = Field(description="queued | running | succeeded | failed")
    status_label: str = ""
    waiting_lab: bool = Field(default=False, description="任务挂起等待化验结果回填")
    lab_tests_needed: list[str] = Field(default_factory=list)
    result: dict | None = Field(default=None, description="terminal 状态时的完整诊疗结果（含证据引用）")
    disclaimer: str


@router.post("/retrieval", response_model=RetrievalResponse)
async def retrieval(
    request: RetrievalRequest,
    client: AlgorithmApiKey = Depends(get_api_key_client),
):
    """医学文献混合检索：向量 + BM25 + 关键词三路召回，RRF 融合 + 双级重排。"""
    import time as _time

    start = _time.perf_counter()
    citations = await asyncio.to_thread(
        knowledge_manager.search_citations, request.query, request.top_k
    )
    latency_ms = int((_time.perf_counter() - start) * 1000)
    return RetrievalResponse(
        query=request.query,
        total=len(citations),
        evidence=[
            RetrievalEvidence(rank=i + 1, source=c.source, location=c.location, snippet=c.snippet)
            for i, c in enumerate(citations)
        ],
        latency_ms=latency_ms,
        disclaimer=DISCLAIMER,
    )


@router.post("/diagnosis/tasks", response_model=DiagnosisTaskResponse)
async def create_diagnosis_task(
    request: StartWorkflowRequest,
    client: AlgorithmApiKey = Depends(get_api_key_client),
):
    """提交临床辅助诊疗任务（异步执行）：返回 task_id 供轮询。

    注意：task 与调用方不绑定归属（算法服务语义），任务 ID 请妥善保存。
    """
    task_id = await asyncio.to_thread(
        medical_workflow.start_workflow,
        0,  # doctor_id=0：算法调用不关联平台用户，审计仍完整落库
        request.patient_name,
        request.patient_age,
        request.patient_gender,
        request.allergy_history,
        request.symptom_description,
        request.use_rag,
    )
    status = await asyncio.to_thread(medical_workflow.get_status, task_id)
    return _to_task_response(task_id, status)


@router.get("/diagnosis/tasks/{task_id}", response_model=DiagnosisTaskResponse)
async def get_diagnosis_task(
    task_id: int,
    client: AlgorithmApiKey = Depends(get_api_key_client),
):
    """查询诊疗任务状态；terminal 状态时返回完整结果（含证据引用）。"""
    status = await asyncio.to_thread(medical_workflow.get_status, task_id)
    if not status.get("success"):
        raise HTTPException(status_code=404, detail=status.get("error", "任务不存在"))
    return _to_task_response(task_id, status)


@router.post("/diagnosis/tasks/{task_id}/lab-results", response_model=DiagnosisTaskResponse)
async def submit_lab_results(
    task_id: int,
    request: LabResultRequest,
    client: AlgorithmApiKey = Depends(get_api_key_client),
):
    """回填化验结果，恢复挂起的诊疗任务（waiting_lab=true 时调用）。"""
    await asyncio.to_thread(medical_workflow.handle_lab_results, task_id, request.lab_results)
    status = await asyncio.to_thread(medical_workflow.get_status, task_id)
    if not status.get("success"):
        raise HTTPException(status_code=404, detail=status.get("error", "任务不存在"))
    return _to_task_response(task_id, status)


def _to_task_response(task_id: int, status: dict) -> DiagnosisTaskResponse:
    """内部状态机 -> 算法任务语义。terminal 时透传完整结果供调用方取用。"""
    main_status = status.get("main_status", 0)
    if main_status == 99:
        task_status = "failed"
    elif main_status == 5:
        task_status = "succeeded"
    elif main_status in (1, 2, 3, 4):
        task_status = "running"
    else:
        task_status = "queued"

    return DiagnosisTaskResponse(
        task_id=task_id,
        status=task_status,
        status_label=status.get("status_label", ""),
        waiting_lab=status.get("waiting_lab", False),
        lab_tests_needed=status.get("lab_tests_needed", []) or [],
        result=status if task_status in ("succeeded", "failed") else None,
        disclaimer=DISCLAIMER,
    )


# ---------- API Key 管理（平台管理员 JWT，非算法调用路径）----------

class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="调用方名称")
    rate_limit_qps: int = Field(default=5, ge=1, le=50)


@router.post("/keys", tags=["admin"])
def create_api_key(
    body: ApiKeyCreate,
    admin: SysUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """签发算法 API Key。明文 key 只在此响应中出现一次，请调用方妥善保存。"""
    raw_key = f"mdrg_{secrets.token_hex(24)}"
    record = AlgorithmApiKey(
        name=body.name,
        key_hash=hash_api_key(raw_key),
        rate_limit_qps=body.rate_limit_qps,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "name": record.name, "api_key": raw_key, "rate_limit_qps": record.rate_limit_qps}


@router.get("/keys", tags=["admin"])
def list_api_keys(
    admin: SysUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    keys = db.query(AlgorithmApiKey).order_by(AlgorithmApiKey.created_at.desc()).all()
    return {
        "keys": [
            {
                "id": k.id,
                "name": k.name,
                "rate_limit_qps": k.rate_limit_qps,
                "is_active": k.is_active,
                "last_used_at": k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else "",
                "created_at": k.created_at.strftime("%Y-%m-%d %H:%M") if k.created_at else "",
            }
            for k in keys
        ]
    }


@router.delete("/keys/{key_id}", tags=["admin"])
def revoke_api_key(
    key_id: int,
    admin: SysUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    record = db.get(AlgorithmApiKey, key_id)
    if not record:
        raise HTTPException(status_code=404, detail="API Key 不存在")
    record.is_active = False
    db.commit()
    return {"success": True, "message": f"API Key「{record.name}」已吊销"}
