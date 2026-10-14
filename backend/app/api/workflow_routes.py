"""诊疗工作流路由：启动/状态/化验回填/历史列表/SSE 状态推送。"""

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_current_user_sse
from app.core.database import get_db
from app.dao.medical import InquiryRecordDAO
from app.medical.workflow import medical_workflow
from app.models.entities import SysUser
from app.schemas.auth import LabResultRequest, StartWorkflowRequest

router = APIRouter(prefix="/api/workflow", tags=["workflow"])

# SSE 轮询间隔：轻于前端 3s 轮询（服务端可做指纹去重，客户端只在变化时收到事件）
_SSE_POLL_SECONDS = 1.5
# 连接最长存活：到期客户端 EventSource 自动重连，防止僵尸连接占住会话
_SSE_MAX_LIFETIME_SECONDS = 600


def _ensure_owned_record(inquiry_id: int, user: SysUser, db: Session):
    """越权防护（IDOR）：问诊记录仅限主治医生本人访问，返回 404 避免暴露记录是否存在。"""
    record = InquiryRecordDAO.get(db, inquiry_id)
    if not record or record.doctor_id != user.id:
        raise HTTPException(status_code=404, detail="问诊记录不存在")
    return record


@router.post("/start")
async def start_workflow(
    request: StartWorkflowRequest,
    user: SysUser = Depends(get_current_user),
):
    """启动诊疗工作流。若需化验，工作流会暂停在等待化验节点。"""
    inquiry_id = await asyncio.to_thread(
        medical_workflow.start_workflow,
        user.id,
        request.patient_name,
        request.patient_age,
        request.patient_gender,
        request.allergy_history,
        request.symptom_description,
        request.use_rag,
    )
    status = await asyncio.to_thread(medical_workflow.get_status, inquiry_id)
    return {
        "success": True,
        "inquiry_record_id": inquiry_id,
        "waiting_lab": status.get("waiting_lab", False),
        "lab_tests_needed": status.get("lab_tests_needed", []),
        "status_label": status.get("status_label"),
    }


@router.get("/{inquiry_id}/status")
async def get_status(
    inquiry_id: int,
    user: SysUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _ensure_owned_record(inquiry_id, user, db)
    status = await asyncio.to_thread(medical_workflow.get_status, inquiry_id)
    if not status.get("success"):
        raise HTTPException(status_code=404, detail=status.get("error", "查询失败"))
    return status


@router.post("/{inquiry_id}/lab-results")
async def submit_lab_results(
    inquiry_id: int,
    request: LabResultRequest,
    background: BackgroundTasks,
    user: SysUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """化验结果回填：恢复暂停的工作流（HTTP 直连；MQ 模式下由消费者触发同一入口）。"""
    _ensure_owned_record(inquiry_id, user, db)

    def _resume():
        medical_workflow.handle_lab_results(inquiry_id, request.lab_results)

    try:
        await asyncio.to_thread(_resume)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    status = await asyncio.to_thread(medical_workflow.get_status, inquiry_id)
    return {"success": True, "message": "化验结果已提交，工作流已恢复", "status": status}


@router.get("/list/history")
async def list_history(
    user: SysUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    page: int = 1,
    page_size: int = 20,
    patient: str = "",
):
    """问诊历史（分页 + 患者名搜索）。

    page_size 无参数时返回全量（兼容旧前端调用）。
    """
    from app.medical.workflow import STATUS_LABELS

    def _row(r):
        return {
            "id": r.id,
            "patient_name": r.patient_name,
            "main_status": r.main_status,
            "status_label": STATUS_LABELS.get(r.main_status, str(r.main_status)),
            "use_rag": r.use_rag,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            "symptom_brief": r.symptom_description[:50],
        }

    if page_size <= 0:
        records = InquiryRecordDAO.list_by_doctor(db, user.id, limit=50)
        return {"items": [_row(r) for r in records], "total": len(records), "page": 1, "page_size": len(records)}

    page_size = min(page_size, 100)  # 上限防拉全表
    records, total = InquiryRecordDAO.page_by_doctor(db, user.id, page, page_size, patient)
    return {
        "items": [_row(r) for r in records],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def _status_fingerprint(status: dict) -> str:
    """状态内容指纹：变化才推送，避免每 1.5s 全量刷一遍。"""
    return json.dumps(
        {
            "ms": status.get("main_status"),
            "wl": status.get("waiting_lab"),
            "lt": status.get("lab_tests_needed"),
            "sym": bool(status.get("symptom_analysis")),
            "hyp": len(status.get("hypothesis_verifications") or []),
            "dia": bool(status.get("diagnosis")),
            "tre": bool(status.get("treatment")),
            "ev": len(status.get("events") or []),
        },
        sort_keys=True,
    )


@router.get("/{inquiry_id}/events")
async def stream_status_events(
    inquiry_id: int,
    user: SysUser = Depends(get_current_user_sse),
    db: Session = Depends(get_db),
):
    """SSE 状态推送：状态指纹变化时向前端推送完整状态快照。

    终态（完成/失败）推送后主动断流，客户端 EventSource 会自动重连，
    重连后因指纹不再变化只会收到一次快照，不会形成事件风暴。
    """
    _ensure_owned_record(inquiry_id, user, db)

    async def event_stream():
        import time

        last_fingerprint = None
        deadline = time.monotonic() + _SSE_MAX_LIFETIME_SECONDS
        try:
            while time.monotonic() < deadline:
                status = await asyncio.to_thread(medical_workflow.get_status, inquiry_id)
                if not status.get("success"):
                    yield f"event: error\ndata: {json.dumps({'error': status.get('error', '查询失败')})}\n\n"
                    return

                fingerprint = _status_fingerprint(status)
                if fingerprint != last_fingerprint:
                    last_fingerprint = fingerprint
                    yield f"event: status\ndata: {json.dumps(status, ensure_ascii=False)}\n\n"

                # 终态：推完最后一份快照即收流（客户端拿到终态后自己 close）
                if status.get("main_status") in (5, 99):
                    return

                await asyncio.sleep(_SSE_POLL_SECONDS)
            # 生命周期到期的保活收尾：通知客户端重连
            yield "event: reconnect\ndata: {}\n\n"
        except asyncio.CancelledError:
            # 客户端断开（EventSource.close / 页面切换）会走到这里，静默退出
            raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 反代时禁用缓冲，保证事件即时下发
        },
    )
