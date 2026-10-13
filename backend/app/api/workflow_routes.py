"""诊疗工作流路由：启动/状态/化验回填/历史列表。"""

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.database import get_db
from app.dao.medical import InquiryRecordDAO
from app.medical.workflow import medical_workflow
from app.models.entities import SysUser
from app.schemas.auth import LabResultRequest, StartWorkflowRequest

router = APIRouter(prefix="/api/workflow", tags=["workflow"])


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
async def get_status(inquiry_id: int, user: SysUser = Depends(get_current_user)):
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
):
    """化验结果回填：恢复暂停的工作流（HTTP 直连；MQ 模式下由消费者触发同一入口）。"""

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
):
    records = InquiryRecordDAO.list_by_doctor(db, user.id, limit=50)
    from app.medical.workflow import STATUS_LABELS

    return [
        {
            "id": r.id,
            "patient_name": r.patient_name,
            "main_status": r.main_status,
            "status_label": STATUS_LABELS.get(r.main_status, str(r.main_status)),
            "use_rag": r.use_rag,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            "symptom_brief": r.symptom_description[:50],
        }
        for r in records
    ]
