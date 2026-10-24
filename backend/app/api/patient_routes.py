"""患者端聊天路由：免登录的「智慧医疗诊疗聊天」入口。

与医生端 JWT 鉴权分离：患者无账号体系，以 session_key（localStorage）关联会话，
按 IP 限流（chat_rate_limit）作为免登录接口防滥用的闸门。
"""

import asyncio

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.rate_limit import _client_ip, chat_rate_limit
from app.medical.patient_chat import new_session_key, patient_chat_service

router = APIRouter(prefix="/api/patient", tags=["patient-chat"])


class PatientChatRequest(BaseModel):
    """单条患者消息。session_id 缺省时视为新会话，由后端签发。"""

    session_id: str = Field(default="", description="会话标识（首次可不传）")
    message: str = Field(..., min_length=1, max_length=2000, description="患者消息")


@router.post("/chat", dependencies=[Depends(chat_rate_limit)])
async def patient_chat(request: PatientChatRequest, http_request: Request):
    """患者消息 -> 意图识别分流 -> 回复（含引用与免责声明）。"""
    session_key = (request.session_id or "").strip() or new_session_key()
    result = await asyncio.to_thread(
        patient_chat_service.handle_chat,
        session_key,
        request.message.strip(),
        _client_ip(http_request),
    )
    return {"success": True, **result}
