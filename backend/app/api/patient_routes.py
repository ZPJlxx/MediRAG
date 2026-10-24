"""患者端聊天路由：免登录的「智慧医疗诊疗聊天」入口。

与医生端 JWT 鉴权分离：患者无账号体系，以 session_key（localStorage）关联会话，
按 IP 限流（chat_rate_limit）作为免登录接口防滥用的闸门。
"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.rate_limit import _client_ip, chat_rate_limit
from app.core.database import SessionLocal
from app.dao.medical import PatientChatDAO
from app.medical.intent import INTENT_LABELS
from app.medical.patient_chat import new_session_key, patient_chat_service
from app.models.entities import PatientChatMessage, PatientChatSession

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


@router.get("/chat/history")
async def patient_chat_history(
    session_id: str = "",
    limit: int = 50,
    http_request: Request = None,
):
    """拉取会话历史（前端刷新后恢复聊天记录）。

    免登录设计下 session_id 即凭证：只返回该会话的消息，limit 夹紧到 [1, 100]。
    未知会话返回空列表（前端按新会话处理，展示欢迎语）。
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id 不能为空")
    limit = max(1, min(100, limit))

    def _load():
        session = SessionLocal()
        try:
            chat_session = PatientChatDAO.get_or_create_session(
                session, session_id, _client_ip(http_request) if http_request else ""
            )
            rows = PatientChatDAO.list_messages(session, chat_session.id, limit=limit * 2)
            messages = []
            for row in rows:
                try:
                    citations = json.loads(row.citations) if row.citations else []
                except (TypeError, json.JSONDecodeError):
                    citations = []
                messages.append({
                    "role": row.role,
                    "content": row.content,
                    "intent": row.intent,
                    "citations": citations,
                })
            # 只保留成对的 user/assistant；尾部孤立 user（回复生成中途崩溃）截掉
            if messages and messages[-1]["role"] == "user":
                messages = messages[:-1]
            return messages[-limit:]
        finally:
            session.close()

    messages = await asyncio.to_thread(_load)
    return {"success": True, "session_id": session_id, "messages": messages}


@router.get("/chat/sessions")
async def patient_chat_sessions(ids: str = ""):
    """会话列表摘要（前端「历史会话」面板用）。

    免登录设计下"我的会话"= 本浏览器 localStorage 记录的 session_id 台账，
    接口按传入 ids 批量返回摘要（首条提问预览 + 消息数 + 最近时间），
    不提供全局会话枚举（跨浏览器/跨用户的会话不可见，隐私边界）。
    空会话（只建未聊）过滤不返回。
    """
    keys = [k.strip() for k in (ids or "").split(",") if k.strip()][:20]
    if not keys:
        return {"success": True, "sessions": []}

    def _load():
        session = SessionLocal()
        try:
            result = []
            for key in keys:
                chat_session = (
                    session.query(PatientChatSession)
                    .filter(PatientChatSession.session_key == key)
                    .first()
                )
                if not chat_session or chat_session.message_count == 0:
                    continue
                first_user = (
                    session.query(PatientChatMessage)
                    .filter(
                        PatientChatMessage.session_id == chat_session.id,
                        PatientChatMessage.role == "user",
                    )
                    .order_by(PatientChatMessage.id.asc())
                    .first()
                )
                result.append({
                    "session_id": key,
                    "message_count": chat_session.message_count,
                    "first_message": first_user.content[:40] if first_user else "",
                    "last_time": (
                        chat_session.updated_at.strftime("%m-%d %H:%M")
                        if chat_session.updated_at else ""
                    ),
                })
            return result
        finally:
            session.close()

    sessions = await asyncio.to_thread(_load)
    return {"success": True, "sessions": sessions}
