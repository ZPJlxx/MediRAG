"""JWT 鉴权依赖与角色控制。"""

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.entities import SysUser


def get_current_user(request: Request, db: Session = Depends(get_db)) -> SysUser:
    """从 Authorization: Bearer <token> 解析当前用户。"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录或令牌缺失")
    token = auth_header.removeprefix("Bearer ").strip()
    return _resolve_user(token, db)


def get_current_user_sse(request: Request, db: Session = Depends(get_db)) -> SysUser:
    """SSE 专用鉴权：EventSource 无法自定义请求头，允许 ?token= 查询参数携带 JWT。

    JWT 校验逻辑与 get_current_user 完全一致，仅多一个取值途径。
    """
    token = request.query_params.get("token", "")
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="未登录或令牌缺失")
    return _resolve_user(token, db)


def _resolve_user(token: str, db: Session) -> SysUser:
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="令牌无效或已过期")

    user = db.query(SysUser).filter(SysUser.id == int(payload["sub"])).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")
    return user


def require_admin(user: SysUser = Depends(get_current_user)) -> SysUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
