"""认证路由：登录（JWT 签发）。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.rate_limit import login_rate_limit
from app.core.database import get_db
from app.core.security import create_access_token, verify_password
from app.models.entities import SysUser
from app.schemas.auth import LoginRequest, LoginResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse, dependencies=[Depends(login_rate_limit)])
def login(form: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(SysUser).filter(SysUser.username == form.username).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="账号已禁用")

    token = create_access_token(user.id, user.username, user.role)
    return LoginResponse(
        token=token,
        user={"id": user.id, "username": user.username, "nickname": user.nickname, "role": user.role},
    )
