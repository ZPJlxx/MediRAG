"""用户管理路由（admin 专属，迁移自 Enterprise-RAG 管理后台）。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.core.database import get_db
from app.core.security import hash_password
from app.models.entities import SysUser
from app.schemas.auth import UserCreate, UserUpdate

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("")
def list_users(_: SysUser = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(SysUser).order_by(SysUser.id.asc()).all()
    return [
        {"id": u.id, "username": u.username, "nickname": u.nickname,
         "role": u.role, "is_active": u.is_active}
        for u in users
    ]


@router.post("")
def create_user(form: UserCreate, _: SysUser = Depends(require_admin), db: Session = Depends(get_db)):
    if db.query(SysUser).filter(SysUser.username == form.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = SysUser(
        username=form.username,
        password_hash=hash_password(form.password),
        nickname=form.nickname or form.username,
        role=form.role,
    )
    db.add(user)
    db.commit()
    return {"success": True, "id": user.id}


@router.put("/{user_id}")
def update_user(
    user_id: int,
    form: UserUpdate,
    _: SysUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = db.get(SysUser, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.nickname = form.nickname or user.nickname
    user.role = form.role
    user.is_active = form.is_active
    if form.password:
        user.password_hash = hash_password(form.password)
    db.commit()
    return {"success": True}


@router.delete("/{user_id}")
def delete_user(user_id: int, _: SysUser = Depends(require_admin), db: Session = Depends(get_db)):
    user = db.get(SysUser, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    if user.id == user_id and user.role == "admin":
        active_admins = (
            db.query(SysUser)
            .filter(SysUser.role == "admin", SysUser.is_active.is_(True), SysUser.id != user_id)
            .count()
        )
        if active_admins == 0:
            raise HTTPException(status_code=400, detail="不能删除最后一个管理员")
    db.delete(user)
    db.commit()
    return {"success": True}
