"""知识库管理路由：上传/列表/删除/状态/检索测试。"""

import asyncio
import hashlib

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.models.entities import KnowledgeFile, SysUser
from app.rag.loader import SUPPORTED_KNOWLEDGE_SUFFIXES
from app.rag.knowledge_manager import knowledge_manager

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB


def _collection_name(filename: str) -> str:
    return hashlib.md5(filename.encode("utf-8")).hexdigest()


def _sync_registered_files(db: Session):
    """把磁盘文件同步进台账（首次启动/手工放入文件时对齐）。"""
    disk_files = set(knowledge_manager.list_files())
    known = {f.filename for f in db.query(KnowledgeFile).all()}
    for name in disk_files - known:
        path = settings.knowledge_dir / name
        db.add(KnowledgeFile(
            filename=name,
            collection_name=_collection_name(name),
            size_bytes=path.stat().st_size,
            uploader_id=0,
        ))
    for name in known - disk_files:
        stale = db.query(KnowledgeFile).filter(KnowledgeFile.filename == name).first()
        if stale:
            db.delete(stale)
    db.commit()


@router.get("/files")
def list_files(user: SysUser = Depends(get_current_user), db: Session = Depends(get_db)):
    _sync_registered_files(db)
    files = db.query(KnowledgeFile).order_by(KnowledgeFile.created_at.desc()).all()
    return {
        "files": [
            {
                "filename": f.filename,
                "size_bytes": f.size_bytes,
                "created_at": f.created_at.strftime("%Y-%m-%d %H:%M") if f.created_at else "",
            }
            for f in files
        ],
        "rag_enabled": settings.medical_rag_enabled,
        "runtime": knowledge_manager.runtime_status(),
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile,
    user: SysUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    filename = file.filename or "untitled"
    suffix = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if suffix not in SUPPORTED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix}，支持: {sorted(SUPPORTED_KNOWLEDGE_SUFFIXES)}",
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="文件超过 100MB 上限")
    if not data:
        raise HTTPException(status_code=400, detail="空文件")

    knowledge_manager.save_uploaded_file(filename, data)

    existing = db.query(KnowledgeFile).filter(KnowledgeFile.filename == filename).first()
    collection = _collection_name(filename)
    if existing:
        existing.size_bytes = len(data)
        existing.uploader_id = user.id
    else:
        db.add(KnowledgeFile(
            filename=filename,
            collection_name=collection,
            size_bytes=len(data),
            uploader_id=user.id,
        ))
    db.commit()
    return {"success": True, "filename": filename, "size_bytes": len(data)}


@router.delete("/files/{filename}")
def delete_file(
    filename: str,
    user: SysUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not knowledge_manager.delete_file(filename):
        raise HTTPException(status_code=404, detail="文件不存在")
    record = db.query(KnowledgeFile).filter(KnowledgeFile.filename == filename).first()
    if record:
        db.delete(record)
        db.commit()
    return {"success": True}


@router.post("/search")
async def test_search(
    payload: dict,
    user: SysUser = Depends(get_current_user),
):
    """知识库检索测试入口（前端调试与评估脚本共用）。"""
    query = (payload or {}).get("query", "").strip()
    top_k = int((payload or {}).get("top_k", 5))
    if not query:
        raise HTTPException(status_code=400, detail="query 不能为空")

    def _run():
        citations = knowledge_manager.search_citations(query, top_k=top_k)
        return [c.model_dump() for c in citations]

    results = await asyncio.to_thread(_run)
    return {"success": True, "query": query, "results": results}
