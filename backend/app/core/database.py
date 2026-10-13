"""数据库引擎与会话管理。

替代三个项目各自的数据库接入方式：
- MediAgent-LLM 的 memory/db_connection.py（基础可用）
- Enterprise-RAG 在 5 个文件里重复硬编码的 DB_CONFIG

dev 模式默认 SQLite（零依赖启动），prod 通过 .env 的 DATABASE_URL 切 MySQL。
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""


def _build_engine():
    if settings.is_sqlite:
        return create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    return create_engine(
        settings.database_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：请求级会话。"""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def create_tables() -> None:
    from app.models import entities  # noqa: F401  确保模型注册

    Base.metadata.create_all(bind=engine)
