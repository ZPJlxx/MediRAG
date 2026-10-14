"""日志配置：loguru 控制台 + 文件双通道 + request_id 上下文注入。

全链路追踪设计：
- RequestIdMiddleware 为每个入站请求生成/透传 X-Request-ID
- contextvars 在异步任务与线程间稳定携带 request_id（uvicorn 的 to_thread 会复制上下文）
- 日志格式统一附带 request_id，一条 ID 可串起 API -> Agent -> LLM 审计记录
"""

import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from app.core.config import BACKEND_DIR

_configured = False

# 请求级上下文：未绑定（后台任务/启动阶段）时显示 "-"
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _format_with_request_id(record: dict) -> str:
    record["extra"].setdefault("request_id", request_id_var.get())
    return (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
        "[{extra[request_id]}] {name}:{function}:{line} - {message}\n"
    )


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    log_dir = BACKEND_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO", format=_format_with_request_id)
    logger.add(
        Path(log_dir) / "medirag_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        format=_format_with_request_id,
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )
    _configured = True


setup_logging()
