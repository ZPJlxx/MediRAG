"""日志配置：loguru 控制台 + 文件双通道（沿用 general-doc-qa-assistant 的日志方案）。"""

import sys
from pathlib import Path

from loguru import logger

from app.core.config import BACKEND_DIR

_configured = False


def setup_logging() -> None:
    global _configured
    if _configured:
        return

    log_dir = BACKEND_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        Path(log_dir) / "medirag_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        enqueue=True,
    )
    _configured = True


setup_logging()
