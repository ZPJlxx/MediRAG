"""算法接口专用鉴权：API Key（服务间调用）+ 每调用方限流。

与用户 JWT（面向登录用户）分离：算法接口的调用方是外部系统，
凭 X-Api-Key 头访问；密钥只存 sha256 哈希，限流按调用方 QPS 配置执行。
"""

import datetime as dt
import hashlib
import threading
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.entities import AlgorithmApiKey


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class _SlidingWindowLimiter:
    """进程内滑动窗口限流（单实例部署足够；多实例部署应换 Redis）."""

    def __init__(self) -> None:
        self._hits: dict[int, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key_id: int, qps: int) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._hits[key_id]
            while window and now - window[0] >= 1.0:
                window.popleft()
            if len(window) >= qps:
                return False
            window.append(now)
            return True


_limiter = _SlidingWindowLimiter()


def get_api_key_client(request: Request, db: Session = Depends(get_db)) -> AlgorithmApiKey:
    """验证 X-Api-Key 并执行限流；返回调用方记录。"""
    raw_key = request.headers.get("X-Api-Key", "")
    if not raw_key:
        raise HTTPException(status_code=401, detail="缺少 X-Api-Key 请求头")

    client = (
        db.query(AlgorithmApiKey)
        .filter(AlgorithmApiKey.key_hash == hash_api_key(raw_key))
        .first()
    )
    if not client or not client.is_active:
        raise HTTPException(status_code=401, detail="API Key 无效或已停用")

    if not _limiter.allow(client.id, client.rate_limit_qps or 5):
        raise HTTPException(status_code=429, detail=f"超过调用频率限制（{client.rate_limit_qps} QPS），请稍后重试")

    client.last_used_at = dt.datetime.now()
    db.commit()
    return client
