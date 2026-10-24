"""用户端接口限流：按客户端 IP 的滑动窗口（防爆破/防滥用）。

与算法接口的按 key 限流（algorithm_deps）互补：
- 登录接口：防密码爆破（每 IP 5 次/分钟级）
- 工作流启动：防滥用刷 LLM 配额（每 IP 级 QPS 限制）
单实例部署内存实现即可；多实例部署应换 Redis。
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class _WindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, scope: str, key: str, limit: int, window_seconds: float) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._hits[(scope, key)]
            while window and now - window[0] >= window_seconds:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True

    def reset(self) -> None:
        """测试用：清空窗口状态。"""
        self._hits.clear()


_limiter = _WindowLimiter()


def _client_ip(request: Request) -> str:
    """取客户端 IP（反代场景优先 X-Forwarded-For 首段）。"""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def login_rate_limit(request: Request) -> None:
    """登录限流：每 IP 每分钟 10 次（含失败尝试），防暴力爆破。"""
    if not _limiter.allow("login", _client_ip(request), limit=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请 1 分钟后再试")


def workflow_rate_limit(request: Request) -> None:
    """工作流启动限流：每 IP 每分钟 5 次（单次问诊要烧多轮 LLM 配额）。"""
    if not _limiter.allow("workflow", _client_ip(request), limit=5, window_seconds=60):
        raise HTTPException(status_code=429, detail="诊疗工作流提交过于频繁，请稍后再试")


def chat_rate_limit(request: Request) -> None:
    """患者聊天限流：每 IP 每分钟 20 次（免登录接口，防配额烧穿的唯一闸门）。"""
    if not _limiter.allow("patient_chat", _client_ip(request), limit=20, window_seconds=60):
        raise HTTPException(status_code=429, detail="发送过于频繁，请稍后再试")
