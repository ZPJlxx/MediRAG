"""可观测性中间件：request-id 全链路追踪 + 结构化访问日志。"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import request_id_var

# 不记录访问日志的路径（探活/静态资源噪音）
_QUIET_PATHS = {"/health", "/favicon.ico"}


class RequestIdMiddleware(BaseHTTPMiddleware):
    """为每个请求生成（或透传上游的）X-Request-ID，并贯穿日志上下文。

    响应头带回归属请求，前端/调用方报障时只需提供该 ID，
    即可在日志文件中 grep 出这条请求的全部处理轨迹。
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 上游网关已生成则透传（微服务链路拼接），否则本地生成
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        request.state.request_id = rid
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = rid
        if request.url.path not in _QUIET_PATHS:
            # 结构化访问日志：一行看全 谁(routing_key) 在哪(path) 花了多久
            from loguru import logger

            logger.info(
                f"ACCESS {request.method} {request.url.path} -> {response.status_code} "
                f"{elapsed_ms:.1f}ms client={_client_ip(request)}"
            )
        return response


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
