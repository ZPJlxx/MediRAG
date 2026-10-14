"""统一错误码与响应体规范。

设计原则：
- 业务错误码集中登记（错误码 = 领域前缀 + 3 位序号），避免散落的魔法字符串
- 所有非 2xx 响应体统一为 {"code": "MED-40401", "message": "...", "request_id": "..."}，
  前端 api.js 的 extractDetail 兼容读取 message 字段
- HTTP 状态码职责不变（401/403/404/429/500...），code 提供更细的机器可读分类
- 未捕获异常统一兜底为 INTERNAL_ERROR，不向前端泄露堆栈（细节进日志，靠 request_id 关联）
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---- 错误码登记表（新增错误在此追加，保持前缀分域）----
ERROR_CODES = {
    # 通用 0xxx
    "INTERNAL_ERROR": ("SYS-0500", "服务内部错误"),
    "VALIDATION_ERROR": ("SYS-0400", "请求参数校验失败"),
    # 认证 1xxx
    "UNAUTHORIZED": ("AUTH-1401", "未登录或令牌无效"),
    "FORBIDDEN": ("AUTH-1403", "无权访问"),
    "ACCOUNT_DISABLED": ("AUTH-1413", "账号已禁用"),
    "RATE_LIMITED": ("AUTH-1429", "请求过于频繁"),
    # 业务 2xxx
    "RECORD_NOT_FOUND": ("MED-2404", "记录不存在"),
    "WORKFLOW_QUEUE_TIMEOUT": ("MED-2429", "工作流排队超时"),
    "WORKFLOW_STATE_ERROR": ("MED-2400", "工作流状态错误"),
    "FILE_TOO_LARGE": ("MED-2413", "文件超出大小限制"),
    "UNSUPPORTED_FILE_TYPE": ("MED-2415", "不支持的文件类型"),
    # 算法服务 3xxx
    "API_KEY_INVALID": ("ALG-3401", "API Key 无效或已停用"),
    "TASK_NOT_FOUND": ("ALG-3404", "任务不存在"),
}


class BizError(HTTPException):
    """业务异常：携带注册表中的错误码，由全局处理器渲染为统一响应体。"""

    def __init__(self, error_name: str, message: str | None = None, status_code: int | None = None):
        code, default_msg = ERROR_CODES[error_name]
        super().__init__(
            status_code=status_code or int(code.split("-")[1]),
            detail=message or default_msg,
        )
        self.error_code = code


def _error_body(request: Request, code: str, message: str) -> dict:
    return {
        "code": code,
        "message": message,
        "request_id": getattr(request.state, "request_id", None),
    }


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(BizError)
    async def _biz_error_handler(request: Request, exc: BizError):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.error_code, str(exc.detail)),
        )

    @app.exception_handler(HTTPException)
    async def _http_error_handler(request: Request, exc: HTTPException):
        # 既有路由仍用原生 HTTPException：透传 detail，配通用错误码前缀
        code = f"HTTP-{exc.status_code}"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, code, str(exc.detail)),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []))
        msg = f"{loc}: {first.get('msg', '参数无效')}" if loc else "参数无效"
        code, message = ERROR_CODES["VALIDATION_ERROR"]
        return JSONResponse(
            status_code=422,
            content=_error_body(request, code, message) | {"detail": msg},
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception):
        # 兜底：堆栈只进日志（request_id 关联），响应体不泄露内部细节
        from loguru import logger

        logger.exception(f"未处理异常 {exc.__class__.__name__}: {exc}")
        code, message = ERROR_CODES["INTERNAL_ERROR"]
        return JSONResponse(
            status_code=500,
            content=_error_body(request, code, message),
        )
