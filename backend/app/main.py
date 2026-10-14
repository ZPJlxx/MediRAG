"""MediRAG 后端主入口。

三个来源项目的服务形态整合：
- Enterprise-RAG 的 3 个 FastAPI + 3 个 Streamlit -> 一个 FastAPI + 一个 React 前端
- MediAgent-LLM 的 RabbitMQ 消费者按配置可选启动（dev 默认 HTTP 回传）
"""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.algorithm_routes import router as algorithm_router
from app.api.auth_routes import router as auth_router
from app.api.knowledge_routes import router as knowledge_router
from app.api.user_routes import router as user_router
from app.api.workflow_routes import router as workflow_router
from app.core.config import settings
from app.core.database import create_tables
from app.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    # ---- 启动自检：逐项检查配置/连通性，fail 级问题阻断启动 ----
    checks = settings.run_startup_checks()
    failed = [c for c in checks if c[1] == "fail"]
    for item, level, desc in checks:
        icon = {"ok": "OK ", "warn": "WARN", "fail": "FAIL"}[level]
        logger.info(f"自检 [{icon}] {item:14s} {desc}")
    if failed:
        raise RuntimeError(f"启动自检未通过（{len(failed)} 项 fail），服务拒绝启动。见上方日志。")

    settings.ensure_runtime_directories()
    settings.validate_security()
    create_tables()
    logger.info(
        f"MediRAG 启动完成 env={settings.app_env}, "
        f"db={'sqlite' if settings.is_sqlite else 'mysql'}, "
        f"rag={'on' if settings.medical_rag_enabled else 'off'}"
    )

    # 可选：RabbitMQ 化验结果消费者（prod 模式；失败不阻断启动，HTTP 回传始终可用）
    if settings.rabbitmq_enabled:
        def _start_consumer():
            try:
                from app.medical.workflow import medical_workflow

                import pika

                def on_message(ch, method, properties, body):
                    import json

                    try:
                        message = json.loads(body.decode("utf-8"))
                        medical_workflow.handle_lab_results(
                            message["inquiry_record_id"], message.get("lab_results", [])
                        )
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as exc:
                        logger.error(f"MQ 化验结果处理失败: {exc}")
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=settings.rabbitmq_host,
                        port=settings.rabbitmq_port,
                        credentials=pika.PlainCredentials(
                            settings.rabbitmq_username, settings.rabbitmq_password
                        ),
                        heartbeat=600,
                    )
                )
                channel = connection.channel()
                channel.queue_declare(queue="lab_results", durable=True)
                channel.basic_consume(queue="lab_results", on_message_callback=on_message)
                logger.info("RabbitMQ 化验结果消费者已启动（队列 lab_results）")
                channel.start_consuming()
            except Exception as exc:
                logger.warning(f"RabbitMQ 消费者启动失败（HTTP 回传仍可用）: {exc}")

        threading.Thread(target=_start_consumer, daemon=True).start()

    yield
    logger.info("MediRAG 关闭")


app = FastAPI(
    title="MediRAG 智慧医疗辅助诊疗平台",
    description="RAG 增强的多 Agent 医疗辅助诊疗系统：症状分析 -> 化验中断 -> 诊断 -> 治疗，全程证据可溯源",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 可观测性：request-id 生成/透传 + 访问日志（放 CORS 内层，所有业务请求都覆盖）
from app.core.middleware import RequestIdMiddleware  # noqa: E402

app.add_middleware(RequestIdMiddleware)

# 统一错误码与响应体：{code, message, request_id}
from app.core.errors import register_error_handlers  # noqa: E402

register_error_handlers(app)

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(knowledge_router)
app.include_router(workflow_router)
app.include_router(algorithm_router)


# ---------- 前端静态资源托管（单服务部署模式）----------
# 存在 ../frontend/dist 时由后端直接伺服（API 与静态资源同源，免去独立 Nginx）；
# 开发模式（Vite 5173）或未构建时不挂载，不影响原有流程。
_FRONTEND_DIST = settings.project_root / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_spa():
        return FileResponse(_FRONTEND_DIST / "index.html")

    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
async def root():
    return {
        "name": "MediRAG 智慧医疗辅助诊疗平台",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    import datetime as dt

    return {"status": "healthy", "timestamp": dt.datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    # dev 环境自动热重启：改完 Python 代码保存即生效，无需手动重启服务
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=settings.app_env == "dev")
