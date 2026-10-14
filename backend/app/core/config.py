"""MediRAG 统一配置中心。

整合三个来源项目的配置方式，收敛为单一入口：
- Enterprise-Knowledge-Rag-Agent：yaml + 环境变量混用、DB 配置硬编码 5 处 -> 统一消除
- MediAgent-LLM：.env + os.getenv 散落各文件
- general-doc-qa-assistant：.env + models.py 里的检索侧配置解析

所有模块只从这里读取配置（Settings 单例），密钥只出现在 .env。
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env", override=False)

DASHSCOPE_COMPATIBLE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_URL = "https://api.deepseek.com/v1"


class Settings(BaseSettings):
    """全局配置。字段名与环境变量一一对应。"""

    model_config = SettingsConfigDict(env_file=str(BACKEND_DIR / ".env"), extra="ignore")

    # ---- 运行环境 ----
    app_env: str = "dev"
    app_name: str = "MediRAG"

    # ---- 数据库 ----
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'database' / 'medirag.db').as_posix()}"

    # ---- JWT ----
    jwt_secret: str = "dev-secret-change-me"
    jwt_expire_minutes: int = 720
    jwt_algorithm: str = "HS256"

    # ---- LLM 网关（OpenAI 兼容）----
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    dashscope_api_key: str = ""
    deepseek_api_key: str = ""

    # ---- 检索侧 embedding / rerank（沿用 general-doc-qa-assistant 的配置约定）----
    retrieval_embedding_provider: str = "dashscope"
    retrieval_embedding_model: str = "text-embedding-v3"
    retrieval_embedding_base_url: str = ""
    retrieval_embedding_api_key: str = ""
    retrieval_rerank_provider: str = "dashscope"
    retrieval_rerank_model: str = "gte-rerank-v2"
    retrieval_rerank_api_key: str = ""

    # ---- 医疗 RAG ----
    medical_rag_enabled: bool = True
    rag_top_k: int = 5

    # ---- 工作流并发控制（保护 LLM 网关配额）----
    # 单 worker 同时执行的工作流上限，超出排队；多进程部署时总并发 = 该值 × worker 数
    max_concurrent_workflows: int = 3
    # 排队等待槽位的超时秒数，超时直接报错（防请求无限堆积）
    workflow_queue_timeout_seconds: int = 120

    # ---- 文档切分 ----
    chunk_size: int = 500
    chunk_overlap: int = 200

    # ---- RabbitMQ ----
    rabbitmq_enabled: bool = False
    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = 5672
    rabbitmq_username: str = "guest"
    rabbitmq_password: str = "guest"

    # ---- 派生路径 ----
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def knowledge_dir(self) -> Path:
        return BACKEND_DIR / "knowledge_docs"

    @property
    def vector_store_dir(self) -> Path:
        return BACKEND_DIR / "database" / "vector_store"

    @property
    def checkpoint_db_path(self) -> Path:
        return BACKEND_DIR / "database" / "workflow_checkpoints.db"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def resolve_llm(self) -> dict:
        """解析医疗 Agent 实际使用的聊天模型配置。

        优先级：显式 LLM_API_KEY/LLM_BASE_URL > DashScope(qwen) > DeepSeek。
        三个来源项目各自写死模型的问题在此统一为一个可切换网关。
        """
        if self.llm_api_key and self.llm_base_url:
            return {
                "api_key": self.llm_api_key,
                "base_url": self.llm_base_url,
                "model": self.llm_model or "custom-model",
                "provider": "custom",
            }
        if self.dashscope_api_key:
            return {
                "api_key": self.dashscope_api_key,
                "base_url": DASHSCOPE_COMPATIBLE_URL,
                "model": self.llm_model or "qwen-plus",
                "provider": "dashscope",
            }
        if self.deepseek_api_key:
            return {
                "api_key": self.deepseek_api_key,
                "base_url": DEEPSEEK_URL,
                "model": self.llm_model or "deepseek-chat",
                "provider": "deepseek",
            }
        return {"api_key": "", "base_url": "", "model": self.llm_model or "", "provider": "none"}

    def retrieval_embedding_api_key_resolved(self) -> str:
        """embedding 专用 key：优先专用配置，回退 DashScope / OpenAI 主 key。"""
        if self.retrieval_embedding_api_key:
            return self.retrieval_embedding_api_key
        if self.retrieval_embedding_provider == "dashscope":
            return self.dashscope_api_key
        return self.llm_api_key

    def retrieval_rerank_api_key_resolved(self) -> str:
        if self.retrieval_rerank_api_key:
            return self.retrieval_rerank_api_key
        return self.dashscope_api_key

    def ensure_runtime_directories(self) -> None:
        for directory in (
            self.knowledge_dir,
            self.vector_store_dir,
            BACKEND_DIR / "database",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # 已知的不安全默认密钥（config 默认值与 .env.example 示例值），生产环境禁止使用
    _INSECURE_JWT_SECRETS = frozenset({"dev-secret-change-me", "change-me-in-production-9f8e7d6c5b4a"})

    def validate_security(self) -> None:
        """JWT 密钥安全检查：prod 用已知默认值直接拒绝启动（fail-fast），dev 仅告警。"""
        if self.jwt_secret in self._INSECURE_JWT_SECRETS:
            message = (
                "JWT_SECRET 仍为默认示例值，令牌可被伪造。"
                "请生成随机密钥（如 python -c \"import secrets; print(secrets.token_hex(32))\"）"
                "并写入 backend/.env 的 JWT_SECRET。"
            )
            if self.app_env == "prod":
                raise RuntimeError(f"[安全拦截] {message}")
            import warnings

            warnings.warn(f"[安全告警] {message}（dev 环境允许启动）", stacklevel=2)

    def run_startup_checks(self) -> list[tuple[str, str, str]]:
        """启动自检：返回 [(项, 状态(ok/warn/fail), 说明)] 清单。

        fail 级问题由调用方（main.lifespan）决定是否阻断启动；
        warn 仅提示。集中在此避免校验逻辑散落各模块。
        """
        checks: list[tuple[str, str, str]] = []

        # 1. LLM 网关配置
        llm = self.resolve_llm()
        if llm["provider"] == "none":
            checks.append(("llm", "fail", "未配置任何 LLM 网关（LLM_API_KEY / DASHSCOPE_API_KEY / DEEPSEEK_API_KEY 均为空），诊疗工作流不可用"))
        else:
            checks.append(("llm", "ok", f"provider={llm['provider']}, model={llm['model']}"))

        # 2. 检索侧 embedding
        if self.retrieval_embedding_provider == "disabled":
            checks.append(("embedding", "warn", "检索 embedding 已禁用，RAG 证据链不可用（LLM 将无参考作答）"))
        elif not self.retrieval_embedding_api_key_resolved():
            key_desc = "RETRIEVAL_EMBEDDING_API_KEY / DASHSCOPE_API_KEY"
            checks.append(("embedding", "warn", f"检索 embedding 缺少 API Key（{key_desc}），首次检索将失败"))
        else:
            checks.append(("embedding", "ok", f"{self.retrieval_embedding_provider}::{self.retrieval_embedding_model}"))

        # 3. 数据库连通性
        try:
            from sqlalchemy import create_engine, text

            probe = create_engine(self.database_url, pool_pre_ping=True)
            with probe.connect() as conn:
                conn.execute(text("SELECT 1"))
            probe.dispose()
            db_kind = "mysql" if not self.is_sqlite else "sqlite"
            checks.append(("database", "ok", f"{db_kind} 连通"))
        except Exception as exc:
            checks.append(("database", "fail", f"连接失败: {exc}（MySQL 模式请先 docker compose start mysql）"))

        # 4. 运行目录可写
        try:
            self.ensure_runtime_directories()
            checks.append(("runtime_dirs", "ok", "knowledge/vector_store/database 目录就绪"))
        except Exception as exc:
            checks.append(("runtime_dirs", "fail", f"运行目录创建失败: {exc}"))

        # 5. JWT 密钥
        if self.jwt_secret in self._INSECURE_JWT_SECRETS:
            checks.append(("jwt_secret", "warn" if self.app_env != "prod" else "fail", "使用默认示例密钥"))
        else:
            checks.append(("jwt_secret", "ok", "已配置独立密钥"))

        return checks


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
