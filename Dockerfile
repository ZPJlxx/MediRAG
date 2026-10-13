# syntax=docker/dockerfile:1
# ============================================================
# MediRAG 单镜像：前端 dist + 后端 FastAPI 同容器伺服
# 构建:  docker build -t medirag .
# 运行:  docker run -p 8000:8000 --env-file backend/.env medirag
# 数据:  建议挂载卷保存 /app/backend/database（SQLite/向量库/检查点）
# ============================================================

# ---- 阶段 1：构建前端 ----
FROM node:20-alpine AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 阶段 2：后端 + 前端产物 ----
FROM python:3.11-slim
WORKDIR /app/backend

# 系统依赖（bcrypt/cryptography 有 wheel，一般无需编译链；保留 gcc 兜底）
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/alembic.ini ./
COPY backend/alembic ./alembic
COPY --from=frontend-build /build/dist /app/frontend/dist

# 运行时数据卷（SQLite / 向量库 / 检查点）
VOLUME ["/app/backend/database"]
VOLUME ["/app/backend/knowledge_docs"]

EXPOSE 8000
# 启动前先跑迁移，保证表结构与代码版本一致
CMD ["sh", "-c", "python -m alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]
