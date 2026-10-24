# -*- coding: utf-8 -*-
"""sqlite 向量库 -> qdrant 迁移脚本（向量直接搬运，不重新调 embedding API，零费用）。

用法：
    cd backend
    # 演习（只统计不动数据）
    python scripts/migrate_vector_to_qdrant.py --dry-run
    # 执行迁移（默认写入 qdrant local：database/vector_store/<指纹>/qdrant_local/）
    python scripts/migrate_vector_to_qdrant.py
    # 迁移到 Docker 服务端（先 docker compose up -d qdrant）
    QDRANT_URL=http://localhost:6333 python scripts/migrate_vector_to_qdrant.py

迁移完成后把 .env 里 VECTOR_BACKEND=qdrant（服务端模式再填 QDRANT_URL）即可切换。
原 sqlite 库文件保留不动，回退只需把 VECTOR_BACKEND 改回 sqlite。
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402


def iter_sqlite_stores(vector_store_dir: Path):
    """遍历所有指纹目录下的 sqlite 向量库：yield (fingerprint, collection, sqlite_path)。"""
    if not vector_store_dir.exists():
        return
    for fingerprint_dir in sorted(vector_store_dir.iterdir()):
        if not fingerprint_dir.is_dir():
            continue
        for db_file in sorted(fingerprint_dir.glob("*.sqlite")):
            yield fingerprint_dir.name, db_file.stem, db_file


def migrate(dry_run: bool = False) -> int:
    from app.core.config import settings

    if settings.vector_backend == "qdrant" and not dry_run:
        logger.warning("当前 VECTOR_BACKEND 已是 qdrant：迁移脚本读 sqlite 文件不受影响，继续")

    total_rows = 0
    total_collections = 0
    t0 = time.time()

    for fingerprint, collection, db_file in iter_sqlite_stores(settings.vector_store_dir):
        conn = sqlite3.connect(str(db_file))
        try:
            rows = conn.execute(
                "SELECT doc_id, content, metadata, vec FROM vectors"
            ).fetchall()
            meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
        finally:
            conn.close()

        if not rows:
            logger.info(f"[跳过] {fingerprint}/{collection}：空库")
            continue

        logger.info(
            f"[迁移] {fingerprint}/{collection}：{len(rows)} 条向量，"
            f"{len(meta_rows)} 条元数据"
        )
        total_collections += 1
        total_rows += len(rows)
        if dry_run:
            continue

        # 构造目标 store（qdrant 后端，落同一指纹目录）
        from app.rag.model_clients import get_embedding_client
        from app.rag.vector_store import QdrantVectorStore

        embedding = get_embedding_client()

        target = QdrantVectorStore(
            collection_name=collection,
            db_path=settings.vector_store_dir / fingerprint / f"{collection}.sqlite",
            embedding_function=embedding,
        )

        doc_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        metadatas = [json.loads(r[2]) for r in rows]
        vectors = [
            np.frombuffer(r[3], dtype=np.float32).tolist() for r in rows
        ]

        # 指纹元数据先搬（clear 语义不覆盖 meta，但迁移幂等起见直接写）
        for key, value in meta_rows:
            target.set_meta(key, value)

        # 分批直写（upsert 幂等，重跑安全）
        batch = 500
        for start in range(0, len(rows), batch):
            target.add_embeddings(
                doc_ids[start : start + batch],
                texts[start : start + batch],
                metadatas[start : start + batch],
                vectors[start : start + batch],
            )
            logger.info(
                f"  已写入 {min(start + batch, len(rows))}/{len(rows)}"
            )

        assert target.count() == len(rows), (
            f"迁移校验失败：期望 {len(rows)}，实际 {target.count()}"
        )

    mode = "（演习，未写数据）" if dry_run else ""
    logger.success(
        f"迁移完成{mode}：{total_collections} 个 collection / {total_rows} 条向量，"
        f"耗时 {time.time() - t0:.1f}s"
    )
    if not dry_run and total_rows:
        logger.info("下一步：.env 中设置 VECTOR_BACKEND=qdrant 后重启服务")
    return 0 if total_collections or dry_run else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sqlite 向量库迁移到 qdrant")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写数据")
    args = parser.parse_args()
    sys.exit(migrate(dry_run=args.dry_run))
