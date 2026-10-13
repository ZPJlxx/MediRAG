"""开源医学语料导入脚本。

来源（均已核实可直接下载）：
1. 真实临床指南 16 份：github.com/lanciSTar/chinese-clinical-guidelines（研究用途）
2. Huatuo26M-Lite 医疗百科 QA：FreedomIntelligence/Huatuo26M-Lite（Apache-2.0，
   走 hf-mirror 镜像下载，本机直连 huggingface.co 超时）

用法（backend 目录）：
    python scripts/import_corpus.py guidelines            # 只导指南
    python scripts/import_corpus.py huatuo --sample 20000 # 抽样 2 万条冒烟
    python scripts/import_corpus.py huatuo                # 全量 17.8 万条
    python scripts/import_corpus.py all --sample 20000    # 两者都导

注意：Huatuo 首次索引会产生 DashScope embedding 批量调用——
全量约 7,000 批次（约 1-2 小时、费用约 10 元级），建议先 --sample 20000 冒烟。
中断重跑安全：增量写入，已索引 chunk 自动跳过。
"""

import argparse
import io
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

GUIDELINES_ZIP = "https://codeload.github.com/lanciSTar/chinese-clinical-guidelines/zip/refs/heads/main"
HF_MIRROR = "https://hf-mirror.com"
HUATUO_REPO = "FreedomIntelligence/Huatuo26M-Lite"
ANSWER_MAX_CHARS = 800


def _download(url: str, dest: Path, timeout: int = 300) -> Path:
    """流式下载带进度日志。"""
    import urllib.request

    logger.info(f"下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "MediRAG-import/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    logger.info(f"已下载 {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


def import_guidelines() -> int:
    """下载指南仓库 zip，提取 PDF / 已提取 txt 到 knowledge_docs/guidelines/。"""
    from app.core.config import settings
    from app.core.database import SessionLocal
    from app.models.entities import KnowledgeFile

    guidelines_dir = settings.knowledge_dir / "guidelines"
    guidelines_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "guidelines.zip"
        _download(GUIDELINES_ZIP, zip_path)

        extract_dir = Path(tmp) / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # 仓库结构：<repo-main>/<子目录>/...；收集全部 pdf 与 *_extracted.txt
        pdfs = list(extract_dir.rglob("*.pdf"))
        txts = list(extract_dir.rglob("*_extracted.txt"))
        logger.info(f"指南仓库内容: {len(pdfs)} 份 PDF, {len(txts)} 份已提取 txt")

        imported = 0
        seen_names = set()
        for pdf in pdfs:
            target = guidelines_dir / pdf.name
            shutil.copyfile(pdf, target)
            imported += 1
            seen_names.add(pdf.stem)
        # 无对应 PDF 的 extracted txt 兜底入库（同一指南 PDF 优先时跳过同名 txt）
        for txt in txts:
            stem = txt.name.replace("_extracted.txt", "")
            if stem in seen_names:
                continue
            shutil.copyfile(txt, guidelines_dir / txt.name)
            imported += 1

    # 登记台账（相对路径作为 filename，与 list_files 的 POSIX 相对路径一致）
    session = SessionLocal()
    try:
        from app.rag.knowledge_manager import get_md5

        for path in sorted(guidelines_dir.iterdir()):
            if not path.is_file():
                continue
            rel = path.relative_to(settings.knowledge_dir).as_posix()
            existing = session.query(KnowledgeFile).filter(KnowledgeFile.filename == rel).first()
            if existing:
                continue
            session.add(KnowledgeFile(
                filename=rel,
                collection_name=get_md5(rel),
                size_bytes=path.stat().st_size,
                uploader_id=0,
            ))
        session.commit()
    finally:
        session.close()

    logger.info(f"指南导入完成: {imported} 份 -> knowledge_docs/guidelines/")
    return imported


def import_huatuo(sample: int | None) -> int:
    """下载 Huatuo26M-Lite（format_data.jsonl, 138MB, Apache-2.0），
    流式合成单一大 md 入 knowledge_docs/huatuo/。

    单一大文件设计：全库一个 collection，聚合检索只做一次向量扫描，
    避免 17.8 万条小文件 × 重复建索引/重复 embed 的开销。
    """
    import os

    os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)
    from huggingface_hub import hf_hub_download

    from app.core.config import settings
    from app.core.database import SessionLocal
    from app.models.entities import KnowledgeFile

    local = hf_hub_download(
        repo_id=HUATUO_REPO, filename="format_data.jsonl", repo_type="dataset"
    )
    logger.info(f"已下载 Huatuo 原始文件: {local}")

    huatuo_dir = settings.knowledge_dir / "huatuo"
    huatuo_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_s{sample}" if sample else ""
    corpus_path = huatuo_dir / f"huatuo26m_lite_corpus{suffix}.md"
    rel = corpus_path.relative_to(settings.knowledge_dir).as_posix()

    # 去重护栏：换抽样档位会生成新文件；同档位重跑直接跳过（想重建先删文件）
    if corpus_path.exists():
        logger.warning(f"语料文件已存在，跳过生成: {rel}（如需重建请先删除该文件）")
    else:
        count = 0
        with open(local, "r", encoding="utf-8") as src, open(corpus_path, "w", encoding="utf-8") as out:
            out.write("# Huatuo26M-Lite 医疗百科问答语料（Apache-2.0）\n\n")
            for line in src:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 数据集字段：question/answer 单条 + label(科室) + related_diseases
                question = str(item.get("question") or "").strip()
                answer = str(item.get("answer") or "").strip()[:ANSWER_MAX_CHARS]
                if len(question) < 4 or len(answer) < 8:
                    continue
                department = str(item.get("label") or "").strip()
                disease = str(item.get("related_diseases") or "").strip()
                tags = "、".join(t for t in (department, disease) if t)
                header = f"### 问：{question}"
                if tags:
                    header += f"（{tags}）"
                out.write(f"{header}\n\n**答**：{answer}\n\n")
                count += 1
                if sample and count >= sample:
                    break
        logger.info(f"合成语料完成: {rel}，{count} 条 QA，"
                    f"{corpus_path.stat().st_size / 1024 / 1024:.1f} MB")

    session = SessionLocal()
    try:
        from app.rag.knowledge_manager import get_md5

        existing = session.query(KnowledgeFile).filter(KnowledgeFile.filename == rel).first()
        if not existing:
            session.add(KnowledgeFile(
                filename=rel,
                collection_name=get_md5(rel),
                size_bytes=corpus_path.stat().st_size,
                uploader_id=0,
            ))
            session.commit()
    finally:
        session.close()

    logger.info(f"Huatuo 导入完成 -> {rel}（首次检索时会自动建索引，产生 embedding 调用）")
    return 0


def main():
    global logger
    from loguru import logger as _logger
    logger = _logger

    parser = argparse.ArgumentParser(description="MediRAG 开源医学语料导入")
    parser.add_argument("target", choices=["guidelines", "huatuo", "all"], help="导入目标")
    parser.add_argument("--sample", type=int, default=None, help="Huatuo 抽样条数（默认全量）")
    args = parser.parse_args()

    if args.target in ("guidelines", "all"):
        import_guidelines()
    if args.target in ("huatuo", "all"):
        import_huatuo(args.sample)


if __name__ == "__main__":
    main()
