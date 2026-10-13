# MediRAG 进度说明

> 更新：2026-09-29。知识库扩容与性能优化任务全部完成，本文档为状态存档。

## 全部已完成

- **知识库扩容**：6 份真实临床指南 PDF（21MB）+ Huatuo26M-Lite 2 万条（13.7MB），
  索引 112MB / 12 collection，检索 P50=0.71s / MAX=1.11s
- **embedding 并行化**：`EMBED_CONCURRENCY=4`（可调 1-8），限流自适应退避；
  实测 1 万段落建索引约 2 分钟（串行约 8 倍耗时）
- **评测全套**：
  - retrieval_eval（14 条标注，多来源白名单）：recall@5=100%，MRR=0.857
  - A/B（3 自制 + 12 CMB 真实病例 ×2 组）：RAG 组 75 引用 vs 对照 0；确定性 90.1 vs 86.1
  - CmedqaRetrieval 公开基准（200 查询 × 1 万段构造性抽样子库）：
    recall@5=49.5%，recall@10=63.0%，MRR@10=0.386，0.24s/查询（无 rerank 口径）
- **过程修复**（详见 architecture.md 第五节）：非重入锁死锁、大库倒排降级、
  embedding 8192 截断、文件间并发检索、限流退避

## 后续可选

- Cmedqa 全量 10 万段基准：需提升 DashScope 配额后
  `python -m app.eval.bench_eval --n 200`（去掉 --corpus-sample）
- 云服务器部署：docker-compose + .env 切 MySQL/RabbitMQ 即可（见 README）
- LLM 换 qwen-turbo 提速单次问诊（.env 改 LLM_MODEL 即可 A/B）
