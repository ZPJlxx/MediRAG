# MediRAG 智慧医疗辅助诊疗平台

> RAG 增强的多 Agent 医疗辅助诊疗系统：症状分析 → 化验中断 → 诊断推理 → 治疗建议，**每条建议都能给出知识库出处**。

## 项目简介

MediRAG 是三个项目整合二次创新的成果：

| 来源项目 | 提供的核心能力 |
|---|---|
| **MediAgent-LLM**（医疗多Agent） | LangGraph 诊疗工作流骨架、`interrupt()/Command(resume=)` 人机协同（等化验结果）、全链路审计 |
| **general-doc-qa-assistant**（文档问答） | 三路混合检索（向量+BM25+关键词打分/RRF融合/重排）、证据位置标签（页码/行号/章节）、防幻觉设计 |
| **Enterprise-Knowledge-Rag-Agent**（企业知识库） | 用户/角色体系与管理后台形态（本次重写为 JWT + bcrypt 安全版） |

**核心创新点**：
1. **证据可溯源的医疗 AI**——诊断/治疗建议强制携带 `citations`（来源文件 + 位置 + 原文片段），LLM 只能引用检索提供的编号，无法编造出处
2. **RAG 注入多 Agent 工作流**——症状/诊断/治疗三个 Agent 各自构造检索查询（症状描述→疾病名+症状→主诊断+鉴别诊断），参考资料块注入 Prompt
3. **鉴别假设并行核验（Send fan-out / map-reduce）**——诊断前对 Top-3 鉴别假设各派生一个独立核验 Agent **并行执行**（同一超步分发，`Annotated[list, operator.add]` reducer 聚合），对照患者症状、化验结果与知识库分别打支持度分，榜单注入诊断 Agent 综合裁决——即医疗领域的 second opinion（第二意见）机制；主流程保持固定 DAG 以确保可审计（每次核验调用均落 `agent_session_his`）
4. **检查点持久化的人机协同**——SqliteSaver 替代内存检查点，化验结果回填可跨进程恢复工作流
5. **修复并增强了三个源项目的已知缺陷**（详见下文"修复清单"）

## 架构

```
React 医疗工作台 (5173)
 ├─ 登录(JWT) / 诊疗工作台 / 知识库管理 / 用户管理(admin)
 └─ 证据引用卡片 · 状态机步骤条 · 化验回填表单 · 流程时间线
        │ /api/*（代理到 8000）
FastAPI 单服务 (8000)
 ├─ auth/knowledge/users/workflow 四组路由
 ├─ LangGraph 诊疗工作流（SqliteSaver 检查点持久化）
 │    create_inquiry → symptom_analysis → [interrupt 等化验]
 │      → prepare_hypotheses ──Send×N──► verify_hypothesis（并行超步，reducer 聚合）
 │      → merge_verifications（支持度榜单）→ diagnosis → treatment → complete
 │    每个 Agent：审计 → RAG检索 → 参考资料注入 → LLM → Pydantic校验 → 引用映射 → 落库
 ├─ RAG 引擎（app/rag/）
 │    loader(中文感知切分+位置标签) / SqliteVectorStore(SQLite+numpy余弦)
 │    KeywordPriorityRetriever(向量+BM25[jieba]+关键词打分, RRF融合) / DashScope gte-rerank-v2
 │    evidence(Citation 构造 / 编号映射防编造)
 └─ 存储：SQLite(dev)/MySQL(prod) + 向量库(SQLite文件) + 工作流检查点(SQLite)
LLM 网关：OpenAI 兼容协议（DashScope qwen-plus 默认，可切 DeepSeek/任意兼容端点）
```

## 快速开始（开发模式，零外部服务依赖）

**Windows 一键方式**：
1. 首次使用：双击 `start-init.bat`（建 venv、装依赖、建表、种子数据，需已装 Python 3.11+ 和 Node 18+；结束后编辑 `backend/.env` 填入 DASHSCOPE_API_KEY）
2. 日常启动：双击 `start-dev.bat`（两个窗口分别拉起后端 8000 与前端 5173）

**手动方式**：

```bash
# 1. 后端
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # 填入 DASHSCOPE_API_KEY（或 LLM_API_KEY/LLM_BASE_URL）
python -m app.seed                # 建表 + 种子用户(admin/admin123, doctor/doctor123) + 3份样例医学知识库
python -m uvicorn app.main:app --port 8000

# 2. 前端（注意：在 frontend/ 目录下，不是项目根目录）
cd frontend
npm install
npm run dev                       # http://localhost:5173
```

生产模式（可选）：`docker-compose up -d` 启动 MySQL + RabbitMQ，`.env` 中
`DATABASE_URL=mysql+pymysql://root:medirag123@localhost:3306/medirag`、
`RABBITMQ_ENABLED=true`（化验结果走 MQ 异步回传，HTTP 接口始终可用）。

## 使用流程

1. `doctor/doctor123` 登录 → 诊疗工作台
2. 录入患者信息 + 症状描述（可选填过敏史）→ 启动工作流
3. 症状分析完成；若需化验，工作流**暂停**在"等待化验"步骤
4. 化验结果回填（表格输入）→ 工作流自动恢复 → 诊断 → 治疗
5. 每个阶段结果下方展示**证据引用卡片**（来源文档 + 位置 + 原文片段）
6. 知识库页可上传临床指南/药品说明书（pdf/docx/txt/md/csv），或用检索测试验证效果

## 知识库数据集

| 语料 | 规模 | 许可证 | 入库方式 |
|---|---|---|---|
| 自编演示文档（高血压/肺炎/糖尿病要点） | 3 份 / 5KB | 自编 | `python -m app.seed` |
| **真实临床指南**（lanciSTar/chinese-clinical-guidelines）：中国糖尿病防治指南2024、中国高血压临床实践指南、基层高血压/糖尿病管理指南2025、高尿酸与痛风诊疗指南2024 等 | 6 份 PDF / 21MB | 研究用途（版权归原发布方） | `python scripts/import_corpus.py guidelines` |
| **Huatuo26M-Lite** 医疗百科 QA（FreedomIntelligence） | 抽样 2 万条 / 13.7MB 合成语料（全集 17.8 万条可 `--sample` 调整） | **Apache-2.0** | `python scripts/import_corpus.py huatuo --sample 20000` |

索引规模：约 3.7 万 chunk / 112MB 向量库（`database/vector_store/`，按 embedding 指纹隔离）。
证据引用实测同时命中真实指南 PDF（带页码，如《中国高血压临床实践指南》第 15 页联合用药方案）与 Huatuo 真实问答。

**评测数据集**：
- C-MTEB **CmedqaRetrieval**（医疗问诊垂类，corpus 10 万段 + 3,999 查询 + dev qrels）——`python -m app.eval.bench_eval --n 200`
- **CMB-Clin**（FreedomIntelligence，Apache-2.0）真实病例 74 例抽取 12 例扩充 A/B——`python -m app.eval.build_cmb_cases --n 12`
- 自制 14 条标注查询（回归冒烟）——`python -m app.eval.retrieval_eval`

检索性能（3.7 万 chunk、10 个 collection、含 DashScope rerank 网络往返）：**P50=0.71s，MAX=1.11s**。

```bash
cd backend
python -m app.eval.retrieval_eval   # 检索质量：recall@5 / MRR / 关键词覆盖率
python -m app.eval.ab_compare       # A/B：RAG 开 vs 关（证据引用数 / 用药一致率 / 确定性）
```

**评估结果（扩容后：3.7 万 chunk / 12 collection）**：

| 指标 | 结果 | 口径 |
|---|---|---|
| 检索 recall@5 | **100%** | 14 条标注查询，多来源白名单（指南PDF/演示文档/Huatuo） |
| 检索 MRR | 0.857 | 同上（真实指南入池后排序竞争加剧，属正常） |
| 关键词覆盖率 | 78.3% | 命中片段含期望关键词 |
| 检索延迟 P50 / MAX | **0.71s / 1.11s** | 含 DashScope rerank 网络往返 |
| A/B 证据引用 | RAG 组 75 条 / 对照组 **0** 条 | 15 病例（3 自制 + 12 CMB 真实）×2 组，30 次工作流全部跑通 |
| A/B 诊断确定性 | RAG 90.1 vs 无 RAG 86.1 | 同上 |
| A/B CMB 诊断一致率 | 23.1%（两组同） | 口径严格：需 LLM 措辞与标准答案字面匹配；知识库暂未覆盖 CMB 外科领域，该指标当前主要反映难度基线 |
| **CmedqaRetrieval 基准**（公开，可对外引用） | 消融四组（200 查询×1 万段）：①v3 纯混合 recall@5=49.5%/MRR=0.386；②v3+rerank **75.0%/0.633**；③bge-m3 纯混合 38.5%/0.307；④bge-m3+rerank 47.0%/0.430 | 结论：**rerank 是最大杠杆（+25.5pt）**；bge-m3 经硅基流动无指令前缀调用在本基准弱于 v3（该基准段落多为短问句、m3 的长文档优势发挥不出且缺 query 指令前缀），生产保持 DashScope v3 + gte-rerank-v2 组合 |

## 对源项目缺陷的修复清单

| 源项目缺陷 | MediRAG 修复 |
|---|---|
| MediAgent：过敏史写死 `[]`，禁忌检查形同虚设 | 过敏史随问诊记录入库并传入治疗 Agent，程序化双向匹配核对 + Prompt 约束 |
| MediAgent：状态接口写死 `"running"` | 数据库状态机 + 检查点快照合成真实状态 |
| MediAgent：MemorySaver 重启即失效 | SqliteSaver 持久化检查点，thread_id 落库 |
| MediAgent：同步 SDK 阻塞事件循环 | 路由层 `asyncio.to_thread` |
| MediAgent：Pydantic v1 写法 | 迁移 v2（`field_validator` / `model_dump`） |
| MediAgent：硬校验"≥5种疾病"导致整体失败 | 放宽为 ≥1，数量要求由提示词约束 |
| doc-qa：BM25 中文按空格分词失效 | jieba 分词预处理 |
| doc-qa：无删除知识库能力 | 删除联动清理文件/向量库/缓存 |
| doc-qa：会话仅内存 | 审计与结果全部落库 |
| Enterprise-RAG：明文密码、每请求带密码 | bcrypt 哈希 + JWT Bearer 鉴权 |
| Enterprise-RAG：DB 配置硬编码 5 处 | 统一 `app/core/config.py`（pydantic-settings） |
| Enterprise-RAG：3后端+3前端进程割裂 | 单 FastAPI + 单 React |

## 已知限制与说明

- **向量库**：chromadb 的原生 wheel（chroma-hnswlib / rust frontend）在部分 Windows + Python 3.12 环境下 `add_items` 段错误（faulthandler 定位于原生层），本项目改用自实现 `SqliteVectorStore`（SQLite 存储 + numpy 暴力余弦检索）。医学知识库规模（万级 chunk 内）延迟毫秒级，零原生依赖。若在 Linux/其他环境可自行换回 Chroma（接口面已隔离在 `create_indexes`）。
- 患者身份以"姓名"近似关联历史（演示用途），生产应换真实患者主键。
- **合规**：仅供演示与辅助参考，不能替代执业医师的诊断与治疗决策。
