# MediRAG 技术架构文档

## 一、总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│  React 医疗工作台（frontend/，Vite + React 18）                     │
│  Login · Workspace(诊疗) · KnowledgePanel(知识库) · UsersPanel     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ REST /api/*（JWT Bearer）
┌──────────────────────────────▼───────────────────────────────────┐
│  FastAPI 单服务（backend/app/）                                    │
│                                                                    │
│  api/        auth_routes · user_routes · knowledge_routes ·       │
│              workflow_routes · deps(JWT/角色)                      │
│                                                                    │
│  medical/    workflow(LangGraph 状态机) · llm_client(Prompt+解析)  │
│              agents/ base(公共骨架) + 症状/诊断/治疗三Agent         │
│                                                                    │
│  rag/        loader · vector_store · retriever · model_clients ·  │
│              evidence · knowledge_manager                          │
│                                                                    │
│  core/       config(pydantic-settings) · database(SQLAlchemy2.0)  │
│              security(bcrypt+JWT) · logging(loguru)               │
│                                                                    │
│  dao/ models/ schemas/ eval/                                       │
└───────┬──────────────┬──────────────┬─────────────────────────────┘
        │              │              │
   SQLite/MySQL    向量库文件      工作流检查点
   (业务数据)      (SQLite+numpy)  (SqliteSaver)
```

## 二、核心链路

### 2.1 诊疗工作流（LangGraph StateGraph）

```
create_inquiry ──► symptom_analysis ──► check_lab_needed
                        │                     │
                        │          需化验且无结果 │ 无需化验
                        │                     ▼
                        │              wait_for_lab  ←── interrupt() 暂停
                        │                     │              ▲
                        │                     ▼              │ HTTP /lab-results
                        │           prepare_hypotheses      │ 或 RabbitMQ 消费者
                        │                │        │         │
                        │     有Top-N假设 │        │无假设    │
                        │    Send×N ──►──┘        │(跳过)    │ Command(resume=lab_results)
                        │        ▼                          │
                        │  verify_hypothesis ×N（并行超步）   │
                        │        │  reducer: operator.add   │
                        │        ▼                          │
                        │  merge_verifications（榜单+摘要）   │
                        │        │                          │
                        │        ▼                          │
                        │  diagnosis_analysis ◄─────────────┘
                        │        │
                        │        ▼
                        │  treatment_advice
                        │        │
                        │        ▼
                        │  complete_workflow ──► END
                        ▼
                  handle_error ──► END（任何节点失败统一路由）
```

### 2.1.1 鉴别假设并行核验（Send fan-out / map-reduce）

- **map**：`prepare_hypotheses` 从症状分析的 `possible_diseases` 取 Top-3（按
  probability 降序）写入 `hypotheses`；`_fan_out_or_skip` 返回 `Send` 列表，
  每个假设一个分支，LangGraph 在**同一超步并行调度**（实测三分支同毫秒启动）
- **分支**：`verify_hypothesis` 收到的 state 即 Send payload（假设+患者信息+化验
  结果），核验 Agent 独立执行 RAG 检索 + LLM 对照评分；分支只返回
  `{"hypothesis_verifications": [结果]}`——`hypothesis_verifications` 声明为
  `Annotated[list, operator.add]`，各分支结果自动累加（并行写同键必须 reducer，
  否则 InvalidUpdateError；且不能整包返回 payload，非 channel 键会校验失败）
- **reduce**：`merge_verifications` 按支持度排序生成榜单 + 文本摘要，摘要注入
  诊断 Agent 的 Prompt（`verification_block`）；榜单存入
  `verification_ranking`，`get_status` 从检查点快照读出供前端展示
- **容错**：单分支失败不拖垮整体（0 分占位 + rationale 标注失败原因）；
  每个分支的调用审计独立落 `agent_session_his`（agent_type=4）
- **医疗语义**：second opinion（第二意见）机制——多个假设被独立、无偏见地
  核验后再裁决，而非单一 LLM 串行自证


- 检查点：`SqliteSaver`（`database/workflow_checkpoints.db`），thread_id 写入
  `inquiry_record.thread_id`，进程重启后仍可通过化验回传恢复
- 每阶段流转写 `workflow_event` 表（前端时间线数据源）
- async 路由通过 `asyncio.to_thread` 调用同步 graph，避免阻塞事件循环

### 2.2 RAG 证据链（核心创新）

```
Agent 检索查询构造（三阶段各不同）
  症状Agent: 症状描述原文
  诊断Agent: 高概率疾病×3 + 关键症状×4
  治疗Agent: 主诊断 + 鉴别诊断×2
        │
        ▼
KnowledgeManager.search()（跨全部知识文件聚合）
  每文件 KeywordPriorityRetriever:
    ① 向量检索（SqliteVectorStore, k=8）
    ② BM25（jieba 分词, k=8）
    ③ 本地关键词打分（倒排索引候选 ∪ 召回）
    ④ RRF 融合（只看排名，规避分数量纲） → top_k×2
    ⑤ gte-rerank-v2 精排（失败回退） → top_k
  全局：doc_id 去重 → 可选全局 rerank
        │
        ▼
document_to_citation(): chunk → Citation{source, location, snippet}
location 由 loader 生成：PDF=页码 / MD=标题+行号 / DOCX=章节路径 / CSV=行号
        │
        ▼
format_reference_block(): 编号参考资料注入 Prompt
  "[1] 《高血压诊疗要点.md》（二、分级标准 · 行 9-27）：……"
        │
        ▼
LLM 输出 citations: [{"ref": 1, "quote": "……"}]
        │
        ▼
map_llm_citations(): ref 编号 → 真实 Citation（非法编号丢弃）
  ⇒ LLM 只能"选择"证据，不能编造出处
        │
        ▼
落库：*_result 表 citations 列（JSON）→ 前端证据卡片
```

### 2.3 安全

- 密码 bcrypt 哈希；登录签发 JWT（HS256，默认 12h）
- 所有业务路由 `get_current_user` 依赖；用户管理 `require_admin`
- 上传文件名 `Path.name` 防路径穿越；大小 ≤100MB；后缀白名单

## 三、数据库表

| 表 | 说明 |
|---|---|
| sys_user | 用户（admin/doctor），bcrypt 哈希 |
| knowledge_file | 知识文件台账（collection=md5(文件名)） |
| inquiry_record | 问诊主状态机（0创建/1症状/2待化验/3诊断/4治疗/5完成/99失败），含患者信息、过敏史、thread_id、use_rag |
| symptom/diagnosis/treatment 三张结果表 | agent_result + citations + 医生复核字段（is_agreed/disagree_reason/doctor_result） |
| agent_session_his | Agent 调用审计（含 rag_enabled / retrieval_query） |
| workflow_event | 阶段流转事件（时间线） |

## 四、配置（backend/.env）

| 键 | 说明 |
|---|---|
| DATABASE_URL | dev 默认 SQLite；prod 用 MySQL |
| DASHSCOPE_API_KEY / DEEPSEEK_API_KEY / LLM_* | LLM 网关（优先级：显式 LLM_* > DashScope > DeepSeek） |
| RETRIEVAL_EMBEDDING_* / RETRIEVAL_RERANK_* | 检索侧模型（provider: dashscope/openai, disabled） |
| MEDICAL_RAG_ENABLED / RAG_TOP_K | 全局 RAG 开关与证据条数（问诊级 use_rag 优先） |
| RABBITMQ_ENABLED | 化验结果 MQ 异步回传（dev 关闭走 HTTP） |

## 五、关键设计取舍

1. **SqliteVectorStore 替代 Chroma**：chroma-hnswlib 原生 wheel 在 Win+Py3.12
   段错误（faulthandler 定位），自实现 SQLite+numpy 零原生依赖，接口隔离在
   `create_indexes` 便于日后替换
2. **大规模语料的三层性能设计**（3.7 万 chunk 实测 P50=0.71s）：
   - 内存向量矩阵缓存：首查把全量向量读入 float32 矩阵常驻，之后单次矩阵乘出
     全量余弦分（`VECTOR_MATRIX_CACHE=on|off` 可关，省内存回退流式）
   - 大库 keyword 路降级：chunk 数 > `keyword_index_max_docs`(3,000) 时跳过多粒度
     滑窗倒排（构建复杂度不可接受），keyword 路退化为查询短语包含匹配 + 召回集
     重打分——Huatuo 2.9 万 chunk 单文件检索从 282s 降到亚秒级
   - 文件间并发检索：`KnowledgeManager.search()` 用线程池并发跑各 collection
     检索器（矩阵懒加载有锁保护，加载后只读），配合 embed-once（查询向量只调
     一次 embedding API）消除 N 倍重复调用
   - **embedding 并行入库**：`_embed_batch` 按 25 条/批切分后线程池并发调
     DashScope（`EMBED_CONCURRENCY`，默认 4，夹紧 [1,8]），实测 1 万段落建索引
     约 2 分钟（串行约 15-20 分钟）；撞 429/Throttling 时未完成批次指数退避
     （4s→120s 封顶）且并发减半至 1，配额型限流以时间换吞吐
   - 教训记录：矩阵缓存的失效函数曾在持锁路径内再次拿锁，造成非重入锁自我
     死锁（卡在 rebuild_if_changed→clear→_invalidate_matrix）——修复为只清引用
     不加锁，锁调用点需审计嵌套
3. **问诊级 use_rag**：A/B 评估要求同批次病例对照，开关随 inquiry_record 落库
   而非仅全局配置
4. **Agent 公共基类**：三 Agent 的审计/检索/校验/落库骨架合一，子类只实现
   4 个钩子（检索查询/LLM调用/校验/持久化）
5. **Pydantic 校验前剥离 LLM 的 citations 声明**：LLM 返回的 `{ref, quote}`
   不是完整 Citation，校验后再由 map_llm_citations 映射
6. **患者历史按姓名近似关联**：演示用途的最简实现，生产应换患者主键
