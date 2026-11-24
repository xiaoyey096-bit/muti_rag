# 格物智答 · 多产品知识库 RAG 智能问答平台

> 一套代码承载多个产品知识库：上传产品手册 → 自动解析、切片、向量化入库 → 用户选择产品智能体提问 → 结合本地知识库与按需联网，给出带配图、可溯源的答案。

格物智答（muti_rag）是面向运营/售后场景的检索增强问答系统。产品之间数据与会话按 `kb_id` 完全隔离，同一套服务可并行承载任意多个产品知识库。

## 核心能力

- **多产品隔离**：三路检索均按 `kb_id + item_name` 过滤，产品间不串扰；入口一次切换产品智能体。
- **图片全链路**：MinerU 抽取文档图片 → 视觉模型生成摘要 → MinIO 持久化 → 摘要随切片入向量库；问答时图题分层门控、答案白名单回显，命中配图可走 VL 读图作答。
- **混合检索**：BGE-M3 dense + sparse 双路、HyDE 假设文档检索、RRF 融合、BGE-reranker 重排，按断崖动态裁剪 top_k，另有召回排名保底防止误杀。
- **查询理解**：一次模型调用输出结构化 `query_plan`（子问题拆分、联网需求、图片意图），每个子问题独立检索、独立重排、独立作答。
- **答案真实性**：证据门控逐子问判定 local / web / refusal；模型判无可依据时输出 `NO_EVIDENCE`，先合并兄弟子问证据、再按需联网，仍无证据才拒答；联网答案强制标注来源。
- **稳定性工程**：MongoDB 持久化导入任务、断点续跑与崩溃接管、分片续传与秒传、错误分级重试、依赖级熔断、embedding/图片摘要缓存、`/ops/metrics` 可观测指标、临时数据 TTL 清理。

## 系统架构

```
导入服务 :8002  /front/import.html
  上传 → entry → pdf_to_md(MinerU，大文件按页分片) → md_to_img(VLM+MinIO)
       → split → item_name → embedding(BGE-M3) → Milvus → suggest_questions

查询服务 :8003  /front/chat.html
  查询理解 → 本地检索(混合+HyDE) ∥ 联网(按需) → RRF → reranker
          → 证据门控 → 答案生成(含 VL 读图) → SSE 流式输出
```

导入与问答均为异步任务，前端可看到节点级进度；一次导入完成后产物（chunks、md、推荐问题）落盘，可从中断处继续。

## 技术栈

| 类别 | 选型 |
| --- | --- |
| 对话 / 查询理解 | 通义千问 `qwen-flash`（DashScope） |
| 视觉模型 | `qwen3-vl-plus`（导入期图片摘要 + 查询期读图作答） |
| 向量模型 | BGE-M3（本地，1024 维，dense + sparse） |
| 重排模型 | BGE-reranker-large（本地） |
| 向量库 / 文档库 / 对象存储 | Milvus、MongoDB、MinIO |
| 文档解析 | MinerU（pipeline 后端，支持按页分片） |
| 编排 / 服务 | LangGraph、FastAPI、SSE |
| 联网搜索 | DashScope WebSearch MCP |

## 目录结构

```
muti_rag/
├── api/                  # 导入 / 查询两个 FastAPI 应用与路由
├── service/              # 任务队列、执行器、上传服务、查询服务
├── processor/
│   ├── import_processor/ # 导入图谱与节点
│   └── query_processor/  # 查询图谱与节点
├── utils/                # 客户端、配置、意图识别、清理、自检等
├── prompts/              # 提示词
├── schema/               # 数据结构
├── core/                 # 路径等基础设施
├── front/                # import.html / chat.html
└── PRD.md                # 需求、参数与版本变更记录
```

## 快速开始

### 1. 环境

- Python 3.11
- 本地 embedding / rerank / MinerU 默认使用 CUDA（可在 `.env` 改为 CPU）

### 2. 安装依赖

```bash
pip install fastapi uvicorn langgraph langchain-core langchain-openai langchain-text-splitters \
  pymilvus pymongo minio FlagEmbedding torch transformers httpx requests pypdf python-dotenv \
  pydantic beautifulsoup4 pillow

# 仅 PDF 导入需要 MinerU（Markdown 导入可跳过）
pip install mineru
```

### 3. 准备基础设施

需要可用的 Milvus、MongoDB、MinIO 实例，地址在 `.env` 中配置（`.env.example` 给出了完整字段与示例值）。

### 4. 配置

```bash
cp .env.example .env
# 填入 LLM / VLM 密钥、Milvus / MongoDB / MinIO 地址、本地模型路径
```

### 5. 启动（在包含 `muti_rag` 包的父目录下执行）

```bash
python -m muti_rag.api.import_router   # 导入服务，默认端口 8002（IMPORT_PORT）
python -m muti_rag.api.query_router    # 查询服务，默认端口 8003（QUERY_PORT）
```

### 6. 访问页面

- 导入页：`http://localhost:8002/front/import.html`
- 问答页：`http://localhost:8003/front/chat.html`

首次使用先在导入页新建产品知识库并上传 PDF/Markdown，再到问答页选择该产品智能体提问。

## 关键设计

- **检索**：每子问题独立执行 dense+sparse 混合检索（WeightedRanker 0.5/0.5）与 HyDE（词汇覆盖率/空召回自适应触发），RRF 仅融合本地两路（k=60，上限 10）；联网结果在重排阶段并入。
- **重排与截断**：BGE-reranker 批量打分，断崖阈值 gap ≥ 0.15 动态裁剪；本地切片优先保底 `min_top_k=3`、上限 `max_top_k=10`，RRF 前 4 名本地切片强制保留。
- **图片链路**：仅保留 sigmoid ≥ 0.5 的含图切片（无保底），再断崖裁剪；按（子问题，图片描述）逐图门控，强相关最多回显 2 张；答案层以「重排前 3 篇文档图片 ∪ 门控通过图」为白名单过滤，越界图片一律剔除。
- **证据门控**：无本地证据且允许联网时才补检索，避免无谓的外部请求；图片意图无图直接走"暂无配图"模板，不编造。
- **大文档**：超过阈值的 PDF 按页分片交给 MinerU 串行解析，199 页 / 191MB 扫描件实测零 OOM；嵌入结果与图片摘要均有磁盘缓存，续跑/重导不重复计算。

## 效果（同机对照测评）

- 50 题问答测评：完全正确率 **90% → 94%**，8 道知识库外问题全部正确拒答（零编造）。
- 单次查询模型往返 **3.37 → 2.28 次/题**，平均响应 **5.6s → 4.2s**，P95 **14.7s → 7.9s**。
- 199 页 / 191MB 扫描件导入：4 片串行解析，零 OOM，257 个切片正常入库。

## 文档

- [PRD.md](./PRD.md)：完整功能列表、真实参数配置、已知限制与 v1.0 → v4.4 变更记录。

## 安全说明

`.env` 含真实密钥，已被 `.gitignore` 排除，仓库只提交 `.env.example` 占位模板；上传产生的临时数据（`temp_data/`）与运行日志同样不入库。

---

> 背景：项目源自尚硅谷大模型课程实践（掌柜智库），后重构为多产品通用的统一 RAG 包。