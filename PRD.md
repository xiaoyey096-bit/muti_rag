# 格物智答 muti_rag 产品需求文档（PRD）

> 版本：v4.0（2026-11-17；v3.0 / v2.0 / v1.0 同日早前）
> 适用代码：`muti_rag/`（统一多产品 RAG 包，取代旧 `dgx/`、`knowledge/` 双副本）
> 说明：本文所有参数均取自当前代码与 `.env` 实际配置，仅描述能力与限制，不含任何密钥/口令。

---

## 1. 产品概述

格物智答（原名掌柜智库）是一个**多产品、知识库隔离**的检索增强问答（RAG）系统。运营/售后人员把产品手册（PDF/Markdown）导入系统，系统自动解析、切片、向量化并入库；终端用户在前端选择某个产品智能体后提问，系统结合**本地知识库 + 联网搜索**给出带参考图片的答案。

核心价值：
- 一套代码支撑任意多个产品，产品之间数据与会话完全隔离（`kb_id`）。
- 文档中的图片被识别、描述、持久化，并能在问答中**回显原图**。
- 导入与问答均为异步任务，前端可见**节点级进度**。

### 1.1 角色
| 角色 | 诉求 |
| --- | --- |
| 知识管理员 | 上传/追加文档、新建/改名/删除产品知识库、查看导入进度 |
| 终端用户 | 选择产品智能体、提问、查看带图答案与处理过程、管理会话历史 |

### 1.2 两个服务与两个页面
| 服务 | 端口（环境变量） | 页面 | 职责 |
| --- | --- | --- | --- |
| 导入服务 | `IMPORT_PORT`（默认 8002） | `/front/import.html` | 上传、知识库管理、导入进度 |
| 查询服务 | `QUERY_PORT`（默认 8003） | `/front/chat.html` | 产品选择、问答、流式输出、历史 |

---

## 2. 支持的文件与内容

### 2.1 上传文件格式（硬性白名单）
| 格式 | 后缀 | 处理路径 |
| --- | --- | --- |
| PDF | `.pdf` | MinerU 解析为 Markdown（含图片/表格/公式）→ 后续切片 |
| Markdown | `.md` | 直接进入图片处理与切片（跳过 PDF 解析） |

- 其它后缀在 `entry_node` 直接报错拒绝（`ValidationError`）。
- 前端上传控件 `accept=".pdf,.md"`，与后端白名单一致。

### 2.2 文件大小上限
- **代码层未设置硬性大小上限**：上传接口与保存逻辑不做 max-size 校验。
- 实际可处理规模受 **MinerU 解析的显存/内存** 与 **宿主 Windows 提交额度（commit charge）** 约束；超大 PDF 可能解析失败或 OOM。
- 运维建议：大文档**串行导入**（并发会抢 GPU 致崩溃）；导入前确保宿主内存/提交额度充足。

### 2.3 图片支持
- 支持图片格式（`ImportConfig.image_extensions`）：`.jpg .jpeg .png .gif .bmp .webp`。
- 图片处理链路（`md_to_img_node`）：
  1. 从 Markdown 中抽取图片；
  2. 调用**视觉大模型**为每张图生成文字描述（base64 传入）；
  3. 原图上传 **MinIO** 持久化；
  4. 用 MinIO 公网可读 URL 替换 Markdown 中的图片引用；
  5. 图片描述 + 上下文进入切片，供检索与答案回显。
- 图片上下文长度上限 `img_content_length = 200` 字。
- 图片总结 API 速率限制 `requests_per_minute = 15`。
- 问答回显（v3）：答案统一 Markdown 输出，图片以 `![描述](url)` 内嵌；代码层按「重排前 3 篇文档的图片 + 图片级相关性门控」白名单过滤后由前端安全渲染 `<img>`。
- MinIO 桶策略：图片对象匿名可读、`origin_files/` 原始文件私有（显式 Deny）。

### 2.4 表格与公式
- 表格：MinerU 输出 Markdown 表格，随切片入库。
- 公式：MinerU `method=auto` 默认开启公式识别；可用 `--formula false` 跳过以省显存。

---

## 3. 功能清单

### 3.1 知识库管理（导入服务）
| 功能 | 接口 | 说明 |
| --- | --- | --- |
| 列表 | `GET /kbs` | 返回全部知识库（含文档数、产品名数、展示名、提示） |
| 新建 | `POST /kbs` | 先建产品身份（`product_name`/`display_name`/`hint`），再上传文档 |
| 改名/改提示 | `PATCH /kbs/{kb_id}` | 更新展示名/提示/产品名 |
| 删除 | `DELETE /kbs/{kb_id}` | 清理 Milvus 该 kb 的 chunks 与商品名记录，再删注册表 |

### 3.2 文档导入（导入服务）
| 功能 | 接口 | 说明 |
| --- | --- | --- |
| 上传 | `POST /upload` | `kb_id` 追加 / `product_name` 新建 / 缺省用文件名兜底（provisional） |
| 进度 | `GET /status/{task_id}` | 节点级 done/running/耗时；前端 1.5s 轮询 |

导入图谱节点（PDF 全链路，共 9 步含上传）：`upload_file → entry_node → pdf_to_md_node → md_to_img_node → document_split_node → item_name_recognition_node → embedding_chunks_node → import_milvus_node → suggest_questions_node`（MD 跳过 `pdf_to_md_node`，共 8 步）。`suggest_questions_node` 由 LLM 生成"用户常问 3 问"写入注册表，失败不影响导入主流程。

### 3.3 问答（查询服务）
| 功能 | 接口 | 说明 |
| --- | --- | --- |
| 产品列表 | `GET /kbs` | 供入口筛选选择 |
| 提问 | `POST /query` | `is_stream=true` 返回 `task_id` 走 SSE；否则同步返回答案 |
| 流式 | `GET /stream/{task_id}` | SSE：`progress` / `delta` / `final`/`final_answer` / `error` |
| 历史 | `GET /history/{session_id}` | 默认 `limit=50` |
| 清空历史 | `DELETE /history/{session_id}` | 删除该会话历史 |

查询图谱（v3）：`item_name_confirmed_node →（无澄清短路时）query_split_node →（并行）hybrid_vector_search_node / hyde_vector_search_node / web_mcp_search_node → rrf_merge_node → reranker_node → answer_output_node`。多问题输入先拆分为独立子问题，**每个子问题独立检索、独立 RRF、独立重排 top_k、独立生成答案**，最后按 `## 子问题` 分节拼装。

### 3.4 前端能力
- 导入页：下拉选择导入目标 / 新建产品、拖拽或点击上传、进度条与运行日志（面向用户的步骤文案）、知识库列表管理（搜索过滤 + 删除）、**下拉筛选框进入问答**。
- 问答页：**下拉筛选框选择产品智能体**（入口 + 顶栏常驻切换）、动态品牌化（标题/头像/欢迎语）、**欢迎语下推荐问题按钮（点击即问）**、流式开关、**轻量思考动画**（v4.1，见 §9.4）、**Markdown 安全渲染**（标题/列表/表格/代码/图片/链接）、时间标注 `年.月.日 时:分`、会话历史加载与清空、Enter 发送 / Shift+Enter 换行。

---

## 4. 关键参数与上限（真实配置）

### 4.1 导入 / 切片（`ImportConfig`）
| 参数 | 值 | 含义 |
| --- | --- | --- |
| `max_content_length` | 1000 | 单个切片最大字数 |
| `min_content_length` | 500 | 合并短内容的最小字数 |
| `overlap_sentences` | 1 | 句子级切分重叠句数 |
| `img_content_length` | 200 | 图片上下文最大字数 |
| `item_name_chunk_k` | 3 | 商品名识别使用切片数 |
| `item_name_chunk_size` | 2500 | 商品名识别切片内容长度 |
| `embedding_dim` | 1024 | 向量维度（BGE-M3） |
| `embedding_batch_size` | 8 | 向量化批大小 |
| `requests_per_minute` | 15 | 图片总结 API 速率限制 |

### 4.2 检索 / 融合 / 重排 / 生成（`QueryConfig`）
| 参数 | 值 | 含义 |
| --- | --- | --- |
| `embedding_search_limit` | 5 | 混合向量检索每路 top |
| `hyde_search_limit` | 5 | HyDE 检索 top |
| 联网搜索 `count` | 3 | WebSearch MCP 返回网页数 |
| `rrf_k` | 60 | RRF 平滑参数 |
| `rrf_max_results` | 10 | RRF 融合保留上限 |
| `rerank_min_top_k` | 3 | 重排保底保留数（本地优先，仅普通问答） |
| `rerank_max_top_k` | 10 | 重排保留上限 |
| 断崖阈值 | gap ≥ 0.15 | 动态 top_k 断崖裁剪 |
| 图片分数下限 | 0.5 | 图片意图重排：sigmoid ≥ 0.5 才保留（**无保底**，避免无关图混入） |
| 图片断崖阈值 | gap ≥ 0.15 | 图片意图重排的断崖裁剪 |
| 图片级门控 | 0.5 | 对每张图按 (子问题, 图片描述/标题) 重排打分，≥0.5 才进 `matched_images` |
| 单子问题配图上限 | 4 | 单个子问题最多回显 4 张图 |
| 子问题上限 | 4 | 多问题拆分最多 4 个独立子问题 |
| `max_context_chars` | 12000 | 送入 LLM 的上下文字数上限（每子问题独立） |
| 图片回显白名单 | 前 3 篇 + 门控 | 重排前 3 文档图片 ∪ 图片级门控通过图 |
| 历史 `limit` | 50 | 单会话历史返回条数 |
| 进度轮询间隔 | 1.5s | 导入页轮询 |

### 4.3 模型与存储（`.env`，不含密钥）
| 项 | 值 |
| --- | --- |
| 对话/商品名 LLM | 通义千问 `qwen-flash` 系列 |
| 视觉模型（导入期图片描述 + 查询期识图作答） | `qwen3-vl-plus` |
| 向量模型 | BGE-M3（本地，1024 维，dense+sparse 混合） |
| 重排模型 | BGE-reranker-large（本地） |
| 联网搜索 | DashScope WebSearch MCP |
| 向量库 | Milvus：`rag_chunks_v1`、`rag_item_names_v1` |
| 注册表/历史 | MongoDB 库 `rag_kb` |
| 对象存储 | MinIO 桶 `rag-knowledge-base-v1` |
| MinerU | backend=pipeline，device=cuda:0，method=auto |

### 4.4 检索隔离与排序策略
- 三路检索均按 `kb_id == {kb_id} and item_name in {item_names}` 过滤，产品间不串扰。
- 混合检索 = dense + sparse，WeightedRanker(0.5, 0.5)。
- RRF 仅融合本地两路（embedding + HyDE）；联网结果在重排阶段并入。
- **每子问题独立**：拆分后的每个子问题各自向量化、各自三路检索、各自 RRF、各自重排 top_k，互不共用候选。
- 普通问答重排「本地优先」：前 `min_top_k` 个本地切片保底入选排前，其余名额按分数补（防止联网长文挤掉本地配图）。
- 图片意图重排**不设保底**：仅保留分数 ≥0.5 的含图本地切片，再按断崖裁剪，最后逐图门控——修复"问一张图却回两张无关图"。
- 联网搜索词自动拼上产品名（`item_names`），避免短查询搜回无关内容。

---

## 5. 非功能与运维约束
- **并发导入**：两份 PDF 同时导入会抢 GPU 致 MinerU 崩溃，应串行。
- **宿主内存**：MinerU 崩溃多因 Windows 提交额度耗尽；主页面文件已移 E 盘、C 盘留小份以缓解。
- **MinIO 时钟**：VM 用 chrony 对时（`makestep 1.0 -1`），避免 RequestTimeTooSkewed。
- **文件名乱码**：Windows curl 上传中文名按 GBK 还原（`_normalize_filename`）。
- **安全**：`.env` 含真实密钥，已被 `.gitignore` 排除，禁止提交公开仓库；MinIO 原始文件私有、仅图片公开读。

---

## 6. 已知限制（v1.0）
- 上传仅 PDF/MD，不支持 docx/txt/图片直传。
- 无上传大小硬校验，超大文档依赖解析资源，可能失败。
- 图片回显受「前 3 文档白名单 + 图片级门控」限制，极靠后或描述不相关的配图不会回显。
- 联网搜索依赖 DashScope MCP 可用性与配额。
- 会话历史按 `session_id + kb_id` 隔离，跨设备不漫游（session 存于浏览器 localStorage）。
- **问答为纯文本输入，不支持用户上传图片提问**（用户图识答为可选后续项，未实施）；但命中知识库配图的问题会走查询期 VL 读图通道（v4，见 §9），基于真实图片像素作答。

---

## 7. v2.0 变更与回答策略

### 7.1 品牌与界面
- 项目更名**格物智答**（入口页、顶栏、标题、导入页同步）；产品智能体名仍按知识库动态显示。
- 机器人头像改为统一生成的机器人图标（`front/assets/robot-avatar.png`），不再用截断的产品名文字做头像。
- 答案正文**不再显示图片 URL / markdown 图片 / 裸链接等泄露内容**；图片仍以 `<img>` 回显；原「阶段进度」面板（运行状态 + 节点路径）已于 v4.1 移除，改为轻量思考动画（§9.4）。
- 流式输出期间**允许自由上下滑动**：仅当用户停留在底部附近才自动跟随；上滑后出现「回到最新」悬浮按钮。
- 输入框旁新增**联网搜索筛选开关**（开/关），透传 `enable_web` 到查询链路。

### 7.2 联网开关链路
`QueryRequest.enable_web`(默认 true) → `query_router` → `QueryService.run_query_graph` → `state.enable_web` → `web_mcp_search_node` 开头判断：关闭则直接跳过联网分支（返回 `{}`），其余两路本地检索不受影响。

### 7.3 图片意图回答策略（修复"答非所问"；v3 已升级为 §8.2 的图片门控策略）
- 意图识别 `utils/query_intent_util.is_image_query`：匹配"有/有没有/展示/给我…图/图片/照片/示意图/接线图…"或"…图吗/图片有哪些"等存在性/展示类问法（对任意产品通用）。
- 命中图片意图时：
  1. `web_mcp_search_node` 跳过联网（联网综述只会灌产品百科废话）；
  2. `answer_output_node._prioritize_image_docs` 重排 `reranked_docs`：本地含图切片优先、本地无图切片兜底、剔除联网文档（本地以 `chunk_id` 识别），使上下文与配图白名单同序；
  3. 改用 `IMAGE_ANSWER_PROMPT`：第一句直答有无图、禁止百科铺垫、全文≤3 句、有主题一致配图才追加【图片】区块。
- 未命中图片意图的普通技术问答仍走原 `ANSWER_PROMPT` 与三路检索，行为不变。

### 7.4 当前查询算法（总览）
`item_name_confirmed_node`（商品名确认/澄清短路）→ 并行三路 `hybrid_vector_search`(dense+sparse WeightedRanker 0.5/0.5) / `hyde_vector_search` / `web_mcp_search`(count=3，可被开关/图片意图跳过) → `rrf_merge_node`(RRF k=60、max 10、仅本地两路) → `reranker_node`(BGE-reranker-large sigmoid、断崖 gap≥0.15、本地优先 min3/max10) → `answer_output_node`(上下文≤12000 字、配图白名单=重排前 3 切片、流式 SSE / 非流式队列)。

---

## 8. v3.0 变更（输出算法与输出结构优化）

### 8.1 多问题拆分 + 每问独立 top_k
- 新增 `query_split_node`（`utils/query_split_util.split_queries`）：
  - 单问题快速通道：无多问提示词（？/；/换行/另外/还有/以及/同时…）直接 `[原问题]`，零 LLM 开销；
  - 命中提示则 LLM 按 `QUERY_SPLIT_PROMPT` 输出 `{"questions":[...]}`（≤4 个、自包含、不拆句内并列），失败降级规则切分。
- 状态新增 `sub_queries` 与 `*_per_q` 系列字段；hybrid / hyde / web / rrf / reranker 全部按子问题循环，**每个子问题独立检索与独立 top_k**（旧标量字段保留为 per_q[0] 供调试）。
- `answer_output_node` 逐子问题生成，多问时以 `## 子问题` 分节、`---` 分隔拼装；单问不加节标题。

### 8.2 图片意图：分数下限 + 断崖 + 逐图门控（修复"一图答两图"）
- 图片意图时候选只取**本地含图切片**；`_image_cutoff`：sigmoid ≥ 0.5 才保留，再按 gap≥0.15 断崖裁剪，**取消本地保底 min_top_k**（保底正是无关图混入的根因）。
- `_gate_images`：对每张候选图按 (子问题, 图片描述/标题) 批量重排打分，≥0.5 写入 `doc['matched_images']`。
- 图片子问题的答案**优先 VL 实时读图**（v4，见 §9）；VL 失败/无图时回退代码直答"有，知识库中包含以下相关配图："+ `![描述](url)`（无相关图则直答暂无），真实且零幻觉。

### 8.3 输出标准化 Markdown
- `ANSWER_PROMPT` 强制 md：结论句先行 + `##` 分节/列表/加粗/表格/代码；图片仅允许主题一致的 `![描述](url)` 原样引用；单资源回答用一级标题分层（如 `# 相关配图`）。
- `answer_output_node._sanitize_images`：按白名单（前 3 文档图片 ∪ matched_images）过滤 md 图片/裸 URL/旧【图片】块，越界图片一律剔除。
- 前端 chat.html 内置**安全 md 渲染器**（先 escapeHtml 再拼标签，仅 http/https 链接）：标题/有序无序列表/表格/代码块/引用/分隔线/图片(带描述)/链接；流式期间增量重渲染。

### 8.4 时间标注
- 会话气泡与日期分隔线统一 `年.月.日 时:分`（如 2026.11.17 20:45），日期线显示当天 `年.月.日`。

### 8.5 导入期推荐问题 + 入口快捷提问
- 导入图谱末尾新增 `suggest_questions_node`：LLM 按 `SUGGEST_QUESTIONS_*` 提示词从文档内容预测**最多 3 个**用户常问问题（覆盖操作/规格/注意事项/配图不同主题、自包含、≤25 字），写入注册表 `suggested_questions`；失败仅记日志不阻断导入。
- **真实性第一，宁可无也不可错**：每个问题必须能被文档明确回答；提示词注入【文档实际配图】清单（从切片图片引用收集图题），配图类问题只能针对清单中真实存在的图，严禁把示意图/面板图当产品外观、严禁生成文档没有的"外观图/实物照片"类问题；配图清单只约束配图类问题，操作/规格类依据正文；有依据时尽量给满 3 个，不足可少返回甚至空数组。
- `GET /kbs` 返回 `suggested_questions`；问答页欢迎语下渲染为可点击按钮，点击即发送该问题。**原静态提示行（kb.hint"提示：可以问：…"）已从问答页移除**，避免与推荐问按钮重叠。
- 存量 3 个知识库（DGX Spark / RS-12 / HUAWEI MateStation S）已用导入缓存切片按上述规则回填。

### 8.6 前端文案用户化
- 导入页/问答页去除开发视角话术（"界面不会随产品增多而膨胀"、"PDF 经 MinerU"、"1.5s 轮询"等），改为面向用户的功能描述；导入日志改为"×× 已完成 / 正在处理：××…"。
---

## 9. v4.0 变更：查询期识图 + 答案真实性硬约束

### 9.1 查询期 VL 识图通道（方案 A）
- 新增 `utils/vl_answer_util.answer_with_images`：把命中配图的**真实像素** + 用户问题 + 文档上下文送 `qwen3-vl-plus`（`VL_MODEL`）读图作答，返回 Markdown。
- **内网图片转 base64**：MinIO 地址是内网 URL，DashScope 服务端无法回源抓取，故由本机 httpx 下载图片字节 → base64 data URL 再传模型（单次最多 4 张，下载失败跳过该图）。
- 触发条件（两条路径）：
  1. 图片意图（`is_image_query`，如"有没有后面板接口图"）：`_vl_image_body` 读图作答 + 附原图；VL 失败/无图回退 v3 代码直答。
  2. 普通问句提及图片（新增 `mentions_image`，如"解释布局图里每个接口的作用"）：reranker 在普通分支也跑 `_gate_images`，答案节点见 `matched_images` 非空即走 VL（图片 + 完整检索上下文），失败回退普通文本 LLM。
- VL 提示词（`VL_ANSWER_SYSTEM_PROMPT`）：只描述图中实际可见内容 + 文档上下文；看不清/没有的明说"图中无法确认"，禁止凭常识编造型号参数；无法回答直说暂无；md 输出、≤400 字、不输出 URL（配图另行展示）。
- 实测：DGX"有没有后面板接口图"→ VL 列出 ON/OFF/Power/4×USB-C/HDMI/RJ-45(10GbE)/2×QSFP 真实接口 + 附图；"解释每个接口作用"→ 读图 + 文档补充，并诚实标注图中无法确认的细节。

### 9.2 答案真实性硬约束（不知道就说不知道）
- `ANSWER_PROMPT` 规则 3 升级为硬约束：答案必须完全来自【参考内容】；即使模型自身知识知道，参考内容没有就必须说"抱歉，知识库未涵盖该信息"，不猜测、不编造、不用"一般来说/通常"引入外部知识；参考内容完全无关时直接回答不知道。
- 联网关闭时参考内容仅本地文档，约束自然生效；联网开启时联网结果也属参考内容。
- 实测：web off 问"DGX Spark 的 CPU 主频"（文档无）→"抱歉，知识库未涵盖该信息。缺少关于 DGX Spark CPU 主频的具体参数说明。"；文档内问题（接电自动启动）正常作答不受影响。

### 9.3 未实施（可选后续）
- 方案 B：用户上传图片问答（前端上传入口 + 后端接收通道 + 同一 VL 调用），模型侧已具备能力，待需求确认。

### 9.4 轻量思考动画（替代阶段进度面板）
- 移除 `details.progress` 面板（"阶段进度（已完成N，进行中N）"+节点名列表），改为单行 `.thinking`：三个跳动圆点 + 一句呼吸动画友好文案。
- 文案按后端进度事件映射（`renderProgress`）：确认问题产品/问题拆分→"正在理解你的问题…"；切片搜索/网络搜索→"正在查阅产品文档…"；倒排融合/重排序→"正在筛选最相关的内容…"；生成答案→"正在整理答案…"；兜底"正在思考…"。不出现任何技术节点名/计数。
- 首个答案 delta / FINAL / 错误 / 完成时自动移除思考行；流式与轮询两条路径统一。
- 实测（CDP think_check.mjs，端口9339）：等待期文案随阶段推进、无技术词泄露；答案出现即消失；识图答案 md+配图正常、0 console error。

### 9.5 大 PDF 分片解析（v4.2，修复大扫描件导入 OOM）
- 问题：MinerU 的 pipeline 后端会把**整份 PDF 的全部页面一次性渲染进内存**（200dpi），再叠加 OCR 中间结果；199 页/191MB 扫描件在 16GB 主机上于 OCR 96% 处提交内存耗尽（`numpy ArrayMemoryError`，连 1.41MB 都分配失败），同机虚拟机也因资源紧张出现 vmdk 操作失败弹窗。
- 方案：`pdf_to_md_node` 内置分片解析——pypdf 读取总页数，超过 `MINERU_CHUNK_THRESHOLD_PAGES`（默认 80 页）时按 `MINERU_CHUNK_PAGES`（默认 60 页）切片，每片带 `-s/-e` 页码参数单独调用 MinerU；各片 md 拼接、`images` 合并到标准产物路径，`md_to_img` 及下游节点无感。
- 内存防护：同一进程内 MinerU 子进程串行执行（`_MINERU_LOCK`），避免多个导入任务并发解析叠加内存峰值。
- 附带优化：章节号恢复仅在 md 含空字节（`\x00`）时才抽取 PDF 标题，扫描件省去一次全本 pypdf 遍历。
- 实测：《广东技术师范大学学生手册（2024年版）》199 页 → 4 片（每片 1–2 分钟），零 OOM；合并 md 324KB、225 个标题；导入 257 个切片，问答"学生转学公示几天"→"在学校网站公示5个工作日"，内容来自正文。

### 9.6 导入任务断点续跑（v4.3 P0）
- 任务状态从进程内存迁移到 MongoDB `import_tasks`：`status/done_nodes/running_nodes/durations/error/resumable/resume_count/artifacts` 等字段持久化；Mongo 不可用时自动降级纯内存，不阻断导入。
- 上传完成即建任务文档（含 `file_sha256`、`file_dir`、`import_file_path`）；每个节点结束后把 `chunks`（含向量）写入 `<file_dir>/import_chunks.json`（路径记入 `artifacts`），md 路径/商品名/推荐问写任务文档。
- 服务启动自检：遗留的 `queued/processing` 任务统一标记为 `failed + resumable`（前端提示"服务重启导致导入中断，可继续导入"）；同时校验并兜底提升 pypdf 流长度上限（MinerU 子进程由 `sitecustomize.py` 承担）。
- 新接口 `POST /tasks/{task_id}/resume`：重建 state（md 优先取 `_new.md`，chunks 优先取落盘文件），从第一个未完成且产物缺失的节点继续执行；任务不存在 404、运行中 409、原始文件丢失 410。
- 幂等：Milvus 切片新增 `file_sha256` 动态字段，入库前按 `kb_id + (file_title 或 file_sha256)` 清理旧切片；插入前剔除自增主键 `chunk_id`（续跑时旧 id 残留会导致字段数不匹配）。
- 前端（import.html）：失败时展示后端错误原因；`resumable=true` 显示"继续导入"按钮，点击调用续跑接口并恢复轮询。
- 实测：小文档导入完成 → 人为回退到"向量生成前失败" → 续跑后任务完成、`resume_count` 递增、Milvus 切片数不变（1 条，带 sha256），无重复入库；运行中续跑返回 409；重启后中断任务被正确标记为可续跑。

### 9.7 分片断点续传（v4.4 P1）
- 问题：191MB 文件走单请求 `POST /upload`，网络抖动/超时即整份重传；重复上传同一文件重复解析。
- 上传协议：`POST /uploads/init` → `PUT /uploads/{upload_id}/parts/{n}`（5MB/片，原始二进制）→ `POST /uploads/{upload_id}/complete`（合并 + sha256 校验）→ `POST /files/{file_id}/tasks` 建导入任务；`POST /uploads/{upload_id}/abort` 可放弃。会话与分片写入 Mongo `upload_sessions` + `temp_data/_uploads/<upload_id>`，会话 24h TTL。
- 秒传：上传前先算 sha256，`init` 命中 `files` 中完整文件直接返回 `completed + file_id`，不传字节；同 sha 的未完成会话返回已收分片列表，只补传缺失部分；同文件 + 同知识库 + 同标题的导入任务自动去重。
- 完整性：`complete` 缺片返回可操作的 `missing_parts`，前端自动补传（最多 3 轮）；sha256 不一致返回 `sha256_mismatch` 并让前端重传，防止拼接错位。
- 前端（import.html）：浏览器内增量 SHA-256（8MB 切片，不整文件读入内存）；3 并发分片上传，单片失败按 1/2/4s + 抖动重试（各 3 次）；上传进度 0–10%，之后由导入节点进度接管；`localStorage` 记录未完成的上传/任务，刷新后自动恢复——未完成上传显示"待续传 + 选择文件继续上传"（复用原会话与已传分片），导入中任务刷新后继续轮询。
- 状态渲染：`queued`→"排队中"、`waiting_retry`→"等待重试（步骤/资源紧张/倒计时/尝试次数）"、`processing`→"处理中"、任务记录被清理→"任务已过期"并提示重传。
- 实测（无头 Chrome 驱动真实页面）：首传走分片通道（1 片 PUT + 合并 COMPLETED）；同文件第二次上传走秒传且任务去重；预置"已传 1/1 片未合并"会话后刷新页面出现"待续传"，选择原文件后复用同一 `upload_id`、跳过分片、合并成功；导入中刷新后卡片恢复并轮询到"已完成"，全程无重复上传、`localStorage` 收尾清空、无前端异常。

### 9.8 依赖级熔断与任务队列（v4.4 P2）
- 错误分类：异常统一归为 `TRANSIENT`（网络/超时/5xx）、`RESOURCE`（显存/内存/限流/磁盘）、`PERMANENT`（参数/格式/不存在）三类；`PERMANENT` 立即终止，另两类按步骤配额重试（pdf_to_md 3、md_to_img 3、split 1、item_name 3、embedding 4、milvus 3、suggest 2，`RESOURCE` 额外 +1）。
- 退避与止损：退避 `5·2ⁿ + 抖动`，上限 5 分钟；任务总尝试 ≤15 次，连续 3 次失败且无新增完成节点 → 终态失败（`resumable` 视产物而定）。
- 熔断器（按依赖：dashscope/minio/bge/milvus）：60s 滑动窗口，失败率 ≥50% 且样本 ≥10（或连续 5 次失败）→ 打开 60s，翻倍至上限 600s；半开只放 1 个探测请求；熔断期间步骤回 `waiting_retry`，不消耗页面重试次数。
- 任务队列：Mongo 抢占式队列（`queued/waiting_retry` 可领取），2 个 worker 消费；心跳 10s / 租约 60s，进程崩溃后任务被重新领取并从断点继续；同一进程内 GPU 解析（MinerU/VLM）串行，embedding 与 Milvus 写入并发上限 2。
- 缓存：embedding 结果按内容哈希落盘 `<file_dir>/embedding_cache.json`（续跑/重导不重复计算）；图片摘要按"图片哈希 + 文档/上下文哈希"写入全局 `cache/image_summary_cache.json`，跨文档复用、相同图片不再调用 VLM。
- 可观测：`GET /ops/metrics` 汇总队列深度/活跃数、近 24h 各步骤成功率与平均耗时、熔断器状态、上传会话数、磁盘余量；日志为 JSON 结构化输出。
- 临时数据 TTL：janitor 线程每小时清理过期/中止/完成的上传会话与孤儿分片目录、超过保留期（默认 7 天）的终态任务目录；也可 `python -m muti_rag.utils.cleanup_util` 手动执行。
- 实测：小文档端到端导入 7 步全部成功（embedding 步骤输出"嵌入缓存：命中 0，新计算 1"），`/ops/metrics` 四个熔断器 `closed`、队列空闲、步骤耗时齐全；图片摘要缓存命中时不调用 VLM；janitor 正确清理过期会话、孤儿目录与过期任务目录。
