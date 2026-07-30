# MIRIX: Multi-Agent Memory System for LLM-Based Agents（arXiv 2025，开源）

- 论文：https://arxiv.org/abs/2507.07957
- 本地论文全文：`research/MIRIX/MIRIX_Multi-Agent_Memory_System_for_LLM-Based_Agents.md`
- 仓库：https://github.com/Mirix-AI/MIRIX
- commit：905984e（本文所有代码行号仅在该 commit 下有效）
- stars：3547（2026-06-11 查询）
- 本文读者假设：没完整读过论文，也没读过 MIRIX 源码

## 0. 先给我一个结论

MIRIX 想解决的问题是：现有 agent 记忆系统大多把所有东西塞进一个扁平向量库，适合存几条文本事实，但不适合长期、多模态、真实个人助手场景。它的核心办法是把记忆拆成六类：Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault，再用一个 Meta Memory Manager 决定新输入该交给哪些记忆子代理处理。

如果只记住一个点：**MIRIX 的卖点不是某个新检索算法，而是“记忆类型分工 + 多 agent 路由 + 主动检索进 system prompt”这一整套工程架构**。实验上，它在 ScreenshotVQA 中用很小的结构化数据库替代海量原图存储，在 LOCOMO 中拿到 85.38 的 LLM-as-Judge 总分，接近 Full-Context 的 87.52。但代码里也有明显落差：Reflexion/Background 基本没接上，默认主链路所谓 BM25 其实是 PostgreSQL 全文检索，主动检索里的 embedding 分支默认不可达。

## 1. 论文故事：作者为什么要做这件事

### 1.1 背景

LLM agent 如果没有长期记忆，每次对话都像重新认识用户。用户上周说过的偏好、昨天看过的网页、常用流程、通讯录、账号信息，都要重新提供。现有记忆系统已经有一些做法，比如：

- Mem0 把对话抽成事实，存进向量库。
- Letta/MemGPT 把上下文窗口看成主存，把外部数据库看成外存。
- Zep/Graphiti 用时序知识图谱保存实体和 fact。

MIRIX 认为这些方法都有一个共同问题：**记忆结构太扁平**。如果所有内容都变成一堆文本 chunk 或一堆 fact，系统很难区分“用户身份信息”“某天发生的事件”“一套操作流程”“一份文档”“必须逐字保存的敏感信息”。真实个人助手面对的输入也不只是聊天文本，还包括屏幕截图、文件、网页、应用界面和长期活动轨迹。

### 1.2 问题定义

MIRIX 要做的是一个“长期个人记忆系统”：持续接收用户对话和屏幕活动，把不同类型的信息分别写进不同记忆模块；用户提问时，系统自动检索相关记忆，并把检索结果注入模型上下文，让回答基于用户过去的真实经历，而不是只靠模型参数知识。

它和普通 RAG 的区别在于：

- 普通 RAG 常常是“文档切块 → 向量检索 → 拼进 prompt”。
- MIRIX 是“先判断这段信息属于什么记忆类型 → 由专门的记忆代理更新结构化字段 → 回答时跨六类记忆主动检索 → 按来源标签拼进 system prompt”。

举个简单例子：用户截图里出现一份报销流程文档。普通 RAG 可能只存一个文档 chunk；MIRIX 理想上会把“这个文档本身”放到 Resource Memory，把“如何报销”的步骤放到 Procedural Memory，把“用户最近在处理报销”放到 Episodic Memory。

### 1.3 核心挑战

- **信息类型不同**：事件、事实、流程、文档、敏感信息不是同一种东西，用同一个表存会让检索和更新都变粗糙。
- **多模态输入太大**：论文里的 ScreenshotVQA 每个用户有 5000 到 20000 张 2K-4K 截图，直接塞进长上下文或保存原图都不现实。
- **模型不一定主动查记忆**：如果用户问“Twitter CEO 是谁”，模型可能直接凭旧参数回答，而不是去查用户之前告诉它的新信息。
- **写入成本高**：六类记忆都可能要 LLM 判断和工具调用，写入质量变好，但每次更新会很重。
- **维护机制难做**：长期运行后会出现重复、冲突、过期、无用记忆。论文说有避免冗余和 rewrite，但代码里的自动维护并不完整。

## 2. 方法主线：论文到底提出了什么

### 2.1 总体思路

MIRIX 的直觉来自人类记忆分工：有些东西是“我是谁/你是谁”的核心设定，有些是“某天发生了什么”的事件，有些是“某个概念是什么”的知识，有些是“怎么做一件事”的流程。既然这些东西性质不同，就不该全放在一个扁平向量库里。

所以 MIRIX 做了两层分工：

1. **存储分工**：六类记忆各自有结构化字段。
2. **代理分工**：一个 Meta Memory Manager 负责路由，多个 Memory Manager 分别维护自己的记忆类型。

回答问题时，它不等用户显式说“请查记忆”，而是使用 Active Retrieval：先让 agent 生成当前话题，再用话题去检索六类记忆，把结果按 `<episodic_memory>`、`<semantic_memory>` 这类标签放进 system prompt。

### 2.2 三件事拆解

| 问题 | 论文怎么做 | 直觉解释 | 可能代价 |
|---|---|---|---|
| 存记忆 | 六类结构化记忆 + Meta Memory Manager 路由 + 各 Memory Manager 并行更新 | 不同信息交给不同“抽屉”，以后更容易找，也更容易维护 | 写入要多次 LLM 调用，路由错了会漏存或错存 |
| 查记忆 | Active Retrieval：先生成 topic，再跨六类记忆检索 top-k；另提供 embedding/BM25/string_match 工具 | 不等模型想起来查，而是每轮自动把相关记忆找出来 | topic 抽错会带偏检索；默认检索仍像 RAG，缺全局理解 |
| 写进 prompt | 检索结果按来源标签注入 system prompt，如 `<episodic_memory>` | 模型能看到“这条信息来自哪类记忆”，减少把事件、事实、敏感信息混为一谈 | prompt 会膨胀；检索错的信息会更强地影响回答 |
| 维护/遗忘/整合 | 论文提到避免冗余、Core 超 90% rewrite、事件合并、Reflexion 归纳 | 长期记忆不能只增不管，需要压缩、合并、去重 | 代码里很多维护机制默认关闭或很弱，实际更多依赖 LLM 自觉 |

### 2.3 六类记忆分别是什么

- **Core Memory**：永远重要、应常驻的身份信息。例如 agent persona、用户姓名、长期偏好。
- **Episodic Memory**：带时间戳的事件。例如“2025-03-05 10:15 用户在看某个论文网页”。
- **Semantic Memory**：抽象概念、实体、关系。例如“John 是用户的朋友，住在 San Francisco，喜欢 jogging”。
- **Procedural Memory**：怎么做某件事的步骤。例如“如何通过 OpenTable 订餐”。
- **Resource Memory**：用户正在阅读或使用的文档/文件/媒体内容。例如项目提案、PDF、语音转写。
- **Knowledge Vault**：需要精确保留的敏感信息。例如地址、手机号、API key、账号信息。论文说高敏感度内容应有访问控制，避免随便检索泄漏。

### 2.4 用一个例子走一遍

假设用户今天说：“我下周要按 Alice 发来的流程申请会议报销，链接在这个 PDF 里。”同时屏幕截图里打开着报销 PDF。

MIRIX 理想上的处理是：

1. Meta Memory Manager 判断这条输入涉及多个记忆类型。
2. Episodic Memory 记录一个事件：用户今天在准备下周会议报销。
3. Semantic Memory 可能记录 Alice 是与用户报销流程相关的人。
4. Procedural Memory 提取“申请会议报销”的步骤。
5. Resource Memory 保存 PDF 标题、摘要和内容片段。
6. 如果里面有账号、地址等精确信息，Knowledge Vault 保存敏感字段。

几天后用户问：“我报销会议费要先做什么？”Active Retrieval 先生成话题“会议报销流程”，检索到 Procedural/Resource/Episodic 相关条目，再把这些条目放进 system prompt。模型回答时就不需要重新读取原 PDF，也不需要用户再解释上下文。

## 3. 实验与结果

### 3.1 ScreenshotVQA

这是 MIRIX 自建的多模态基准。数据来自 3 名博士生的电脑截图：

- Student 1：1 天 5886 张截图。
- Student 2：20 天 18178 张截图。
- Student 3：一个多月 5349 张截图。
- 总问题数：87 个，由用户基于自己的截图历史手动构造并检查可答性。

对比方法：

- **Gemini 长上下文**：把最近约 3600 张压缩到 256×256 的截图塞进 prompt。
- **SigLIP@50 + Gemini**：用 SigLIP 检索 top-50 原始截图，再交给 Gemini 回答。
- **MIRIX**：不保存原图，只保存从截图中抽出的结构化记忆，存储用 sqlite.db 大小衡量。

论文 Table 1 的总体结果：

| 方法 | Overall Acc | Overall Storage |
|---|---:|---:|
| Gemini | 0.1166 | 236.70MB |
| SigLIP@50 | 0.4410 | 15.07GB |
| MIRIX | 0.5950 | 15.89MB |

论文想证明的是：对大规模截图历史，直接长上下文不够，图像检索也很贵；如果把视觉活动抽成结构化记忆，可以更准且极大省存储。按论文说法，MIRIX 比 SigLIP RAG 准确率高 35%，存储少 99.9%；比 Gemini 长上下文准确率高 410%，存储少 93.3%。

### 3.2 LOCOMO

LOCOMO 是长对话记忆 benchmark。每个 conversation 约 600 轮、26000 tokens，平均约 200 个问题，分为 single-hop、multi-hop、temporal、open-domain 等类型。论文沿用 Mem0 的设置，排除了 adversarial 类问题。

论文 Table 2 的关键结果：

| 方法 | Single-Hop | Multi-Hop | Open-Domain | Temporal | Overall |
|---|---:|---:|---:|---:|---:|
| Zep（gpt-4.1-mini） | 79.43 | 69.16 | 73.96 | 83.33 | 79.09 |
| LangMem（gpt-4.1-mini） | 74.47 | 61.06 | 67.71 | 86.92 | 78.05 |
| Mem0（gpt-4.1-mini） | 62.41 | 57.32 | 44.79 | 66.47 | 62.47 |
| MIRIX | 85.11 | 83.70 | 65.62 | 88.39 | 85.38 |
| Full-Context | 88.53 | 77.70 | 71.88 | 92.70 | 87.52 |

最值得注意的是 multi-hop：MIRIX 83.70，比表中最强 baseline 高很多，甚至超过 Full-Context 的 77.70。论文解释是，MIRIX 在写入阶段会把分散信息整合成“Caroline moved from her hometown, Sweden, 4 years ago”这样的记忆，回答时不必再现场做多跳拼接。

### 3.3 实验局限

- ScreenshotVQA 只有 3 个用户、87 个问题，规模不大，且问题由用户自己构造。
- ScreenshotVQA 的 baseline 并不是其它成熟 memory system，因为 Mem0/Letta 等不能处理大规模图像输入；这能说明 MIRIX 的多模态实用性，但横向对比不完全公平。
- LOCOMO 排除了 adversarial 问题，不能说明系统会拒答不可答问题。
- 评测主要用 GPT-4.1 做 LLM-as-Judge，仍有 judge 偏差。
- MIRIX 写入成本很高，论文效果没有和“每轮多少 LLM 调用/多少钱/延迟多少”放在同等地位比较。

## 4. 写作结构可借鉴

MIRIX 的写作故事很清晰：先批评 flat memory，再提出“真实个人助手需要多类型记忆”，然后用 ScreenshotVQA 把“文本记忆系统不够用”这个问题具象化。它的好处是场景感很强：屏幕截图、可穿戴设备、个人助手、记忆可视化，这些都比抽象地说“提升长期记忆”更容易让读者理解价值。

可借鉴的地方：

- **问题叙事具体**：不是泛泛说记忆重要，而是说“高分辨率截图太多，长上下文和图像 RAG 都扛不住”。
- **方法图容易讲**：六类记忆组件天然适合画架构图，也方便做横向对比。
- **实验故事有冲击力**：存储从 GB 级降到 MB 级，容易形成亮点。

需要谨慎学习的地方：

- 论文有较强产品宣传味，比如 marketplace、wearable 等段落和核心算法关系不紧。
- “八个 agent”“多种检索函数”“Reflexion”等宣称，在代码默认路径里并不都完整落地。
- 它强调架构完整性，但对维护机制、成本、失败案例讲得不够细。

## 5. 代码实现：论文方法在源码里长什么样

### 5.1 源码里的核心模块

MIRIX 代码是一个完整服务端应用，不只是论文 demo。整体链路是：HTTP/SDK 请求进入 FastAPI → AsyncServer 调度 → MetaAgent 管理多个子代理 → 每个子代理走统一 `Agent.step()` 循环 → 记忆工具调用对应 service manager → manager 写 PostgreSQL/pgvector 和 Redis 缓存。

| 论文概念 | 代码位置 | 作用 | 备注 |
|---|---|---|---|
| Meta Memory Manager | `mirix/agent/meta_agent.py:89-144` | 注册和初始化多个记忆相关代理 | 代码里是 9 个：6 类记忆 + meta_memory + reflexion + background |
| Memory Manager 通用循环 | `mirix/agent/agent.py:1491` | 所有子代理复用的 step 入口 | 区别主要来自 system prompt 和工具集 |
| 六类记忆分类规则 | `mirix/prompts/system/base/meta_memory_agent.txt:6-112` | 告诉 meta 代理什么信息该进哪类记忆 | 路由逻辑主要写在 prompt 里 |
| 写入入口 | `mirix/server/rest_api.py:2005` | `/memory/add` 接收待写入消息 | 默认异步入队 |
| 队列入口 | `mirix/queue/queue_util.py:76` | 将消息放入 Kafka/内存队列 | 写入不阻塞前端 |
| 路由工具 | `mirix/functions/function_sets/memory_tools.py:954` | `trigger_memory_update` 选择要更新的记忆类型 | 子代理并行执行在 `memory_tools.py:1187-1188` |
| Active Retrieval | `mirix/agent/agent.py:1726` | `build_system_prompt_with_memories` 检索记忆并重建 prompt | 默认每类 top-10 |
| Topic 抽取 | `mirix/agent/agent.py:2098` | `_extract_topics_from_messages` 生成检索话题 | 独立 LLM 调用，强制工具调用 |
| Prompt 拼装 | `mirix/agent/agent.py:1966` | 将检索结果拼成 XML 标签块 | 如 `<episodic_memory>`、`<knowledge_vault>` |
| Episodic 写入 | `mirix/services/episodic_memory_manager.py:529` | `insert_event` 落库 | 默认为 summary/details 算 embedding |
| Episodic 检索 | `mirix/services/episodic_memory_manager.py:645` | `list_episodic_memory` | Redis → PostgreSQL → SQLite 回退 |
| 主动工具检索 | `mirix/functions/function_sets/base.py:84` | `search_in_memory` | agent 可显式按 memory_type/search_method 搜 |

### 5.2 一条写入数据流

以 SDK 或 HTTP 调 `client.add()` / `POST /memory/add` 为例：

1. 请求进入 `mirix/server/rest_api.py:2005`，然后通过 `put_messages()` 进入队列（`mirix/queue/queue_util.py:76`）。
2. worker 取出消息，进入 `server._step()`（`mirix/server/server.py:693`），加载 `meta_memory_agent`。
3. meta 代理 step 0 先抽话题：`Agent._extract_topics_from_messages`（`mirix/agent/agent.py:2098-2163`）发起一次 LLM 调用，强制调用 `update_topic`。
4. 每个 `inner_step` 都会重建 system prompt：`build_system_prompt_with_memories`（`mirix/agent/agent.py:1726`）按 topic 从各类记忆里检索 top-10。
5. meta 代理看着“新输入 + 已检索记忆”，调用 `trigger_memory_update(memory_types=[...])`（`mirix/functions/function_sets/memory_tools.py:954`），决定交给哪些子代理。
6. `trigger_memory_update` 用 `asyncio.gather` 并行跑多个子代理（`memory_tools.py:1187-1188`）。子代理拿到原始消息和父代理已检索的 `retrieved_memories`（`agent.py:1075-1081`、`memory_tools.py:1107-1112`）。
7. 每个子代理通常只做一次工具调用，比如 `episodic_memory_insert`、`semantic_memory_update`、`resource_memory_insert`，然后由对应 manager 写 DB。
8. 写入后，记忆代理清空自己的对话历史，只保留 system 消息（`mirix/agent/agent.py:1252-1283`）。也就是说，记忆代理本身不靠聊天历史保持状态，状态都在数据库里。

默认配置下，每次写入大概需要：

- topic 抽取 1 次 LLM；
- meta 代理 1-2 次 LLM（chaining 默认开）；
- 被选中的每类记忆子代理各 1 次 LLM；
- 落库条目还会触发 embedding，例如 episodic 的 summary/details 各算一次（`episodic_memory_manager.py:557-566`）。

所以 MIRIX 的写入成本常见是 3-9 次 LLM 调用。这解释了为什么它架构很强，但作为 baseline 时必须报告成本口径。

### 5.3 一条读取数据流

用户向 chat agent 提问时，MIRIX 默认走 Active Retrieval：

1. 先抽当前 topic，例如用户问“我上周在看哪篇 memory 论文”，topic 可能是“memory paper last week”。
2. `build_system_prompt_with_memories`（`mirix/agent/agent.py:1726`）用 topic 检索六类记忆。
3. 主链路里 `search_method` 硬编码为 `"bm25"`（`mirix/agent/agent.py:1754`），每类最多拿 `MAX_RETRIEVAL_LIMIT_IN_SYSTEM=10`（`mirix/constants.py:87`）。
4. 检索结果在 `build_system_prompt`（`mirix/agent/agent.py:1966-2052`）里拼成 XML 标签块，替换或重建 system prompt。
5. LLM 最终回答时，看到的是“用户问题 + system prompt 里的结构化记忆”。

此外还有两类显式检索入口：

- agent 工具：`search_in_memory(memory_type, query, search_field, search_method)`（`mirix/functions/function_sets/base.py:84`），支持 bm25/embedding/string_match。
- HTTP API：`/memory/retrieve/conversation`、`/memory/retrieve/topic`、`/memory/search`（`mirix/server/rest_api.py:2524/2684/2802`）。

### 5.4 论文方法和代码的逐项对应

| 论文里的方法点 | 代码是否实现 | 关键证据 | 我的理解 |
|---|---|---|---|
| 六类记忆组件 | 是 | `mirix/services/` 下各 memory manager；`meta_agent.py:89-144` | 结构确实存在，不是只在论文里画图 |
| Meta Memory Manager 路由 | 是 | `memory_tools.py:954`、`meta_memory_agent.txt:6-112` | 路由主要靠 LLM + prompt 规则 |
| 子 Memory Manager 并行更新 | 是 | `memory_tools.py:1187-1188` | 写入时可并行调多个记忆代理 |
| Active Retrieval | 是 | `agent.py:2098`、`agent.py:1726`、`agent.py:1966` | 每轮先抽 topic，再检索并写入 system prompt |
| 多种检索函数 | 部分实现 | `base.py:84`；但 active retrieval 主链路 `agent.py:1754` 硬编码 bm25 | 工具可用，默认自动检索路径较窄 |
| Core Memory 过载 rewrite | 部分实现 | `memory_tools.py:57-67` | 超限时返回错误让 LLM 自己 rewrite，不是确定性压缩器 |
| 去重/避免冗余 | 很弱 | `memory_tools.py:625-647` 等 | 多数是全字段精确相等才跳过，语义重复靠 LLM 自觉 |
| Reflexion 归纳/全库整理 | 默认关闭 | `app_constants.py:28`、`reflexion_agent.txt:9-47` | prompt 写得很漂亮，但无默认调度 |
| Background agent | 基本空壳 | `app_constants.py:31`、`background_agent.py` | 默认关闭，且实现很少 |
| 自动遗忘/时间衰减 | 基本无 | `jobs/cleanup_raw_memories.py:20` 只清 raw memory 且需外部 cron | 六类长期记忆没有默认衰减或淘汰 |

### 5.5 最值得读的代码入口

- `mirix/agent/meta_agent.py:89-144`：先看这里，能知道 MIRIX 实际有哪些子代理。
- `mirix/prompts/system/base/meta_memory_agent.txt:6-112`：这里比很多 Python 代码更重要，六类记忆的路由规则都写在 prompt 里。
- `mirix/functions/function_sets/memory_tools.py:954`：`trigger_memory_update` 是写入路由的核心工具。
- `mirix/agent/agent.py:1726`：Active Retrieval 的核心入口，解释了记忆如何自动进 prompt。
- `mirix/agent/agent.py:1966-2052`：看检索结果如何被格式化成 XML 标签块。
- `mirix/services/episodic_memory_manager.py:645`：看一个具体 memory manager 如何做缓存、全文检索、向量检索和回退。

## 6. 论文宣称 vs 代码实际

### 6.1 与论文一致的地方

- **六类记忆 + 多 agent 管理**是真实现了。每类记忆有自己的 manager、schema、工具和 prompt。
- **Meta 路由 + 并行更新**是真实现了。meta 代理调用 `trigger_memory_update`，再并行调多个子代理。
- **Active Retrieval**是真实现了。代码确实每轮抽 topic，并把检索结果按 XML 标签注入 system prompt。
- **多模态应用方向**在产品代码里有痕迹。桌面端截图流通过 `TemporaryMessageAccumulator` 攒满 20 条触发记忆更新（`temporary_message_accumulator.py:446-501`），对应论文应用场景。

### 6.2 不一致或容易误读的地方

- **论文说多种检索函数，但默认 Active Retrieval 只走 bm25**：`agent.py:1754` 把 `search_method` 硬编码为 `"bm25"`，1757 行附近的 embedding 分支在默认主链路不可达。向量检索主要通过显式工具和 HTTP search API 使用。
- **“BM25”名不副实**：PostgreSQL 路径实际用 `to_tsvector + ts_rank_cd`（`episodic_memory_manager.py:1094-1118`），不是标准 BM25。真正的 `BM25Okapi` 只在 SQLite 回退分支（`episodic_memory_manager.py:948-971`）。
- **Reflexion 代理默认是摆设**：prompt 宣称它可做全库去重和用户行为模式归纳，但 `WITH_REFLEXION_AGENT=False`（`app_constants.py:28`），全仓库也没看到默认定时触发。
- **Background 代理基本是 stub**：默认关闭（`app_constants.py:31`），实现非常薄。
- **prompt 与代码常数不一致**：episodic 代理 prompt 说会展示最多 50 条最近和 50 条相关事件，但代码常数是每类 top-10（`constants.py:87`）。
- **并行开关不可配置**：`CALL_MEMORY_AGENT_IN_PARALLEL`（`constants.py:237`）定义后没有实际引用，并行行为由代码固定。

### 6.3 维护机制的真实强度

MIRIX 很适合作为“类型化记忆架构”的代表，但不适合作为“长期记忆自动维护”的强代表。

- 去重：很多 insert 工具会 `list_*(query="", limit=1000)` 拉全量，然后字段完全相等才跳过（如 `memory_tools.py:625-647`）。这只能挡住逐字重复。
- 更新：多数 `*_update` 是删旧插新（如 `semantic_memory_update`，`memory_tools.py:708-744`），不是精细版本管理。
- 矛盾处理：没有统一矛盾检测器，靠子代理 LLM 在一次工具调用里判断。
- 遗忘：六类长期记忆没有自动 TTL、衰减、容量淘汰。`raw_memory` 有 14 天 TTL 清理脚本（`jobs/cleanup_raw_memories.py:20`），但 raw memory 不是论文六类长期记忆，而且需要外部 cron 或手动 HTTP 端点触发。

## 7. 局限与问题空间

### 7.1 论文自己暴露出的局限

- Open-domain 问题上 MIRIX 低于 Full-Context，论文也承认这反映了 RAG 类方法缺少全局理解。
- Single-hop 某些问题会因为 MIRIX 存了“确认发生的事件”而覆盖“原计划”，导致和标准答案口径不一致。比如用户原计划六月露营，后来十月真的去了，MIRIX 可能优先答实际发生事件。
- 系统依赖强 function calling 模型，论文在 LOCOMO 上专门换成 gpt-4.1-mini，因为它比 gpt-4o-mini 更适合多轮函数调用。

### 7.2 我从代码看出来的局限

- 写入路径太重：一次写入可能 3-9 次 LLM 调用，适合异步后台，不适合低成本高频在线更新。
- 很多“智能维护”停在 prompt 层，没有默认闭环调度。
- 默认检索方法偏保守，Active Retrieval 主要是全文检索，不是论文读起来那种“多检索策略智能选择”。
- 六类记忆虽然分工清楚，但跨记忆类型的一致性管理较弱。例如一个信息同时进入 Episodic 和 Semantic 后，后续怎么同步更新并不清晰。

### 7.3 对我的研究可能有价值的问题

- **类型化记忆是否真的带来收益，还是 benchmark 刚好偏向这种抽取？** LOCOMO multi-hop 的提升很大，但需要拆 ablation：六类里到底哪几类贡献最大？
- **写入阶段整合 vs 查询阶段推理**：MIRIX 的 multi-hop 强项来自写入时把分散信息整合成一条记忆。这说明“提前整合”可能比“检索后现场多跳推理”更稳，但也可能产生错误合并。
- **高成本记忆系统的 Pareto 评测**：MIRIX 效果好，但成本高。可以做“准确率-写入成本-存储增长-延迟”的多目标比较。
- **维护机制缺口仍然明显**：类型化存储解决了“放哪里”，但没真正解决“什么时候过期、冲突怎么处理、哪些该删”。

## 8. 对我的启发

- 可借鉴的设计：把检索结果按来源标签注入 system prompt。`<episodic_memory>` 和 `<knowledge_vault>` 这样的标签，能让模型知道信息性质，不只是看到一堆相似文本。
- 可借鉴的设计：写入用 meta 代理路由，子代理无状态化，更新完清空历史，把状态都放到 DB。这个工程模式清晰，适合长期服务端系统。
- 可借鉴的实验：ScreenshotVQA 的故事很好，证明“多模态长期记忆”不是普通文本 memory benchmark 能覆盖的。
- 不宜盲跟的方向：单纯再加一种记忆类型，容易变成 MIRIX 的小变体；没有明确评测收益，很难讲成新论文。
- 可能发展成 idea 的点：对类型化记忆做成本敏感 ablation，研究“什么信息值得写入哪类记忆，以及写错类型的代价”；或者研究类型化记忆下的矛盾/过期生命周期。

## 9. 术语小词典

- **Active Retrieval**：主动检索。不是等用户或模型说“查记忆”，而是每轮先生成 topic，再自动检索相关记忆放进 prompt。
- **Meta Memory Manager**：元记忆管理器。它不直接保存所有记忆，而是判断新输入该交给哪些记忆子代理。
- **Memory Manager**：某一类记忆的专职代理，比如 Episodic Memory Manager 只负责事件记忆。
- **Episodic Memory**：情节记忆，记录“什么时候发生了什么”。
- **Semantic Memory**：语义记忆，记录概念、实体和关系，不一定绑定某个具体时间。
- **Procedural Memory**：程序性记忆，记录“怎么做一件事”的步骤。
- **Resource Memory**：资源记忆，保存用户正在用的文档、文件、转写内容等。
- **Knowledge Vault**：知识金库，保存需要逐字保留或带敏感性的事实。
- **LLM-as-Judge**：让另一个 LLM 根据标准答案判断模型回答是否正确的评测方式。

## 10. 原源码审计要点保留

这部分是为了以后快速查代码证据。

| 机制环节 | 文件路径 | 类/函数名 | 行号 | 一句话作用 |
|---|---|---|---|---|
| 子代理注册 | `mirix/agent/meta_agent.py` | `MEMORY_AGENT_CONFIGS` / `MetaAgent` | 89-144, 147 | 定义并初始化记忆相关子代理 |
| 写入入口 | `mirix/server/rest_api.py` | `add_memory` | 2005 | `/memory/add` 进 Kafka/队列异步处理 |
| 路由决策 | `mirix/functions/function_sets/memory_tools.py` | `trigger_memory_update` | 954 | meta 代理选记忆类型，并行 step 子代理 |
| 路由规则 | `mirix/prompts/system/base/meta_memory_agent.txt` | - | 6-112 | 六类记忆的分类规则 |
| 话题抽取 | `mirix/agent/agent.py` | `_extract_topics_from_messages` | 2098 | 独立 LLM 调用强制触发 `update_topic` |
| Active Retrieval | `mirix/agent/agent.py` | `build_system_prompt_with_memories` | 1726 | 按 topic 检索六类记忆 |
| prompt 拼装 | `mirix/agent/agent.py` | `build_system_prompt` | 1966 | 检索结果按 XML 标签块拼进 system prompt |
| episodic 写入 | `mirix/services/episodic_memory_manager.py` | `insert_event` | 529 | 落库，默认算 summary/details embedding |
| episodic 检索 | `mirix/services/episodic_memory_manager.py` | `list_episodic_memory` | 645 | Redis 缓存 → PG 全文/向量 → SQLite 回退 |
| PG 全文检索 | `mirix/services/episodic_memory_manager.py` | `_postgresql_fulltext_search` | 1015 | `ts_rank_cd` 加权排序，AND 失败回退 OR |
| 事件合并 | `mirix/functions/function_sets/memory_tools.py` | `episodic_memory_merge` | 168 | 延续事件覆盖式合并 |
| 写入去重 | `mirix/functions/function_sets/memory_tools.py` | `semantic_memory_insert` | 625-647 | 拉全量逐条精确比对，完全相等才跳过 |
| 删除/更新 | `mirix/functions/function_sets/memory_tools.py` | `semantic_memory_update` | 685 | 删旧插新，`new_items=[]` 即纯删除 |
| 代理工具检索 | `mirix/functions/function_sets/base.py` | `search_in_memory` | 84 | chat/记忆代理的主动检索工具 |
| 历史清空 | `mirix/agent/agent.py` | `_handle_ai_response` | 1252-1283 | 记忆代理更新完即清空对话历史 |
| 上下文摘要 | `mirix/agent/agent.py` | `summarize_messages_inplace` | 2601 | 75% 阈值触发对话压缩 |
| TTL 清理 | `mirix/jobs/cleanup_raw_memories.py` | `delete_stale_raw_memories_async` | 20 | raw memory 14 天 TTL，需外部 cron |
