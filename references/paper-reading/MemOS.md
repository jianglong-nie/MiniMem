# MemOS

- 仓库：https://github.com/MemTensor/MemOS
- commit：b60616d（本文所有行号仅在该 commit 下有效）
- stars：9777（2026-06-11 查询）
- 对应论文：MemOS: A Memory OS for AI System，arXiv:2507.03724；前置框架论文 arXiv:2505.22101

---

## 一、论文讲了一个什么故事

### 1.1 问题：LLM 的"数字失忆症"

一个典型的 AI 助手今天帮你记录了健康目标，明天对话重启，它把这件事忘得干干净净。不仅如此：

- **上下文窗口有限**：模型每次只能"看到"最近 N 个 token 的对话（token 可以理解为文字片段），超出就截断。
- **训练权重固定**：模型学到的知识在训练结束后就冻结了，新发生的事它不知道。
- **现有补丁方案无根治**：RAG（Retrieval-Augmented Generation，检索增强生成——在回答问题前先去外部库里搜索相关文本，再拼入 prompt）虽然能补充外部知识，但每次对话都是临时拼凑，没有任何长期记忆的"生命周期"：没有更新、没有遗忘、没有演化。

### 1.2 想法：把记忆当成操作系统资源来管理

论文的核心比喻是：**AI 系统缺的不是记忆能力，而是记忆管理的操作系统**。

操作系统（OS）做什么？它不创造 CPU 和内存，但它统一调度和管理这些资源——内存可以分配、回收、换页（把内存里的数据临时存到硬盘再换回来）。MemOS 想对 AI 的"记忆"做同样的事：

| 操作系统概念 | MemOS 对应概念 |
|---|---|
| 进程（程序运行单位） | AI Agent（智能体） |
| 内存（运行时可读写的快速存储） | Activation Memory（KV 缓存） |
| 硬盘（持久化存储） | Textual Memory（明文图节点） |
| 固件/程序本体 | Parametric Memory（LoRA 参数） |
| 可移动磁盘 | MemCube（可装卸的记忆容器） |
| OS 调度器 | MemScheduler |

这个隐喻的意义在于：记忆从此有了**生命周期**（创建 → 使用 → 更新 → 遗忘），而不是一个静态的文本堆。

### 1.3 论文提出的架构

MemOS 分三层：
- **接口层（Interface Layer）**：对外暴露统一 API（add/search/chat/delete），把用户输入翻译成记忆操作。
- **操作层（Operation Layer）**：MemScheduler 异步调度记忆任务，MemOperator 负责记忆结构的组织和演化。
- **基础设施层（Infrastructure Layer）**：MemVault 负责持久化，MemGovernance 负责权限和安全。

核心抽象是 **MemCube**：一个自包含的记忆容器，里面封装了内容（明文 / KV 缓存 / LoRA 权重）和元数据（来源、版本号、访问权限、过期时间）。论文宣称 MemCube 可以像移动硬盘一样在不同 agent、用户、项目之间"插拔"。

---

## 二、三类记忆——理论与现实

### 2.1 Textual Memory（明文记忆）——**真实可用**

这是论文和代码里唯一完整实现的记忆类型。

**是什么**：把 AI 系统中需要长期保存的知识，以"事实句"的形式存进一个图数据库（graph database，一种把数据组织成"节点+关系"的数据库，类似人脑中概念之间的联系网络）。每条记忆是一个图节点，节点之间可以有 PARENT（从属）、MERGED_TO（合并）等关系边。

**为什么用图而不是普通列表**：图结构允许把相似的记忆聚合成"主题"节点，形成"具体事实 → 抽象主题"的层级，就像记笔记时把细节条目归纳到章节标题下面。

### 2.2 Activation Memory（激活记忆）——**实现了但默认关闭**

**是什么**：KV 缓存（Key-Value Cache）是 Transformer 神经网络的内部中间计算结果。每次模型处理一段文本，都会产生这组数值，下次再处理相关文本时可以直接复用，不必重新计算，从而大幅加速推理。

论文宣称把 KV 缓存作为"工作记忆"来管理，类比 RAM——频繁访问的记忆直接以计算结果的形式保存在内存里，无需每次重新理解。

**实际情况**：`enable_activation_memory` 默认为 False（[configs/mem_os.py:53-56](../research/MemOS/src/memos/configs/mem_os.py)）；目前只有 HuggingFace 本地后端支持，且只取第一个 MemCube 的第一条缓存，代码里有 TODO 自注。vLLM 版本的"KV cache"实际上只是存了 prompt 字符串，靠服务端的前缀缓存（prefix caching）来加速，根本不是可移植的 KV 张量。

### 2.3 Parametric Memory（参数记忆）——**完全是占位符**

**是什么**：LoRA（Low-Rank Adaptation）是一种轻量级微调技术——不改动整个模型的巨大参数矩阵，而是在每一层旁边插入一对小矩阵，只训练这对小矩阵来让模型"学会"新知识。论文设想把不同用户或领域的 LoRA 权重作为一种可切换的"参数记忆"。

**实际情况**：`lora.py` 开头注释明写"currently serves as a placeholder, do not use"，`dump()` 方法只写一个字节串 `b"Placeholder"`（[src/memos/memories/parametric/lora.py:37-41](../research/MemOS/src/memos/memories/parametric/lora.py)）。论文三大支柱之一，代码里是空壳。

---

## 三、一条记忆条目长什么样

理解"记忆条目的数据结构"是读懂整个系统的关键，因为后面所有的写入、检索、更新操作都是在操作这个数据对象。

### 3.1 完整字段

明文记忆的核心数据类是 `TextualMemoryItem`（[src/memos/memories/textual/item.py:299](../research/MemOS/src/memos/memories/textual/item.py)），结构如下：

```
TextualMemoryItem
├── id: str                    # 唯一 UUID，例如 "a3f2e1..."
├── memory: str                # 记忆正文，一条完整的事实句
└── metadata: TreeNodeTextualMemoryMetadata
    ├── memory_type: str       # 分桶类型，见下表
    ├── key: str               # 记忆标题，例如 "Tom的项目截止日期"
    ├── tags: list[str]        # 关键词标签，例如 ["项目","截止日期","会议"]
    ├── embedding: list[float] # 向量表示，约 1024 维浮点数（用于语义检索）
    ├── sources: list[SourceMessage]  # 来源溯源，记录这条记忆从哪段对话/文档来的
    ├── confidence: float      # 可信度，0~100
    ├── status: str            # "activated" / "archived" / "deleted"
    ├── is_fast: bool          # 是否是 fast 模式的原始文本（未经 LLM 精炼）
    ├── version: int           # 版本号，更新时递增
    ├── history: list[ArchivedTextualMemory]  # 历史版本快照
    ├── created_at / updated_at: str          # ISO 8601 时间戳
    ├── user_id / session_id: str             # 归属哪个用户、哪次对话
    └── usage: list[str]       # 使用历史（目前未更新，见"论文与代码的差距"）
```

**memory_type 的分桶**：这个字段决定记忆放在哪个"桶"里，影响检索策略和容量限制：

| memory_type | 容量上限 | 含义 |
|---|---|---|
| WorkingMemory | 20 条 | 当前对话高度相关的临时记忆，按需动态替换 |
| LongTermMemory | 1500 条 | 跨会话的通用事实知识 |
| UserMemory | 480 条 | 用户个人偏好、计划、经历 |

**embedding 是什么**：把一句话变成一串数字，数字之间的距离反映句子语义的远近——类似的句子数字相近，不同的句子数字相远。有了 embedding，系统才能"凭意思"找到相关记忆，而不只是靠关键词匹配。

**sources 字段**：记录这条记忆是从哪段原始对话中提取来的，每个 source 有 `role`（user/assistant）、`content`（原文片段）、`chat_time` 等字段，方便回溯和审计。

### 3.2 具体例子：一段对话如何变成记忆条目

**原始输入**（用户和助手的对话）：
```
user: 我今天和团队开了个会，项目截止日期是12月15日，但后端到12月10日才能完成，测试时间会很紧。
assistant: 要不要考虑把截止日期延到1月5日？
user: 好主意，我明天早上9点半的会议上提这个建议。
```

**第一步（切窗口）**：`_iter_chat_windows`（[simple_struct.py:303](../research/MemOS/src/memos/mem_reader/simple_struct.py)）把对话按 1024 token 切成滑动窗口。上面这段对话很短，是一个窗口。

**第二步（LLM 抽取）**：把整个窗口文本作为 prompt 的 `${conversation}` 部分，发给 LLM（大语言模型）。LLM 被要求输出这样的 JSON：
```json
{
  "memory list": [
    {
      "key": "项目会议与截止日期讨论",
      "memory_type": "LongTermMemory",
      "value": "用户与团队开会讨论项目，原截止日期为2025年12月15日，但后端要到12月10日才能完成，测试时间不足。",
      "tags": ["项目", "截止日期", "会议", "后端"]
    },
    {
      "key": "计划申请延期",
      "memory_type": "UserMemory",
      "value": "用户计划在次日（2025年X月X日）上午9:30的会议上提议将项目截止日期延至2026年1月5日。",
      "tags": ["计划", "截止日期变更", "会议"]
    }
  ],
  "summary": "用户参加了项目进度会议，发现原截止日期12月15日过于紧张，打算次日上午提议延期至1月5日。"
}
```

**第三步（embedding）**：对每条抽取出的 `value` 字符串，调用 embedding 模型（如 text-embedding-ada-002），把它变成约 1024 维的向量。

**第四步（写入图数据库）**：每条记忆变成一个 `TextualMemoryItem` 对象，写入图数据库（Neo4j 或兼容数据库）。最终存储的数据大致是：

```
节点 A:
  id: "a3f2..."
  memory: "用户参加了项目进度会议，发现原截止日期12月15日过于紧张..."
  memory_type: "LongTermMemory"
  key: "项目会议与截止日期讨论"
  tags: ["项目", "截止日期", "会议", "后端"]
  embedding: [0.023, -0.041, 0.118, ...]   # 1024个浮点数
  sources: [{role: "user", content: "我今天和团队开了个会..."}, ...]
  status: "activated"
  created_at: "2025-12-14T10:30:00"
  user_id: "user_001"
```

**第五步（进入 prompt）**：下次用户问"我项目截止日期有没有问题"时，系统检索出这些节点，拼成：
```
## Memories:
1. 用户参加了项目进度会议，发现原截止日期12月15日过于紧张...
2. 用户计划在次日上午9:30的会议上提议将项目截止日期延至2026年1月5日。
```
这段文字被附加到 system prompt 里，让 LLM 在回答时"有记忆可参考"。

---

## 四、实验与结果

### 4.1 评测基准说明

论文在四个基准上评测，覆盖不同维度的"记忆能力"：

| 基准 | 考察什么 | 形式 |
|---|---|---|
| **LongMemEval** | 长期对话中的事实一致性与知识更新 | 多轮 QA，问题依赖历史对话中的细节 |
| **LoCoMo** | 长期会话的多跳推理和时序理解（"多跳"指答案需要串联多个分散的事实才能推导出来） | 多轮对话 + 问答 |
| **PersonaMem** | 个性化响应准确率——能否记住并运用用户特征 | 给定用户画像，评价回答的个性化程度 |
| **PrefEval** | 用户偏好的遵从率——能否记住"我不喜欢 X"并持续执行 | 测试在 0 轮和 10 轮对话后偏好的保留情况 |

### 4.2 主要结果

与主要基线（OpenAI Memory、Mem0、Zep、MIRIX 等）相比：

- **LongMemEval 整体均值**：MemOS +40.43%（vs. 最强基线）
- **LoCoMo 整体准确率**：+38.97%，其中时序推理任务 +159%
- **PersonaMem 精确率**：+40.75%
- **PrefEval-10（10轮对话后偏好保留）**：+2568%（相对数字，基线接近 0）
- **Token 消耗节省**：35.24%（LoCoMo 上 token 开销降低 60.95%）

**如何解读这些数字**：LoCoMo 上 159% 的时序推理提升说明系统真的在维护时间线信息（"用户在 6 月提到了 X，7 月又说了 Y"），而不是把所有历史信息堆在 prompt 里靠 LLM 自己梳理。Token 节省来自"精确检索相关记忆再放入 prompt"而非"把整段历史都塞进去"。

### 4.3 实验背后的设计选择

论文里"MemOS-1031"是参加评测的版本（数字代表内部版本号），在所有四个基准上排第一。但需要注意：这套系统的优势很大程度上来自**精细的明文记忆**（LLM 抽取 + 图结构管理），而不是论文宣称的三类记忆统一框架——因为另外两类记忆在当前代码里基本是空的。

---

## 五、源码实现：系统是怎么运转的

### 5.1 核心对象三层

```
MOSCore（core.py）
    ├── GeneralMemCube（general.py）
    │   ├── text_mem: TreeTextMemory    ← 真正在用的
    │   ├── act_mem: KVCacheMemory      ← 默认关闭
    │   ├── para_mem: LoRAMemory        ← 占位符
    │   └── pref_mem: PreferenceMemory  ← 偏好记忆
    └── MemScheduler（异步任务队列）
```

`MOSCore` 是面向使用者的编排层，提供 `add/search/chat/get/delete` 接口。它不直接操作记忆，而是把任务分发给各个 MemCube，并向 MemScheduler 投递异步任务消息。

`TreeTextMemory` 是明文记忆的主体，底层对接图数据库（Neo4j 或 PolarDB）。

### 5.2 写入路径

**同步精确模式（sync + fine）**，每次 add 的完整流程：

```
MOSCore.add(messages)
  → mem_reader.get_memory()
    → SimpleStructMemReader._process_chat_data()
       ① 按 1024 token 切对话窗口
       ② 每个窗口：1次 LLM 调用 → 若干条 {key, value, tags, memory_type}
       ③ 每条记忆：1次 embedding
  → TreeTextMemory.add(memory_items)
    → MemoryManager._add_memories_batch()
       ④ 批量写图节点，挂 working_binding 标记
       ⑤ （若 reorganize=True）推消息给后台整理线程
```

**关键权衡——fast vs fine 模式**：

这是 MemOS 最核心的工程设计之一，用来解决"快速响应"和"记忆质量"之间的矛盾：

- **fine 模式**（精确）：每个窗口调用一次 LLM 抽取事实，慢但质量高。一次普通对话约 1 次 LLM 调用 + 2~6 次 embedding。
- **fast 模式**（快速）：0 次 LLM，直接把原文窗口作为一条"粗糙记忆"写进图节点，打上 `is_fast=True` 和 `[working_binding:<id>]` 标记。同时给 MemScheduler 发送 MEM_READ 任务。

Fast 模式写入后，后台的 `MemReadMessageHandler` 会把粗糙节点取出，用 LLM 精抽取得到干净的事实句，写入新的节点，然后**删除原来的粗糙节点**（[mem_read_handler.py:401-413](../research/MemOS/src/memos/mem_scheduler/task_schedule_modules/handlers/mem_read_handler.py)）。这是"先快后精、异步升级"的策略——不堵塞用户，精炼在后台默默发生。

### 5.3 读取路径

**检索是多路并行召回，不是单一向量搜索**：

```
MOSCore.search(query)
  → TreeTextMemory.search()
    → Searcher.search()
       ① 解析 query（fine模式：1次LLM解析意图/关键词/标签/改写）
       ② 同时embedding query 得到查询向量
       ③ 多路并行召回（线程池）：
          Path A: WorkingMemory 全量取出（只有20条，直接全拿）
          Path B: LongTerm+User：
                  - 图元数据召回（key精确匹配 + tags 重叠≥2条）
                  - 向量召回（余弦相似度）
                  - 可选：BM25全文检索（稀疏检索，基于词频统计的经典算法）
          Path C: 互联网检索（可选）
       ④ rerank（重新排序）：默认余弦相似度本地打分，0次LLM
       ⑤ 按文本字符串去重 → 按分数排序 → 截取 top_k
  → top_k 条 TextualMemoryItem 返回
```

**三路召回为什么这样设计**：
- WorkingMemory 全量取出：只有 20 条，当前会话最相关，直接全要。
- 图元数据（key/tags）：高精度精确匹配，避免向量检索在细节上"漏网"（例如标签"项目A"能精确命中，而向量检索可能把"项目B"也拉进来）。
- 向量召回：处理语义相近但用词不同的情况（"截止日期"和"deadline"是同义词，向量空间里距离近）。
- BM25 全文检索：基于词频的老牌算法，在关键词精确匹配上补向量检索的盲区。

**检索结果如何进入 prompt**：`MOSCore.chat` 调用 `_build_system_prompt`（[core.py:354](../research/MemOS/src/memos/mem_os/core.py)），把 top_k 条记忆编号排列，拼在 system prompt 末尾：
```
## Memories:
1. 用户计划在次日上午9:30的会议上提议将项目截止日期延至2026年1月5日。
2. 用户参加了项目进度会议，发现原截止日期12月15日过于紧张...
```

### 5.4 维护路径（记忆的演化与遗忘）

**去重**：三层机制，默认只有后两层生效：
1. 写入时（仅 `reorganize=True`）：embedding 相似度 > 0.8 的节点逐对由 LLM 判断是否冗余，冗余则融合成新节点——**默认关闭**。
2. 读取时：按文本字符串完全匹配去重，防止同一句话重复出现在 prompt 里。
3. fast→fine 精化时：旧的 fast 节点被标记为 `status="archived"` 并删除。

**矛盾处理**（仅 `reorganize=True`）：新节点写入时，如果发现 embedding 相似但内容矛盾（例如"截止日期是12月"vs"截止日期是1月"），LLM 尝试融合两条，生成一条新的合并节点；融合失败则按 `updated_at` 时间戳删除较旧的那条。

**层级摘要整合**（仅 `reorganize=True`）：后台线程每 100 秒触发一次 `optimize_structure`（[reorganizer.py:151](../research/MemOS/src/memos/memories/textual/tree_text_memory/organize/reorganizer.py)）：把图节点的 embedding 做 MiniBatchKMeans 聚类（一种把相似向量分组的算法），然后让 LLM 给每个簇写摘要，生成"主题节点"，用 PARENT 边连接到下属具体事实节点。这就形成了"事实 → 主题"的两层层级。

**遗忘**：纯容量限制，无智能衰减：
- WorkingMemory：每次同步 add 后裁到 20 条（FIFO，先进先出）。
- LongTerm/User：达到容量上限的 80% 时触发清理，删到上限（[manager.py:527](../research/MemOS/src/memos/memories/textual/tree_text_memory/organize/manager.py)）——但这个清理只在异步路径里触发，纯同步写入不会裁剪 LTM。
- 没有时间衰减、没有重要性打分、`_update_usage_history` 函数体整体被注释掉了（[searcher.py:1290](../research/MemOS/src/memos/memories/textual/tree_text_memory/retrieve/searcher.py)），节点的 usage 字段从不更新。

---

## 六、关键代码位置速查

| 机制 | 文件 | 函数/类 | 行号 | 说明 |
|---|---|---|---|---|
| 总入口 add | src/memos/mem_os/core.py | MOSCore.add | 684 | 分发抽取任务、投递 scheduler 消息 |
| 记忆进 prompt | src/memos/mem_os/core.py | _build_system_prompt | 354 | 编号列表拼 system prompt |
| 四槽位 MemCube | src/memos/mem_cube/general.py | GeneralMemCube.__init__ | 24-48 | text/act/para/pref 装配 |
| LLM 抽取 prompt | src/memos/templates/mem_reader_prompts.py | SIMPLE_STRUCT_MEM_READER_PROMPT | 1 | 明文提示词，含完整示例 |
| 对话切窗口 | src/memos/mem_reader/simple_struct.py | _iter_chat_windows | 303 | 按 token 数滑动切分 |
| fast/fine 分支 | src/memos/mem_reader/simple_struct.py | _process_chat_data | 347 | fast 存原文，fine 调 LLM |
| LLM 调用 | src/memos/mem_reader/simple_struct.py | _get_llm_response | 268 | 失败时以原文兜底 |
| 批量写图节点 | src/memos/memories/.../organize/manager.py | _add_memories_batch | 138 | 写节点、挂 working_binding |
| 容量清理 | src/memos/memories/.../organize/manager.py | _cleanup_memories_if_needed | 527 | 80% 阈值触发 FIFO 裁剪 |
| 冲突/冗余检测 | src/memos/memories/.../organize/handler.py | NodeHandler.detect | 30 | emb>0.8 候选逐对 LLM 判断 |
| 冲突消解 | src/memos/memories/.../organize/handler.py | NodeHandler.resolve | 76 | LLM 融合；失败按时间删旧 |
| 周期结构整理 | src/memos/memories/.../organize/reorganizer.py | optimize_structure | 211 | KMeans+LLM 聚类摘要 |
| 关系推理（空转） | src/memos/memories/.../organize/relation_reason_detector.py | process_node | 26-86 | 四步逻辑全被注释，恒返回空 |
| 多路并行检索 | src/memos/memories/.../retrieve/searcher.py | _retrieve_paths | 335 | Working/LTM/internet 并行召回 |
| 混合召回 | src/memos/memories/.../retrieve/recall.py | GraphMemoryRetriever.retrieve | 35 | 图元数据∪向量∪BM25 |
| 文本去重+截断 | src/memos/memories/.../retrieve/searcher.py | _deduplicate_results | 1096 | 字符串去重+按分排序截 top_k |
| 深度检索 | src/memos/memories/.../retrieve/advanced_searcher.py | deep_search | 232 | 多阶段 LLM 自评+扩展短语 |
| 异步精化 | src/memos/mem_scheduler/.../mem_read_handler.py | _process_memories_with_reader | 110 | fast 节点精抽取后删原始节点 |
| KV 缓存合并 | src/memos/memories/activation/kv.py | KVCacheMemory._concat_caches | 200 | 多段 DynamicCache 逐层 torch.cat |
| LoRA 占位符 | src/memos/memories/parametric/lora.py | LoRAMemory | 14-41 | dump 写 b"Placeholder" |

---

## 七、论文宣称 vs 代码实际

### 7.1 重要不一致（影响如何引用这篇论文）

**① 参数记忆是空壳**
论文把 Parametric Memory 列为三大支柱之一，给出了完整的理论框架；代码里 `lora.py` 第一行注释就说"do not use"，`dump()` 只写 `b"Placeholder"`。引用本文讲"三类记忆统一框架"时要注意：目前只有一类是真实的。

**② 激活记忆条件严苛**
论文用大量篇幅讲 KV 缓存管理的优势；代码默认关闭，只支持本地 HuggingFace 推理，且 vLLM 版"KV cache"实际存的是字符串而非张量，无法跨进程迁移。

**③ 图谱关系推理是空转**
论文描述了 INFERS（推断）、FOLLOWS（时序）、AGGREGATE_TO（聚合）等丰富的边类型；`RelationAndReasoningDetector.process_node` 中这四步逻辑全被三引号注释掉（[relation_reason_detector.py:49-80](../research/MemOS/src/memos/memories/textual/tree_text_memory/organize/relation_reason_detector.py)），恒返回空字典。论文图里的"知识图谱"在运行时实际上只有 PARENT 和 RELATED 边。

**④ 检索 pipeline 文档串与代码不符**
`search` 函数的 docstring 写"MemoryReranker → MemoryReasoner → Final output"；`MemoryReasoner.reason()` 从未在检索流程中被调用，实例化了但没用。

**⑤ 冲突检测/结构整理默认完全不生效**
`reorganize` 配置默认为 False；`MOSCore.mem_reorganizer_on()` 方法体是空 `pass`。论文里讲的"矛盾消解""层级摘要"需要手动打开这个开关才能运行。

**⑥ 使用频率记忆管理无从谈起**
`_update_usage_history` 整个函数体被注释成 docstring，节点的 `usage` 字段永远不会更新。论文里提到的"基于使用频率的智能遗忘"实际不存在。

**⑦ update 接口对主力后端不可用**
`TreeTextMemory.update` 直接 `raise NotImplementedError`；`MOSCore.update` 只打 warning。

### 7.2 论文宣称与代码一致的部分

- **MemCube 容器抽象**：four-slot（text/act/para/pref）结构确实存在，可独立 load/dump，实现了"记忆即可移动盘"的概念。
- **统一 add/search/delete API**：MOSCore 确实提供了统一接口。
- **fast/fine 双速写入 + 异步精化**：fast 先落库 → 后台精化 → 删原始节点，这套流程完整可用。
- **WorkingMemory 动态调度**：`memory_update_handler` 确实根据 query 意图动态替换 WorkingMemory 内容（[memory_update_handler.py:36](../research/MemOS/src/memos/mem_scheduler/task_schedule_modules/handlers/memory_update_handler.py)）。
- **图结构层级摘要**：当 `reorganize=True` 时，KMeans + LLM 聚类摘要确实可以运行，生成 PARENT 边层级。

---

## 八、对研究的启发

**fast/fine 双速写入**是值得借鉴的工程设计：写入延迟和记忆质量是一对矛盾，先存原文保证实时性、后台异步精化保证质量，这个设计可以作为 baseline 写进论文（记忆写入的延迟-质量权衡）。

**多路并行召回**比单一向量检索更鲁棒：图元数据（精确匹配）+ 向量（语义匹配）+ BM25（词频匹配）三路并行再 rerank，是实际系统中处理检索覆盖率的成熟思路，比"把向量检索当万能钥匙"更可靠。

**记忆分桶 + 容量预算**是最简单可控的遗忘策略：Working/LongTerm/User 各有配额，80% 阈值 FIFO 清理，不追求"智能"但足够稳定。

**系统复杂度与实现差距**是值得警惕的教训：MemOS 在论文层面架构完整，但代码里大量核心功能要么未实现要么被注释掉，却仍在评测中排名第一——说明当前 agent 记忆领域的 benchmark 可能主要测"明文记忆的精确性"，而不是论文里宣称的多类记忆联合管理能力。**这意味着如果你的 idea 专注在明文记忆的某个具体环节（抽取质量、检索准确率、遗忘策略），就能和这套系统直接竞争，而不必担心它的"三类记忆"理论框架。**
