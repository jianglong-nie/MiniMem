# MemGPT（Letta）

- 仓库：https://github.com/letta-ai/letta
- commit：1131535（本文代码行号均在该 commit 下有效）
- stars：23267（2026-06-11 查询）
- 论文：MemGPT: Towards LLMs as Operating Systems，arXiv:2310.08560，UC Berkeley，2024

**演变说明**：MemGPT 已商业化为 Letta 平台（"Letta (formerly MemGPT)"），功能大幅扩展（多 agent、服务端状态、计费等）。本文聚焦其记忆机制，论文原始设计与现代码实现的对应关系在"论文宣称 vs 代码实际"一节中逐条说明。

---

## 一、论文讲了什么故事

### 1.1 问题：LLM 只有一个有限的"短期记忆"

用 ChatGPT 聊天时，它只能记住当前这次对话的内容。一旦对话太长，或者开了新会话，之前说过的事就全忘了。这不是能力问题，而是架构限制：大语言模型（LLM）每次处理的文本长度有严格上限，叫做"上下文窗口"（context window）。

2023 年论文发表时，主流模型的上下文窗口大概是：
- Llama 2：4k tokens（约 60 条消息）
- GPT-4（原始版）：8k tokens（约 140 条消息）
- GPT-4 Turbo：128k tokens（约 2600 条消息）

"token"可以理解为约 0.75 个单词，一条普通聊天消息大概 50 个 token。8k 限制意味着 GPT-4 只能记住最近 140 条消息，更长的历史就被自动丢弃了。

这导致两类任务非常困难：（1）**长期对话伴侣**：跨多次会话记住用户的偏好、习惯、经历；（2）**长文档分析**：法律文件、财务报告动辄百万 token，远超任何模型上限。

### 1.2 已有方法的局限

当时的常见做法是"摘要"：把旧对话压缩成一段简短摘要，塞进新对话的开头。这样可以省空间，但有代价——摘要不可避免地丢失细节，而且模型看的是别人总结的内容，不是原始对话。如果你问 AI"上次我去夏威夷买了什么？"，摘要版的 AI 很可能给出错误答案，因为那个细节在压缩时被丢掉了。

另一个方向是扩展模型本身的上下文长度，但这会让计算量以"二次方"速度增长（即上下文长度翻倍，计算量变 4 倍），代价极高。而且即便做到了，研究发现超长上下文里模型对中间部分的内容反而利用效率很低（所谓"lost in the middle"问题）。

### 1.3 MemGPT 的核心思路：把 LLM 当成操作系统

这篇论文的出发点是一个类比：操作系统（OS）是怎么让电脑处理远超物理内存大小的数据的？

答案是"虚拟内存分页"：内存（RAM）是有限的快速存储，磁盘是无限的慢速存储。OS 自动在两者间搬运数据：常用的放内存，不常用的挪到磁盘，需要时再换回来。程序感觉上好像有无限内存，其实后面靠 OS 不断换页。

论文把这个思路搬到 LLM：
- **上下文窗口 = 物理内存（RAM）**：速度快，但容量有限，LLM 只能处理这里的内容
- **外部数据库 = 磁盘**：容量无限，但 LLM 无法直接"看见"，必须显式检索

关键创新是：**让 LLM 自己通过工具调用（function call）管理自己的记忆换页**，而不是让系统强制推管理。具体来说，给 LLM 配备若干"记忆工具"，让它能主动：
1. 把重要信息写入核心记忆（永远可见）
2. 把需要保留但暂不关键的信息存进外部数据库
3. 搜索外部数据库，把需要的信息取回上下文
4. 在给用户回复之前，连续调用多个工具（函数链）

这样，一个固定上下文窗口的模型就有了"无限记忆"的能力。

### 1.4 实验设置和结果

论文在两类任务上验证了效果：

**任务一：多会话聊天（Multi-Session Chat）**

使用 MSC（Multi-Session Chat）数据集，里面是真实用户扮演固定人设进行 5 轮对话的记录。论文新增了第 6 轮，用来测试：
- **深度记忆检索（DMR）**：问一个只有真正看过前 5 轮才能回答的具体问题（比如"上次你提到你们第一次见面是在哪里？"），考察一致性
- **对话开场白**：让 agent 基于积累的记忆主动写一个有个性的开场白，考察互动感

结果（DMR 任务）：
| 模型 | 准确率（无 MemGPT）| 准确率（+MemGPT）|
|---|---|---|
| GPT-3.5 Turbo | 38.7% | 66.9% |
| GPT-4 | 32.1% | 92.5% |
| GPT-4 Turbo | 35.3% | 93.4% |

没有 MemGPT 时，所有模型的基线是把 5 次会话摘要塞进上下文，仍然大幅落后。MemGPT 可以搜索原始完整对话记录，准确率大幅提升。

对话开场白任务中，MemGPT 生成的开场白与人工编写的参考答案相似度相当，部分指标超过了人类。

**任务二：文档分析**

把 2000 万条维基百科段落预先存入"外部记忆"（archival storage），让 MemGPT 自己搜索来回答问题。固定上下文的基线必须依赖一次性检索顶 K 个文档，但上下文放不下太多，准确率随文档数增加会下降（文档被截断）。MemGPT 可以反复翻页搜索，准确率基本不受文档总量影响。

**嵌套键值检索（Nested KV Retrieval）**：考察多跳检索能力——给一个键 A，它的值 B 本身又是另一个键，找到 B 的值 C，C 还是另一个键……如此嵌套 4 层。GPT-4 在 3 层就降到 0% 准确率，MemGPT+GPT-4 在 4 层仍然保持高准确率。

### 1.5 作者想证明什么

论文核心论点：通过**操作系统式的分层记忆 + LLM 自驱动的记忆管理**，可以在不改变模型本身的前提下，给固定上下文窗口的 LLM 提供"无限上下文"的幻觉。这个思路在长期对话和长文档分析两个场景里都有显著效果。

---

## 二、三层记忆架构：设计思路

MemGPT 的记忆分三层，对应 OS 的内存层次：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      上下文窗口（主存 / RAM）                            │
│                                                                         │
│  [系统指令]  [Core Memory / 工作记忆]  [FIFO 消息队列]  [外存元数据摘要] │
│  （静态，只读）（关键事实，可自编辑）  （最近对话，会溢出） （统计信息）  │
└─────────────────────────────────────────────────────────────────────────┘
                              ↕ 工具调用
┌─────────────────────────────────────────────────────────────────────────┐
│                         外部存储（磁盘）                                 │
│                                                                         │
│   Recall Storage（回溯记忆）        Archival Storage（档案记忆）         │
│   全部对话历史，自动记录             agent 主动存入的事实/段落            │
│   关键词/语义搜索                   向量语义搜索                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**三层的作用分工**：
1. **Core Memory（核心记忆）**：随时要用的、需要始终可见的关键信息。比如"用户叫 Alice，喜欢徒步，有个男友叫 James"。每次生成回复时 LLM 都能直接"看见"这里的内容，不需要检索。
2. **Recall Memory（回溯记忆）**：全部对话历史的数据库，自动保存，LLM 需要时可以搜索翻出来。类似"对话日志存档"。
3. **Archival Memory（档案记忆）**：LLM 主动存入的任意文本——可以是文档段落、事实摘要，也可以是自己整理的笔记。支持语义检索（用意思相似来找，而不是关键词精确匹配）。

**控制流**：LLM 输出的是普通文本，MemGPT 解析其中的函数调用指令并执行，结果反馈给 LLM。如果 LLM 在调用里加了 `request_heartbeat=true` 标志，系统会立刻再运行一次 LLM（而不是等用户下一条消息），允许 LLM 连续执行多步记忆操作再给用户回复——这叫"函数链"（function chaining）。

---

## 三、记忆条目设计：一条记忆实际长什么样

### 3.1 Core Memory Block（核心记忆块）

核心记忆由若干个"块"（Block）组成，每个块是一段自由文本。默认创建的 agent 有两个块：`human`（关于用户的信息）和 `persona`（agent 的人设）。

**数据结构**（`letta/schemas/block.py:13-78`）：
```
Block:
  id:          block-abc123          # 数据库唯一 ID
  label:       "human"               # 块的名字，决定它在 prompt 里的 XML 标签名
  value:       "Name: Alice\nBirthday: Feb 7\nBoyfriend: James"  # 实际内容，自由文本
  limit:       100000                # 字符数上限（仅展示，不硬性执行，见第六节）
  description: "Key information about the human user."  # 这个块是做什么用的
  read_only:   False                 # agent 是否可以编辑这个块
  tags:        []
```

这个块在进入 LLM 的 system prompt 之前，会被 `Memory.compile()` 方法（`letta/schemas/memory.py:688`）渲染成 XML 格式：

```xml
<memory_blocks>
The following memory blocks are currently engaged in your core memory unit:

<human>
<description>
Key information about the human user.
</description>
<metadata>
- chars_current=57
- chars_limit=100000
</metadata>
<value>
Name: Alice
Birthday: Feb 7
Boyfriend: James
</value>
</human>

<persona>
<description>
The persona of the agent.
</description>
...
</persona>
</memory_blocks>
```

这段 XML 是 system prompt 的一部分，所以 LLM 每次都能"看见"它，不需要检索。

**设计特点**：块里的 `value` 是完全无结构的自由文本，LLM 自己决定怎么组织内容（逐行写、子标题划分都行）。没有规定"用 JSON 格式存"或"按字段分行"——这提高了灵活性，也意味着内容质量完全依赖 LLM 的自觉性。

### 3.2 Archival Memory Passage（档案记忆段落）

当 LLM 调用 `archival_memory_insert` 工具时，传入的文本会被存成一个 `Passage`。

**数据结构**（`letta/schemas/passage.py:35-47`）：
```
Passage:
  id:               passage-xyz789
  text:             "User mentioned they first met James at Six Flags on 2023-10-12"
  embedding:        [0.023, -0.187, 0.341, ...]   # 1536 维浮点数向量，用于语义相似度计算
  embedding_config: {model: "text-embedding-3-small", dim: 1536, ...}
  created_at:       2024-01-15T14:23:00Z
  tags:             ["user-facts", "relationships"]   # 可选标签，用于过滤
  archive_id:       archive-123    # 归属哪个 agent 的档案库
  metadata:         {}
```

**什么是 embedding（嵌入向量）**：把文本转换成一串数字（比如 1536 个浮点数）的技术。语义相近的文本转换后得到的数字串"距离"也近（用余弦相似度量）。这样检索时可以用"意思相似"而不是"关键词匹配"来找相关内容。例如搜索"用户的感情生活"，能找到含有"James"和"boyfriend"的记录，即使没有用到"感情生活"这个词。

**写入过程**：文本不经过任何 LLM 提取或改写，原样存入（`passage_manager.py:566-567` 甚至有 TODO 注释说忘了检查 token 上限）。系统调用 embedding 模型把文本转成向量，然后存入 SQL 数据库，如果配了向量数据库（pgvector 或 Turbopuffer）则同步写入。写什么内容完全由 LLM 决定，系统不做任何格式化或提炼。

### 3.3 Recall Memory Message（回溯记忆消息）

每条对话消息（用户说的、agent 回的、工具调用和结果等）都自动保存到 `messages` 数据库表中。这里没有特殊的条目结构设计，就是原始消息记录：

```
Message:
  id:         msg-000111
  role:       "user"           # user / assistant / tool / system
  content:    "James and I actually first met at Six Flags"
  created_at: 2023-10-12T20:00:00Z
  agent_id:   agent-abc
  model:      "gpt-4"
```

Recall memory 的特点是：
- **自动记录**，LLM 不需要主动存，每轮对话都自动落库
- **永久保留**，即使消息从上下文窗口被"驱逐"，在数据库里仍然存在
- **被驱逐时不丢失**：当消息队列溢出，系统会把旧消息替换成摘要，但原始消息仍在数据库，可以通过 `conversation_search` 搜出来

---

## 四、一个具体例子：记忆从输入到写入的完整流程

假设用户说："James 和我分手了。"

```
用户输入: "actually james and i broke up"
          ↓
LettaAgentV3._step() 触发 LLM 推理（letta_agent_v3.py:895）
          ↓
LLM 读取 system prompt（里面包含 core memory 块）：
    <human>
    <value>
    Name: Alice
    Birthday: Feb 7
    Boyfriend: James   ← LLM 看到了"Boyfriend: James"
    </value>
    </human>
          ↓
LLM 决定：用户刚分手，应该更新 core memory。
LLM 输出（工具调用）:
    {
      "function": "memory_replace",
      "args": {
        "label": "human",
        "old_str": "Boyfriend: James",
        "new_str": "Ex-boyfriend: James"
      }
    }
          ↓
LettaCoreToolExecutor.execute() 执行工具（core_tool_executor.py:41-56）
          ↓
memory_replace() 校验 "Boyfriend: James" 在 human 块里唯一存在（core_tool_executor.py:381-391）
校验通过 → 修改内存中 Block.value
          ↓
update_memory_if_changed_async()（agent_manager.py:1747）
比对修改前后编译的字符串，确认有变化
→ block_manager.update_block_async() 写入数据库（block_manager.py:211）
          ↓
rebuild_system_prompt_async()（agent_manager.py:1523）
重新 Memory.compile() 生成新的 XML 内容
原地替换 message_ids[0]（即 system 消息），不追加新消息
          ↓
下次 LLM 调用时 system prompt 已更新：
    <value>
    Name: Alice
    Birthday: Feb 7
    Ex-boyfriend: James   ← 已更新
    </value>
          ↓
LLM 给用户回复: "Sorry to hear that - hope you're OK 💔"
```

整个过程：0 次额外 LLM 调用（编辑在当前这次 step 内完成），0 次 embedding 调用，修改立即生效，下一轮就能用新的记忆。

---

## 五、源码实现：系统如何运转

### 5.1 整体架构：服务端有状态 agent

Letta 的 agent 不是"无状态"的函数调用，而是持久化的服务端对象。每个 agent 的全部状态——blocks、消息历史、档案记忆——都在数据库里，每次处理用户消息时从数据库重建。

入口：用户消息进来 → `AgentLoop.load()`（`letta/agents/agent_loop.py:20-43`）根据 agent 类型选择处理器：
- 普通 agent（letta_v1 类型）→ `LettaAgentV3`（`agents/letta_agent_v3.py`）
- 启用了后台整理的 agent → `SleeptimeMultiAgentV4`（V3 的子类）

### 5.2 写入路径

**(a) Core Memory（核心记忆写入）**

LLM 调用记忆编辑工具时触发，两套工具族：
- **经典版**（`core_memory_append/replace`，`core_tool_executor.py:319/328`）：字符串追加/精确替换，原 MemGPT 论文的工具。
- **v2 版**（`memory_replace/insert/rethink`，`core_tool_executor.py:346/683/743`）：借鉴了 Anthropic 文本编辑器设计——`memory_replace` 要求 old_string 在整个块里唯一才允许替换（防止误改），否则报错并列出所有出现位置；`memory_rethink` 整块重写，适合大范围重组内容。

写完后立即触发 `rebuild_system_prompt_async()`（`agent_manager.py:1523`）重新编译 system prompt，并原地替换消息队列里的 system 消息（`agent_manager.py:1602-1608`），保证记忆更新对下一步 LLM 调用立即可见。

**成本：0 次额外 LLM 调用，0 次 embedding，在主 agent 当前 step 内完成。**

**(b) Archival Memory（档案记忆写入）**

LLM 调用 `archival_memory_insert` 工具 → `insert_passage()`（`passage_manager.py:543`）：
1. 把文本原样存为一个 Passage，**不做切分、不做摘要、不做格式化**
2. 调用 embedding 模型把文本转成向量（`passage_manager.py:574-579`）
3. 写入 SQL（PostgreSQL 或 SQLite），如果配了 Turbopuffer 同步双写
4. 写完后强制重建 system prompt，更新外存元数据里的"archival 共 N 条"计数

**成本：0 次额外 LLM 调用，1 次 embedding 调用。**

**(c) Recall Memory（回溯记忆写入）**

每条消息（用户的、agent 的、工具调用和结果）都自动落库，LLM 无需主动操作，**0 次 LLM 调用，0 次 embedding**（除非开了 `embed_all_messages` 配置才给消息打 embedding）。

**(d) Compaction（上下文溢出时的被动摘要）**

当消息队列太长、上下文窗口快满时触发。有两种触发条件：
- **报错触发**：LLM 推理时返回"context too long"错误，立刻压缩再重试（`letta_agent_v3.py:1218-1284`）
- **主动触发**：每步结束后检查，如果当前 token 估算 > 上下文窗口 × 0.9，主动压缩（`letta_agent_v3.py:1439-1505`；阈值在 `constants.py:83` 定义为 0.9）

压缩逻辑在 `compact_messages()`（`summarizer/compact.py:135`）：默认用"sliding window"模式，把最老的一批消息拿出来，调用一次 LLM 生成摘要（默认路由到便宜的小模型如 GPT-4o-mini 或 claude-haiku，`summarizer_config.py:26-32`），摘要以特殊的 `role=summary` 消息插在消息队列最前面，原始消息退出上下文但数据库里永久保留。

**成本：1 次 LLM 调用（用便宜小模型），不产生 embedding。**

这里有个重要的细节：论文里说接近上限时会给 LLM 发"内存压力警告"（memory pressure warning），让 LLM 主动把重要内容转移到 core memory 或 archival。**实际代码里这个机制已被废弃**：只存在于旧版 `agent.py:944-972`，新版 `LettaAgentV3` 完全没有，且 `settings.py:103` 里默认关闭。现在是系统静默地帮 LLM 压缩，LLM 不知道"要溢出了"。

**(e) Sleeptime 后台写入**

这是 Letta 新增的、论文里没有的机制（来自他们后来的 sleep-time compute 工作）：`SleeptimeMultiAgentV4`（`groups/sleeptime_multi_agent_v4.py`）在每次主 agent 完成处理后，异步启动一个后台 agent 来做记忆整理。

工作方式：把上次整理以来的所有对话 transcript 拼成一条消息，喂给一个独立的"sleeptime agent"（实际上是另一个 `LettaAgentV3`，共享相同的 memory blocks）。这个 agent 的工具集只有 `memory_replace/insert/rethink/finish_edits`（`constants.py:135-139`），它会多步编辑核心记忆，直到调 `memory_finish_edits` 工具表示整理完毕。

**成本：一整次 agent run，≥1 次 LLM 调用，通常多步。`enable_sleeptime` 默认关闭（`schemas/agent.py:318`）。**

这相当于"在线/离线"的记忆分工：主 agent 的在线路径只做低成本的滑窗摘要，语义级的记忆整合、精简、重组下放给离线的 sleeptime agent。

### 5.3 读取路径

**(a) Core Memory 读取**：零成本，始终在 system prompt 里，LLM 每次推理都能看见，不需要任何操作。

**(b) system prompt 里的外存提示**：`<memory_metadata>` 段（`prompt_generator.py:26-89`）会告诉 LLM "你有 recall memory 共 3841 条消息，archival memory 共 142 条，可用 tags: [user-facts, relationships]"。这是让 LLM 知道"外面还有更多内容可以搜索"的关键提示，对应论文里的 memory pressure 机制的信息感知部分。

**(c) Archival Memory 检索**：`archival_memory_search` 工具 → `query_agent_passages_async()`（`agent_manager.py:2416`）：
- 配了 Turbopuffer：向量检索 + FTS 全文检索，RRF 算法融合两个排名（`agent_manager.py:2457-2473`）
- 未配 Turbopuffer：用 pgvector 的余弦距离排序（`services/helpers/agent_manager_helper.py:1245`），SQLite 环境下用自定义余弦函数（`:1250-1255`）
- 默认取 top-5（`constants.py:458`）
- 支持 tags 过滤（any/all 模式）

**什么是"FTS 全文检索"和"RRF 融合"**：FTS（Full-Text Search，全文检索）是根据关键词精确匹配，速度快但找不到同义词。向量检索是根据语义相似性，能找同义词但精确词可能漏。RRF（Reciprocal Rank Fusion，倒数排名融合）把两个排名列表合并：某条内容在两个列表里都靠前，它在合并结果里就排更高，兼顾精确和语义。

检索结果作为工具返回值（JSON 字符串）进入消息流，格式例如：
```json
{
  "results": [
    {
      "id": "passage-xyz789",
      "timestamp": "2023-10-12 20:00 UTC",
      "content": "User mentioned they first met James at Six Flags",
      "tags": ["user-facts"]
    }
  ],
  "count": 1
}
```
这只是临时注入上下文，要永久保留需要 LLM 再次调工具写进 core 或 archival。

**(d) Recall Memory 检索**：`conversation_search` 工具 → `search_messages_async()`（`message_manager.py:1142`）：
- 配了 Turbopuffer + 开了 `embed_all_messages`：hybrid 语义+关键词
- **默认情况（未配 Turbopuffer）：SQL 的 `ILIKE '%query%'` 子串匹配**（`message_manager.py:978-993`），只能精确匹配关键词，无语义理解

这意味着工具文档声称的"hybrid 语义+全文检索"在默认自托管部署下并不生效，是付费云版功能。

### 5.4 维护路径

**去重**：无任何自动去重机制。Archival 插入时不查有没有相似内容；核心 block 编辑时也不检查相似度。唯一约束是 sleeptime prompt 用自然语言告诉 LLM "不要包含重复和过时信息"，靠 LLM 自觉。

**矛盾处理**：无自动矛盾检测。当用户说"James 和我分手了"，LLM 需要自己判断 core memory 里有冲突信息，并主动调工具修改。能不能及时发现和修正，完全取决于 LLM 的理解和指令遵循能力。

**遗忘**：无衰减、无打分、无自动清理。
- Core block：agent 可以调 `memory_delete`（实为 detach block，`core_tool_executor.py:778-806`）或用空字符串覆盖。
- Archival：**没有 agent 可调用的删除工具**（`function_map` 里没有 archival delete，`core_tool_executor.py:41-56`），工具文档明确说"persists indefinitely"（`functions/function_sets/base.py:176`）。
- Recall：消息从不真正删除，只有逻辑标记（`is_deleted == False` 过滤，`message_manager.py:944`）。

**Block 版本历史**：有 checkpoint/undo/redo 功能链（`block_manager.py:842/952/1004`），但只供管理端人工回滚，agent 自己不能用。

---

## 六、论文宣称 vs 代码实际

### 宣称与实现一致的部分

- **自编辑 core memory**：工具真实可用，立即触发重编译，下一步即可见。✅
- **递归摘要驱逐**：`compact.py:427-465` 的实现与论文 queue manager 描述基本一致。✅
- **被驱逐消息可从 recall 搜回**：消息永久在数据库，只是离开了上下文，搜索仍可找到。✅
- **Archival 向量检索**：pgvector 余弦相似度真实存在，不是摆设。✅

### 重要偏差

**1. Archival memory 已从默认工具集移除**

论文的核心机制之一是 `archival_memory_insert/search`，但现代码中这两个工具被列为 `DEPRECATED_LETTA_TOOLS`（`services/helpers/agent_manager_helper.py:1298-1302`，`constants.py:116`），默认不加入 agent 工具集。Sleeptime 工具集里 archival 相关工具也被注释掉（`constants.py:140-142`）。

**影响**：论文说的"两级换页"（core memory ↔ archival memory）如今是可选项，团队实践重心已转向 memory blocks + sleeptime。如果只安装默认配置，archival 功能不会自动启用。

**2. Memory pressure warning 已废弃**

论文宣称系统接近上限时给 LLM 发警告，让 LLM 主动决定转移什么内容。实际代码只在旧版 `agent.py:944-972` 里保留，新主路径 `LettaAgentV3` 没有这个机制，改为系统静默压缩。**LLM 不再主动意识到"内存快满了，该做决策了"**，这个自主性设计在实践中被简化掉了。

**3. Recall search 默认是字符串子串匹配，不是语义搜索**

工具文档写"hybrid search (text + semantic similarity)"（`functions/function_sets/base.py:96`），但需要 `use_tpuf + tpuf_api_key + openai_api_key + embed_all_messages` 全部开启才能生效（`helpers/tpuf_client.py:208-215`），而 `use_tpuf/embed_all_messages` 默认 False（`settings.py:442-445`）。默认自托管部署退化为 `SQL ILIKE '%query%'` 子串匹配。

**4. Block 字符上限不硬性执行**

论文把"core memory 有限"作为 agent 必须取舍的约束——满了就要主动清理。实际代码里 `chars_limit` 只是展示给 LLM 看的数字，系统不做任何检查（`block_manager.py:825-827` 直接赋值，无检查；`schemas/block.py:51-64` 验证器也不校验）。默认上限是 100000 字符（`constants.py:435`），基本不可能写满。

**5. 函数体 `raise NotImplementedError` 是误导**

`functions/function_sets/base.py` 里的 `memory`、`archival_memory_insert/search` 函数体是 `raise NotImplementedError`，看起来像"未实现"，但实际上这些函数只提供工具的 JSON Schema 描述，真正的实现在 `core_tool_executor.py` 里。这是一个代码结构上的坑，读码时别误判。

---

## 七、关键代码位置表

| 机制 | 文件路径 | 函数/类 | 行号 | 对应论文概念 |
|---|---|---|---|---|
| 核心工具分发入口 | `letta/services/tool_executor/core_tool_executor.py` | `LettaCoreToolExecutor.execute` | 41-56 | function executor |
| Core memory 追加/替换（经典） | 同上 | `core_memory_append / core_memory_replace` | 319/328 | working context 自编辑 |
| Core memory 精确编辑（v2） | 同上 | `memory_replace` | 346-401 | old_string 唯一性校验 |
| Core memory 整块重写 | 同上 | `memory_rethink` | 743 | 整合/重组 working context |
| Block 写入数据库 | `letta/services/agent_manager.py` | `update_memory_if_changed_async` | 1747 | 持久化 |
| System prompt 重编译 | `letta/services/agent_manager.py` | `rebuild_system_prompt_async` | 1523-1612 | 确保记忆更新立即可见 |
| Blocks 渲染为 XML | `letta/schemas/memory.py` | `Memory.compile` | 688-732 | working context 进入 prompt |
| 外存元数据提示 | `letta/prompts/prompt_generator.py` | `compile_memory_metadata_block` | 26-89 | 让 LLM 感知外存存在 |
| Archival 写入 | `letta/services/passage_manager.py` | `insert_passage` | 543-637 | archival storage 写入 |
| Archival 检索 | `letta/services/agent_manager.py` | `query_agent_passages_async` | 2416-2530 | archival_storage.search() |
| pgvector 余弦排序 | `letta/services/helpers/agent_manager_helper.py` | `build_agent_passage_query` | 1242-1255 | 语义相似度检索 |
| Recall 检索 | `letta/services/message_manager.py` | `search_messages_async` | 1142-1260 | recall_storage.search() |
| 子串匹配兜底 | `letta/services/message_manager.py` | `list_messages` | 977-993 | 默认 recall 搜索退化 |
| Compaction 触发 | `letta/agents/letta_agent_v3.py` | `_step`（溢出重试/步后检查） | 1218-1284 / 1439-1505 | queue manager 驱逐策略 |
| Compaction 实现 | `letta/services/summarizer/compact.py` | `compact_messages` | 135-472 | recursive summary |
| Sleeptime 编排 | `letta/groups/sleeptime_multi_agent_v4.py` | `run_sleeptime_agents` | 132-168 | 论文 sleep-time compute |
| Agent 类型路由 | `letta/agents/agent_loop.py` | `AgentLoop.load` | 20-43 | 选择 V3/Sleeptime 处理器 |

---

## 八、值得借鉴的地方

**1. System prompt 原地重编译，不追加消息**

记忆更新后不是在消息队列里追加"[MEMORY_UPDATE]"之类的消息，而是直接替换 system 消息（`message_ids[0]`），并且比较编译前后字符串，没变化就不写数据库（`agent_manager.py:1562-1567`）。这样做的好处：
- 记忆更新不会"污染"对话历史，消息队列里看不到记忆操作的痕迹
- 对 LLM 来说，记忆就像是一直"在那里"，而不是"刚被修改了"
- 兼容 prefix cache（因为 system prompt 是消息的第一条，保持稳定有利于缓存命中）

**2. `memory_replace` 的唯一性强制 + 行号只读视图防幻觉**

老问题：让 LLM 编辑自己的记忆时，它可能"幻觉"出一个不存在的 old_string，导致替换失败或替换错地方。Letta 的解决方案：
- 要求 old_string 在整个块里只出现一次，否则报错并列出所有位置（`core_tool_executor.py:381-391`）
- 渲染时给每行加行号（`1→ ...`，`2→ ...`），但校验时严格拒绝包含行号的 old_string（`:357-368`）
- 行号是"帮你定位"的提示，不是"你要输入的内容"的一部分

这是一个工程上很扎实的防幻觉设计，把"LLM 可靠地编辑文本"这个难题分解为"提供位置信息 + 强制唯一性校验 + 报错引导"。

**3. 在线/离线记忆分工**

这是整个 Letta 演化的核心设计决策：
- **在线路径**：主 agent 处理用户请求时，只做必要的低成本操作（记忆工具调用是当前 step 的副产品，compaction 用最便宜的小模型）
- **离线路径**：sleeptime agent 在对话结束后异步跑，不影响响应延迟，可以用更多步骤、更好的 LLM 来做高质量的记忆整理

这种"写路径延迟容忍"的分工值得在自己的系统里借鉴：不是所有记忆操作都需要同步完成，语义级整合可以异步、按频率做。

---

## 九、已知的粗糙之处

- **Archival 不切块**：不管文本多长，一整段存成一个 Passage（`passage_manager.py:566-567` 的 TODO 说"应该检查 token 数但还没做"）。长文本 embedding 可能被 embedding 模型截断，导致向量不能代表完整内容。
- **Archival 无去重**：反复存语义相近的事实会稀释检索质量，没有机制阻止这一点。
- **SQL 路径 tag 过滤低效**：先取回 top-k 结果再在内存里按 tag 过滤（`agent_manager.py:2507-2527`），如果带 tag 的结果在 top-k 之外就会漏掉。作者自己留了 TODO。
- **代码历史包袱重**：仓库里同时存在 4 代 agent loop（`agent.py/v1/v2/v3`）、4 代 sleeptime 编排、3 套记忆工具集、新旧两套 summarizer，`rebuild_system_prompt_async` 上方有作者自评注释"这可能是我写过最烂的代码"（`agent_manager.py:1519-1520`）。
