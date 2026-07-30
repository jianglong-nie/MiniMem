# A-MEM：Agentic Memory for LLM Agents

- 仓库：https://github.com/WujiangXu/A-mem-sys（系统实现）；https://github.com/WujiangXu/AgenticMemory（论文复现）
- 本地路径：`research/A-mem-sys`（commit f303dfc）；`research/AgenticMemory`（commit 0c8039f）
- 论文：arXiv:2502.12110，已收录 NeurIPS 2025
- 作者：Xu Wujiang 等，Rutgers University

---

## 一、论文讲了一个什么故事

### 背景：给 LLM Agent 加记忆，现有方法哪里不够

LLM（大语言模型）本身没有持久记忆：每次对话结束后，它对之前发生过什么一无所知。为了让 agent 能利用历史经验，通常会给它配一个外部记忆库——把历史对话或事实存进去，回答问题时先检索再作答。

最常见的做法是把每句话直接存入**向量数据库**（一种能根据语义相似度快速查找的数据库——可以理解成"能理解意思的搜索引擎"，不只是关键词匹配）。检索时，用当前问题去找语义最相似的历史记录，然后把结果塞进 LLM 的输入里让它参考。

这个做法有两个根本问题：

- **每条记忆是孤立的文本片段**。系统不知道"这条记忆和那条记忆有关联"，没有人去维护记忆之间的关系网络。
- **元数据缺失**。纯文本只能靠字面语义检索，处理"需要综合多条对话才能推断出答案"的问题时效果很差。

### A-MEM 的想法：把 LLM 变成知识管理员

论文的灵感来自 **Zettelkasten（卡片盒笔记法）**——德国社会学家卢曼发明的方法，他几十年里用这套方法写出了海量学术著作。核心思想是：每条笔记是一张独立的卡片，卡片上除了内容，还有关键词、主题摘要、分类标签；更重要的是，卡片之间相互链接——新卡片入库时，你要思考它和哪些旧卡片有关联，并建立双向链接。整个卡片盒因此形成一个互相连接的知识网络。

A-MEM 把这个方法引入到 LLM 记忆系统：

> **每条记忆入库时，让 LLM 自动给它生成元数据（关键词、摘要、标签），判断它和哪些旧记忆有关联，并反过来更新旧记忆的描述。**

论文把这套系统概括为三步：

1. **Note Construction（笔记构建）**：LLM 读取新记忆的原始内容，生成三类元数据——keywords（关键词）、context（一句话摘要）、tags（分类标签）。
2. **Link Generation（链接生成）**：向量检索找到最相关的旧记忆，LLM 判断新记忆应该和哪些旧记忆建立连接（strengthen 操作）。
3. **Memory Evolution（记忆进化）**：同一次 LLM 调用中，LLM 还可以决定更新旧记忆的 context 和 tags（update_neighbor 操作），让旧记忆在有了新邻居后"进化"出更准确的描述。

---

## 二、一条记忆长什么样

### 数据结构

A-MEM 里，一条记忆是一个结构化的**笔记对象**（`MemoryNote`），不是裸文本，也不是 QA 问答对，而是这样的结构（[memory_system.py:24-81](../../research/A-mem-sys/agentic_memory/memory_system.py#L24-L81)）：

```
id:               "a1b2c3d4-5678-..."     # UUID，系统自动生成的唯一标识
content:          "Speaker Alice says: I started working at Google last March"
keywords:         ["Google", "employment", "March"]      # LLM 提取的关键词
context:          "Alice mentioned starting a new job at a tech company"  # LLM 生成的一句话摘要
tags:             ["career", "employment", "technology"] # LLM 生成的分类标签
links:            ["旧记忆uuid1", "旧记忆uuid2"]          # 和哪些旧记忆有关联（进化阶段填入）
timestamp:        "202403010900"           # 写入时间，格式 YYYYMMDDHHmm
last_accessed:    "202403010900"           # 最近访问时间（实际上从不更新，是死字段）
retrieval_count:  0                        # 检索次数（同样从不更新）
evolution_history: []                      # 进化历史（同样从不写入）
```

原始输入是一句对话文本（比如 "Speaker Alice says: ..."），写入后才附带 LLM 生成的三类元数据和系统维护的时间戳。

**有三个字段要特别注意**：`last_accessed`、`retrieval_count`、`evolution_history` 在创建时赋初值，但代码里没有任何地方更新它们。它们是论文设计上有（暗示着遗忘机制和访问频率跟踪），但实际从未生效的死字段。

### 写入前：内容经历了哪些变换

**第一步——LLM 生成元数据**

原始对话文本被送给 LLM，提取 keywords + context + tags。这让记忆从"一句话"变成"一张有标签的笔记"。

**第二步——元数据增强嵌入（enhanced embedding）**

嵌入（embedding）是把文本转换成向量的过程——可以理解为把文字变成一串数字，让计算机能比较两段文字的语义相似度。

普通做法只把原始内容做嵌入。A-MEM 的做法是把内容和元数据拼接在一起再做嵌入（[retrievers.py:38-55](../../research/A-mem-sys/agentic_memory/retrievers.py#L38-L55)）：

```
enhanced_document = "Speaker Alice says: I started working at Google last March"
                  + " context: Alice mentioned starting a new job at a tech company"
                  + " keywords: Google, employment, March"
                  + " tags: career, employment, technology"
```

这段拼接文本才是真正被向量化、存入数据库的内容。

**为什么这样设计**：检索时，关键词、摘要、分类标签都参与了语义相似度计算，而不只是字面内容。代价是：如果 LLM 生成的元数据不准确，向量也会偏离真实含义。

### 两个仓库对 links 的存储方式不同

- **A-mem-sys（系统实现仓）**：`links` 存的是 **UUID 字符串**——每条旧记忆的唯一 id。即使记忆库有增删，UUID 仍能准确定位那条记忆。
- **AgenticMemory（论文复现仓）**：`links` 存的是**整数下标**——记忆在 `list(self.memories.values())` 中的位置。如果记忆库顺序发生变化，下标会错位，链接会指向错误的记忆。

---

## 三、实验：论文想证明什么

### 数据集和评测方式

实验使用 **LoCoMo**（Long-term Conversational Memory）数据集，包含多人之间的长期对话和基于对话内容的问答题，分5类：

| 类型 | 考察点 |
|------|------|
| Multi-hop（多跳推理） | 需要综合多条对话才能回答 |
| Temporal（时间推理） | "这件事发生在什么时候" |
| Open-domain（开放域） | 开放性问答 |
| Single-hop（单跳） | 一条对话里就有答案 |
| Adversarial（对抗性） | 考验系统不乱编（答案可能是"对话中没有提到"） |

评测流程：把一段对话逐条存入记忆系统 → 给出问题 → 系统检索记忆并拼成上下文 → LLM 根据上下文作答 → 和标准答案比较。评估指标包括 Exact Match、F1 分数、ROUGE、BERT F1 等。

### 结果，以及一个严重的前提问题

论文称在6个基础模型（包括 GPT-4o-mini 等）上，A-MEM 在各类别问答上都超过了现有系统（MemGPT、Zep、Mem0、ReadAgent 等）。

**但有一个绕不过去的问题**：论文的评测代码在 AgenticMemory 仓库的 `test_advanced.py` 里，它使用的是 `memory_layer.py`——也就是存在严重代码缺陷的那个版本（见下文"论文与代码的偏差"第1点）。

这意味着：论文报告的实验数字，是在 **Note Construction（笔记构建）步骤实质上失效**的情况下跑出来的——所有记忆都以空关键词、无标签、默认 context（"General"）存储。论文宣称的"带进化的记忆系统"，在实验时实际上接近于"纯文本存储"。

能否在修复 bug 后复现论文数字，目前无法核实。这也引出了一个有意思的研究问题：既然 Note Construction 没工作，A-MEM 的胜出究竟来自哪里？

---

## 四、源码是怎么运转的

### 整体架构

A-mem-sys 只有三个核心文件：

```
agentic_memory/
├── memory_system.py   # 核心：AgenticMemorySystem + MemoryNote
├── retrievers.py      # 向量检索：ChromaRetriever（封装 ChromaDB）
└── llm_controller.py  # LLM 后端：OpenAI / Ollama / SGLang / OpenRouter
```

记忆同时存在两个地方（双写）：
- `self.memories`（Python 字典，id → MemoryNote）：权威副本，存完整的 MemoryNote 对象，随着进化实时更新
- ChromaDB 内存集合：存 enhanced_document 的向量 + 序列化的 metadata

ChromaDB 是一个向量数据库，这里用的是内存模式（`chromadb.Client(Settings(allow_reset=True))`），进程退出后数据全丢，没有持久化到磁盘。

---

### 写入流程：一条对话消息完整走一遍

以一条实际的评测输入为例：

> 原始输入："Speaker Alice says: I started working at Google last March"（时间戳："202403010900"）

**Step 1：创建 MemoryNote 对象** ([memory_system.py:237-242](../../research/A-mem-sys/agentic_memory/memory_system.py#L237-L242))

用原始文本和时间戳创建笔记对象，keywords/context/tags 默认为空 / "General" / 空。

```python
note = MemoryNote(content="Speaker Alice says: ...", timestamp="202403010900")
# → note.keywords = []
# → note.context = "General"
# → note.tags = []
```

**Step 2：LLM 生成元数据（analyze_content）** ([memory_system.py:244-261](../../research/A-mem-sys/agentic_memory/memory_system.py#L244-L261))

检查：keywords 是否为空、context 是否是默认值 "General"、tags 是否为空。只要有任何一个条件成立，就调用 `analyze_content()` 让 LLM 生成。

LLM 接到的 prompt 大意是"分析这段内容，提取关键词、写一句话摘要、给出分类标签，以 JSON 格式返回"。A-mem-sys 使用 `response_format={"type": "json_schema", ...}` 强制 LLM 返回结构化 JSON（这是 OpenAI API 的一个功能，能保证输出格式是干净的 JSON，不会带额外文字）。

LLM 返回：
```json
{
  "keywords": ["Google", "employment", "March"],
  "context": "Alice mentioned starting a new job at a tech company",
  "tags": ["career", "employment", "technology"]
}
```

笔记对象的三个字段被填入。

**Step 3：记忆进化决策（process_memory）** ([memory_system.py:625-754](../../research/A-mem-sys/agentic_memory/memory_system.py#L625-L754))

1. 用当前笔记的 content 作为查询，去 ChromaDB 里找5条最相关的旧记忆（向量检索）
2. 把这5条旧记忆的内容和元数据格式化成文本，连同新笔记一起喂给 LLM：

   ```
   新记忆 content: "Speaker Alice says: I started working at Google..."
   新记忆 context: "Alice mentioned starting a new job..."
   新记忆 keywords: ['Google', 'employment', 'March']
   
   5条近邻记忆（每条格式如下）：
   memory_id:abc123  talk start time:202302010900  memory content: ...  memory context: ...
   memory_id:def456  ...
   ```

3. LLM 返回进化决策：

   ```json
   {
     "should_evolve": true,
     "actions": ["strengthen", "update_neighbor"],
     "suggested_connections": ["abc123"],
     "tags_to_update": ["career", "tech_company"],
     "new_context_neighborhood": ["updated context for neighbor 1", ...],
     "new_tags_neighborhood": [["tag1", "tag2"], ...]
   }
   ```

4. 按决策执行（[memory_system.py:711-743](../../research/A-mem-sys/agentic_memory/memory_system.py#L711-L743)）：
   - `strengthen`：把 `suggested_connections` 里的 UUID 追加进新笔记的 `links` 字段，并用 `tags_to_update` 覆盖新笔记的 tags
   - `update_neighbor`：直接修改各旧记忆在 `self.memories` 里的 tags 和 context（**只改内存 dict，不同步到 ChromaDB**——这是关键缺陷，见"偏差"章节）

**Step 4：写入内存 dict** ([memory_system.py:268](../../research/A-mem-sys/agentic_memory/memory_system.py#L268))

```python
self.memories[note.id] = note
```

**Step 5：写入 ChromaDB** ([memory_system.py:270-284](../../research/A-mem-sys/agentic_memory/memory_system.py#L270-L284))

先拼接 enhanced_document，再调 ChromaDB 的 add 接口（ChromaDB 内部用 SentenceTransformer 做向量化）：

```
"Speaker Alice says: I started working at Google last March
 context: Alice mentioned starting a new job at a tech company
 keywords: Google, employment, March
 tags: career, employment, technology"
```

**注意写入顺序**：Step 3（进化）在 Step 5（入库）之前完成。所以新笔记被 `strengthen` 修改过的 tags，能进入自己的 enhanced embedding。但 `update_neighbor` 修改的旧记忆，**不会**重新生成向量——旧记忆在 ChromaDB 里的向量是写入当时的快照，进化后的新 context/tags 不会反映在检索时的相似度计算中。

**每条记忆写入消耗的资源：**
- LLM API 调用：2次（analyze_content 1次 + process_memory 进化决策 1次）
- 本地 SentenceTransformer embedding（不走 API）：1次查近邻 + 1次写入 Chroma = 2次
- 若 keywords/context/tags 已有值，则跳过 analyze_content，省去1次 LLM 调用
- 记忆库为空时（第一条记忆），跳过 process_memory，省去1次 LLM 调用

---

### 检索流程：问题是怎么得到答案的

评测代码（AgenticMemory 仓库的 `test_advanced.py`）中，完整的问答流程如下（[test_advanced.py:95-196](../../research/AgenticMemory/test_advanced.py#L95-L196)）：

**第一步**：LLM 把问题扩展成关键词（`generate_query_llm`，第95-125行，在第129行被调用）

```
输入问题："Where does Alice work?"
LLM 返回："Alice cosmos work cosmos employer cosmos company cosmos job"
（关键词用 'cosmos' 分隔成一个字符串）
```

**第二步**：用这段关键词字符串调 `find_related_memories_raw` 检索（`retrieve_memory`，第60-61行，在第134行被调用）

ChromaDB 会把这段关键词字符串向量化，然后找最相似的记忆。`find_related_memories_raw` 的输出是一段拼接文本（[memory_system.py:345-374](../../research/A-mem-sys/agentic_memory/memory_system.py#L345-L374)）：

```
talk start time:202403010900  memory content: Speaker Alice says: I started working at Google...
memory context: Alice mentioned starting a new job...
memory keywords: ['Google', 'employment', 'March']
memory tags: ['career', 'employment']
talk start time:202302010900  memory content: Speaker Alice says: I love technology...  ...
```

如果某条记忆有 `links`，还会把链接到的旧记忆内联追加在后面（从 `self.memories` 取最新值，包含进化后的 context/tags）。

**第三步**：把这段文本塞进 QA prompt 的 Context 段，让 LLM 根据上下文作答（第140-145行）

---

### 四个检索接口，各有什么区别

| 方法 | 用途 | 实际行为 |
|---|---|---|
| `find_related_memories(query, k)` | 进化流程内部 | 向量检索 top-k，从 Chroma metadata 取值，返回 (str, ids) |
| `find_related_memories_raw(query, k)` | 评测/prompt 构建 | 向量检索 top-k，主结果读 Chroma 旧值，链接邻居读 `self.memories` 新值，内联拼接 |
| `search(query, k)` | 对外 API | 向量检索 top-k，元数据从 `self.memories` 取（含进化后新值） |
| `search_agentic(query, k)` | 带链接扩展 | 向量检索 top-k + 追加 links 邻居，但最终 `[:k]` 截断导致邻居几乎必出局 |

`search()` 的 docstring 写着 "hybrid retrieval"（混合检索——通常指向量检索和关键词检索两路并用），但代码只有向量检索。BM25（一种基于词频的关键词检索算法）相关的 `from rank_bm25 import BM25Okapi` 被 import 进来（[memory_system.py:9](../../research/A-mem-sys/agentic_memory/memory_system.py#L9)）但从未被任何函数调用。

---

### 没有实现的功能

- **无去重**：相同内容写两次 `add_note` 会生成两条独立记忆，没有相似度去重
- **无矛盾处理**：Alice 今天说"我在北京"，明天说"我在上海"，两条都存入，不做冲突检测
- **无遗忘机制**：`retrieval_count` 和 `last_accessed` 初始化后从不更新（[memory_system.py:77-80](../../research/A-mem-sys/agentic_memory/memory_system.py#L77-L80)），没有基于访问频率或时间的淘汰
- **无持久化**：ChromaDB 用内存模式，进程退出数据全丢

---

## 五、论文宣称与代码实际的偏差

这是这篇论文最值得注意的部分，直接影响你怎么理解和引用它。

### 偏差1：AgenticMemory（论文复现仓）的 Note Construction 永远失效

**原因**：`memory_layer.py` 第380行调用了 `re.sub(...)`，但整个文件没有 `import re`。

更值得注意的是，这个 `re.sub` 本来就是**多余的**：LLM 调用（第339行）使用了 `response_format={"type": "json_schema", ...}`——这是 OpenAI 的结构化输出功能，保证 LLM 返回的就是干净的 JSON，不会带 markdown 代码块。代码本可以直接 `json.loads(response)`，但有人加了一步"去除 markdown 代码围栏"的 `re.sub`——偏偏 `re` 没有导入。

**错误链**：
1. `re.sub(...)` → `NameError: name 're' is not defined` → 进入内层 `except:` 块（第382行）
2. 内层 except 里 `print(f"...{e}")` → 变量 `e` 在此作用域未定义 → `NameError` 再次抛出
3. 被最外层 `except Exception as e:` 兜住 → 返回 `{"keywords": [], "context": "General", "tags": []}`

**结果**：所有记忆都以空关键词、默认 context、无标签存储，Note Construction 从未正常运行。

**对论文的影响**：论文展示的实验数字，是在 Note Construction 失效（相当于无元数据的纯文本记忆）的条件下跑出来的。A-mem-sys（系统实现仓）修复了这个问题：`analyze_content` 直接解析结构化输出，没有 `re.sub`。

---

### 偏差2：`update_neighbor` 只改内存 dict，不改 ChromaDB

`update_neighbor` 操作会修改 `self.memories` 里的旧记忆对象（tags 和 context 被直接赋新值），但**不更新 ChromaDB 里该记忆的向量和 metadata**（[memory_system.py:719-743](../../research/A-mem-sys/agentic_memory/memory_system.py#L719-L743)）。

正确的做法应该是调 `update()` 方法（[memory_system.py:387-426](../../research/A-mem-sys/agentic_memory/memory_system.py#L387-L426)）——这个方法先 delete 再 add，能同步更新 ChromaDB。但 `update_neighbor` 没有这样做。

**实际后果**：进化修改了旧记忆的 context/tags，但 ChromaDB 里的向量仍是写入当时生成的，检索排序不受任何影响。通过 `search()` 和 `find_related_memories_raw`（链接邻居部分）得到的 context/tags 文本是新值，但哪条记忆被检索到这件事，完全由写入时的原始向量决定。

**`consolidate_memories()` 是补救机制，但两个仓库行为完全不同**（[memory_system.py:292-312](../../research/A-mem-sys/agentic_memory/memory_system.py#L292-L312)）：

代码每 100 次进化后调用一次 `consolidate_memories()`，意图是重建向量索引。

A-mem-sys 的写法：
```python
def consolidate_memories(self):
    # Reset ChromaDB collection（注释写着 reset，但没有调 client.reset()）
    self.retriever = ChromaRetriever(collection_name="memories", model_name=self.model_name)
    for memory in self.memories.values():
        self.retriever.add_document(memory.content, metadata, memory.id)
```

问题：`ChromaRetriever.__init__` 使用 `get_or_create_collection`，如果 `chromadb.Client()` 在同一进程内共享内存存储（证据：`AgenticMemorySystem.__init__` 明确调用了 `temp_retriever.client.reset()` 才创建真正的 retriever——如果客户端之间真正独立，这个 reset 就毫无意义），那新建的 `ChromaRetriever` 拿到的是同一个已满的 collection，用相同 ID 再 add 就是 no-op 或报错，consolidate 什么都没做。

对比：AgenticMemory 原仓的 `consolidate_memories`（[memory_layer.py:729-751](../../research/AgenticMemory/memory_layer.py#L729-L751)）用的是 `SimpleEmbeddingRetriever`——这是一个自定义的纯 Python 类，`SimpleEmbeddingRetriever(model_name)` 每次创建的都是全新空实例，然后从 `self.memories` 重新 add 所有记忆。**原仓的 `consolidate_memories` 确实能重建索引，A-mem-sys 的版本可能是 no-op。**

即便 A-mem-sys 的 `consolidate_memories` 有效，它每 100 次进化才触发一次，前 99 次进化的结果在检索层面仍然是不可见的。

---

### 偏差3：`search_agentic` 的链接扩展是摆设

`search_agentic` 先取 top-k 主结果，再遍历 links 追加邻居，最后返回 `memories[:k]`（[memory_system.py:620](../../research/A-mem-sys/agentic_memory/memory_system.py#L620)）——只要主结果已满 k 条，所有追加的邻居都被截断，links 形同虚设。links 只在 `find_related_memories_raw` 里真正被内联到检索结果（但主结果读的是 ChromaDB 里存的旧 metadata，不是进化后的最新值）。

---

### 偏差汇总

| 论文宣称 | 代码实际情况 | 影响 |
|---|---|---|
| Note Construction 生成语义元数据 | AgenticMemory 原仓永远失效；A-mem-sys 正常 | 原仓实验结果基于空元数据 |
| Memory evolution 更新旧记忆 | 只改 `self.memories`，不改 ChromaDB | 进化对向量检索排序无即时影响 |
| consolidate 定期重建索引 | A-mem-sys 可能是 no-op；AgenticMemory 原仓确实有效 | A-mem-sys 进化结果可能永远进不了向量索引 |
| Hybrid retrieval | 纯向量检索，BM25 import 了但从未使用 | 名不副实 |
| Links 在检索时扩展邻居 | `search_agentic` 截断；仅 `find_related_memories_raw` 有部分效果 | 链接扩展实际覆盖面很有限 |
| retrieval_count/last_accessed 跟踪使用 | 初始化后从不更新 | 无法支撑遗忘机制，是死字段 |

---

## 六、值得借鉴和追问的地方

### 值得借鉴

**元数据增强 embedding**（[retrievers.py:38-55](../../research/A-mem-sys/agentic_memory/retrievers.py#L38-L55)）：把 LLM 生成的 context/keywords/tags 拼进原始内容再 embed，让抽象语义参与向量检索。实现成本极低（几行字符串拼接），原理上能提升检索精度，可以直接用在其他系统上。

**进化 prompt 的工程设计**（[memory_system.py:131-161](../../research/A-mem-sys/agentic_memory/memory_system.py#L131-L161)）：用 JSON schema 强约束 LLM 输出结构，要求 `new_tags_neighborhood` 的数组长度等于邻居数，要求按输入顺序对位。这是 LLM 批量操作多个对象时保证对齐的实用做法。

**用 UUID 而非整数下标定位记忆**（A-mem-sys 对 AgenticMemory 的改进）：原仓用 `list(self.memories.values())[indices[i]]`（按全局插入顺序的整数下标），记忆库动态变化时下标极易错位；A-mem-sys 改成了真实的 UUID。这体现了 LLM 操作动态数据结构时"引用稳定性"的重要性。

### 值得追问的研究问题

1. **Note Construction 失效时系统仍能赢，这说明了什么？** 论文在无元数据条件下仍超过基线，意味着 A-MEM 的胜出可能主要来自更好的检索 prompt 设计或 memory evolution 的链接结构，而不是元数据本身。这值得单独用实验拆开验证。

2. **进化真的有用吗？** 在 A-mem-sys 里，进化结果很可能进不了向量检索索引（consolidate 可能是 no-op）。如果彻底修复这些 bug，让进化结果真正影响检索，效果会提升还是下降？这是一个没被验证的假设，可以作为实验切入口。

3. **每条写入调2次 LLM，成本值吗？** 对话很长时，写入延迟会是瓶颈。论文没有报告写入延迟或 LLM 调用次数对比。

4. **无去重、无矛盾处理是假设还是遗漏？** 对话系统里，同一信息会被重复说，矛盾信息也很常见。A-MEM 完全没处理这些情况，这是一个明确的改进方向。

5. **元数据生成质量没有独立评估**：论文只看最终 QA 分数，没有单独验证 keywords/context/tags 的质量。如果 LLM 生成的元数据质量差，增强 embedding 可能引入噪声而非提升精度。
