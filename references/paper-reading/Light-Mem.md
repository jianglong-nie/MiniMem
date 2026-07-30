# LightMem：轻量高效的记忆增强生成

- 论文：LightMem: Lightweight and Efficient Memory-Augmented Generation（ICLR 2026，arXiv:2510.18866）
- 仓库：https://github.com/zjunlp/LightMem（commit 579ee76，stars 918，2026-06-11）
- 机构：浙江大学 / 南京大学 / 新加坡国立大学

---

## 一、论文的故事：它为什么存在？

### 背景

LLM（大型语言模型，也就是 ChatGPT 这类 AI）有一个根本性的限制：它们是"无状态"的。每次你打开新对话，它对你上次说的话一无所知。如果你跟 AI 聊了一个小时，告诉它你叫 Alice、你在学钢琴、你下周要去东京，然后下次打开对话继续问，它完全忘了这些。

为了解决这个问题，研究界开发了"记忆系统"（Memory System）：把对话历史里的关键信息提炼、存储到外部数据库，下次对话时先去检索相关记忆，再拼给 AI 作为背景知识。这类系统已经有不少：Mem0、A-MEM、MemoryOS 等。

### 已有方法的三个痛点

LightMem 认为现有记忆系统太"重"了，具体体现在：

**痛点一：原始数据太噪声。** 用户的输入里充斥着大量废话。比如"嗯嗯好的我知道了，那你说的是啊……我想问一下，就是关于……"这类填充词对后续记忆构建毫无帮助，但现有系统把它们原样喂给 LLM 处理，浪费 token（token 是 LLM 按量计费的单位，可以理解为"字数钱"）。

**痛点二：切分粒度不合理。** 现有系统要么按"每一轮"切（太细，LLM 调用次数多），要么按"整个 session"切（太粗，把不同话题混一起，AI 抽取的记忆乱七八糟）。比如你在一次对话里先聊了工作，又聊了旅游，又聊了饮食，如果把这三个话题的消息打包成一个任务让 AI 提炼记忆，它很可能输出一些混杂不清的结果。

**痛点三：在线实时更新太慢。** 用户每说一句话，现有系统就立刻调用一次 LLM 来更新记忆（"今天的东京计划和上次的东京计划矛盾吗？要删除旧记忆吗？"）。这个仲裁过程既慢又贵，而且是串行的——必须一条条处理完才能进行下一轮对话。

### LightMem 的核心想法

作者从人类记忆的 **Atkinson–Shiffrin 模型**（一个心理学经典理论）获得灵感。这个模型把人类记忆分为三级：

- **感觉记忆（Sensory Memory）**：感官信息进来后在最短时间内被自动过滤，不重要的马上丢掉（例如你看到路边一棵普通的树，几秒内就忘了）
- **短期记忆（Short-Term Memory）**：少量重要信息暂时保留，攒到一定量后进入长期记忆
- **长期记忆（Long-Term Memory）**：睡觉时大脑会整理当天信息，合并矛盾的记忆、遗忘不重要的——这就是"睡眠巩固"

LightMem 把这套机制搬到 AI 记忆系统里：
1. 用一个**本地小模型**（LLMLingua-2，一个只有 BERT 大小的模型，不需要 API，本地跑，不花钱）先过滤噪声 → 对应感觉记忆
2. 过滤后的内容**按话题攒批**，满了才调一次 LLM → 对应短期记忆
3. 新记忆先**直接存入数据库**（不做任何冲突处理），等对话结束后"睡眠期"再批量离线整合 → 对应长期记忆

### 实验证明了什么？

论文在两个权威评测集上对比了六个基线方法：

**LongMemEval**（500 道题，模拟 ~115k token 的超长对话）：
- 准确率比最强基线（A-MEM）高 **2%–7.7%**
- 总 token 消耗少 **10×–38×**（算上离线更新）
- API 调用次数少 **3.6×–30×**
- 如果只看在线实时成本，token 省 **105×**，API 调用少 **159×**

**LoCoMo**（现实长对话评测）：
- 准确率高 **6%–29%**
- 总 token 少 **3×–21×**，API 调用少 **13×–55×**

核心结论：**更便宜、更快、效果还更好**。这是因为 topic 分组让每次 LLM 调用处理的上下文更集中、语义更纯粹，减少了"混杂话题导致的记忆质量下降"。

---

## 二、一条记忆长什么样？

### 记忆条目（MemoryEntry）的结构

LightMem 里一条记忆是一个**事实句**，加上一堆元数据。定义在 [src/lightmem/memory/utils.py:13-32](../../research/LightMem/src/lightmem/memory/utils.py#L13-L32)：

```python
@dataclass
class MemoryEntry:
    id: str                # 唯一标识符（UUID）
    time_stamp: str        # ISO 格式时间戳，如 "2023-05-20T00:44:00.000"
    float_time_stamp: float # 时间戳的浮点版本，方便数值比较
    weekday: str           # 星期几，如 "Sat"
    memory: str            # 核心内容：一条事实句，如 "User is planning a trip to Tokyo."
    topic_id: int          # 这条记忆属于哪个话题段（用整数编号）
    speaker_id: str        # 说话人 ID
    update_queue: List     # 睡眠期维护用：记录哪些更新者可能覆盖这条记忆
    consolidated: bool     # 是否已被 summarize() 整合过
    # 以下字段定义了但实际为空（详见"论文 vs 代码"一节）
    original_memory: str
    compressed_memory: str
    category / subcategory / memory_class / bam_tags ...
```

**最关键的字段是 `memory`**：它就是一个独立的事实句，能脱离上下文独立理解。

- 不是原始消息（"I'm planning a trip to Tokyo"）
- 不是摘要（一段话）
- 不是 QA 对
- 而是一条**改写成第三人称的独立事实**（"User is planning a trip to Tokyo next month."）

### 写入前经历了什么变换？

一条用户消息变成 MemoryEntry 要经历五个阶段：

1. **时间戳归一化**：把 "2023/05/20 (Sat) 00:44" 这类人类友好格式统一解析成 ISO 格式，同一 session 内的多条消息按 500ms 递增生成独立时间戳（这样后续可以用时间戳判断"谁比谁新"）
2. **token 压缩**：LLMLingua-2 把消息里的冗余词去掉，保留最重要的词（比如 "I'm planning a trip to Tokyo next month, I think it will be great fun" 可能压缩到 "planning trip Tokyo next month"）
3. **话题切分**：压缩后的消息攒在感觉缓冲里，满了就按话题切段
4. **LLM 批量抽取**：一批 topic 段一次性送给 LLM，让它逐条抽取事实句，输出 JSON 格式的 `{source_id, fact}` 列表
5. **构建 MemoryEntry**：每条 `fact` 加上时间戳、星期、topic_id 等元数据，生成 MemoryEntry，embed（用 all-MiniLM-L6-v2 模型把文本转成向量）后存入 Qdrant（向量数据库）

---

## 三、系统怎么运转的？

### 一个具体例子贯穿全程

假设你在和 AI 聊天，在两个不同的话题里说了话：

**话题 A（旅游）**：
- "I'm planning a trip to Tokyo next month, visiting Shibuya and Shinjuku."
- "My budget is around 3000 dollars."

**话题 B（工作）**：
- "I started a new job at Google last week as a software engineer."

以下跟踪这些消息如何一步步变成存储在数据库里的记忆。

---

### 第一关：感觉缓冲（Sensory Memory Buffer）

**代码位置**：[src/lightmem/factory/memory_buffer/sensory_memory.py](../../research/LightMem/src/lightmem/factory/memory_buffer/sensory_memory.py)，[src/lightmem/factory/pre_compressor/llmlingua_2.py](../../research/LightMem/src/lightmem/factory/pre_compressor/llmlingua_2.py)

调用 `add_memory(messages)` 后，第一步是**时间戳归一化**（[lightmem.py:276](../../research/LightMem/src/lightmem/memory/lightmem.py#L276)），把原始消息规范化：

```
输入：{"role": "user", "content": "I'm planning a trip to Tokyo next month...", "time_stamp": "2023/05/20 (Sat) 00:44"}
输出：{"role": "user", "content": "...", "time_stamp": "2023-05-20T00:44:00.000", "weekday": "Sat", "session_time": "2023/05/20 (Sat) 00:44"}
```

接着是**LLMLingua-2 压缩**（[lightmem.py:278-298](../../research/LightMem/src/lightmem/memory/lightmem.py#L278-L298)）。

LLMLingua-2 是一个基于 BERT 的轻量分类器（BERT 是一种常见的小型 NLP 模型，约 137M 参数，本地跑 <2GB 显存，零 API 费用）。它的任务是对每个 token 打分：这个词值不值得保留？按压缩率阈值保留最重要的 50%–80% token。

压缩后（以 60% 保留率为例）：

```
原文：I'm planning a trip to Tokyo next month, visiting Shibuya and Shinjuku.
压缩：planning trip Tokyo next month visiting Shibuya Shinjuku
```

压缩后的消息进入 **`SenMemBufferManager`**（感觉缓冲管理器，[sensory_memory.py:15-38](../../research/LightMem/src/lightmem/factory/memory_buffer/sensory_memory.py#L15-L38)）。缓冲区只计算 user 消息的 token 数（不算 assistant 回复），上限 512 token。只要缓冲不满，消息就继续累积，**什么也不做**。

**当缓冲满了**，触发**两阶段话题切分**（[sensory_memory.py:43-113](../../research/LightMem/src/lightmem/factory/memory_buffer/sensory_memory.py#L43-L113)）：

- **粗边界（注意力矩阵）**：把缓冲里所有 user 消息送进 LLMLingua-2 的 BERT 层，取第 8-11 层的注意力矩阵（不额外调用 API，这是模型前向计算的副产物）。看相邻两句之间的"注意力峰值"——当第 k 句对前一句的注意力分数是局部极大值时，说明第 k 句与前一句的关联度突然降低，这里有个话题边界（[topic_segmenter/llmlingua_2.py:107-120](../../research/LightMem/src/lightmem/factory/topic_segmenter/llmlingua_2.py#L107-L120)）。
  
- **细边界（Embedding 相似度）**：再用 all-MiniLM-L6-v2 对每个对话轮计算语义向量（向量是一种把句子意义编码成数字列表的表示方式），计算相邻轮向量的余弦相似度（相似度越低 = 语义越不相关）。相似度低于阈值（从 0.2 开始，逐步升到 0.5 直到找到边界）的位置也是候选边界（[sensory_memory.py:72-79](../../research/LightMem/src/lightmem/factory/memory_buffer/sensory_memory.py#L72-L79)）。

- **对齐取交**：两种边界如果位置差 ≤3 轮，就认为是同一个真实边界，取其交集（[sensory_memory.py:87-94](../../research/LightMem/src/lightmem/factory/memory_buffer/sensory_memory.py#L87-L94)）。

对于我们的例子，话题 A（旅游）和话题 B（工作）之间相似度很低，注意力也突然下降，会在这里切出边界，得到两个 topic 段：

```
Segment 0（话题 A）：[旅游相关的 user/assistant 消息...]
Segment 1（话题 B）：[工作相关的 user/assistant 消息...]
```

> **这个设计解决的问题**：不需要 LLM 来切分话题（省 API 调用），而是复用了已经在做压缩任务的 BERT 模型的注意力信息（"一模两用"）。代价是：切分结果有时不精确（实验显示准确率约 80%），不如 LLM 切分稳定。

---

### 第二关：短期缓冲（Short-Term Memory Buffer）

**代码位置**：[src/lightmem/factory/memory_buffer/short_term_memory.py:36-57](../../research/LightMem/src/lightmem/factory/memory_buffer/short_term_memory.py#L36-L57)

切出的 topic 段进入 `ShortMemBufferManager`（短期记忆缓冲管理器）。这个缓冲的阈值硬编码为 512 token（可配置 `th` 参数，默认 512），只有当**新加入的段会让缓冲超过阈值**时，才把当前缓冲里所有的段打包，触发一次 LLM 调用。

这就是"攒批触发"的精髓：如果你的对话稀疏（每天聊几句），多天的消息都攒在缓冲里，触发一次 LLM 调用处理大量内容；如果你聊得密集，可能每隔几个 topic 段就触发一次。

**关键设计**：一次触发 = 一次 LLM API 调用（flat 模式），而不是每条消息一次调用。

---

### 第三关：LLM 批量抽取事实（核心一次 API 调用）

**代码位置**：[src/lightmem/factory/memory_manager/openai.py:143-206](../../research/LightMem/src/lightmem/factory/memory_manager/openai.py#L143-L206)，[src/lightmem/memory/prompts.py:1-52](../../research/LightMem/src/lightmem/memory/prompts.py#L1-L52)

触发时，把所有 topic 段拼成一个 prompt，格式如下：

```
--- Topic 0 ---
[2023-05-20T00:44:00.000, Sat] 0.User: planning trip Tokyo next month visiting Shibuya Shinjuku
[2023-05-20T00:44:00.500, Sat] 1.Assistant: That sounds great! Tokyo is wonderful...
[2023-05-20T00:44:01.000, Sat] 2.User: budget around 3000 dollars
--- Topic 1 ---
[2023-05-20T00:44:01.500, Sat] 3.User: started new job Google last week software engineer
```

注意：**只有 user 消息才进抽取（默认配置 `messages_use="user_only"`）**，assistant 的回复不参与，进一步省 token。

LLM 被要求逐条消息抽取事实，输出 JSON：

```json
{"data": [
  {"source_id": 0, "fact": "User is planning a trip to Tokyo next month."},
  {"source_id": 0, "fact": "User plans to visit Shibuya and Shinjuku in Tokyo."},
  {"source_id": 2, "fact": "User has a budget of around 3000 dollars for the Tokyo trip."},
  {"source_id": 3, "fact": "User started a new job at Google last week as a software engineer."}
]}
```

`source_id` 对应 prompt 里消息的序号，用来追溯这条记忆来自哪条原始消息（取其时间戳）。

这一步的 prompt（`METADATA_GENERATE_PROMPT`，[prompts.py:1-52](../../research/LightMem/src/lightmem/memory/prompts.py#L1-L52)）设计要点：
- 要求"事实句要能独立理解"（light contextual completion）
- 哪怕是"User drank coffee this morning"这种小细节也必须保留
- 只跳过"Hi"、"lol"、"thanks"这类完全无信息量的内容

---

### 第四关：入库（写入 Qdrant 向量数据库）

**代码位置**：[src/lightmem/memory/lightmem.py:363-443](../../research/LightMem/src/lightmem/memory/lightmem.py#L363-L443)，[src/lightmem/memory/utils.py:206-286](../../research/LightMem/src/lightmem/memory/utils.py#L206-L286)

每条 `{source_id, fact}` 经过 `convert_extraction_results_to_memory_entries()` 转换成 MemoryEntry：

```python
# 以第一条为例
entry = MemoryEntry(
    id = "3f2a9b1c-...",          # 随机 UUID
    time_stamp = "2023-05-20T00:44:00.000",  # 来自 source_id=0 的消息时间戳
    weekday = "Sat",
    memory = "User is planning a trip to Tokyo next month.",
    topic_id = 0,
    speaker_id = "User",
    update_queue = [],
    consolidated = False
)
```

然后用 all-MiniLM-L6-v2 把 `memory` 字段的文本转成向量（768 维浮点数列表），和 MemoryEntry 的全部字段一起插入 Qdrant。

> **向量数据库是什么**：Qdrant 是一个专门存储"向量"的数据库。向量是文本的数学表示，语义相近的句子向量之间的"距离"更近。这样检索时只需要把问题也转成向量，找出库里距离最近的条目，就能找到语义相关的记忆——这比关键词搜索更灵活，可以匹配同义表达。

**写入是纯追加，不做任何在线去重或冲突检查**。这是 LightMem 的核心设计哲学：在线只追加、零仲裁，把代价高的冲突处理全部推迟到"睡眠期"。

---

### 第五关：检索（回答问题时）

**代码位置**：[src/lightmem/memory/lightmem.py:644-707](../../research/LightMem/src/lightmem/memory/lightmem.py#L644-L707)

用户提问时，`retrieve(query, limit=10)` 做三件事：
1. 把问题文本转成向量
2. 在 Qdrant 里找余弦相似度最高的 top-k 条记忆
3. 把每条记忆格式化成 `"时间戳 星期 记忆文本"` 字符串

输出示例：
```
"2023-05-20T00:44:00.000 Sat User is planning a trip to Tokyo next month."
"2023-05-20T00:44:00.000 Sat User plans to visit Shibuya and Shinjuku in Tokyo."
"2023-05-20T00:44:01.000 Sat User has a budget of around 3000 dollars for the Tokyo trip."
```

调用方（如评测脚本 [experiments/longmemeval/run_lightmem_qwen.py:198-205](../../research/LightMem/experiments/longmemeval/run_lightmem_qwen.py#L198-L205)）把这些字符串直接 `'\n'.join` 进 prompt：

```
Please answer the question based on the following memories:
2023-05-20T00:44:00.000 Sat User is planning a trip to Tokyo next month.
2023-05-20T00:44:00.000 Sat User plans to visit Shibuya and Shinjuku in Tokyo.
...
Question: What are the user's travel plans?
```

检索过程 **0 次 LLM 调用**，只有向量计算（本地完成）。

---

### 第六关：睡眠期更新（离线批处理）

这一步**不在用户交互过程中发生**，需要用户显式调用，通常在对话结束后跑。

**第一步：构建 update_queue**（[lightmem.py:457-537](../../research/LightMem/src/lightmem/memory/lightmem.py#L457-L537)）

对库里每一条记忆 `e_i`，找出所有**时间戳晚于等于它**的记忆里，语义最相似的 top-20 个（用时间过滤保证"新记忆才能更新旧记忆"）。把这 top-20 写进 `e_i` 的 payload 里的 `update_queue` 字段。这步只是向量检索，不调用 LLM。

```python
# e_i 的 update_queue 可能长这样：
[
    {"id": "abc-...", "score": 0.93},  # 更新者：同样关于东京旅行
    {"id": "def-...", "score": 0.85},  # 更新者：关于京都计划（可能相关）
    ...
]
```

**第二步：并行 LLM 仲裁**（[lightmem.py:539-642](../../research/LightMem/src/lightmem/memory/lightmem.py#L539-L642)）

对每条记忆 `e_i`，找出哪些其他记忆的 `update_queue` 里包含指向 `e_i` 的条目（这些才是真正想更新 `e_i` 的"更新者"），且相似度 ≥ 阈值（如 0.8）。

把 `e_i`（旧记忆）和这些更新者一起送给 LLM，用 `UPDATE_PROMPT`（[prompts.py:334-406](../../research/LightMem/src/lightmem/memory/prompts.py#L334-L406)）让它判断：
- **update**：新信息补充了旧信息（合并，改写 `e_i` 的 memory 字段）
- **delete**：新旧信息矛盾，新的正确，删除旧的 `e_i`
- **ignore**：两者无关，什么都不做

所有 `e_i` 的更新独立互不依赖，所以可以**并行**用线程池处理（[lightmem.py:626-627](../../research/LightMem/src/lightmem/memory/lightmem.py#L626-L627)）。

举例：如果用户后来说 "Actually I'm going to Kyoto, not Tokyo"，产生一条新记忆 "User is going to Kyoto"，它的向量与 "User is planning a trip to Tokyo next month" 相似度高（都是日本旅行），且时间戳更新，会出现在旧记忆的 update_queue 里。LLM 判断后选 delete，旧的东京记忆被物理删除。

---

## 四、关键设计决策解析

### 为什么用本地小模型做压缩和切分？

Mem0 等系统每轮对话都要调用 GPT API 来分析、提炼信息。如果聊 100 轮，就调 100 次 API。LightMem 用 LLMLingua-2（<2GB 显存，本地跑）做预处理，只有当缓冲真正积满时才调一次 API，摊到每轮的调用次数可以低到 0.01 次甚至更少。

还有一个隐藏的巧妙：压缩和切分**共用同一个 BERT 模型**（[lightmem.py:169](../../research/LightMem/src/lightmem/memory/lightmem.py#L169)），注意力矩阵是前向计算的"副产品"，拿来做 topic 切分不需要额外推理——这就是 "一模两用，注意力免费"。

### 为什么分两级缓冲（感觉 + 短期），而不是一级？

感觉缓冲（512 token）负责积累和切分，短期缓冲（可配置 `th`）负责决定"什么时候触发 LLM"。两级分离的好处是：可以单独调节 topic 切分的粒度和 LLM 调用的频率。较大的 `th` 让 LLM 一次处理更多 topic 段，API 调用更少但延迟更长；较小的 `th` 则反之。

### 为什么写入不做在线去重，而是推到睡眠期？

在线去重意味着每条新记忆入库时都要查一遍所有旧记忆，判断是否矛盾——这既慢又可能误删（LLM 有时会把"相关但不矛盾"的信息错误地当作冲突删除，如论文 §5.6 的案例分析所示）。推迟到睡眠期的好处是：（1）响应延迟为零；（2）可以并行处理；（3）有更多上下文时判断更准确。

代价是：如果用户长期不触发睡眠期更新，数据库里会堆积矛盾的旧记忆，导致检索到错误信息。这是个"在线实时性 vs 维护成本"的权衡。

---

## 五、论文宣称 vs 代码实际

有几处论文的措辞和代码实际不符，影响对该系统能力的判断：

### 已实现、可信的核心宣称

- 三阶段类人记忆架构：完整实现
- 预压缩 + 话题切分零 LLM 调用：完整实现
- 攒批触发抽取：完整实现
- 睡眠期离线并行更新（含物理删除）：完整实现
- 内建 token/调用计费器：完整实现（[lightmem.py:143-160](../../research/LightMem/src/lightmem/memory/lightmem.py#L143-L160)），效率数字可复现

### 宣称了但未实现（或会崩溃）的功能

**1. `update="online"` 静默失效**：代码里 `online_update()` 直接 `return None`（[lightmem.py:394-395](../../research/LightMem/src/lightmem/memory/lightmem.py#L394-L395)），设置这个参数后记忆根本不会入库，也不报错。**影响**：如果你读了 README 以为可以用在线更新，实际是默默丢数据。

**2. `graph_mem=True` 必然报错**：[memory/graph.py](../../research/LightMem/src/lightmem/memory/graph.py) 全文只有一行 `class GraphMem:` 没有类体，但 README 仍宣称"支持图记忆"（README.md:521）。**影响**：这个功能实际不存在。

**3. KV cache 是摆设**：配置文件里有 `kv_cache` 字段（[configs/base.py:98-105](../../research/LightMem/src/lightmem/configs/base.py#L98-L105)），但全库无任何地方读取或使用它。论文结论部分说"未来计划加速"，但配置表的措辞像已实现。

**4. BM25 检索是空壳**：[factory/retriever/contextretriever/bm25.py](../../research/LightMem/src/lightmem/factory/retriever/contextretriever/bm25.py) 是空文件，对应的 `examples/run_lightmem_bm25.py` 也是空文件。`retrieve()` 只走向量检索路径。

**5. `text_summary=False` 会崩溃**：README 说关掉这个参数会存原始文本，实际代码会触发 NameError（[lightmem.py:344-363](../../research/LightMem/src/lightmem/memory/lightmem.py#L344-L363)）。

**6. `topic_segment=False` 静默丢数据**：关掉话题切分后函数提前 return，不存任何记忆（[lightmem.py:300-309](../../research/LightMem/src/lightmem/memory/lightmem.py#L300-L309)，有自带的 TODO 注释）。

**7. 离线更新改文本后不重算向量**：当 action == "update" 时，代码只修改 payload 里的 `memory` 字段（文本），但 Qdrant 里存的向量还是旧的（[lightmem.py:617-621](../../research/LightMem/src/lightmem/memory/lightmem.py#L617-L621)）。这意味着记忆内容改了，但检索索引没更新，越更新越检索不准。

**8. `MemoryEntry.hit_time` 字段从不被更新**：定义了访问次数统计（[utils.py:29](../../research/LightMem/src/lightmem/memory/utils.py#L29)），但系统里没有任何地方在检索后递增它，也没有基于访问频率的遗忘机制。

> **对引用这篇论文的影响**：论文的核心贡献（三阶段架构、压缩切分、睡眠期更新）是真实实现且可复现的，效率数字可信。但图记忆、BM25 检索、在线更新模式等功能实际不可用，不能作为代码层面的实现参考。

---

## 六、核心代码位置速查

| 机制 | 文件 | 函数/类 | 行号 | 作用 |
|------|------|---------|------|------|
| 写入主流程 | memory/lightmem.py | LightMemory.add_memory | 204–392 | 压缩→切分→攒批→抽取→入库全流程 |
| 记忆数据结构 | memory/utils.py | MemoryEntry | 13–32 | 一条事实句 + 元数据 |
| 抽取 prompt | memory/prompts.py | METADATA_GENERATE_PROMPT | 1–52 | 让 LLM 逐消息抽 {source_id, fact} |
| 预压缩 | factory/pre_compressor/llmlingua_2.py | LlmLingua2Compressor.compress | 38–89 | 本地 BERT token 二分类，≥512 循环再压 |
| 感觉缓冲+切分 | factory/memory_buffer/sensory_memory.py | SenMemBufferManager.cut_with_segmenter | 43–113 | 注意力粗边界 + embedding 细边界，取交 |
| 注意力峰值检测 | factory/topic_segmenter/llmlingua_2.py | LlmLingua2Segmenter.propose_cut | 107–120 | BERT 8-11 层注意力矩阵局部极大值 |
| 短期攒批 | factory/memory_buffer/short_term_memory.py | ShortMemBufferManager.add_segments | 36–57 | 超过 th 阈值才触发批量抽取 |
| 批量事实抽取 | factory/memory_manager/openai.py | OpenaiManager.meta_text_extract | 143–206 | flat 1次调用 / event 2次，线程池并行 |
| 入库 | memory/lightmem.py | LightMemory.offline_update | 397–456 | 逐条 embed 插 Qdrant，纯追加 |
| 更新队列构建 | memory/lightmem.py | construct_update_queue_all_entries | 457–537 | 每条记忆找更新它的候选，时间戳 lte 过滤 |
| 睡眠期并行更新 | memory/lightmem.py | offline_update_all_entries | 539–642 | LLM 裁决 update/delete/ignore，物理删旧 |
| 更新 prompt | memory/prompts.py | UPDATE_PROMPT | 334–406 | 三规则：补充合并/冲突删旧/无关忽略 |
| 检索 | memory/lightmem.py | LightMemory.retrieve | 644–707 | query embed→Qdrant top-k→格式化字符串 |
| Qdrant 操作 | factory/retriever/embeddingretriever/qdrant.py | search/delete/update | 126, 193, 207 | 余弦检索/删点/改 payload |

---

## 七、对研究的启示

**值得借鉴的设计**：

1. **"一模两用，注意力免费"**：压缩和话题切分共用同一个 BERT 模型，注意力矩阵作为压缩的副产物直接拿来做切分，边际成本趋近于零。任何"用小模型做预处理"的方案都可以思考这种复用。

2. **写读维护三路径完全解耦**：在线写入（追加）、在线读取（向量检索）、离线维护（睡眠期更新）完全独立。这种解耦让每条路径都可以单独优化，也避免了在线读写竞争锁的问题。

3. **内建分项计费器**：把 summary/update/embedding 的 token 和调用次数分开统计（[lightmem.py:143-160](../../research/LightMem/src/lightmem/memory/lightmem.py#L143-L160)），效率论文的核心数字由框架本身产出，直接可用于写论文。

**暴露的问题空间**：

- 离线更新改文本不重算向量（文本与索引向量失配）——一个明显的 bug，也是可改进方向
- 睡眠期必须用户手动触发，且是全量遍历——可以探索增量式或触发式维护
- topic 切分假设消息严格 user/assistant 交替，现实中不一定成立
- 没有基于访问频率或时间衰减的遗忘机制——仅有 LLM 仲裁删除
