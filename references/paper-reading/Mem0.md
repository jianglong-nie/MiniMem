# Mem0

- 仓库：https://github.com/mem0ai/mem0
- commit：b819d95d（本文所有行号仅在该 commit 下有效）
- stars：58348（2026-06-11 查询）
- 论文：《Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory》，arXiv:2504.19413，本地 PDF：references/agent-memory/Mem0_Building_Production-Ready_AI_Agents_with_Scalable_Long-Term_Memory.pdf

---

## 一、论文的故事：为什么要做 Mem0

**问题背景**

AI 助手要想做到真正的个性化，必须跨会话"记住"用户。用户上周提到自己对坚果过敏，本周问菜谱时助手不应该再推荐含花生的食谱。但 LLM 本身是无状态的，每次对话都从零开始，不知道用户是谁、说过什么。

解决这个问题最直白的办法是：把所有历史对话塞进 **context window**（模型每次对话能"看到"的文字总量上限，类似短期工作记忆——超过这个上限，早期内容就会被截掉，模型就再也看不到了）。这叫 full-context 方案，在小规模场景下能用，但代价是：
- **token 消耗巨大**：token 是 LLM 计算和计费的基本单位，大约 4 个英文字母 = 1 个 token，一次查询要消耗约 5 万 token，成本高。
- **速度慢**：延迟比精简方案高 91%。
- **不能无限扩展**：对话积累到百万 token 级别时彻底撑不住，因为没有任何模型能一次读完那么多内容。

另一个常见方案是 **RAG（Retrieval-Augmented Generation，检索增强生成）**：把对话内容切成文本块存进数据库，每次提问前先检索最相关的 top-K 块（K 通常是 5 到 20，表示取相关性最高的 K 个片段）填进提示词，让模型看到"最相关的历史"。但这有个根本问题：**检索回来的是原始对话片段，而非结构化的事实**。一段对话可能有 10 条有用信息和大量废话，检索结果把废话也带进去，浪费 context。

Mem0 的答案是：把对话提炼成**自包含的事实条目**，以事实粒度存储和检索。检索结果是精炼过的信息，而非原始文本块。这就是这篇论文的核心想法。

**论文的两个版本**

这篇论文比较特殊：论文 arXiv:2504.19413 描述的是 v2 两段式管线（extract → ADD/UPDATE/DELETE/NONE 决策），但 2026 年 4 月的代码重写（commit a488e190）已把整套 v2 机制移除，替换成 v3 ADD-only 管线。**你读到的论文和跑到的代码描述的是两套不同的算法**。后面会详细解释两者差异。

---

## 二、记忆条目设计：一条记忆长什么样

这是理解 Mem0 最核心的问题。

**一条记忆是什么**

Mem0 里的记忆不是 QA 对、不是摘要段落、不是原始消息、不是图节点。它是一条**自包含的事实陈述句**，能独立被读懂、不依赖对话上下文。

例子：用户说"我最近换了工作，现在在 Stripe 做算法工程师"。Mem0 提取出的记忆条目是：

> "Alice recently transitioned to a new job as a Machine Learning Engineer at Stripe, as of March 2025."

注意：
- 人称代词"我"被替换成了名字 Alice（**自包含**：这条记忆拿出来单独看也能读懂，不需要知道"我是谁"）
- "最近"被解析成了具体日期 March 2025（**时间绝对化**）
- 职位和公司名保留了完整信息（**不泛化**）

**记忆条目包含的字段**

每条记忆在系统里有两套字段：一套是对外暴露的 API 对象（`MemoryItem`，[mem0/configs/base.py:16-26](research/mem0/mem0/configs/base.py#L16-L26)）：

```
id          UUID（通用唯一标识符：一串随机生成的长字符串如 "a3f8-bc1d-..."，在全球范围内几乎不可能重复，每条记忆的"身份证号"）
memory      文本事实本身
hash        MD5 摘要（一种哈希算法：把任意长度文本变成固定长度的32位数字"指纹"，相同文本指纹完全一样，哪怕只差一个字符指纹也完全不同，用于快速判断两条文本是否字面相同）
metadata    自定义扩展字段
score       检索时的相关性分数（0-1，越接近 1 越相关）
created_at  创建时间（ISO 字符串）
updated_at  最近修改时间
```

另一套是实际存入向量库的 payload（[main.py:834-842](research/mem0/mem0/memory/main.py#L834-L842)）：

```
data               文本事实（同 memory）
text_lemmatized    词形还原后的文本（词形还原：把单词还原到词典原形，"running/ran/runs" 都变成 "run"，让关键词检索能匹配各种变形），供 BM25 关键词检索使用
hash               MD5
created_at/updated_at  时间戳
attributed_to      "user" 或 "assistant"（这条记忆来自谁的发言）
user_id/agent_id/run_id  作用域标识，隔离不同用户/会话/智能体
actor_id           具体发言人名称
role               消息角色
```

**写入前经历了什么变换**

从"对话文本"到"记忆条目"，经历了这几步：

1. **LLM 抽取**：一次 LLM 调用，从对话中提取事实列表，格式是 JSON 数组。
2. **自包含化**：prompt 要求 LLM 用真实姓名替换代词，保证每条事实离开对话仍能被读懂。
3. **时间绝对化**：prompt 提供两个日期——"Observation Date"（对话发生的日期）和"Current Date"（当前日期）。LLM 必须把"昨天""上周"这类相对时间解析成绝对日期，依据是 Observation Date 而非 Current Date（[prompts.py:524-540](research/mem0/mem0/configs/prompts.py#L524-L540)）。
4. **去重检查**：MD5 hash 与已有记忆精确对比，相同则跳过（不进行语义级去重）。
5. **词形还原**：spaCy（一个开源自然语言处理 Python 库，提供分词、词性标注、命名实体识别等基础 NLP 能力，无需 LLM）把文本词形还原，存入 `text_lemmatized`，供 BM25 关键词检索用。BM25（Best Matching 25）是一种经典关键词检索算法，不理解语义，只统计词频和文档频率来判断相关性：某个词在这条记忆里越多见、在整个记忆库里越罕见，分数越高。
6. **向量嵌入**：把文本转换成一串高维数字（称为"向量"或"embedding"）——语义相近的文本在这个数字空间里距离也近，计算机可以通过比较数字距离来找语义相似的内容。批量转换后存入向量库（Qdrant，一种开源向量数据库，专门高效存储和检索这类数字向量，比普通数据库快得多）。
7. **实体提取与链接**：spaCy 从记忆文本中识别命名实体（Named Entity：人名、地名、机构名、产品名等有专有含义的名词），存入独立的实体向量库，建立"实体→记忆 ID"的反向链接。

**这种设计的优势和限制**

优势：
- 检索回来的是精炼事实，不携带无关信息，节省 context。
- 自包含设计使记忆可独立使用，不需要还原对话上下文。
- 时间绝对化让系统能区分"2024年3月的偏好"和"2025年3月的偏好"。

限制：
- 事实级别太细时可能丢失因果关系或完整背景。
- 自动去重只能处理字面完全相同的情况，同义改写的事实会重复累积（因为系统是 ADD-only，不做语义去重和合并）。
- 矛盾的旧事实和新事实会并存，靠检索排名来"自然压制"旧版本。

---

## 三、论文原始方法（v2）：两段式决策管线

论文 arXiv:2504.19413 真正描述的方法是这套两段式管线。

**流程概述**

第一段：**事实抽取**

`FACT_RETRIEVAL_PROMPT`（[prompts.py:15](research/mem0/mem0/configs/prompts.py#L15)）让 LLM 从对话里提取事实列表，格式如：

```json
{"facts": ["Name is John", "Is a software engineer", "Likes cheese pizza"]}
```

第二段：**记忆决策（ADD/UPDATE/DELETE/NONE）**

对每条新抽出的事实，系统先向量检索相似的已有记忆，然后由 `DEFAULT_UPDATE_MEMORY_PROMPT`（[prompts.py:176](research/mem0/mem0/configs/prompts.py#L176)）驱动 LLM 做四选一决策：

- **ADD**：新信息，不与现有记忆冲突 → 新建一条记忆。
- **UPDATE**：与某条已有记忆说的是同一件事但更新/更具体 → 覆写该条（保留 ID）。
- **DELETE**：与某条已有记忆矛盾（新事实是正确的）→ 删除旧条。
- **NONE**：已有记忆里已经有完全相同或等价的信息 → 跳过。

具体例子：
- 已有记忆："用户喜欢打板球"；新事实："喜欢和朋友一起打板球" → 决策 UPDATE（更具体）。
- 已有记忆："用户住在上海"；新事实："用户搬去了北京" → 决策 UPDATE（地点变了）。
- 已有记忆：无；新事实："John 是软件工程师" → 决策 ADD。

**这个方法的逻辑**

这套设计让记忆库能"进化"：不会无限堆积，旧的错误信息会被新信息纠正，矛盾信息会被清理。从逻辑上看这是合理的——记忆系统应该像人脑一样能更新、遗忘。

**为什么这套方法被放弃了**

论文没有明说，但代码层面的证据表明：
- LLM 会做出错误的 UPDATE/DELETE 决策，导致记忆被错误覆写或删除。
- 每次写入需要两次 LLM 调用（抽取 + 决策），成本高。
- 系统对 prompt 细节高度敏感，生产环境不稳定。

2026 年 4 月的 v3 重写彻底放弃了这条路，转向更简单但实证上效果更好的 ADD-only 方案。

---

## 四、当前代码实现（v3）：ADD-only 管线

v3 的核心哲学是：**写入零维护，靠检索侧的多信号融合来弥补记忆库不整洁的代价**。

### 系统整体如何运转

一次完整的 `Memory.add()` 调用触发 8 个阶段（[main.py:725 注释](research/mem0/mem0/memory/main.py#L725)）：

```
阶段 0 → 从 SQLite 取最近 10 条原始消息（供 LLM 理解代词指代）
阶段 1 → 把新消息转向量，检索 top-10 语义相似旧记忆
阶段 2 → 唯一一次 LLM 调用：从消息中抽取新事实
阶段 3 → 批量把抽出的事实转向量
阶段 4 → MD5 去重（命中已有记忆则跳过该条）
阶段 5 → 词形还原（为 BM25 关键词检索准备）
阶段 6 → 批量写入向量库 + SQLite history 审计日志
阶段 7 → spaCy 实体抽取 → 写入实体库并建立实体-记忆反链
阶段 8 → 把原始消息存 SQLite（只保留最近 10 条）
```

### 具体例子：一条消息从输入到存储的完整经历

用这个例子从头到尾追踪数据在每个阶段的实际形态：

**输入**（用户发的消息）：
```python
Memory.add(
    messages=[{"role": "user", "content": "我是Alice，刚换工作，现在在Stripe做机器学习工程师，我对花生过敏"}],
    user_id="alice-001"
)
```

---

**阶段 0：从 SQLite 取最近 10 条旧消息**

SQLite 是一种轻量级文件数据库，数据直接存在本地文件里，不需要安装服务器，常用于存储少量结构化记录。这里系统读出 Alice 最近的对话历史，为下一步 LLM 理解"我"指谁提供上下文：

```
（假设 Alice 是首次对话，结果为空列表）
recent_messages = []
```

---

**阶段 1：嵌入新消息 + 检索已有记忆**

消息文本被转换成向量，去向量库查找语义最相近的 top-10 条旧记忆（用于让 LLM 知道"这些事实已经存过了，你提取时参考"）：

```
（首次对话，无已有记忆）
existing_memories = []
```

---

**阶段 2：LLM 调用 → 抽取事实**（整个管线唯一的 LLM 调用）

系统把以下内容打包成 prompt 发给 LLM：
- 系统提示词（约 480 行的 `ADDITIVE_EXTRACTION_PROMPT`，核心规则："只做 ADD，提取事实，用真实姓名替换代词，把相对时间改成绝对日期，每条事实 15-80 词"）
- 旧记忆列表（此处为空）
- 最近 10 条消息（此处为空）
- 新消息内容
- 日期信息：Observation Date: 2026-06-15，Current Date: 2026-06-15

LLM 返回：
```json
{
  "memories": [
    {
      "text": "Alice recently started a new job as a Machine Learning Engineer at Stripe, as of June 2026.",
      "attributed_to": "user",
      "linked_memory_ids": []
    },
    {
      "text": "Alice has a peanut allergy.",
      "attributed_to": "user",
      "linked_memory_ids": []
    }
  ]
}
```

注意发生了什么变换：
- "我"→ "Alice"（代词被替换，事实变得自包含）
- "刚换工作"→ "as of June 2026"（相对时间被绝对化）
- 一条消息被拆成两条独立事实（工作和过敏是不同类型的信息）

管线只取每条记录的 `text` 和 `attributed_to` 字段，`linked_memory_ids` 被丢弃（详见第六节）。

---

**阶段 3：批量嵌入两条事实**

两条文本分别被转换成向量（一串约 1536 维的浮点数，维度取决于使用的嵌入模型），准备写入向量库。

```
"Alice recently started..." → [0.023, -0.187, 0.641, ..., 0.012]  （1536 个数字）
"Alice has a peanut allergy." → [0.184, 0.052, -0.339, ..., 0.267] （1536 个数字）
```

---

**阶段 4：MD5 去重**

对两条文本各自计算 MD5 哈希值（32 位十六进制字符串），查 `alice-001` 已有记忆中是否有相同指纹：

```
"Alice recently started a new job..." → MD5: "a3f89bc1d72e4f8c..."
"Alice has a peanut allergy."         → MD5: "d72e14f8b903a1c2..."
```

查无命中 → 两条都通过，继续处理。

---

**阶段 5：词形还原**（spaCy 处理）

spaCy 把每条文本中的词还原为词典原形，存为 `text_lemmatized` 字段，供 BM25 关键词检索使用：

```
原文：  "Alice recently started a new job as a Machine Learning Engineer at Stripe, as of June 2026."
还原后："Alice recently start a new job as a Machine Learning Engineer at Stripe as of June 2026"
```

"started" → "start"，"has" → "have"，其余不变。这样以后用户查 "start new job" 或 "starting job" 都能命中这条记忆。

---

**阶段 6：写入向量库 + SQLite history**

向量库 Qdrant 写入两个 point，每个 point 包含：
- 向量（用于语义相似检索）
- payload（附带的结构化数据）：

```json
{
  "data": "Alice recently started a new job as a Machine Learning Engineer at Stripe, as of June 2026.",
  "text_lemmatized": "Alice recently start a new job as a Machine Learning Engineer at Stripe as of June 2026",
  "hash": "a3f89bc1d72e4f8c...",
  "created_at": "2026-06-15T10:00:00",
  "updated_at": "2026-06-15T10:00:00",
  "attributed_to": "user",
  "user_id": "alice-001",
  "actor_id": "Alice"
}
```

同时，SQLite 的 history 表写入两条 ADD 事件记录（old_memory=null，new_memory=事实文本，event="ADD"），供日后用 `Memory.history(memory_id)` 查询这条记忆的变更历史。

---

**阶段 7：实体抽取 + 链接**

spaCy 从两条事实中识别命名实体：
```
"Alice"（PERSON，人名）
"Stripe"（ORG，机构名）
"Machine Learning Engineer"（名词复合）
"June 2026"（DATE，日期）
"peanut"（名词复合）
```

每个实体也被转成向量，写入独立的实体向量库（collection 名 `alice-001_entities`）。写入时用余弦相似度（Cosine Similarity：衡量两个向量方向有多接近的数学指标，值域 0-1，越接近 1 说明越相似）去重——阈值 0.95，意味着"几乎完全相同"的实体才会合并：

```
发现 "Alice" 是新实体 → 新建实体 point，linked_memory_ids = [<工作记忆的UUID>, <过敏记忆的UUID>]
发现 "Stripe" 是新实体 → 新建实体 point，linked_memory_ids = [<工作记忆的UUID>]
```

以后检索时，如果查询里提到 "Stripe"，这个实体就会把 boost（额外加分）传播给工作记忆，让它排名更靠前。

---

**阶段 8：保存原始消息到 SQLite**

原始消息（"我是Alice，刚换工作..."）存入 SQLite messages 表，只保留每个用户最近 10 条，供下次写入时的阶段 0 使用。

---

### 如何接收输入

入口是 `Memory.add(messages, user_id=...)` ([main.py:599](research/mem0/mem0/memory/main.py#L599))。

`messages` 可以是字符串或消息列表（`[{"role": "user", "content": "..."}, ...]`）。`user_id`/`agent_id`/`run_id` 三个字段定义记忆的作用域——不同用户的记忆互相隔离，同一 agent 的不同 run 的记忆可以分开管理。

有三条路径：
- `infer=True`（默认）：触发 8 阶段 LLM 抽取管线。
- `infer=False`：跳过 LLM，每条消息原文直接存向量库（适合存结构化日志）。
- `memory_type="procedural_memory"`：另一条路径，1 次 LLM 调用把整段对话总结成一条流程记忆，如"用户设置 Python 环境的步骤"（[main.py:1672-1709](research/mem0/mem0/memory/main.py#L1672-L1709)）。

### 如何判断是否写入记忆

系统不做"应不应该写"的主动判断，而是让抽取 prompt 来控制：

- 没有新事实时，LLM 输出空列表（写入什么都不发生）。
- MD5 hash 命中已有记忆时，该条跳过（[main.py:825-829](research/mem0/mem0/memory/main.py#L825-L829)）。
- 其余情况一律写入（ADD-only）。

### 如何生成/变换记忆条目：唯一的 LLM 调用

阶段 2 的 LLM 调用是整个写入流程的核心（[main.py:751-771](research/mem0/mem0/memory/main.py#L751-L771)）。

系统 prompt 是 `ADDITIVE_EXTRACTION_PROMPT`（[prompts.py:468](research/mem0/mem0/configs/prompts.py#L468)，共约 480 行）。这个 prompt 的工程密度很高：

- 明确写"Your sole operation is ADD"——禁止任何 UPDATE/DELETE 决策。
- 要求每条事实 15-80 词，保留专有名词和数字，宁多勿漏（"When in doubt, extract"）。
- 提供 Observation Date + Current Date 两个时间锚，要求把相对时间转成绝对日期。
- 列举"first topic dominance"等常见抽取失败模式（指 LLM 只关注对话前半段，后半段事实被漏掉），附反例。
- 要求同时抽取 user 和 assistant 的事实（agent 生成的信息也存）。
- 要求输出 `linked_memory_ids`（关联旧记忆的 UUID 列表）——**但这个字段在管线中被丢弃，详见第六节**。

user prompt 由 `generate_additive_extraction_prompt` 组装（[prompts.py:1016-1062](research/mem0/mem0/configs/prompts.py#L1016-L1062)）：拼接 Summary（恒为空）、旧记忆列表、最近 10 条消息、新消息、日期。

### 如何存储

存储分三层：

**① 主向量库**（默认 Qdrant）：每条事实一个向量 point，payload 含全部元数据（data、hash、text_lemmatized、时间戳、作用域字段）。批量插入（[main.py:856-882](research/mem0/mem0/memory/main.py#L856-L882)）。

**② 实体库**：独立的 collection `{collection}_entities`。每个实体有自己的向量，payload 含 `linked_memory_ids` 数组（反向链到引用了它的记忆）。写入时 0.95 余弦相似度查重——命中则追加 memory_id 到现有实体，未命中则新建（[main.py:891-981](research/mem0/mem0/memory/main.py#L891-L981)）。

**③ SQLite**：两张表。`history` 表是 append-only 审计日志（只追加不修改，保证历史不丢失），每次 ADD/UPDATE/DELETE 记一行（[storage.py:108-120](research/mem0/mem0/memory/storage.py#L108-L120)），可用 `Memory.history(memory_id)` 查单条记忆的演变。`messages` 表存最近 10 条原始消息，为下一次写入的阶段 0 提供指代消解上下文（[storage.py:134-141](research/mem0/mem0/memory/storage.py#L134-L141)）。

### 如何检索

`Memory.search(query)` ([main.py:1152](research/mem0/mem0/memory/main.py#L1152)) → `_search_vector_store` ([main.py:1373](research/mem0/mem0/memory/main.py#L1373))，全程 **0 次 LLM 调用**，融合三个信号：

**第一信号：语义分**

query 文本被转成向量后，在向量库里计算余弦相似度，over-fetch `max(limit×4, 60)` 条作为打分候选池。语义分低于 threshold（默认 0.1）的直接过滤掉。这一信号能找到"意思相近但用词不同"的记忆，比如 query 是"food restrictions"时能检索到记录花生过敏的记忆。

**第二信号：BM25 关键词分**

对剩余候选在词形还原文本上做 BM25 关键词检索（[main.py:1392](research/mem0/mem0/memory/main.py#L1392)）。BM25 原始分经 sigmoid 函数（一种 S 形曲线，把任意实数挤压到 0-1 区间）归一化（[scoring.py:16-54](research/mem0/mem0/utils/scoring.py#L16-L54)），sigmoid 的斜率参数随 query 长度自适应——短 query 对关键词命中更敏感。这一信号弥补了语义检索在专有名词上的盲点：比如 query 包含"Stripe"时，语义向量不一定能精确找到含"Stripe"的记忆，但 BM25 一定能找到。

**第三信号：实体 boost**

从 query 中 spaCy 抽实体 → 批量嵌入 → 查实体库 top-500，余弦相似度 ≥ 0.5 的实体把 boost 传播给其 `linked_memory_ids` 里的记忆。"热门实体"（关联记忆数量 n 大的实体）boost 被 `1/(1+0.001×(n-1)²)` 压制，避免高频实体（如"Alice"）把所有检索都带偏（[main.py:1473-1553](research/mem0/mem0/memory/main.py#L1473-L1553)，boost 上限 0.5）。

**融合排序**：`score_and_rank` ([scoring.py:60-139](research/mem0/mem0/utils/scoring.py#L60-L139)) 把三个分数直接相加再除以理论最大值，归一化到 [0,1]：

```
final_score = (semantic_score + bm25_score + entity_boost) / max_possible
```

可选 reranker（一种用 cross-encoder 模型对候选结果重新排序的组件，精度更高但更慢；默认未配置）。

### 如何把检索结果放进 prompt

核心库本身不管注入——这由应用层决定。仓库内的参考实现在 `mem0/proxy/main.py:186-191`：

```python
# 把检索到的每条记忆拼成 "- <text>" 格式，塞进用户最后一条消息之前
relevant = "\n".join(f"- {m['memory']}" for m in memories["results"])
modified_content = f"Relevant Memories/Facts:\n{relevant}\n\nUser Message:\n{original_content}"
```

系统 prompt 用 `MEMORY_ANSWER_PROMPT`（proxy/main.py:151）：要求模型基于记忆中的信息回答，找不到相关信息时也要优雅响应。

### 如何更新/删除/整合记忆

**更新**：只有手动 API `Memory.update(memory_id, data)` → `_update_memory`（[main.py:1711-1774](research/mem0/mem0/memory/main.py#L1711-L1774)）：重新嵌入、保留 `created_at`、更新 `updated_at`、写 history、重建实体链接。**自动管线永远不调用这个函数**。

**删除**：只有手动 `delete`/`delete_all`/`reset`（[main.py:1578-1806](research/mem0/mem0/memory/main.py#L1578-L1806)）。删除时同步清理实体反链，在 history 记 DELETE 事件。

**整合/摘要**：无。没有后台任务把重复事实合并或生成用户画像摘要。prompt 里有 Summary 输入区（[prompts.py:496-498](research/mem0/mem0/configs/prompts.py#L496-L498)），但开源管线调用处传的是空字符串（[main.py:757-762](research/mem0/mem0/memory/main.py#L757-L762)）。

**遗忘/过期**：无。没有 TTL（Time To Live，即"过期时间"：超过设定时间后数据自动删除，常见于缓存系统）、无时间衰减、无容量上限、无自动删除。SQLite messages 表"只保留最近 10 条"是为了控制写入时的上下文窗口大小，不是记忆遗忘机制。

**矛盾处理**：无。ADD-only 下新旧矛盾事实并存，依赖检索时"含绝对日期的新事实通过 BM25/语义排到前面"来自然抑制旧版本。这个设计假设检索结果的时间信号足够强，在实验上被 91.6 的 LoCoMo 分数支撑，但在极端矛盾场景（同一事实多次反转）下可能失效。

---

## 五、实验设置与结果

### 评估基准

论文使用的核心基准：

**LoCoMo**：多轮对话记忆评估数据集，测试模型能否跨会话回答基于历史对话的问题。包含隐式和显式记忆检索任务（evaluation/README.md）。

**LongMemEval**：更长时间跨度的记忆评估，重点测试 assistant 的动作记忆（agent 确认完成某件事后，下次能否正确引用）。

**BEAM (1M/10M)**：生产规模评估，分别在 100 万和 1000 万 token 的对话历史规模下测试记忆系统的性能。

### 基线对比

| 类别 | 系统 |
|------|------|
| 学术论文方法 | ReadAgent、MemoryBank、MemGPT、A-Mem |
| 开源实现 | LangMem |
| RAG | 不同 chunk size（500 词等）+ 不同 top-K 配置 |
| 全上下文 | 完整对话历史塞进 context window |
| 商业系统 | OpenAI Memory（ChatGPT 内置）、Zep |

### 结果

v3 管线（当前代码）在同一模型栈下的结果（README.md:47-53）：

| 基准 | v2 旧算法 | v3 新算法 | Token 消耗 | 延迟 p50 |
|------|-----------|-----------|-----------|---------|
| LoCoMo | 71.4 | **91.6** | 7.0K | 0.88s |
| LongMemEval | 67.8 | **94.8** | 6.8K | 1.09s |
| BEAM (1M) | — | **64.1** | 6.7K | 1.00s |
| BEAM (10M) | — | **48.6** | 6.9K | 1.05s |

延迟 p50 指"第 50 百分位的响应时间"，即一半的请求比这个数快——相当于"典型情况下的延迟"。

评估指标包含：
- **BLEU**：衡量生成文本与参考答案在词序列上的重合度，0-1，越高越好，但只看词面，不理解语义。
- **F1**：综合"找准"（精确率：找到的答案中正确的比例）和"找全"（召回率：所有正确答案中被找到的比例）的调和均值。
- **LLM Judge Score**：用另一个 LLM 判断回答是否正确，输出二元 yes/no，比 BLEU 更接近人的主观判断。

每次查询平均只消耗约 7K token（对比 full-context 的约 50K），单次检索（无 agentic 循环，即不会反复调用工具多步推理）。

v3 vs 全上下文基线：在 LoCoMo 上高约 20 点，同时 token 消耗少约 85%，延迟低约 91%。

评测代码已开源（evaluation/ 目录），可复现。

---

## 六、论文宣称 vs 代码实际：需要注意的重要偏差

这些偏差会影响如何引用这篇论文和如何复现它的方法。

**1. 论文核心机制（两段式 ADD/UPDATE/DELETE/NONE）已整体被移除**

arXiv:2504.19413 的主要贡献是智能记忆决策管线。本 commit 中该机制完全沦为死代码：
- `FACT_RETRIEVAL_PROMPT`（prompts.py:15）— 无任何调用方
- `DEFAULT_UPDATE_MEMORY_PROMPT`（prompts.py:176）— 无任何调用方
- `get_update_memory_messages`（prompts.py:406）— 无任何调用方
- `get_fact_retrieval_messages`（memory/utils.py:15）— 无任何调用方

2026-04-14 commit a488e190 完成替换。**如果你想复现论文方法，要回退到该 commit 之前的版本。**

**2. 论文的图记忆变体 Mem0g 在 Python 包中已被删除**

论文专门讨论了基于图的记忆变体（实体和关系存 Neo4j——一种专门处理节点和边关系网络的图数据库，擅长回答"这个人认识哪些人""这家公司属于哪个行业"这类关系型问题，语义+图关系联合检索）。当前代码中 `mem0/graphs/` 和 `mem0/memory/graph_memory.py` 不存在，`MemoryConfig` 无 graph_store 字段（[configs/base.py:29-57](research/mem0/mem0/configs/base.py#L29-L57)）。proxy/main.py:187 还在读不会再出现的 `relevant_memories["relations"]`，是残留代码。

Graph memory 仍在 server/ 和 openmemory/ 里保留（通过 Neo4j），但 OSS Python SDK 核心路径已删除。

**3. LLM 输出的 Memory Linking 是摆设**

`ADDITIVE_EXTRACTION_PROMPT` 花大量篇幅要 LLM 输出 `linked_memory_ids`（prompts.py:692-701），以建立记忆间关联图（如"这条新记忆与旧记忆 mem-001 矛盾，与 mem-007 是同一主题的延续"）。写入端还建了 `uuid_mapping` 防止 LLM 幻造 ID（[main.py:744-746](research/mem0/mem0/memory/main.py#L744-L746)）。

但管线只取 `text` 和 `attributed_to`（[main.py:821](research/mem0/mem0/memory/main.py#L821)），LLM 输出的 `linked_memory_ids` 与 `uuid_mapping` 双双被丢弃，从未落库。实际生效的记忆间链接只有 spaCy 实体库那条路径（通过共享实体间接关联，不是显式记忆图）。

**4. Summary 和"Recently Extracted Memories"输入恒为空**

`ADDITIVE_EXTRACTION_PROMPT` 把"用户画像摘要"（Summary）和"最近提取的记忆"（Recently Extracted Memories）描述为重要的去重和上下文来源（[prompts.py:496-503](research/mem0/mem0/configs/prompts.py#L496-L503)）。但开源调用点 [main.py:757-762](research/mem0/mem0/memory/main.py#L757-L762) 不传这两个参数，等于宣传的"用户画像"功能在开源版不存在（云端付费版才有）。

**5. "时间感知检索"实际上只是日期字符串参与文本匹配**

README.md:61 宣称"time-aware retrieval that ranks the right dated instance"。但 scoring.py 的融合分完全不含时间项，检索里没有时间衰减、没有时间偏好权重。"时间能力"来自抽取时把相对日期写成绝对日期字符串（"as of March 2025"），然后靠 BM25 的关键词命中或语义相似度自然排序——不是一个独立的时间感知排序器。

**与实现完全一致的宣称**：
- 单次 LLM 调用 ADD-only 抽取 ✓（[main.py:765](research/mem0/mem0/memory/main.py#L765) 唯一调用点）
- 实体链接 boost 检索 ✓（[main.py:891-981、1473-1553](research/mem0/mem0/memory/main.py#L891)）
- 语义+BM25+实体三信号融合 ✓（[scoring.py:60-139](research/mem0/mem0/utils/scoring.py#L60-L139)，spaCy/fastembed 装了才全信号）

---

## 七、实现细节中值得注意的地方

**NLP 能力静默降级**

spaCy 未安装时：词形还原直接返回原文（[lemmatization.py:31-32](research/mem0/mem0/utils/lemmatization.py#L31-L32)）、实体抽取返回空列表（[entity_extraction.py:140-141](research/mem0/mem0/utils/entity_extraction.py#L140-L141)）。fastembed（一个支持稀疏向量的轻量嵌入库，专门用于 BM25 这类基于词频的检索，不做语义理解）未安装时：Qdrant 的 BM25 检索整体禁用（[qdrant.py:96](research/mem0/mem0/vector_stores/qdrant.py#L96)）。三信号检索在依赖不全时会悄悄退化为纯语义，benchmark 数字无法复现时很难察觉。

**去重只有 MD5 精确匹配**

同义改写的事实（"用户是素食主义者" vs "用户不吃肉"）会同时存在。抽取 LLM 被要求跳过语义等价的事实，但它只能看到 top-10 旧记忆（[main.py:738](research/mem0/mem0/memory/main.py#L738)）——库大了之后重复事实必然累积，没有任何后台整合机制处理它们。

**sync/async 两套 ~1400 行近乎复制的实现**

`Memory`（[main.py:348](research/mem0/mem0/memory/main.py#L348)）和 `AsyncMemory`（[main.py:1849](research/mem0/mem0/memory/main.py#L1849)）是同一逻辑的同步/异步两份副本（异步版允许在等待 LLM 或数据库响应时同时处理其他请求，更适合高并发场景），核心差异只是 `await` 关键字。同一逻辑要改两处，死代码多（uuid_mapping 处理、legacy prompts、proxy 的 relations）。

---

## 八、关键代码位置速查

| 机制 | 文件 | 函数/类 | 行号 | 作用 |
|------|------|---------|------|------|
| 写入入口 | mem0/memory/main.py | Memory.add | 599 | 参数校验、分发 procedural/普通路径 |
| V3 批量管线 | mem0/memory/main.py | Memory._add_to_vector_store | 688 | 8 阶段：检索→LLM→去重→入库→实体链接 |
| 抽取 system prompt | mem0/configs/prompts.py | ADDITIVE_EXTRACTION_PROMPT | 468 | ADD-only 规则，约 480 行 |
| user prompt 组装 | mem0/configs/prompts.py | generate_additive_extraction_prompt | 1016 | 拼 Summary/旧记忆/新消息/日期 |
| MD5 去重 | mem0/memory/main.py | （Phase 4/5 内联） | 825 | 命中旧记忆或本批则跳过 |
| 实体链接写入 | mem0/memory/main.py | （Phase 7 内联） | 891 | spaCy 批量抽实体，0.95 查重后 insert/update |
| 检索入口 | mem0/memory/main.py | Memory.search | 1152 | 参数校验、高级过滤、可选 rerank |
| 三信号检索 | mem0/memory/main.py | Memory._search_vector_store | 1373 | 语义+BM25+实体 → 融合排序 |
| 实体 boost 计算 | mem0/memory/main.py | Memory._compute_entity_boosts | 1473 | 实体库 top-500，boost 热门度阻尼 |
| 融合打分 | mem0/utils/scoring.py | score_and_rank | 60 | threshold 过滤后 (sem+bm25+ent)/max |
| BM25 归一化 | mem0/utils/scoring.py | normalize_bm25 | 16 | query 长度自适应 sigmoid 归一化 |
| 实体抽取 | mem0/utils/entity_extraction.py | extract_entities | 123 | 纯 spaCy：专有名词序列/引号/名词复合 |
| 手动更新 | mem0/memory/main.py | Memory._update_memory | 1711 | 重 embed、写 history、重链实体 |
| 手动删除 | mem0/memory/main.py | Memory._delete_memory | 1776 | 删向量、history 记 DELETE、清实体反链 |
| 历史审计表 | mem0/memory/storage.py | SQLiteManager._create_history_table | 102 | old/new memory + event 的 append-only 日志 |
| 消息缓存 | mem0/memory/storage.py | SQLiteManager.save_messages | 257 | 存原始消息且每作用域保留最近 10 条 |
| BM25 接口（基类） | mem0/vector_stores/base.py | VectorStoreBase.keyword_search | 68 | 返回 None 时静默降级纯语义 |
| BM25 实现（Qdrant） | mem0/vector_stores/qdrant.py | Qdrant.keyword_search | 413 | fastembed 稀疏向量，未装则禁用 |
| Procedural 记忆 | mem0/memory/main.py | Memory._create_procedural_memory | 1672 | 1 次 LLM 把对话总结成流程记忆 |
| v2 决策 prompt（死代码） | mem0/configs/prompts.py | DEFAULT_UPDATE_MEMORY_PROMPT | 176 | ADD/UPDATE/DELETE/NONE 逻辑，已无调用 |
| prompt 注入参考实现 | mem0/proxy/main.py | _format_query_with_memories | 181 | 检索结果拼进用户消息 |

---

## 九、对研究的启示

**v2→v3 的转变本身就是有趣的研究素材**

论文提出智能决策管线（v2），实验后发现 ADD-only + 强检索（v3）反而效果更好（+20 LoCoMo）。这意味着"写入时 LLM 维护一致性"的方向在实证上输给了"写入零维护，靠检索弥补"。这个对比可以用来分析什么条件下 in-writing 维护比 in-retrieval 优化更有效。

**实体库是一个轻量图的工程设计**

不用图数据库（如 Neo4j），用第二个向量 collection + `linked_memory_ids` 反链 + 热门实体阻尼，以极低成本拿到关系感知检索的大部分收益，且实体抽取零 LLM 成本（spaCy 规则）。这种"轻量图"设计对资源受限场景有参考价值。

**没有解决的问题**

- **语义去重和合并**：长期运行下记忆库的重复事实累积是已知缺陷，没有机制处理。
- **矛盾事实**：并存的矛盾信息完全靠检索排名自然处理，在极端场景下不可靠。
- **遗忘机制**：没有 TTL、衰减或容量管理——记忆只增不减。
- **开源版缺失功能**：用户画像摘要（Summary）和记忆间显式关联图（linked_memory_ids）在开源版不工作，只在云端付费版存在。

这三个空白（去重整合、矛盾解决、遗忘管理）都是可以做文章的方向。
