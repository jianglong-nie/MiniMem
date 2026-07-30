# Zep（graphiti）

- 仓库：https://github.com/getzep/graphiti
- commit：40eca36（本文所有行号仅在该 commit 下有效）
- stars：27293（2026-06-11 查询）
- 对应论文：Zep: A Temporal Knowledge Graph Architecture for Agent Memory（arXiv:2501.13956）
- 关系说明：Zep 是商业托管产品，其记忆核心引擎是开源的 graphiti。本文只读 `graphiti_core/`，server/ 与 mcp_server/ 只是封装层。

---

## 一、论文的故事：Zep 为什么出现

### 背景：LLM 的记忆天花板

当你和一个 AI 助手聊天时，它能"记住"的内容受限于它的**上下文窗口**（context window）——可以类比为工作时的白板，写满了就擦掉。GPT-4 的上下文窗口大约是 128k 个词，而一个企业用户可能积累了数百万字的对话历史和业务数据，远远超出这个范围。

现有的解决方案叫做 **RAG（Retrieval-Augmented Generation，检索增强生成）**——预先把文档切成小块存起来，回答问题时再从中检索相关片段拼进上下文。这个方法对"静态文档"（法规、产品手册、论文）效果不错，但有一个根本缺陷：**它假设内容不会变**。

企业场景里，信息在持续变化：
- 用户今天说"我在 A 公司工作"，明年可能换了工作
- 本周的会议结论可能推翻上周的决策
- 客户偏好随时间演化

普通 RAG 遇到这类"旧信息被新信息取代"的场景，要么召回过期的旧事实，要么两者都召回导致矛盾，AI 不知道以哪个为准。

### MemGPT 怎么做，哪里不够

2024 年初，MemGPT 提出了给 LLM 加持久记忆的思路：把 LLM 模拟成一个操作系统，让它自己决定何时把对话内容写入"长期记忆"、何时召回。MemGPT 当时在 DMR（Deep Memory Retrieval，深度记忆检索）基准上拿到了 93.4% 的最高分，成为 SOTA（state-of-the-art，该领域当时最好的结果）。

但 Zep 的团队发现 MemGPT 有两个问题：
1. **DMR 基准本身太简单**：每段对话只有 60 条消息，完全可以塞进现代 LLM 的上下文窗口——直接把全文扔给模型反而能拿 94.4%。这意味着 MemGPT 的 93.4% 并没有真正体现记忆系统的价值。
2. **不擅长处理"信息随时间变化"**：MemGPT 主要存摘要，不能精确追踪"这个事实从什么时候开始成立、什么时候失效"。

### Zep 的核心主张

Zep 提出用**知识图谱**（Knowledge Graph，KG）来存记忆，而不是传统的向量数据库或摘要。

**知识图谱是什么？** 可以想象一张人际关系网：每个人是一个圆圈（节点），两个人之间如果有关系（比如"同事"、"家人"），就在他们之间画一条线（边），线上还可以写备注（比如"从 2020 年开始"）。这张网就是知识图谱。

Zep 在传统知识图谱的基础上加了一个关键创新：**双时序模型（bi-temporal model）**——每条关系上不只存"它存在/不存在"，而是存两组时间戳：
- **事件时间线 T**：这件事在现实世界里什么时候发生的（例如"Alice 2020年加入公司"）
- **入库时间线 T'**：我们什么时候把这条信息录入系统的（例如"我们今天从对话里得知这件事"）

这两个时间是不同的。用一个比喻：今天看了一部 1990 年的电影，电影故事发生在 1990 年（事件时间），我看它是今天（入库时间）。这对处理历史信息（"我上个月换工作了"这种回顾性叙述）特别重要。

---

## 二、论文提出的方法

### 图的三层结构

Zep 的知识图谱分三层子图：

**第一层：情节子图（Episodic Subgraph）**

存原始输入数据，不做任何加工。每一条消息、每一段文本、每一个 JSON 数据，进来就存成一个"情节节点"（Episode Node），带着它被接收的原始时间戳。这一层的作用是"保留原文"，确保信息不丢失，出问题时可以追溯原始来源。

**第二层：语义子图（Semantic Entity Subgraph）**

这是核心层。用 LLM 从情节中抽取出：
- **实体节点**（Entity Node）：现实中的人、地点、组织、概念
- **事实边**（Entity Edge）：两个实体之间的关系，用一句话描述，带时间窗口

比如从对话"Alice 在 Acme 公司做工程师，上周晋升了"中，抽出：
- 实体：Alice、Acme 公司
- 事实边：`Alice --[WORKS_AT]--> Acme`，`valid_at: 某年某月`

**第三层：社区子图（Community Subgraph）**

用聚类算法把关联紧密的实体归为一组，生成"社区摘要"。例如把所有关于某个项目的实体归在一起，生成这个项目的总结性描述。这一层提供了宏观的"全局视野"，对回答"给我说说 X 项目整体情况"这类问题有帮助。

### 双时序与矛盾处理

Zep 的事实边上存了四个时间戳：
- `created_at`：何时录入系统（T' 时间线）
- `expired_at`：何时在系统中被标记失效（T' 时间线）
- `valid_at`：这件事从现实中什么时候开始成立（T 时间线）
- `invalid_at`：这件事在现实中什么时候不再成立（T 时间线）

**当新信息和旧事实矛盾时，怎么处理？**

Zep 的关键决策是：**不删除旧事实，而是给它打上"失效"时间戳**。

具体流程：
1. 新事实进来，LLM 判断它是否和已有事实矛盾
2. 如果矛盾，比较时间戳：如果旧事实的 `valid_at` 早于新事实，则旧事实被"失效"——给它写上 `invalid_at = 新事实.valid_at`，`expired_at = 当前时间`
3. 旧事实和新事实都保存在图里，只是旧的多了这两个时间戳

这样设计的好处：历史记录完整保留（方便追溯"那时候的状态是什么"），不会因为误判矛盾而永久丢失信息。代价是图会无限增长，没有任何自动清理机制（这个问题后面会详细说）。

### 检索机制

检索时，Zep 用三种方法并行查找候选结果，再合并：

1. **向量相似搜索**（余弦相似度）：把查询和每条事实都转成向量（一串数字，表示语义），找语义最接近的。类比"找意思相近的"。
   - 什么是向量？把"Alice 是工程师"转成一串1024维的数字，意思相近的句子在数字空间里距离也近。

2. **BM25 全文检索**：一种关键词打分算法，类似更智能的"词频统计"——关键词越罕见且命中越多次，得分越高。类比"找包含相同词汇的"。

3. **图广度优先搜索（BFS）**：从已知的起始节点出发，在图里扩展到附近的实体和边。类比"找关系网里的近邻"。

三路结果用 **RRF（Reciprocal Rank Fusion，倒数排名融合）** 合并：把每条结果在三路中的排名取倒数相加，倒数排名之和最大的排最前。这种方法不需要三路分数可比，鲁棒性强。

最后，结果格式化成一段带时间戳的文本上下文，交给下游对话 LLM 使用。

---

## 三、实验：论文证明了什么

### 实验一：DMR 基准（Deep Memory Retrieval）

DMR 是 MemGPT 论文设立的基准，包含 500 段多轮对话，每段 5 个会话、最多 60 条消息，配一个问答对用于评测。

| 方法 | 分数 |
|---|---|
| 递归摘要（基线） | 35.3% |
| 对话摘要 | 78.6% |
| MemGPT | 93.4% |
| 全文直接塞进上下文 | 94.4% |
| **Zep（gpt-4-turbo）** | **94.8%** |
| 全文上下文（gpt-4o-mini） | 98.0% |
| **Zep（gpt-4o-mini）** | **98.2%** |

Zep 比 MemGPT 高 1.4%，但**直接把所有对话全文塞给 LLM 也能拿 94.4%**——说明 DMR 的对话太短，不需要记忆系统就能搞定。论文作者自己也承认这个基准有缺陷，"最关键的问题是，它远不能代表真实企业场景"。

### 实验二：LongMemEval（更能说明问题的基准）

LongMemEval 的对话平均 115,000 词，远超当时主流 LLM 的上下文窗口。它测试六种类型的问题，其中包括跨会话信息整合、时间推理等复杂任务。

| 方法 | 模型 | 准确率 | 延迟 | 平均上下文词数 |
|---|---|---|---|---|
| 全文上下文 | gpt-4o-mini | 55.4% | 31.3s | 115k |
| **Zep** | **gpt-4o-mini** | **63.8%** | **3.20s** | **1.6k** |
| 全文上下文 | gpt-4o | 60.2% | 28.9s | 115k |
| **Zep** | **gpt-4o** | **71.2%** | **2.58s** | **1.6k** |

关键结论：
- **准确率提升 15~18.5%**：特别是在多会话问题、偏好问题、时间推理问题上改善最明显——这些都是"信息随时间变化"或"需要整合多次对话"的场景，正是 Zep 的设计目标。
- **延迟降低约 90%**：从 30 秒降到 3 秒。根本原因是上下文从 115k 词压缩到 1.6k 词，LLM 处理的文本量大幅减少。
- **唯一变差的类型**：single-session-assistant（助手单次会话问题）下降了 9~18%，原因可能是某些细节在抽取为结构化事实时丢失了。

---

## 四、记忆条目长什么样

这是理解 Zep 的核心。Zep 里的"一条记忆"不是一整段对话摘要，也不是原始消息，而是一条**事实边（EntityEdge）**——两个实体之间的一条带时间戳的关系陈述。

### 一条事实边的完整结构（来自 `edges.py:263-285`）

```python
class EntityEdge:
    name: str           # 关系类型，如 "WORKS_AT", "IS_FRIENDS_WITH"
    fact: str           # 一句话描述这条事实，如 "Alice works at Acme Corp as a senior engineer"
    fact_embedding: list[float]  # 这句话的向量表示（1024 个浮点数）
    episodes: list[str]          # 产生这条事实的原始 episode UUID 列表（溯源）
    # 四个时间戳（双时序模型的核心）
    created_at: datetime    # 入库时间（T'）
    expired_at: datetime    # 在系统中被标记失效的时间（T'）
    valid_at: datetime      # 现实中这件事开始成立的时间（T）
    invalid_at: datetime    # 现实中这件事不再成立的时间（T）
```

### 一条情节节点（EpisodicNode，来自 `nodes.py:318-330`）

```python
class EpisodicNode:
    content: str            # 原始消息全文
    source: EpisodeType     # "message" / "text" / "json"
    source_description: str # 数据来源描述
    valid_at: datetime      # 这条消息发送的时间（参考时间戳）
    entity_edges: list[str] # 这条 episode 产生的所有 fact edge 的 UUID
```

### 一个实体节点（EntityNode）

```python
class EntityNode:
    name: str       # 实体名，如 "Alice"
    summary: str    # 关于这个实体的摘要，随每次 episode 处理而演化
    name_embedding: list[float]  # 名字的向量（用于去重检索）
```

### 具体例子：一条消息变成记忆的全过程

**输入消息：** "Alice: 我上周晋升为高级工程师了"（参考时间戳：2024-03-15）

经过各阶段变换：

**→ 情节节点（EpisodicNode）**
```json
{
  "content": "Alice: 我上周晋升为高级工程师了",
  "source": "message",
  "valid_at": "2024-03-15T10:30:00Z",
  "created_at": "2024-03-15T10:30:01Z"
}
```
这一步：原文存档，一字不改。

**→ LLM 抽取实体**
```
[Alice, 高级工程师职位]
```
这一步：LLM 读消息，找出值得作为节点的实体。说话者（Alice）必然被抽取。

**→ 实体去重（Alice 已在图里？）**

先用向量找候选（"Alice" 的向量和已有实体向量做余弦相似度），再用 MinHash 模糊匹配，最后如果还不确定才让 LLM 判断。假设 Alice 已经在图里，复用旧节点。

**→ LLM 抽取事实边**

LLM 在原消息上下文里提取：
```json
{
  "name": "PROMOTED_TO",
  "fact": "Alice was promoted to Senior Engineer",
  "valid_at": "2024-03-08T00:00:00Z"  // "上周" → 参考时间减7天
}
```
时间戳在这里算出来：参考时间 2024-03-15 减一周 = 2024-03-08。

**→ 矛盾检测**

LLM 发现图里有旧边：
```json
{
  "fact": "Alice works at Acme Corp as Software Engineer",
  "valid_at": "2022-01-01",
  "invalid_at": null
}
```
判定：新事实和旧事实矛盾（职位从"软件工程师"升到"高级工程师"）。

**→ 矛盾失效处理**

旧边被写上时间戳（不删除）：
```json
{
  "fact": "Alice works at Acme Corp as Software Engineer",
  "valid_at": "2022-01-01",
  "invalid_at": "2024-03-08",   // ← 新事实的 valid_at
  "expired_at": "2024-03-15"    // ← 当前时间
}
```

**→ 新边落库**
```json
{
  "fact": "Alice was promoted to Senior Engineer",
  "valid_at": "2024-03-08",
  "invalid_at": null,           // 目前仍然成立
  "expired_at": null,
  "episodes": ["episode-uuid-abc"]  // 溯源到原消息
}
```

**→ 检索时返回给 LLM 的上下文（来自 `search_helpers.py:27-72`）**

```
FACTS and ENTITIES represent relevant context to the current conversation.
<FACTS>
  {"fact": "Alice was promoted to Senior Engineer",
   "valid_at": "2024-03-08", "invalid_at": "Present"}
  {"fact": "Alice works at Acme Corp as Software Engineer",
   "valid_at": "2022-01-01", "invalid_at": "2024-03-08"}
</FACTS>
<ENTITIES>
  {"entity_name": "Alice", "summary": "Alice is a software engineer at Acme..."}
</ENTITIES>
```

下游 LLM 看到两条事实和它们的时间范围，自己判断用哪条（"Present"结尾的是当前有效的）。注意：图谱自己不过滤，把新旧都给 LLM 看，让 LLM 决定。

---

## 五、源码实现：系统怎么运转

### 整体架构一句话

graphiti 是个"图谱维护引擎"：写入时用 LLM 把文本结构化进图，读取时用混合检索从图里拉出相关事实，再拼成上下文字符串。它不负责维持对话，只负责"记忆的存和取"。

### 写入路径：`add_episode`（graphiti.py:980）

一条消息进来，经历以下串行步骤：

```
add_episode(episode_body, reference_time)
  ├─ 1. 存 EpisodicNode（原始消息落库）
  ├─ 2. retrieve_episodes（取最近10条消息作为LLM上下文）
  ├─ 3. extract_nodes（LLM 抽实体）
  ├─ 4. resolve_extracted_nodes（实体去重）
  │    ├─ 向量相似召回候选（top15，阈值≥0.6）
  │    ├─ 确定性匹配（精确名 → MinHash/LSH 模糊）
  │    └─ 未决者 → LLM 仲裁
  ├─ 5. extract_edges（LLM 抽事实三元组+时间窗）
  ├─ 6. resolve_extracted_edges（事实去重+矛盾失效）
  │    ├─ 每条新边做2次混合检索：找去重候选+找失效候选
  │    ├─ LLM 判 duplicate_facts / contradicted_facts
  │    └─ resolve_edge_contradictions（时间窗比较写失效时间戳）
  ├─ 7. extract_attributes_from_nodes（更新实体 summary）
  └─ 8. add_nodes_and_edges_bulk（批量落库，补 embedding）
```

**LLM 调用次数的代价**

一条典型消息大约触发 2~5+ 次 LLM 调用（下限是"抽节点+抽边"两次，但每条新边如果发现矛盾/重复候选就再加一次）。边越多，调用越多，成本越高。这是知识图谱方式相比"直接存全文"的主要代价。

**为什么实体去重要分三步（精确→模糊→LLM）？**

如果每次都直接调 LLM 判断两个名字是否是同一实体，费用高且慢。大多数情况下，"Alice"在图里已经有完全相同的名字，精确匹配就能搞定，根本不需要 LLM。MinHash/LSH 是一种哈希技术，能快速判断两段文本有多少词重叠（比全文比较快得多），用来处理"爱丽丝"和"Alice Smith"这类拼写略不同的情况。只有真正模糊的情况才交 LLM，把计算成本分层。

### 矛盾处理的核心设计（代码里最完整的部分）

矛盾判定 prompt（`dedupe_edges.py:43-100`）同时要求 LLM 返回两个列表：

```
duplicate_facts: [0, 2]   // 纯重复，同一信息
contradicted_facts: [1, 3] // 矛盾（可以和 duplicate 重叠）
```

同一条旧边可以同时出现在两个列表（"同一关系，但值变了——既重复又矛盾"）。这个设计允许 LLM 只做语义判断，时间窗的比较交给后面的确定性代码（`resolve_edge_contradictions`，`edge_operations.py:538-573`）。

```python
# 简化版逻辑：
if old_edge.valid_at < new_edge.valid_at:
    # 旧事实比新事实早，旧的被新的取代
    old_edge.invalid_at = new_edge.valid_at
    old_edge.expired_at = now
else:
    # 旧事实比新事实晚，说明新事实自己已经过期了
    new_edge.expired_at = now   # 新边当场失效
```

语义判断（LLM）和时序裁决（纯代码）分离，是这个系统里最干净的一个设计决策。

### 读取路径：`search()`（graphiti.py:1527）

```
search(query)
  ├─ 把 query 转成向量
  ├─ 三路并行检索：
  │    ├─ 向量相似搜索（边的 fact_embedding）
  │    ├─ BM25 全文检索（边的 fact 字段文本）
  │    └─ （可选）BFS 图遍历
  ├─ RRF 融合（倒数排名求和）
  └─ 格式化成上下文字符串
```

**重要细节：检索默认不过滤失效边**

`SearchFilters` 里，时间过滤字段默认全是 `None`（`search_filters.py:62-65`）。也就是说，一个两年前被标记为"失效"的旧事实，默认情况下也会出现在检索结果里（带着它的 `invalid_at` 时间戳）。库不帮你过滤，而是把时效信息带给下游 LLM，让 LLM 自己判断哪条有效。

这是一个设计权衡：避免库错误地丢弃"对历史查询有用"的旧事实，代价是调用方需要自己构造 `DateFilter` 才能"只看当前有效的事实"。

### 实体摘要演化

每次 episode 处理完，相关实体的 `summary` 会被更新（`node_operations.py:833-910`）。更新策略：如果新 fact + 旧 summary 加起来不超过 2000 字符，就直接把新 fact 字符串拼到 summary 里，不调 LLM（省钱）。超过了才批量调 LLM 重写摘要。

代价：summary 在大多数时候处于"fact 字符串堆积"的状态，并不是真正意义上的"摘要"，直到触发 LLM 重写之前都比较冗余。

### 社区子图：几乎默认关闭

论文用了相当篇幅介绍社区子图，但在代码里：
- `add_episode` 默认参数 `update_communities=False`（graphiti.py:989），不更新社区
- 手动调用 `build_communities` 是全删重建（先 `remove_communities`，再重新跑聚类）
- 增量更新（`update_community`）只在 opt-in 时触发，且方式粗糙（把新实体 summary 和社区 summary 拼在一起让 LLM 重写一次，不重新聚类）

实际上，除非你显式地定期调 `build_communities`，社区子图是不工作的。

---

## 六、论文宣称 vs 代码实际

这部分列出阅读/引用时需要注意的偏差：

| 论文宣称 | 代码实际 | 影响 |
|---|---|---|
| 双时序模型（T 和 T'） | 完全实现，四个时间戳分别对应事件时间线和入库时间线 | 一致，可放心引用 |
| "失效不删除" | 完全实现，失效边和新边一起保存 | 一致 |
| 可在任意时间点查历史状态 | 不是默认行为，要调用方自己构造 DateFilter；默认检索会把过期边也返回 | **打折扣**：论文说"可以查询任意时间点的状态"，实际上没有封装好的"时间点查询"接口 |
| 社区子图动态更新 | 默认关闭；全量重建是全删再重建，不是增量 | **打折扣**：论文强调动态更新，代码里增量更新路径粗糙且默认不启用 |
| cross-encoder reranker | 实际实现是给每个候选发一次 LLM 请求，问"这个结果和 query 相关吗（True/False）"，拿 True 的 logprob 当分数。不是真正的 cross-encoder（真正的 cross-encoder 会把 query 和 passage 拼在一起输入） | **表述不准确**：虽然效果类似，但代价和实现原理不同 |
| IS_DUPLICATE_OF 去重溯源边 | 主写入路径里这条边被直接丢弃（graphiti.py:1131），疑似留给闭源部分使用 | **缺失功能**：图里实际没有去重溯源边 |

---

## 七、实现里的工程瑕疵

以下问题是阅读代码时注意到的，如果你拿 graphiti 做基准对比或引用代码，需要知道这些：

1. **`node_distance_reranker` 说"最短路径"，实际是"是否1-hop"**：注释写"按到中心节点的最短路径排序"，实际 Cypher 只检查是否直接相连（命中得1分，未命中得∞），退化为二值判断，不是路径长度（`search_utils.py:1816-1845`）。

2. **每条新边的 fact 至少被 embed 三次**：落库前算一次，去重检索时再把同一个 fact 作为 query embed 一次，失效候选检索时又 embed 一次（`edge_operations.py:363,392-418` 没有复用已算好的向量）。这是成本浪费。

3. **实体摘要"快速路径"产生 fact 堆积**：直接拼接字符串（不整合、不归纳）的摘要积累久了就不像摘要了（`node_operations.py:873-879`）。

4. **社区摘要两两归并成本高**：`build_community`（`community_operations.py:174-213`）对有 N 个成员的社区做类似归并排序的操作——两两 summarize，O(N) 次 LLM 调用，信息在多层合并中损耗不可控。

5. **无任何自动遗忘/衰减机制**：没有 TTL、容量上限、decay 参数。图只增不减（手动 `remove_episode` 除外）。源码里搜索 `decay/forget/prune/evict/ttl`，只在注释和 prompt 文案里的 "forget" 字样里命中，没有任何机制实现。这意味着长期运行的图会无限增长，是明显的未解决问题。

---

## 八、对研究的启示

**Zep 真正的贡献**：把"时序信息"系统性地引入知识图谱记忆，矛盾处理是当前所有开源 agent 记忆项目里最完整的一套。

**明显的空白**：
- **无遗忘机制**：图无界增长，检索噪声会随时间增加。如何设计遗忘/压缩策略，是可以做文章的方向。
- **写入成本**：每条消息 2~5+ 次 LLM 调用，实际部署代价高。如何用更小模型或规则替代部分 LLM 调用，是工程优化方向。
- **时间过滤未默认封装**：论文宣称的"时间点查询"能力要靠调用方自己实现，是易用性缺口。
- **社区摘要的质量和成本**：两两归并方式信息损耗不可控，且是全量重建，频繁刷新成本很高。

**值得借鉴的设计决策**（做基线或自己设计时可以参考）：
- 矛盾处理的"语义判断交 LLM + 时序裁决用确定性代码"解耦设计
- 去重的"精确匹配 → 模糊匹配 → LLM"三级成本分层
- "失效不删除，把时效判断推给读侧 LLM"的保守保全策略

---

## 关键代码位置速查

| 机制 | 文件 | 行号 | 作用 |
|---|---|---|---|
| 写入主管线 | `graphiti_core/graphiti.py` | 980-1228 | `add_episode` 完整流程 |
| 事实边数据模型 | `graphiti_core/edges.py` | 263-285 | `EntityEdge`，四个时间戳定义 |
| 情节节点数据模型 | `graphiti_core/nodes.py` | 318-330 | `EpisodicNode`，原始消息存档 |
| 实体节点去重-向量召回 | `graphiti_core/utils/maintenance/node_operations.py` | 418-450 | `_semantic_candidate_search` |
| 实体去重-MinHash/精确名 | `graphiti_core/utils/maintenance/dedup_helpers.py` | 220-279 | `_resolve_with_similarity` |
| 事实抽取 | `graphiti_core/utils/maintenance/edge_operations.py` | 117-322 | `extract_edges`，带时间窗 |
| 矛盾+去重判定 prompt | `graphiti_core/prompts/dedupe_edges.py` | 43-100 | `resolve_edge`，双列表协议 |
| 矛盾失效裁决（纯代码） | `graphiti_core/utils/maintenance/edge_operations.py` | 538-573 | `resolve_edge_contradictions` |
| 实体摘要演化 | `graphiti_core/utils/maintenance/node_operations.py` | 833-910 | `_extract_entity_summaries_batch` |
| 检索编排 | `graphiti_core/search/search.py` | 98-460 | 向量+BM25+BFS 并行 |
| RRF 融合 | `graphiti_core/search/search_utils.py` | 1780-1795 | `rrf` 倒数排名求和 |
| 结果格式化进 prompt | `graphiti_core/search/search_helpers.py` | 27-72 | `search_results_to_context_string` |
| 社区构建 | `graphiti_core/utils/maintenance/community_operations.py` | 93-213 | 聚类+两两归并摘要 |
| Saga 增量摘要 | `graphiti_core/graphiti.py` | 438-568 | `summarize_saga`，双水位线 |
