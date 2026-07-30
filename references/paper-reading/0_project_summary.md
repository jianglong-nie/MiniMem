# 0_summary — 13 个 agent memory 项目横向对比

> 说明：每条引用的代码证据格式为 `文件名:行号`，均基于各项目文档开头标注的 commit。
> 笔记中未记录的内容统一写"笔记未记录"。

---

## 一、逐项目记录

### Mem0

**仓库**：mem0ai/mem0，commit b819d95d  
**论文**：arXiv:2504.19413，v3 当前代码与论文描述是两套不同机制（见第 8 条）

---

1. **记忆形式与粒度**

   一条记忆是**自包含的事实陈述句**（15–80 词），要求代词替换为真实姓名、相对时间绝对化，使得每条事实脱离对话上下文也能被独立读懂。

   对外暴露的 API 对象 `MemoryItem`（`mem0/configs/base.py:16-26`）字段：

   | 字段 | 含义 |
   |------|------|
   | `id` | UUID，每条记忆的唯一身份 |
   | `memory` | 事实文本本身 |
   | `hash` | MD5 摘要，用于精确去重 |
   | `metadata` | 自定义扩展字段 |
   | `score` | 检索相关性得分（0–1） |
   | `created_at` / `updated_at` | 时间戳 |

   实际存入向量库的 payload（`main.py:834-842`）还有：`data`（同 `memory`）、`text_lemmatized`（词形还原文本，供 BM25 用）、`attributed_to`（"user"/"assistant"）、`user_id/agent_id/run_id`（作用域隔离）、`actor_id`、`role`。

   **粒度**：细粒度，事实级。每条记忆是一个自包含的陈述句，如"Alice recently started a new job as a Machine Learning Engineer at Stripe, as of June 2026."

   **与其他粒度方案的本质区别**：不是原始对话 chunk（RAG），不是 QA 对，不是摘要段落，不是图节点。核心变换是**代词消解**（"我"→"Alice"）和**时间绝对化**（"上周"→"2026-06-08"），让记忆在离开原始对话后仍可被独立理解和检索。

---

2. **写入机制**

   **触发时机**：每次调用 `Memory.add()` 时立即同步触发（`infer=True` 默认；`infer=False` 跳过 LLM 直存；`memory_type="procedural_memory"` 走摘要路径）。

   **LLM 参与次数**：整个 8 阶段管线中**仅 1 次 LLM 调用**（阶段 2，`main.py:751-771`）。

   **提取过程（8 阶段，`main.py:725` 注释）**：
   - 阶段 0：SQLite 取最近 10 条原始消息（供 LLM 理解代词指代）
   - 阶段 1：嵌入新消息 → 向量检索 top-10 旧记忆（让 LLM 知道已有哪些事实）
   - 阶段 2：**唯一 LLM 调用**，使用 `ADDITIVE_EXTRACTION_PROMPT`（`prompts.py:468`，约 480 行），从消息中提取事实 JSON 数组；prompt 要求"sole operation is ADD"，禁止 UPDATE/DELETE；提供 Observation Date 和 Current Date 让 LLM 把相对时间转换为绝对日期
   - 阶段 3：批量嵌入新事实文本（本地 embedding 模型）
   - 阶段 4：MD5 精确去重（`main.py:825-829`），命中已有记忆则跳过该条
   - 阶段 5：spaCy 词形还原，生成 `text_lemmatized` 字段
   - 阶段 6：批量写入向量库（Qdrant）+ SQLite history 审计日志
   - 阶段 7：spaCy 实体抽取 → 写入独立实体向量库，建立"实体→记忆 ID"反向链接（`main.py:891-981`）
   - 阶段 8：原始消息写入 SQLite（每个作用域只保留最近 10 条）

   **过滤/筛选逻辑**：
   - MD5 精确匹配去重（字面完全相同才跳过，无语义去重）
   - LLM 可返回空列表（无新事实时不写入）
   - Prompt 禁止任何 UPDATE/DELETE 决策，强制 ADD-only

---

3. **检索机制**

   入口 `Memory.search()` → `_search_vector_store()`（`main.py:1373`），全程 **0 次 LLM 调用**，融合三路信号：

   **语义分**：query 文本转向量，余弦相似度，over-fetch `max(limit×4, 60)` 条候选池，过滤相似度 < 0.1 的条目。

   **BM25 关键词分**：对候选池在 `text_lemmatized` 字段上做 BM25 检索（`main.py:1392`），原始分经 sigmoid 归一化（`scoring.py:16-54`），斜率随 query 长度自适应（短 query 对关键词命中更敏感）。弥补语义检索在专有名词（如公司名、人名）上的盲点。

   **实体 boost**：spaCy 从 query 中抽实体 → 批量嵌入 → 查实体库 top-500，余弦相似度 ≥ 0.5 的实体把 boost 传播给其 `linked_memory_ids` 里的记忆。热门实体（关联记忆数多）boost 被 `1/(1+0.001×(n-1)²)` 压制，避免高频实体（如"Alice"）把所有检索结果带偏（`main.py:1473-1553`，boost 上限 0.5）。

   **融合排序**（`scoring.py:60-139`）：
   ```
   final_score = (semantic_score + bm25_score + entity_boost) / max_possible
   ```

   **召回后有无重排**：可选 reranker（cross-encoder），默认未配置。

   **结果数量控制**：`limit` 参数（默认 5–20），候选池 `max(limit×4, 60)` 条。

---

4. **注入 prompt**

   核心库本身不管注入，仅返回记忆列表；注入逻辑由应用层决定。仓库内参考实现在 `mem0/proxy/main.py:186-191`：

   ```python
   relevant = "\n".join(f"- {m['memory']}" for m in memories["results"])
   modified_content = f"Relevant Memories/Facts:\n{relevant}\n\nUser Message:\n{original_content}"
   ```

   **格式**：每条记忆拼成"- `<文本>`"的项目符号列表。  
   **插入位置**：用户最后一条消息的内容之前（user 消息体内，非 system prompt）。  
   **截断/优先级控制**：proxy 实现无截断控制，仅靠检索侧的 limit 参数控制数量。

---

5. **记忆管理**

   **去重**：MD5 精确匹配（`main.py:825-829`）；仅字面完全相同才跳过，同义改写的事实会重复累积。实体库用 0.95 余弦相似度对实体去重（`main.py:891-981`）。**无语义级去重**。

   **矛盾处理**：无。ADD-only 下矛盾事实（"用户住北京"和"用户已搬去上海"）并存，靠检索时"含绝对日期的新事实通过 BM25/语义排到前面"自然抑制旧版本。这个设计假设在极端矛盾场景（同一事实多次反转）下可能失效。

   **遗忘/淘汰**：**无，默认不生效**。无 TTL、无时间衰减、无容量上限、无自动删除。SQLite 只保留最近 10 条消息是为了控制写入时的上下文窗口，不是遗忘机制。删除仅有手动 API：`Memory.delete()`/`delete_all()`/`reset()`（`main.py:1578-1806`）。

   **整合/抽象**：无。没有后台任务把重复事实合并或生成用户画像摘要。Prompt 里虽有 Summary 和"Recently Extracted Memories"两个输入区（`prompts.py:496-503`），但开源版调用处传的是空字符串（`main.py:757-762`），功能仅在付费云端版存在。

---

6. **其他设计**

   - **实体知识图（轻量图替代）**：第二个向量 collection 存命名实体，`linked_memory_ids` 数组建立实体→记忆的反向链，热门实体阻尼防止高频实体带偏检索结果（`main.py:891-981`）。不用图数据库（Neo4j），用第二个向量 collection + 反链，以极低成本拿到关系感知检索的大部分收益。
   - **procedural_memory 路径**：1 次 LLM 把整段对话总结成流程记忆（"用户设置 Python 环境的步骤"），适合存操作流程（`main.py:1672-1709`）。
   - **三存储层**：Qdrant 向量库（主记忆）+ 实体向量库（反链索引）+ SQLite（history 审计日志 append-only + messages 缓存最近 10 条）。
   - **NLP 静默降级**：spaCy 未安装时词形还原返回原文、实体抽取返回空列表；fastembed 未安装时 BM25 整体禁用（`qdrant.py:96`）——三信号在依赖不全时退化为纯语义，benchmark 数字无法复现时很难察觉。

---

7. **核心创新点**

   **论文（v2）宣称**：两段式智能决策管线——先提取事实，再由 LLM 做 ADD/UPDATE/DELETE/NONE 四选一决策，主动维护记忆一致性，防止旧错误信息累积。

   **实际代码（v3）真正采用**：**ADD-only 写入 + 三信号融合检索**（语义 + BM25 + 实体 boost），写入端零维护，靠检索侧弥补。

   **与同类方案的本质区别**：
   - vs 传统 RAG：存的是提炼后的事实句而非原始 chunk，代词消解 + 时间绝对化让记忆自包含
   - vs A-MEM：不做写入时的图更新，用轻量实体反链代替显式知识图谱
   - vs SimpleMem：无缓冲即时写入，三信号检索 vs SimpleMem 的三路检索，但 Mem0 无符号层 SQL 过滤

   v2→v3 的转变本身是一个有趣的研究发现：论文提出智能决策管线（v2），实验后发现 ADD-only + 强检索（v3）反而 LoCoMo 高约 20 点，说明"写入时 LLM 维护一致性"在实证上输给了"检索时多信号融合"。

---

8. **论文 vs 代码差异**

   - **两段式 ADD/UPDATE/DELETE/NONE 整体被移除**：`FACT_RETRIEVAL_PROMPT`（`prompts.py:15`）、`DEFAULT_UPDATE_MEMORY_PROMPT`（`prompts.py:176`）、`get_update_memory_messages`（`prompts.py:406`）、`get_fact_retrieval_messages`（`memory/utils.py:15`）均无任何调用方。2026-04-14 commit a488e190 完成替换。**如需复现论文方法须回退该 commit 之前版本。**

   - **图记忆变体 Mem0g 被删除**：`mem0/graphs/` 目录不存在，`MemoryConfig` 无 `graph_store` 字段（`configs/base.py:29-57`）。`proxy/main.py:187` 读取的 `relevant_memories["relations"]` 是残留死代码（图记忆已删，relations 永远为空）。

   - **Memory Linking 是摆设**：`ADDITIVE_EXTRACTION_PROMPT` 花大量篇幅要求 LLM 输出 `linked_memory_ids`（`prompts.py:692-701`），但管线只取 `text` 和 `attributed_to`（`main.py:821`），`linked_memory_ids` 和 `uuid_mapping` 双双被丢弃，从未落库。

   - **Summary 和 Recently Extracted Memories 恒为空**：开源调用点 `main.py:757-762` 不传这两个参数，"用户画像"功能在开源版不存在。

   - **"时间感知检索"实际只是日期字符串参与文本匹配**：`scoring.py:60-139` 的融合分完全不含时间项，无时间衰减、无时间偏好权重，"时间能力"来自抽取时写入绝对日期字符串，靠 BM25/语义自然排序。

---

9. **实验**

   **数据集**：LoCoMo（多轮对话记忆评估，含隐式/显式记忆检索任务）、LongMemEval（长时间跨度，测 assistant 的动作记忆）、BEAM（生产规模，1M 和 10M token 对话历史）。

   **基线**：ReadAgent、MemoryBank、MemGPT、A-Mem、LangMem；RAG（不同 chunk size + 不同 top-K）；全上下文；OpenAI Memory（ChatGPT 内置）；Zep。

   **核心结论**（v3 管线，`README.md:47-53`，同一模型栈下）：

   | 基准 | v2 旧算法 | v3 新算法 | token 消耗 | 延迟 p50 |
   |------|-----------|-----------|-----------|---------|
   | LoCoMo | 71.4 | **91.6** | 7.0K | 0.88s |
   | LongMemEval | 67.8 | **94.8** | 6.8K | 1.09s |
   | BEAM (1M) | — | **64.1** | 6.7K | 1.00s |
   | BEAM (10M) | — | **48.6** | 6.9K | 1.05s |

   v3 vs 全上下文：LoCoMo 高约 20 点，token 消耗少约 85%，延迟低约 91%。评测代码已开源（`evaluation/` 目录）可复现。单次检索无 agentic 循环。

---

### SimpleMem

**仓库**：aiming-lab/SimpleMem，commit 74174a1  
**论文**：arXiv:2601.02553，SimpleMem: Efficient Lifelong Memory for LLM Agents（2026 年 1 月）

---

1. **记忆形式与粒度**

   一条记忆是 `MemoryEntry` 结构化记录（`simplemem/core/models/memory_entry.py:13`），核心是 `lossless_restatement`（无损改写句）：

   | 字段 | 含义 | 服务哪层检索 |
   |------|------|------------|
   | `lossless_restatement` | 自包含事实陈述句（禁止代词和相对时间） | 语义层（被向量化） |
   | `keywords` | 关键词列表（人名、地点、产品名等） | 词法层（BM25） |
   | `timestamp` | ISO 8601 绝对时间戳（如 `2023-07-02T14:32:00`） | 符号层（时间范围过滤） |
   | `location` | 地点描述 | 符号层（地点过滤） |
   | `persons` | 人名列表 | 符号层（人名过滤） |
   | `entities` | 实体列表（公司、产品、组织） | 符号层（实体过滤） |
   | `topic` | 主题短语 | 辅助理解 |
   | `entry_id` | UUID（系统自动生成） | 去重合并 |

   具体例子（对话"Sarah yesterday signed up for a pottery class"）提炼后：
   ```json
   {
     "lossless_restatement": "Sarah signed up for a pottery class on 2023-07-01 and finds it therapeutic.",
     "keywords": ["Sarah", "pottery class"],
     "timestamp": "2023-07-01T00:00:00",
     "persons": ["Sarah"],
     "topic": "Sarah's pottery hobby"
   }
   ```

   **粒度**：事实级，有丰富的结构化元数据字段（比 Mem0 多出显式时间戳、人物字段，比 A-MEM 更细）。

   **与其他粒度方案的本质区别**：（1）比 Mem0 多出显式结构化字段（timestamp、persons、location 等），支持精确 SQL 过滤，Mem0 的时间信息只以字符串形式嵌在 `memory` 文本里；（2）比 A-MEM 粒度更细，A-MEM 存原始对话文本，SimpleMem 强制提炼成原子事实句；（3）三层索引（向量/BM25/SQL）共用一张 LanceDB 表，无需维护多个存储系统。

---

2. **写入机制**

   **触发时机**：**非即时**，缓冲区积累到 `WINDOW_SIZE=40` 条对话后触发（`config_default.py:60`），每窗口步进 38 条（`OVERLAP_SIZE=2`，`config_default.py:63`），保留最后 2 条作为下一窗口的上下文重叠。

   **LLM 参与次数**：每窗口 **1 次 LLM API 调用**（`memory_builder.py:170`），摊到每条对话约 1/38 次；本地 embedding 模型（Qwen3-Embedding-0.6B）不走 API。

   **提取过程**：
   - `add_dialogue()` 把对话追加到缓冲区（`memory_builder.py:58`），不触发任何模型
   - 达到 40 条时 `process_window()` 被触发（`memory_builder.py:132`）
   - `_generate_memory_entries()` 构建提取 prompt（`memory_builder.py:170`）：要求 Complete Coverage（覆盖所有信息）、Force Disambiguation（禁止代词和相对时间）、提取 6 类元数据（keywords/timestamp/location/persons/entities/topic）；附上前一窗口的前 3 条条目作为"避免重复"的软提示（`memory_builder.py:181`）
   - LLM 返回 JSON 数组，解析失败最多重试 3 次（`memory_builder.py:201`）
   - `VectorStore.add_entries()` 批量向量化并写入 LanceDB（`vector_store.py:121`）；首次写入后额外建立 Tantivy FTS 全文索引（`vector_store.py:74`）

   **过滤/筛选**：**无**。写入侧纯追加（`vector_store.py:143`），无相似度去重。LLM 理论上可以返回空数组（论文称为"语义密度门控"），但 prompt 第一条要求是"Complete Coverage"，与过滤方向相反；实际无任何显式过滤逻辑。

   **并行策略**：超过 80 条对话时用 `ThreadPoolExecutor` 同时处理多个窗口（`memory_builder.py:338`）。副作用：并行时 `previous_entries`（上一窗口前 3 条）在整批完成前不更新，"避免重复"的软提示在并行模式下几乎失效（`memory_builder.py:371`）。

---

3. **检索机制**

   整体流程：LLM 规划信息需求 → 生成子查询 → 三路并行检索 → 去重合并 → 反思补查（可选）→ 生成答案。

   **检索端 LLM 参与次数**：默认配置（`ENABLE_PLANNING=True`，`ENABLE_REFLECTION=True`，`MAX_REFLECTION_ROUNDS=2`）下至少 **2 次**（信息需求分析 + 子查询生成），含反思最多 **4 次**（每轮 1 次判断 + 1 次补充查询）。

   **三路并行检索**（无加权，结果按 `entry_id` 去重合并）：
   - **语义层**（`vector_store.py:150`）：Qwen3-Embedding-0.6B 向量化 → LanceDB 余弦相似度，top-25（`SEMANTIC_TOP_K=25`，`config_default.py:71`）
   - **词法层**（`vector_store.py:167`）：LLM 提取关键词（`hybrid_retriever.py:176`）→ Tantivy BM25，top-5（`KEYWORD_TOP_K=5`）
   - **符号层**（`vector_store.py:185`）：从 query 分析中提取人名/时间范围/地点 → SQL 条件过滤（`array_has_any(persons, make_array('Sarah'))`），top-5（`STRUCTURED_TOP_K=5`）

   **重排**：无。三路结果合并后直接传给答案生成。

   **反思补查**（`hybrid_retriever.py:794`）：最多 2 轮，每轮判断当前信息是否完整，不完整则再生成补充查询。

---

4. **注入 prompt**

   每条 `MemoryEntry` 格式化为带标题的结构化块（`answer_generator.py:85`）：

   ```
   [Context 1]
   Content: Sarah and her kids painted a sunset with palm trees on 2023-06-25.
   Time: 2023-06-25T14:39:00
   Persons: Sarah

   [Context 2]
   Content: Sarah finished painting a horse portrait on 2023-07-14.
   Time: 2023-07-14T19:27:00
   Persons: Sarah
   ```

   **插入位置**：`AnswerGenerator` 的 QA prompt 中的 Context 段（`answer_generator.py:113`），最终调用一次 LLM 生成 JSON 格式答案（含 `reasoning` 和 `answer` 字段）。

   **截断/优先级控制**：无显式截断，进入最终答案 prompt 的条目数量由三路 top-k 的合并结果决定。

---

5. **记忆管理**

   **去重**：**无**。写入纯追加（`vector_store.py:143`），语义重复的事实（"Sarah 喜欢咖啡"和"Sarah 爱喝咖啡"）会同时存在。仅检索侧按 `entry_id` 合并三路结果（`hybrid_retriever.py:409`）。

   **矛盾处理**：**无**。矛盾事实（"Sarah 喜欢咖啡"和"Sarah 说自己不喝咖啡了"）并存，靠最后答案生成 LLM 自己裁断。无版本字段或有效期字段。

   **遗忘/淘汰**：**无，默认不生效**。`VectorStore` 只有整表清空的 `clear()` 方法（`vector_store.py:245-250`），无单条删除，无 importance 字段。`cross/` 子系统（`cross/consolidation.py`）实现了 90 天 decay × 0.9、合并余弦 > 0.95 的条目、清除重要性 < 0.05 的条目，但 session 收尾时 `consolidation_triggered` 被硬编码为 `False`（`cross/session_manager.py:394`），无生产调用路径，仅被测试文件引用。

   **整合/抽象**：核心管线中**无**。已存条目从不被读出来重组。`cross/consolidation.py` 有合并逻辑但无实际调用（见遗忘部分）。

---

6. **其他设计**

   - **三层索引共用一张 LanceDB 表**：向量索引 + Tantivy FTS（BM25）+ 元数据 SQL 过滤，工程成本极低，不需要维护多个存储系统同步（`vector_store.py:74`）。
   - **本地 embedding**：Qwen3-Embedding-0.6B，批量计算不走 API，降低成本且加快速度（`vector_store.py:121`）。
   - **批量并行处理**：超 80 条对话时 `ThreadPoolExecutor` 并行处理多窗口（`memory_builder.py:338`）。
   - **反思补查循环**：最多 2 轮，每轮 LLM 判断信息完整性后决定是否补充查询（`hybrid_retriever.py:794`）。
   - **答案生成 LLM**：返回 JSON 格式，含 `reasoning` 和 `answer` 字段（`answer_generator.py:113`）。

---

7. **核心创新点**

   作者宣称最核心的贡献：**把整理工作前移到写入时（write-time structured compression）**——对话发生时一次性压缩成自包含事实条目，消除代词和相对时间；检索时不需要再回溯或重建上下文，天然省 token。

   三阶段管线（虽有论文与代码的落差，见第 8 条）：
   1. 语义结构化压缩（Semantic Structured Compression）
   2. 在线语义合成（Online Semantic Synthesis）—— 论文概念，代码中未实现
   3. 意图感知检索规划（Intent-Aware Retrieval Planning）

   **与同类方案的本质区别**：
   - vs Mem0：显式结构化字段（timestamp/persons/location 等）支持 SQL 精确过滤；Mem0 的时间信息只是文本字符串；SimpleMem 用缓冲批量写入摊薄 LLM 成本
   - vs A-MEM：写一次 LLM 对比 A-MEM 最多写两次；SimpleMem 无记忆间链接
   - 优势在时间推理（显式绝对时间戳）和 token 效率（531 vs Mem0 的 ~973 token/次）

---

8. **论文 vs 代码差异**

   - **"在线语义合成"基本不存在**：论文 Section 2.2 用公式 $F_{syn}(O_{session}, C_{context}; f)$ 描述"把当前 session 内相关碎片合并为统一高密度条目，在写入数据库之前完成"，给出了三条碎片合并为一条的例子。代码里没有任何读出已存条目再合并的步骤，写入是纯追加。`MemoryBuilder` 类注释（`memory_builder.py:29`）偷换概念，把"合成（synthesis）→更少条目"改成了"生成足够多的条目（generating enough）→更多条目"，方向完全相反。最接近"合成"的两处软机制（附上前 3 条条目作为软提示 + prompt 要求"Complete Coverage"）都在写入当前窗口时生效，既不读已存条目，也不合并已存条目。

   - **"语义密度门控 Φ_gate"只是一条 prompt 指令**：论文描述为"输出空集代表纯寒暄被过滤"的显式门控机制，代码里只是 LLM 可以选择返回空数组（`memory_builder.py:170-227`），无额外分类器、阈值或决策逻辑；而且 prompt 第一条要求是"Complete Coverage"，与过滤方向相反。

   - **"动态检索深度 d"实际是固定常数**：论文描述规划器输出 d 代表检索深度，n ∝ d，系统动态调整取多少条候选。实际三路检索的 top_k 均来自配置文件的固定常数（`SEMANTIC_TOP_K=25`，`KEYWORD_TOP_K=5`，`STRUCTURED_TOP_K=5`，`config_default.py:71-77`），LLM 规划只影响子查询数量（1–4 条），不影响每路取多少条。

   - **`dialogue_ids` 字段从未写入条目**：传参路径中存在，但实际落库时被丢弃（`memory_builder.py:187`），记忆条目和原始对话之间没有反向链接。

   - **`cross/` 子系统 decay/merge/prune 无生产调用**：`cross/README.md:76-77` 宣称自动管理，但 `cross/session_manager.py:394` 硬编码 `consolidation_triggered = False`，仅被测试文件引用。

   **消融实验的可解释性问题**：论文 Table 5 中"w/o Online Synthesis"导致 Multi-hop F1 下降 31.3%，但论文没有说明这个消融具体改了什么代码（因为该功能在代码中根本不存在）。数字本身可信（某个改动确实导致了性能下降），但不能证明"在线语义合成有效"这个解释。

---

9. **实验**

   **数据集**：LoCoMo（多轮对话记忆评估，200–400 轮/样本，评测集 1986 题，4 类：多跳推理/时间推理/开放域/单跳）、LongMemEval-S（极长上下文，跨 session 用户偏好和时间事件）。

   **基线**：全文塞入（Full Context）、ReadAgent、MemoryBank、MemGPT、A-MEM、LightMem、Mem0。

   **LLM 后端**：GPT-4o、GPT-4.1-mini、Qwen3-Plus、Qwen2.5-1.5B/3B、Qwen3-1.7B/8B。

   **核心结论（LoCoMo + GPT-4.1-mini）**：

   | 指标 | SimpleMem | Mem0 | Full Context |
   |------|-----------|------|-------------|
   | Average F1 | **43.24** | 34.20 | 18.70 |
   | Temporal F1 | **58.62** | 48.91 | — |
   | tokens/次 | **531** | ~973 | ~16,910 |
   | 构建速度（s/样本） | **92.6** | 1350.9 | — |

   - A-MEM 构建速度 5140.5 s/样本，SimpleMem 快 55 倍，原因是 A-MEM 做图更新，SimpleMem 每 40 条只调 1 次 LLM。
   - 小模型适用：Qwen2.5-1.5B + SimpleMem（25.23 F1）> Qwen3-1.7B + Mem0（21.19 F1）。
   - 消融实验（Table 5）：去掉"语义压缩"后 Temporal F1 从 58.62 跌到 25.40（-56.7%），说明写入时统一时间戳对时间推理最关键。
   - 论文附录 B.3（Table 6）超参数分析：k 从 1 到 20 时 SimpleMem 在 k=3 达到峰值 99%，说明条目信息密度高。

---

### A-MEM

**仓库**：WujiangXu/A-mem-sys（系统实现，commit f303dfc）；WujiangXu/AgenticMemory（论文复现，commit 0c8039f）  
**论文**：arXiv:2502.12110，已收录 NeurIPS 2025

---

1. **记忆形式与粒度**

   一条记忆是 `MemoryNote` 结构化笔记对象，灵感来自 Zettelkasten 卡片盒笔记法（`memory_system.py:24-81`）：

   | 字段 | 含义 | 备注 |
   |------|------|------|
   | `id` | UUID，系统自动生成 | 稳定唯一标识 |
   | `content` | **原始对话文本**（如"Speaker Alice says: ..."） | 核心内容，被 enhanced embedding |
   | `keywords` | LLM 提取的关键词列表 | 参与 enhanced embedding |
   | `context` | LLM 生成的一句话摘要 | 参与 enhanced embedding |
   | `tags` | LLM 生成的分类标签列表 | 参与 enhanced embedding；进化时可被更新 |
   | `links` | 关联旧记忆的 UUID 列表（进化阶段填入） | A-mem-sys 存 UUID；AgenticMemory 原仓存整数下标（危险！） |
   | `timestamp` | 写入时间（格式 `YYYYMMDDHHmm`） | 无标准时区 |
   | `last_accessed` | 最近访问时间 | **死字段**，从不更新 |
   | `retrieval_count` | 检索次数 | **死字段**，从不更新 |
   | `evolution_history` | 进化历史 | **死字段**，从不写入 |

   **粒度**：对话级（`content` 是原始对话消息的字面文本），比 Mem0 和 SimpleMem 粒度粗。但附有 LLM 生成的语义元数据和显式链接，形成知识网络结构。

   **与其他粒度方案的本质区别**：（1）存的是原始对话文本而非提炼后的事实句（粒度粗），Mem0 和 SimpleMem 都要求代词消解和时间绝对化，A-MEM 不做；（2）有显式的记忆间链接字段（`links`），形成知识网络，其他方案是孤立条目；（3）元数据由 LLM 生成且参与向量化（元数据增强 embedding），而非单纯存储元数据。

---

2. **写入机制**

   **触发时机**：每次调用 `add_note()`/`add_memory()` 时**立即同步触发**，无缓冲。

   **LLM 参与次数**：最多 **2 次**（`analyze_content` 1 次 + `process_memory` 进化决策 1 次）。若 `keywords/context/tags` 已有值则跳过 `analyze_content`（省 1 次）；记忆库为空时跳过 `process_memory`（省 1 次）。

   **提取过程（`memory_system.py:237-284`）**：

   - Step 1：创建 `MemoryNote` 对象，`keywords=[]`，`context="General"`，`tags=[]`
   - Step 2：**LLM 调用 `analyze_content`**（`memory_system.py:244-261`）：使用 `response_format={"type": "json_schema", ...}` 强制结构化输出，提取 `keywords/context/tags`
   - Step 3：**LLM 调用 `process_memory`（进化决策）**（`memory_system.py:625-754`）：
     - 用 `content` 向量检索 5 条最相关旧记忆
     - 把 5 条旧记忆 + 新笔记一起喂给 LLM，决定是否进化和如何进化
     - LLM 返回：`should_evolve`、`actions`（strengthen / update_neighbor）、`suggested_connections`（要链接的旧记忆 UUID）、`tags_to_update`、`new_context_neighborhood`、`new_tags_neighborhood`
     - 执行：`strengthen` → 把 `suggested_connections` 追加进新笔记的 `links` 字段并用 `tags_to_update` 覆盖新笔记 tags；`update_neighbor` → 直接修改旧记忆在 `self.memories` 里的 `tags` 和 `context`（**只改内存 dict，不同步到 ChromaDB**）
   - Step 4：写入 `self.memories[note.id] = note`（`memory_system.py:268`）
   - Step 5：构建 `enhanced_document`（`content + " context: " + context + " keywords: " + keywords + " tags: " + tags`）→ 写入 ChromaDB（`memory_system.py:270-284`，SentenceTransformer 向量化）

   **过滤/筛选**：**无**。相同内容写两次 `add_note` 会生成两条独立记忆，没有去重检查。

---

3. **检索机制**

   **方式**：**纯向量检索**（ChromaDB 余弦相似度，`SentenceTransformer` 向量化）。

   **检索前**：LLM 把问题扩展成关键词字符串（`generate_query_llm`，关键词用 `cosmos` 分隔），**1 次 LLM 调用**；再用这个关键词字符串向量化后查 ChromaDB。

   **召回后处理**：`find_related_memories_raw` 把主结果中的 `links` 字段指向的旧记忆内联追加，形成带邻居的检索结果（`memory_system.py:345-374`）——主结果读 ChromaDB 旧 metadata，邻居读 `self.memories` 最新值（含进化后的 context/tags）。

   **BM25**：`from rank_bm25 import BM25Okapi`（`memory_system.py:9`）被 import 但**从未被任何函数调用**。`search()` 方法的 docstring 写着"hybrid retrieval"，实际只有向量检索。

   **重排**：无。

   **检索端 LLM 参与**：**1 次**（问题扩展为关键词）。

   **结果数量控制**：top-k（默认 5）。`search_agentic` 追加 `links` 邻居后返回 `[:k]`（`memory_system.py:620`），只要主结果满 k 条，邻居全被截断。

---

4. **注入 prompt**

   `find_related_memories_raw` 返回拼接文本字符串（`memory_system.py:345-374`），每条记忆格式：
   ```
   talk start time:202403010900  memory content: Speaker Alice says: ...
   memory context: Alice mentioned starting a new job...
   memory keywords: ['Google', 'employment', 'March']
   memory tags: ['career', 'employment']
   ```
   若有 `links`，关联旧记忆的内容内联追加在后面。

   **插入位置**：QA prompt 的 Context 段（`test_advanced.py:140-145`），再加上用户问题让 LLM 作答。

   **截断**：`search_agentic` 的 `[:k]` 截断导致 `links` 邻居几乎必出局（`memory_system.py:620`）；`find_related_memories_raw` 内联邻居但无截断控制。

---

5. **记忆管理**

   **去重**：**无**。相同内容写两次会生成两条独立记忆，无任何相似度检查。

   **矛盾处理**：**无**。矛盾事实（"Alice 在北京"和"Alice 已搬去上海"）并存，不做冲突检测。`update_neighbor` 可以更新旧记忆的描述，但这是"语义进化"而非"矛盾处理"，且只改内存 dict 不影响向量检索。

   **遗忘/淘汰**：**无，默认不生效**。`delete()` 仅手动 CRUD API（`memory_system.py:428-443`），全库无自动调用方。`retrieval_count` 和 `last_accessed` 初始化后从不更新（`memory_system.py:77-80`），无法支撑任何基于使用频率的淘汰。

   **整合/抽象**：`consolidate_memories()` 每 100 次进化后触发一次（`memory_system.py:292-312`），意图是重建向量索引以反映进化后的状态。但 A-mem-sys 版可能是 no-op（`ChromaRetriever` 使用 `get_or_create_collection`，在共享内存存储中相同 ID 的 add 是 no-op）；AgenticMemory 原仓版用 `SimpleEmbeddingRetriever`（全新空实例）确实能重建索引，但每 100 次才触发一次，前 99 次进化结果在检索层面不可见。

---

6. **其他设计**

   - **元数据增强 embedding**（A-MEM 最值得借鉴的工程设计）：把 LLM 生成的 `context/keywords/tags` 拼进原始 `content` 再做向量化，让抽象语义参与相似度计算（`retrievers.py:38-55`）。实现成本极低（几行字符串拼接），原理上能提升检索精度。
   - **进化 prompt 的工程设计**（`memory_system.py:131-161`）：用 JSON Schema 强约束 LLM 输出结构，要求 `new_tags_neighborhood` 的数组长度等于邻居数，要求按输入顺序对位。这是 LLM 批量操作多个对象时保证对齐的实用做法。
   - **ChromaDB 内存模式**（`chromadb.Client(Settings(allow_reset=True))`）：进程退出后数据全丢，无持久化到磁盘。
   - **两个仓库对 `links` 的存储方式不同**：A-mem-sys 存 UUID（稳定，记忆增删后仍能准确定位）；AgenticMemory 原仓存整数下标（记忆库顺序变化时下标会错位，链接指向错误记忆）。

---

7. **核心创新点**

   作者宣称的最核心贡献：**Zettelkasten 卡片盒笔记法引入 AI 记忆系统**，通过三步（Note Construction → Link Generation → Memory Evolution）使记忆形成可进化的知识网络，而不是孤立的文本条目。

   - **Note Construction**：LLM 自动生成三类元数据（keywords/context/tags），让记忆从"一句话"变成"有标签的笔记"
   - **Link Generation**：向量检索找相关旧记忆，LLM 建立新记忆与旧记忆的显式链接
   - **Memory Evolution**：LLM 在同一次调用中决定是否更新旧记忆的 context 和 tags（让旧记忆在有了新邻居后"进化"出更准确的描述）

   **与同类方案的本质区别**：（1）有显式记忆间链接（其他方案是孤立条目）；（2）写入时主动修改已有记忆（与 Mem0 的 ADD-only 完全相反）；（3）元数据增强嵌入（LLM 生成的摘要/关键词/标签参与向量化）。

---

8. **论文 vs 代码差异**

   - **AgenticMemory（论文复现仓）Note Construction 永远失效**（`memory_layer.py:380-389`）：`re.sub(...)` 调用但整个文件无 `import re` → `NameError` → 内层 `except` 里 `e` 未定义 → 再次 `NameError` → 最外层 `except` 兜住 → 返回 `{"keywords": [], "context": "General", "tags": []}`。所有记忆以空关键词、默认 context、无标签存储，Note Construction 从未正常运行。**论文报告的实验数字在此 bug 下跑出来，相当于在纯文本存储条件下测试。**A-mem-sys 修复了此问题（直接解析结构化输出，无 `re.sub`）。

   - **`update_neighbor` 只改内存 dict，不改 ChromaDB**（`memory_system.py:719-743`）：进化操作改写了旧记忆的 `tags` 和 `context`，但不更新 ChromaDB 里该记忆的向量和 metadata。正确做法应调 `update()` 方法（`memory_system.py:387-426`，先 delete 再 add），但 `update_neighbor` 没有这样做。**进化对向量检索排序零即时影响。**

   - **`consolidate_memories` 在 A-mem-sys 版可能是 no-op**（`memory_system.py:292-312`）：`ChromaRetriever.__init__` 使用 `get_or_create_collection`，在同一进程共享内存存储时，新建 `ChromaRetriever` 拿到的是同一个已满的 collection，用相同 ID 再 add 是 no-op 或报错，consolidate 可能什么都没做。AgenticMemory 原仓用 `SimpleEmbeddingRetriever`（每次创建全新空实例）确实有效（`memory_layer.py:729-751`）。

   - **`search()` 声称"hybrid retrieval"但只有向量检索**：`BM25Okapi` import 了（`memory_system.py:9`）但从未被任何函数调用。

   - **`search_agentic` 的链接扩展是摆设**（`memory_system.py:620`）：追加 `links` 邻居后返回 `[:k]`，只要主结果已满 k 条，所有邻居被截断。

   - **三个死字段**（`memory_system.py:77-80`）：`last_accessed`、`retrieval_count`、`evolution_history` 初始化后从不更新，无法支撑任何遗忘或访问追踪机制。

---

9. **实验**

   **数据集**：LoCoMo（Long-term Conversational Memory，包含 Multi-hop/Temporal/Open-domain/Single-hop/Adversarial 5 类问题）。

   **基线**：MemGPT、Zep、Mem0、ReadAgent 等。

   **核心结论**：论文称在 6 个基础模型（含 GPT-4o-mini）上，A-MEM 在各类别问答上都超过了现有系统。**具体 F1 分项数字笔记未记录**。

   **值得注意的细节**：
   - 实验在 Note Construction 失效（相当于无元数据的纯文本存储）的条件下跑出来，论文宣称的"带进化的记忆系统"在实验时接近"纯文本存储 + 进化 prompt"。能否在修复 bug 后复现论文数字，目前无法核实。
   - SimpleMem 笔记引用了 A-MEM 的构建速度：5140.5 s/样本（比 SimpleMem 的 92.6 s 慢 55 倍），主要原因是每条消息最多 2 次 LLM 调用（analyze_content + process_memory）。
   - 一个有趣的研究问题：Note Construction 失效时系统仍能赢，说明 A-MEM 的胜出可能主要来自更好的检索 prompt 设计或 memory evolution 的链接结构，而不是元数据本身——这值得单独用实验拆开验证。

---

### MemoryOS

**仓库**：BAI-LAB/MemoryOS，commit 1d71706  
**论文**：Memory OS of AI Agent，arXiv:2506.06326，EMNLP 2025 main track

---

1. **记忆形式与粒度**

   MemoryOS 是三层结构，每层的记忆条目形式完全不同：

   **短期记忆（STM）**：最简单的原始 QA 对，三个字段：
   ```json
   { "user_input": "...", "agent_response": "...", "timestamp": "2025-06-10 14:23:11" }
   ```
   FIFO 双端队列，默认最多 10 条，满了弹出最老的送往中期。

   **中期记忆（MTM）**：两级结构，Page 是经过增强的 QA 对，Segment 是同主题 Page 的容器。

   **一条 Page 的字段**（`mid_term.py`）：
   | 字段 | 含义 |
   |------|------|
   | `page_id` | UUID |
   | `user_input` / `agent_response` | 原始 QA 对文本 |
   | `timestamp` | 原始时间戳 |
   | `page_embedding` | SentenceTransformer 向量（384维+），检索基础 |
   | `page_keywords` | LLM 生成的关键词列表，用于写入时归入 segment |
   | `meta_info` | LLM 递推生成的对话链摘要（覆盖连续多条的概览） |
   | `pre_page` / `next_page` | 对话链前后指针（dialogue chain） |
   | `analyzed` | 是否已晋升到 LPM |
   | `preloaded` | 笔记未记录具体作用 |

   **一个 Segment 的字段**（`mid_term.py`）：
   | 字段 | 含义 |
   |------|------|
   | `summary` / `summary_keywords` | LLM 生成的主题摘要和关键词 |
   | `summary_embedding` | 摘要向量，粗匹配用 |
   | `details` | 所属 Page 列表 |
   | `L_interaction` | segment 内 page 数量 |
   | `N_visit` | 被检索命中次数 |
   | `R_recency` | 时效性 = exp(-Δt / 24h) |
   | `H_segment` | 热度 = N_visit + L_interaction + R_recency |
   | `access_count_lfu` | 纯检索命中计数（淘汰时用） |

   **长期画像记忆（LPM）**：用户画像（90 维度纯文本字符串，每次晋升全量重写）+ 用户知识库（deque maxlen=100，每条是事实句 + 时间戳 + embedding）+ 助手知识库（同结构）。

   **粒度**：三层递减——STM 最粗（原始 QA 对），MTM 中等（QA 对 + 链摘要 + 主题聚合），LPM 最抽象（90 维画像 + 原子事实）。

   **与其他粒度方案的本质区别**：（1）保留了原始 QA 对（不提炼事实句），而 Mem0/SimpleMem 强制要求代词消解和时间绝对化；（2）Segment 是一个真正的"主题容器"，而不是孤立的条目或纯平铺列表；（3）对话链（pre/next_page + meta_info）让单条记忆携带上下文线索，其他方案的记忆条目是孤立的。

---

2. **写入机制**

   **触发时机**：每次调用 `add_memory(user_input, agent_response)`（`memoryos.py:226`）时立即触发。

   **LLM 参与次数**：
   - STM **未满**（< 10 条）：直接 append，**0 次 LLM 调用**。
   - STM **已满**（≥ 10 条）：弹出最老的 QA 对迁往 MTM，触发 **3 次 LLM 调用**：
     1. **连续性判断**（`utils.py:359-374`）：判断当前 page 与上一条迁移 page 话题是否连续，返回 `true/false`
     2. **链摘要生成**（`utils.py:376-386`）：连续则递推更新 meta_info，不连续则单独生成概览
     3. **多主题摘要**（`utils.py:251-263`）：把这批 page 拼文本，返回最多 2 个主题的 JSON（theme + keywords + content）
   - LPM 晋升时（热度 ≥ 5.0）：额外 **2 次并行 LLM 调用**（`memoryos.py:169-172`）：画像分析（90 维全量重写）+ 知识抽取（用户 + 助手知识库各 append）

   **从对话到记忆的提取过程**：3 次 LLM 中，核心是多主题摘要。LLM 返回主题列表后，对每个主题计算 embedding，与现存所有 segment 的 `summary_embedding` 做向量内积 + Jaccard 关键词相似（F_score），F_score ≥ 0.6 并入已有 segment，否则新建 segment（`mid_term.py:190-279`）。

   **过滤/筛选**：无。同一批 page 若 LLM 生成 2 个主题，page 被原样插入 2 次（`updater.py:174-185`），同一 page_id 可出现在多个 segment 中，无去重兜底。

---

3. **检索机制**

   `get_response(query)` 调用时，并行启 3 个线程（`retriever.py:110`）同时检索三层：

   **STM**：全量返回所有短期 QA 对，无筛选。

   **MTM 两级检索**（`retriever.py:56-66`，`mid_term.py:302-306`）：
   - 第 1 级：query 转 embedding → FAISS（`IndexFlatIP`）扫描所有 segment 的 `summary_embedding`，取 top-5 个 segment
   - 第 2 级：在 top-5 segment 内，对每条 page embedding 做内积，跨 segment 取全局 top-7 条 page
   - **副作用**：命中的 segment `N_visit+1`、`access_count_lfu+1`、热度重算、`mid_term.json` 全量重写——检索行为直接推高热度

   **LPM 知识检索**（`retriever.py:70-90`）：query embedding 临时建 FAISS，在用户知识库和助手知识库各取 top-20（论文说 top-10，代码实际 top-20，阈值 0.01 形同虚设）。

   **召回后有无重排**：无。三路结果直接拼接进 prompt。

   **检索端 LLM 参与次数**：**0 次**。

   **结果数量控制**：STM 全量（≤ 10 条），MTM top-7，LPM top-20 × 2 库。

---

4. **注入 prompt**

   三段结构（`memoryos.py:315-327`）：

   ```
   [System] 你是 friend 角色，以下是助手知识：
   - Assistant recommended wetland park on 2025-06-10

   [User]
   <CONTEXT>
   最近对话（STM 全量）：
   User: ...  Assistant: ...  (Time: ...)

   <MEMORY>
   【Historical Memory】（MTM top-7 page，含 meta_info）
   User: ...  Assistant: ...  Time: ...
   Conversation chain overview: ...

   <USER TRAITS>
   【User Profile】
   Health Concern (High): ...
   【Relevant User Knowledge Entries】
   - 用户上周去了湿地公园 (Recorded: ...)

   请用最多 30 词、必须英文回复用户：{query}
   ```

   **插入位置**：system prompt（助手知识）+ user 消息体（STM + MTM + 画像 + 知识库）。

   **截断/优先级控制**：硬编码 `maximum 30 words, must be in English`（`prompts.py:26`），这是针对 LoCoMo 评测集调的，实际应用中不合适。数量由各层 top-k 固定控制，无动态优先级。

---

5. **记忆管理**

   **去重**：**无**。同一 page 若归入多个主题会被重复插入多个 segment；LPM 知识库直接 append，相同内容写多少次存多少次（`long_term.py:50-69`）。

   **矛盾处理**：**无显式矛盾检测**。LPM 知识库新旧矛盾条目并存，直到被 FIFO 挤出；用户画像靠 LLM 全量重写隐式消解矛盾（每次晋升 LLM 被指令"整合新旧画像"），无显式冲突检测逻辑。

   **遗忘/淘汰**：
   - STM：FIFO，满 10 条弹出最老
   - MTM：segment 数量超 2000 时删除 `access_count_lfu` 最小的 segment（`mid_term.py:71-101`，LFU 策略）
   - LPM：用户知识库和助手知识库各是 `deque(maxlen=100)`，append 时若满自动丢弃最老（`long_term.py:18-19`）
   - **没有时间自然衰减**：`rebuild_heap` 里的重算热度那行被注释掉（`mid_term.py:184-186`），segment 不被访问时热度冻结，不随时间下降

   **整合/抽象**：LPM 晋升时 LLM 全量重写 90 维画像（覆盖已有内容）+ 追加知识条目（不合并）。无后台摘要合并或跨 segment 整合。

---

6. **其他设计**

   - **对话链（Dialogue Chain）**：pre/next_page 指针 + LLM 递推更新 meta_info。检索到单条 page 时，meta_info 随 page 一起进 prompt，提供对话链概览，低成本缓解单条记忆缺乏上下文的问题。
   - **检索驱动热度**：命中的 segment `N_visit+1` 并重算热度，让"被查得多的记忆更值得晋升"。但热度同时喂给晋升（H ≥ 5.0）和 LFU 淘汰（`access_count_lfu`）两个独立机制，热度与淘汰实际走不同字段，逻辑混乱。
   - **全 JSON 文件持久化**：所有层记忆以 JSON 文件存储（`short_term.json`，`mid_term.json`，`long_term_*.json`）；每次写入/检索都全量重写 mid_term.json，FAISS 索引每次查询临时重建，数据量大时性能会明显下降。

---

7. **核心创新点**

   作者宣称的最核心贡献：**三层分级记忆 + 热度驱动晋升 + 对话链机制**，将操作系统内存管理（寄存器/RAM/磁盘）完整映射到 AI 对话记忆，形成"记忆操作系统"。

   与同类方案的本质区别：
   - vs MemGPT：同样用 OS 隐喻，但 MemOS 用**主题聚合**（segment）取代 MemGPT 的 FIFO 平铺队列，长对话后主题不混杂；MemGPT 的检索是 LLM 主动调工具，MemoryOS 是系统自动并行检索
   - vs A-MEM：写入时每次至多 3 次 LLM（vs A-MEM 约 13 次），效率高；A-MEM 存原始文本 + 图链接，MemoryOS 存 QA 对 + 主题聚合 + 画像
   - vs Mem0/SimpleMem：保留原始 QA 对粒度（不做代词消解），但通过两级检索（segment → page）实现主题过滤

---

8. **论文 vs 代码差异**

   - **淘汰策略不一致**：论文称"按热度最低删"，代码实际是 LFU——按 `access_count_lfu`（纯检索命中次数）最小删（`mid_term.py:75`）。热度里的 L_interaction 和 R_recency 与淘汰决策无关。
   - **时间常数差两个数量级**：论文热度公式时间常数 µ = 1e7 秒（约 116 天），代码 `RECENCY_TAU_HOURS = 24`（`mid_term.py:24`）。且 `rebuild_heap` 里热度重算那行被注释掉（`mid_term.py:184-186`），时间维度在不访问时冻结不变，时间常数选择几乎没有实际影响。
   - **用户画像的结构化字段**：论文描述 User Profile 含固定属性（gender、name、birth year），代码里画像是 LLM 生成的纯字符串（`long_term.py:16`），没有结构化字段。
   - **"transferred to LPM"是夸大**：论文称热 segment"被转移到 LPM"，代码里 segment 不删不迁，只是从中抽取内容写入 LPM 后标 `analyzed=True`、热度清零，segment 永远留在中期（`memoryos.py:207-218`）。
   - **LPM 知识检索 top-k 不符**：论文 top-10，代码默认 `top_k_knowledge=20`，阈值 0.01 形同虚设（`retriever.py:96,98`）。
   - **关键词机制半摆设**：`search_sessions` 里 `query_keywords = set()` 恒为空（`mid_term.py:292`），page 级关键词相似度代码被注释（`mid_term.py:336-339`）。关键词只在写入时归入 segment（Jaccard 计算）时生效，检索端完全不用。

---

9. **实验**

   **数据集**：GVD（自建，15 个虚拟用户和 AI 在 10 天内的多轮对话，评测 Acc/Corr/Cohe 三维度，DeepSeek-R1 自动打分）；LoCoMo（平均 300 轮约 9K tokens，单跳/多跳/时序/开放域 4 类）。

   **基线**：TiM（Think-in-Memory）、MemoryBank、MemGPT、A-Mem。

   **核心结论**（GPT-4o-mini + LoCoMo）：

   | 方法 | 平均 F1 | 平均 LLM 调用次数 | token 消耗 |
   |------|---------|-----------------|-----------|
   | MemoryBank | 6.84 | 3.0 | 432 |
   | TiM | 18.01 | 2.6 | 1,274 |
   | MemGPT | 29.13 | 4.3 | 16,977 |
   | A-Mem | 26.55 | 13.0 | 2,712 |
   | **MemoryOS** | **36.23** | **4.9** | **3,874** |

   Temporal（时序推理）子任务提升最大（约 +119% vs A-Mem），说明对话链机制和时间戳对时序问题帮助显著。消融实验：去掉 MTM 损失最大，去掉对话链影响最小。

   **值得注意的细节**：论文评测使用 GPT-4o-mini，多主题摘要 LLM 最多生成 2 个主题，同一 page 被重复插入多个 segment 的去重问题在评测环境中未被控制。

---

### MemGPT

**仓库**：letta-ai/letta（原 MemGPT，已商业化为 Letta 平台），commit 1131535  
**论文**：MemGPT: Towards LLMs as Operating Systems，arXiv:2310.08560，UC Berkeley，2024

---

1. **记忆形式与粒度**

   三层记忆，每层的存储形式完全不同：

   **Core Memory Block（核心记忆块）**（`letta/schemas/block.py:13-78`）：自由文本块，默认有 `human`（用户信息）和 `persona`（agent 人设）两个块，内容格式由 LLM 自己决定，无固定结构：

   | 字段 | 含义 |
   |------|------|
   | `id` | 数据库唯一 ID |
   | `label` | 块名（决定 prompt 里的 XML 标签名） |
   | `value` | 核心内容，自由文本（如 "Name: Alice\nEx-boyfriend: James"） |
   | `limit` | 字符数上限（仅展示，**不硬性执行**） |
   | `description` | 这个块是做什么用的 |
   | `read_only` | agent 是否可以编辑 |

   **Archival Memory Passage（档案记忆段落）**（`letta/schemas/passage.py:35-47`）：LLM 主动存入的文本，原样存储：

   | 字段 | 含义 |
   |------|------|
   | `id` | UUID |
   | `text` | 原始文本，不做切分/摘要 |
   | `embedding` | 1536 维浮点向量（text-embedding-3-small） |
   | `created_at` | 时间戳 |
   | `tags` | 可选标签（用于过滤） |

   **Recall Memory Message（回溯记忆消息）**：每条对话消息（user/assistant/tool/system）自动落库，字段：id, role, content, created_at, agent_id, model。

   **粒度**：Core 是粗粒度的"长期摘要块"（LLM 自由写作，无原子事实约束）；Archival 是 LLM 决定的任意粒度（不做格式化）；Recall 是最细粒度的原始消息级。

   **与其他粒度方案的本质区别**：（1）Core Memory 是始终可见的自由文本，而 Mem0/SimpleMem/A-MEM 都需要检索才能看到记忆；（2）Archival 存什么、粒度多细完全由 LLM 决定（系统不做提炼），对比其他方案系统自动提炼事实句；（3）有三层层级（始终可见 vs 按需检索 vs 自动历史），而非平铺向量库。

---

2. **写入机制**

   **Core Memory 写入**：LLM 主动调用工具触发，无外部条件驱动：
   - `core_memory_append/replace`（经典版，`core_tool_executor.py:319/328`）：字符串追加/精确替换
   - `memory_replace/insert/rethink`（v2 版，`core_tool_executor.py:346/683/743`）：`memory_replace` 要求 old_string 在块内唯一才允许替换（防幻觉）；`memory_rethink` 整块重写
   - 写完立即触发 `rebuild_system_prompt_async()`（`agent_manager.py:1523`），原地替换 system 消息（不追加新消息）
   - **成本：0 次额外 LLM 调用，0 次 embedding，当前 step 内完成**

   **Archival Memory 写入**（`passage_manager.py:543`）：LLM 主动调用 `archival_memory_insert` 工具，文本**原样存入，不做切分/摘要/格式化**，调 1 次 embedding 模型转向量，写入 SQL（PostgreSQL 或 SQLite），配了 Turbopuffer 则同步双写。**成本：0 次额外 LLM，1 次 embedding。**

   **Recall Memory 写入**：每条消息自动落库，LLM 无需主动操作。**成本：0 次 LLM，0 次 embedding（默认）。**

   **Compaction（上下文溢出摘要）**（`summarizer/compact.py:135`）：消息队列超上下文 × 0.9 时触发，把最老一批消息用 1 次 LLM（小模型如 GPT-4o-mini / claude-haiku）生成摘要，以 `role=summary` 消息插入队列前部，原始消息退出上下文但数据库永久保留。**成本：1 次 LLM（小模型），0 次 embedding。**

   **Sleeptime 后台写入**（`groups/sleeptime_multi_agent_v4.py`，默认关闭）：主 agent 处理完后，异步启动独立 sleeptime agent，把上次整理以来的对话 transcript 喂给它，多步编辑核心记忆直到调 `memory_finish_edits` 工具。**默认 `enable_sleeptime=False`（`schemas/agent.py:318`），成本：≥ 1 次 LLM（多步）。**

   **过滤/筛选**：无。Core 写入靠 LLM 的自然语言指令"不要重复/过时信息"；Archival 无去重检查。

---

3. **检索机制**

   **Archival 检索**（`agent_manager.py:2416` `archival_memory_search` 工具）：
   - 配了 Turbopuffer：向量检索 + FTS 全文检索，RRF 融合（`agent_manager.py:2457-2473`）
   - 未配 Turbopuffer（默认）：pgvector 余弦距离排序（`agent_manager_helper.py:1245`）；SQLite 下用自定义余弦函数（`:1250-1255`）
   - 默认 top-5（`constants.py:458`），支持 tags 过滤

   **Recall 检索**（`message_manager.py:1142` `conversation_search` 工具）：
   - 配了 Turbopuffer + `embed_all_messages`：hybrid 语义+关键词
   - **默认情况（未配 Turbopuffer）：SQL `ILIKE '%query%'` 子串匹配**（`message_manager.py:978-993`），仅精确关键词匹配，无语义理解

   **召回后有无重排**：无。

   **检索端 LLM 参与次数**：LLM 自己决定何时调工具检索，工具调用本身 0 次额外 LLM（检索在当前 step 的函数调用中执行）。

   **结果数量控制**：Archival top-5，Recall 笔记未记录具体上限。

---

4. **注入 prompt**

   **Core Memory**：始终在 system prompt 里，以 XML 格式渲染（`memory.py:688`）：
   ```xml
   <memory_blocks>
   <human>
   <metadata>chars_current=57  chars_limit=100000</metadata>
   <value>Name: Alice\nEx-boyfriend: James</value>
   </human>
   <persona>...</persona>
   </memory_blocks>
   ```
   **每次 LLM 推理都能直接"看见"，不需要检索。**

   **外存元数据提示**（`prompt_generator.py:26-89`）：system prompt 里有 `<memory_metadata>` 段，告诉 LLM "recall memory 共 N 条，archival 共 M 条，可用 tags"，让 LLM 感知外存存在。

   **Archival/Recall 检索结果**：以工具返回值（JSON）形式进入消息流，临时可见，要永久保留需 LLM 再次写入 core 或 archival。

   **截断/优先级控制**：Core Memory 字符上限（`limit`）只展示给 LLM，系统不做检查（`block_manager.py:825-827`，默认 100000 字符）。Archival 检索由 LLM 自主决定何时搜索，每次取 top-5。

---

5. **记忆管理**

   **去重**：**无自动去重**。Archival 插入不检查相似内容；Core 编辑靠 LLM 自觉不写重复信息。sleeptime prompt 用自然语言指令"不要包含重复和过时信息"，靠 LLM 判断。

   **矛盾处理**：**无自动矛盾检测**。当用户说"James 和我分手了"，LLM 需自己识别 core memory 里有冲突信息并主动调工具修改。能否及时发现和修正完全取决于 LLM 能力。

   **遗忘/淘汰**：
   - Core Block：agent 可调 `memory_delete`（实为 detach block，`core_tool_executor.py:778-806`）
   - **Archival：没有 agent 可调用的删除工具**（`function_map` 里无 archival delete，`core_tool_executor.py:41-56`），工具文档明确写"persists indefinitely"（`functions/function_sets/base.py:176`）
   - Recall：消息从不真正删除，只有逻辑标记（`is_deleted == False` 过滤，`message_manager.py:944`）
   - 无衰减、无打分、无自动清理

   **整合/抽象**：Sleeptime agent（默认关闭）做语义级记忆整合和精简；主路径 compaction 只做滑窗摘要，不做语义合并。Block 版本历史（checkpoint/undo/redo，`block_manager.py:842/952/1004`）供管理端人工回滚，agent 自己不能用。

---

6. **其他设计**

   - **System prompt 原地重编译**：记忆更新后不追加消息，直接替换 system 消息（`message_ids[0]`），不污染对话历史，对 LLM 来说记忆"一直在那里"，且有利于 prefix cache 命中（`agent_manager.py:1602-1608`）。
   - **`memory_replace` 唯一性强制 + 行号只读视图防幻觉**（`core_tool_executor.py:346-401`）：要求 old_string 在整个块内唯一才允许替换，否则报错并列出所有位置；渲染时给每行加行号（"1→ ..."）供 LLM 定位，但校验时严格拒绝包含行号的 old_string。
   - **在线/离线记忆分工**：主 agent 在线路径只做低成本操作（compaction 用小模型），语义级整合下放给离线 sleeptime agent（不影响响应延迟）。
   - **函数链（function chaining）**：LLM 可在工具调用里加 `request_heartbeat=true`，让系统立刻再运行一次 LLM，允许连续执行多步记忆操作再给用户回复。

---

7. **核心创新点**

   作者宣称的最核心贡献：**仿 OS 虚拟内存分页——LLM 自驱动记忆换页**，通过给 LLM 配备记忆工具（函数调用），让 LLM 自己决定何时读/写/搜索外部记忆，使固定上下文窗口的 LLM 获得"无限上下文"的幻觉。

   与同类方案的本质区别：
   - vs Mem0/SimpleMem：Mem0 是系统自动提炼事实并在检索时注入，MemGPT 是 LLM 自主管理（Core Memory 主动编辑 + Archival 主动搜索），写什么粒度由 LLM 决定
   - vs MemoryOS：MemoryOS 是系统自动分层迁移，MemGPT 是 LLM 主动调工具换页；MemGPT 无主题聚合，历史消息平铺
   - vs A-MEM：A-MEM 是写入时系统自动建知识图，MemGPT 的 Archival 是 LLM 手动存文本，无图结构

---

8. **论文 vs 代码差异**

   - **Archival 已从默认工具集移除**：`archival_memory_insert/search` 被列为 `DEPRECATED_LETTA_TOOLS`（`agent_manager_helper.py:1298-1302`，`constants.py:116`），默认不加入 agent 工具集。论文核心机制在默认部署下不会自动启用。
   - **Memory pressure warning 已废弃**：论文宣称接近上限时给 LLM 发警告让其主动转移记忆；新主路径 `LettaAgentV3` 完全没有这个机制（只在旧版 `agent.py:944-972` 保留），改为系统静默压缩。**LLM 不再主动意识到"内存快满了"。**
   - **Recall search 默认是子串匹配，非语义搜索**：工具文档写"hybrid search (text + semantic similarity)"，实际需要 Turbopuffer + `embed_all_messages` 全部开启（`settings.py:442-445` 默认均为 False）才生效，默认自托管退化为 `SQL ILIKE '%query%'`（`message_manager.py:978-993`）。
   - **Block 字符上限不硬性执行**：论文把"core memory 有限"作为 agent 必须取舍的约束，实际代码里 `chars_limit` 只展示给 LLM，系统不做任何检查（`block_manager.py:825-827`），默认 100000 字符基本不可能写满。
   - **函数体 `raise NotImplementedError` 是误导**：`functions/function_sets/base.py` 里的工具函数体是 `raise NotImplementedError`，看起来像"未实现"，实际这些函数只提供 JSON Schema 描述，真正实现在 `core_tool_executor.py` 里。

---

9. **实验**

   **数据集**：MSC（Multi-Session Chat，真实用户固定人设进行 5 轮对话，新增第 6 轮测试）；文档分析（2000 万条维基百科段落预存 Archival Memory）；嵌套键值检索（4 层 KV 多跳）。

   **基线**：把前 5 轮对话摘要塞进上下文（固定上下文基线）；直接全量放入（受上下文长度限制）。

   **核心结论（DMR 深度记忆检索任务）**：

   | 模型 | 无 MemGPT（摘要基线）| +MemGPT |
   |------|---------------------|---------|
   | GPT-3.5 Turbo | 38.7% | **66.9%** |
   | GPT-4 | 32.1% | **92.5%** |
   | GPT-4 Turbo | 35.3% | **93.4%** |

   嵌套 KV 检索：GPT-4 在 3 层降到 0% 准确率，MemGPT+GPT-4 在 4 层仍保持高准确率。文档分析：基线准确率随文档数增加而下降（被截断），MemGPT 可以反复翻页搜索，准确率基本不受文档总量影响。

   **值得注意的细节**：实验基于 2023 年论文，当时主流模型上下文只有 4k–128k；现代 LLM 上下文窗口已达 100k–1M，部分 MemGPT 的动机（"上下文不够用"）已被长上下文模型部分缓解，但长上下文"lost in the middle"问题使结构化记忆仍有价值。

---

### MemOS

**仓库**：MemTensor/MemOS，commit b60616d  
**论文**：MemOS: A Memory OS for AI System，arXiv:2507.03724；前置框架论文 arXiv:2505.22101

---

1. **记忆形式与粒度**

   MemOS 论文宣称三类记忆（Textual / Activation / Parametric），但只有 Textual Memory 完整实现（见第 8 条）。

   **Textual Memory 的核心数据类 `TextualMemoryItem`**（`src/memos/memories/textual/item.py:299`）：

   | 字段 | 含义 |
   |------|------|
   | `id` | 唯一 UUID |
   | `memory` | 记忆正文，一条完整的 LLM 提炼后的事实句 |
   | `memory_type` | 分桶：WorkingMemory（20条）/ LongTermMemory（1500条）/ UserMemory（480条） |
   | `key` | 记忆标题（如"Tom的项目截止日期"），用于精确匹配检索 |
   | `tags` | 关键词标签列表，用于精确匹配检索 |
   | `embedding` | 约 1024 维浮点向量，语义检索基础 |
   | `sources` | 来源溯源（原始对话片段 + role + chat_time） |
   | `confidence` | 可信度 0~100 |
   | `status` | "activated" / "archived" / "deleted" |
   | `is_fast` | 是否是 fast 模式的原始文本（未经 LLM 精炼） |
   | `version` | 版本号，更新时递增 |
   | `history` | 历史版本快照列表 |
   | `created_at` / `updated_at` | ISO 8601 时间戳 |
   | `user_id` / `session_id` | 归属作用域 |
   | `usage` | 使用历史（**从不更新**，`_update_usage_history` 被注释掉） |

   记忆以图节点形式存入图数据库（Neo4j 或 PolarDB），节点间可有 PARENT（从属）、MERGED_TO（合并）等关系边。图结构允许"事实 → 主题"两层层级（reorganize 模式下 KMeans + LLM 生成 PARENT 边）。

   **粒度**：事实句级别——LLM 从对话中提炼的独立事实，一条记忆是自包含的陈述句，附有结构化元数据和来源溯源。fine 模式下粒度最细；fast 模式存原始文本窗口（较粗），后台异步精化升级。

   **与其他粒度方案的本质区别**：（1）显式 memory_type 分桶（WorkingMemory/LTM/UserMemory），支持按桶分配容量预算和检索策略，其他方案无分桶；（2）图节点 + PARENT 边的两层层级结构（reorganize 模式），比纯平铺向量库多一层主题聚合；（3）sources 字段记录来源溯源，可追溯每条记忆的原始对话，其他方案多无此字段。

---

2. **写入机制**

   **触发时机**：每次调用 `MOSCore.add(messages)`（`core.py:684`）时触发，同步执行（fine/fast 共同入口）。

   **fine 模式（精确，默认）**：
   - `SimpleStructMemReader._iter_chat_windows` 按 1024 token 切对话滑动窗口（`simple_struct.py:303`）
   - 每个窗口：**1 次 LLM 调用**（`simple_struct.py:268`），返回若干条 `{key, value, tags, memory_type}` JSON 数组；失败时以原文兜底
   - 每条记忆：**1 次 embedding**
   - 批量写图节点，挂 `working_binding` 标记（`manager.py:138`）
   - 若 `reorganize=True`，推消息给后台整理线程
   - **成本：一次普通对话约 1 次 LLM + 2~6 次 embedding**

   **fast 模式（快速）**：
   - **0 次 LLM**，直接把原文窗口作为一条粗糙记忆写进图节点，打上 `is_fast=True` 标记（`simple_struct.py:347`）
   - 同时给 MemScheduler 发送 MEM_READ 任务
   - 后台 `MemReadMessageHandler` 取出粗糙节点，用 LLM 精抽取，写入新节点，**删除原始粗糙节点**（`mem_read_handler.py:401-413`）
   - **先快后精、异步升级**——不堵塞用户

   **过滤/筛选**：LLM 可返回空数组（无新事实时不写入），无显式过滤器或相似度检查。

---

3. **检索机制**

   多路并行召回（`searcher.py:335` `_retrieve_paths`）：

   **Path A - WorkingMemory**：全量取出（最多 20 条），直接用。

   **Path B - LongTermMemory + UserMemory**（`recall.py:35` `GraphMemoryRetriever.retrieve`）：
   - 图元数据召回：key 精确匹配 + tags 重叠（≥ 2 条）
   - 向量召回：query embedding → 余弦相似度
   - 可选 BM25 全文检索（稀疏检索，基于词频统计）

   **Path C - 互联网检索**（可选，默认关闭）。

   **召回后处理**：
   1. 按文本字符串完全匹配去重（`searcher.py:1096` `_deduplicate_results`），防止同一句话重复出现
   2. 按分数排序，截取 top_k
   3. 可选 deep_search（`advanced_searcher.py:232`）：多阶段 LLM 自评 + 扩展短语，笔记记录"用于更复杂查询"

   **重排**：默认余弦相似度本地打分，**0 次 LLM**（fine 模式查询解析时 1 次 LLM）。

   **检索端 LLM 参与次数**：fine 查询模式 **1 次**（解析意图/关键词/标签/改写）；fast 查询模式 **0 次**。

   **结果数量控制**：top_k 参数截断；WorkingMemory 全量（≤ 20 条）。

---

4. **注入 prompt**

   `MOSCore._build_system_prompt`（`core.py:354`）把 top_k 条记忆编号排列，拼在 system prompt 末尾：

   ```
   ## Memories:
   1. 用户计划在次日上午9:30的会议上提议将项目截止日期延至2026年1月5日。
   2. 用户参加了项目进度会议，发现原截止日期12月15日过于紧张...
   ```

   **插入位置**：system prompt 末尾（而非 user 消息体内）。

   **截断/优先级控制**：top_k 参数控制数量；文本字符串去重后排序截断；WorkingMemory 全量优先（始终纳入，不参与 top_k 竞争）。无动态优先级或 token 预算控制。

---

5. **记忆管理**

   **去重**：三层机制：
   1. **写入时**（仅 `reorganize=True`）：embedding 相似度 > 0.8 的节点逐对由 LLM 判断是否冗余，冗余则融合——**默认关闭**（`handler.py:30`）
   2. **读取时**：字符串完全匹配去重（`searcher.py:1096`），防 prompt 里重复——**默认生效**
   3. **fast→fine 精化时**：旧的 fast 节点被标记 `status="archived"` 后删除（`mem_read_handler.py:401-413`）——**默认生效**

   **矛盾处理**（仅 `reorganize=True`）：新节点写入时若发现 embedding 相似但内容矛盾，LLM 尝试融合生成新节点；融合失败则按 `updated_at` 删除较旧的那条（`handler.py:76`）。**默认完全不生效**（`reorganize` 默认 False，且 `MOSCore.mem_reorganizer_on()` 方法体是空 `pass`）。

   **遗忘/淘汰**：
   - WorkingMemory：每次同步 add 后 FIFO 裁到 20 条
   - LongTerm/User：达到容量上限 80% 时触发清理，FIFO 删到上限（`manager.py:527`）——但**只在异步路径触发，纯同步写入不裁剪 LTM**
   - 无时间衰减、无重要性打分
   - `_update_usage_history` 整个函数体被注释成 docstring（`searcher.py:1290`），`usage` 字段从不更新

   **整合/抽象**（仅 `reorganize=True`）：后台线程每 100 秒触发 `optimize_structure`（`reorganizer.py:151`）：对图节点 embedding 做 MiniBatchKMeans 聚类，LLM 给每个簇写摘要，生成"主题节点"，用 PARENT 边连接下属具体事实节点，形成"事实 → 主题"两层层级。

---

6. **其他设计**

   - **fast/fine 双速写入**：先存原文保证实时性，后台异步精化保证质量，解决"写入延迟 vs 记忆质量"的矛盾（`simple_struct.py:347`）。
   - **MemScheduler 异步任务队列**：解耦写入和整理，后台处理 MEM_READ/MEM_REORGANIZE/MEM_UPDATE 等任务，不阻塞主请求路径。
   - **MemCube 容器抽象**（`general.py:24-48`）：四槽位（text/act/para/pref）的自包含记忆容器，可独立 load/dump，论文宣称可跨 agent/用户/项目"插拔"——容器抽象本身实现了，但三类记忆中只有 text 真实可用。
   - **WorkingMemory 动态替换**：`memory_update_handler`（`memory_update_handler.py:36`）根据 query 意图动态替换 WorkingMemory 内容，使其始终保持"当前对话最相关的 20 条"。

---

7. **核心创新点**

   作者宣称的最核心贡献：**MemCube（可移植记忆容器）+ 三类记忆统一框架（Textual/Activation/Parametric）**，将记忆从"文本堆"提升为有生命周期（创建→使用→更新→遗忘）的操作系统资源，类比 OS 的内存管理调度。

   与同类方案的本质区别：
   - vs Mem0/SimpleMem/A-MEM：MemOS 有图数据库图节点 + PARENT 边的两层层级（reorganize 模式），而其他方案是纯平铺向量库；有 memory_type 分桶容量预算
   - vs MemoryOS：都有分层，但 MemoryOS 是 QA 对主题聚合，MemOS 是 LLM 提炼事实句后存图节点；MemOS 的整合（聚类摘要）是后台异步的，MemoryOS 的整合（画像重写）是同步触发的
   - vs MemGPT：MemGPT 靠 LLM 自驱动（主动调工具），MemOS 靠系统自动分层+调度（LLM 只负责抽取和融合判断）

---

8. **论文 vs 代码差异**

   - **参数记忆是空壳**：论文把 Parametric Memory（LoRA 权重）列为三大支柱之一；代码里 `lora.py` 开头注释明写"currently serves as a placeholder, do not use"，`dump()` 只写 `b"Placeholder"`（`src/memos/memories/parametric/lora.py:37-41`）。
   - **激活记忆条件严苛且默认关闭**：`enable_activation_memory` 默认 False（`configs/mem_os.py:53-56`）；只支持本地 HuggingFace 后端；vLLM 版"KV cache"实际存的是 prompt 字符串而非可移植的 KV 张量，无法跨进程迁移。
   - **图谱关系推理空转**：论文描述了 INFERS（推断）、FOLLOWS（时序）、AGGREGATE_TO（聚合）等丰富边类型；`RelationAndReasoningDetector.process_node` 中这四步逻辑全被三引号注释掉（`relation_reason_detector.py:49-80`），恒返回空字典，运行时实际只有 PARENT 和 RELATED 边。
   - **检索 pipeline 与 docstring 不符**：`search` 函数 docstring 写"MemoryReranker → MemoryReasoner → Final output"；`MemoryReasoner.reason()` 从未在检索流程中被调用，实例化了但没用。
   - **冲突检测/结构整理默认完全不生效**：`reorganize` 默认 False；`MOSCore.mem_reorganizer_on()` 方法体是空 `pass`。论文里讲的"矛盾消解""层级摘要"需要手动打开才能运行。
   - **使用频率记忆管理无从谈起**：`_update_usage_history` 整个函数体被注释成 docstring（`searcher.py:1290`），`usage` 字段永远不更新，基于使用频率的智能遗忘在代码里不存在。
   - **update 接口对主力后端不可用**：`TreeTextMemory.update` 直接 `raise NotImplementedError`；`MOSCore.update` 只打 warning。

---

9. **实验**

   **数据集**：LongMemEval（长期对话事实一致性）、LoCoMo（多跳推理+时序，平均 300 轮约 9K tokens）、PersonaMem（个性化响应准确率）、PrefEval（用户偏好遵从率，0轮和10轮后对比）。

   **基线**：OpenAI Memory、Mem0、Zep、MIRIX 等。

   **核心结论**（与最强基线相比）：

   | 基准 | MemOS 提升 |
   |------|-----------|
   | LongMemEval 均值 | +40.43% |
   | LoCoMo 准确率 | +38.97% |
   | LoCoMo 时序推理 | **+159%** |
   | PersonaMem 精确率 | +40.75% |
   | PrefEval-10（10轮后偏好保留）| +2568% |
   | LoCoMo token 消耗 | -60.95% |

   **值得注意的细节**：参加评测的版本是"MemOS-1031"（内部版本号）；优势很大程度来自精细的明文记忆抽取质量，而非论文宣称的三类记忆联合管理——因为另外两类记忆基本是空的。这意味着当前 benchmark 主要测"明文记忆精确性"，而非多类记忆统一调度能力。PrefEval-10 +2568% 的极端数字因为基线接近 0，相对提升数字夸张，需结合绝对值解读。

---

## 批次一横向对比（Mem0 / SimpleMem / A-MEM）

| 维度 | Mem0（v3） | SimpleMem | A-MEM |
|------|-----------|-----------|-------|
| **记忆粒度** | 事实句（15–80词） | 事实句 + 结构化元数据 | 原始对话文本 + LLM元数据 |
| **写入时 LLM 次数** | 1 次 | 每40条对话1次（摊≈1/38） | 最多2次/条 |
| **写入是否即时** | 即时 | 批量（40条触发） | 即时 |
| **检索方式** | 语义+BM25+实体boost | 语义+BM25+SQL符号层 | 纯向量（BM25 import但未用） |
| **检索端 LLM** | 0次 | 2–4次（规划+反思） | 1次（问题扩展） |
| **去重** | MD5精确去重 | 无 | 无 |
| **矛盾处理** | 无（ADD-only并存） | 无 | 无 |
| **遗忘** | 无 | 无（cross/有但未接线） | 无 |
| **整合/抽象** | 无 | 无（synthesis未实现） | consolidate可能no-op |
| **记忆间链接** | 实体反链（间接） | 无 | 显式links字段（直接） |
| **持久化** | Qdrant+SQLite | LanceDB | ChromaDB内存模式（重启丢失） |
| **核心论文vs代码偏差** | 两段式决策管线已删；图记忆已删；linked_memory_ids被丢弃 | synthesis未实现；gate和动态d是固定常数 | 论文复现仓Note Construction永远失效；update_neighbor不同步向量索引 |

**共同点**（三家均符合"扁平向量存储 + 原子事实"大类的特征）：
- 记忆库为扁平结构（或接近扁平），无深度层级
- 写入时至少有一次 LLM 调用负责"从对话中提炼事实"（粒度不同但方向一致）
- **去重、矛盾处理、遗忘三项机制全部缺失或摆设**——这是三家共同的核心弱点，也是明确的研究空白

---

## 批次二横向对比（MemoryOS / MemGPT / MemOS）

| 维度 | MemoryOS | MemGPT（Letta） | MemOS |
|------|----------|----------------|-------|
| **记忆粒度** | QA对（STM/MTM）+ 事实句（LPM知识库）+ 90维画像 | 自由文本块（Core）/ 原样存文本（Archival）/ 原始消息（Recall） | 事实句 + 结构化元数据（key/tags/embedding/sources）+ 图节点 |
| **层级结构** | 三层（STM/MTM/LPM）+ Segment两级聚合 | 三层（Core/Archival/Recall），Core始终可见 | 三桶（Working/LTM/User）+ 可选PARENT边两层层级 |
| **写入时 LLM 次数** | STM未满0次；满时3次/条（连续性+链摘要+多主题）；LPM晋升+2次 | Core：0次额外；Archival：0次额外；Compaction：1次（小模型）；Sleeptime：多步（默认关闭） | fine模式：1次/窗口；fast模式：0次（后台异步精化） |
| **写入是否即时** | 即时（STM）/ 触发迁移（STM满） | 即时（LLM主动调工具）/ 溢出触发compaction | 即时（但fast模式质量延迟） |
| **检索方式** | 三路并行：STM全量+MTM两级FAISS+LPM向量 | Archival：向量（pgvector）或hybrid（需Turbopuffer）；Recall：SQL ILIKE子串（默认）；Core：始终可见无需检索 | 多路并行：WorkingMem全量+图元数据（key/tags精确）+向量+可选BM25 |
| **检索端 LLM** | 0次 | 0次（LLM自主决定何时调工具） | fine查询1次（意图解析）；fast查询0次 |
| **去重** | 无（同page多主题重复插入） | 无自动去重，靠LLM自觉 | 读取时字符串去重；写入去重默认关闭；fast→fine自动清理粗糙节点 |
| **矛盾处理** | 画像全量重写隐式消解；知识库新旧并存 | LLM自主判断并调工具修改（无系统保障） | LLM融合或按时间删旧（仅reorganize=True，默认关闭） |
| **遗忘** | STM FIFO；MTM LFU（access_count_lfu）；LPM deque maxlen=100；无时间衰减 | 无自动衰减；Archival无删除工具；Recall逻辑标记；Sleeptime可整合（默认关闭） | WorkingMem FIFO裁20条；LTM/User 80%阈值FIFO（异步路径）；usage字段从不更新 |
| **整合/抽象** | LPM晋升时90维画像全量重写 | Sleeptime agent多步编辑（默认关闭）；主路径只做滑窗摘要 | KMeans+LLM聚类摘要生成PARENT边主题层级（reorganize=True，每100秒，默认关闭） |
| **记忆间链接** | pre/next_page对话链指针 | 无显式链接；Block版本历史（管理端用） | PARENT边（主题→事实，reorganize时）；MERGED_TO边 |
| **持久化** | JSON文件（全量重写，含FAISS临时重建） | PostgreSQL/SQLite + 可选pgvector/Turbopuffer | Neo4j / PolarDB图数据库 |
| **核心论文vs代码偏差** | 淘汰按热度→实为LFU；时间常数差100倍且冻结；segment不会"迁移"只是抽取 | Archival已从默认工具集移除；memory pressure warning废弃；Recall默认字符串匹配非语义 | LoRA是占位符；图谱关系推理空转；冲突检测/聚类摘要默认关闭；usage字段从不更新 |

**批次二共同特征（OS 隐喻 + 多层分级记忆）**：
- 三个方案都使用操作系统隐喻（STM/MTM/LPM、Core/Archival/Recall、Working/LTM/User），但实现深度差异极大
- **共同弱点**：遗忘机制均为容量上限 FIFO，无真正的重要性衰减；矛盾处理均依赖 LLM 自觉或默认关闭的后台模块
- **MemGPT 的核心差异**：唯一一个让 LLM 自驱动记忆管理（LLM 主动调工具读写），其他两个是系统自动分层迁移
- **论文实现落差规律**：三个项目均有大量"默认关闭"或"空转"的特性，实际测评表现来自相对简单的核心路径（MemoryOS 的 MTM 两级检索、MemGPT 的 Core+Archival、MemOS 的 fine 模式明文记忆抽取）

---

### Zep（graphiti）

**仓库**：getzep/graphiti，commit 40eca36  
**论文**：arXiv:2501.13956，Zep: A Temporal Knowledge Graph Architecture for Agent Memory

---

1. **记忆形式与粒度**

   Zep 的"一条记忆"是**事实边（EntityEdge）**，即两个实体之间的一条关系陈述，而非对话摘要或事实句。完整字段（`edges.py:263-285`）：

   | 字段 | 含义 |
   |------|------|
   | `name` | 关系类型（如 `WORKS_AT`、`IS_FRIENDS_WITH`） |
   | `fact` | 一句话事实描述（如 "Alice works at Acme Corp as a senior engineer"） |
   | `fact_embedding` | 1024 维浮点向量，检索基础 |
   | `episodes` | 产生这条事实的原始 EpisodeNode UUID 列表（溯源） |
   | `created_at` | 入库时间（T' 时间线） |
   | `expired_at` | 在系统中被标记失效的时间（T' 时间线） |
   | `valid_at` | 现实中这件事开始成立的时间（T 时间线） |
   | `invalid_at` | 现实中这件事不再成立的时间（T 时间线） |

   另有两个辅助结构：
   - **EpisodicNode**（`nodes.py:318-330`）：原始消息原文 + 来源类型（message/text/json）+ `valid_at` 参考时间戳；是"原文存档层"，内容一字不改。
   - **EntityNode**：实体名 + 关于该实体的演化摘要 + name_embedding。

   **粒度**：关系三元组级（`主语实体 --[关系]--> 宾语实体`），是所有已研究方案里最细的粒度。Mem0/SimpleMem 存事实句，MemoryOS 存 QA 对，A-MEM 存原始对话文本；Zep 存的是从文本中抽取的单条关系断言。

   **与其他粒度方案的本质区别**：四个时间戳实现**双时序模型**（bi-temporal）——T 时间线记录"这件事现实中什么时候成立"，T' 时间线记录"我们什么时候把这条信息录入系统"，两者独立。其他方案要么无时间戳（A-MEM），要么只有 `created_at` 入库时间，无法区分"今天得知一件三年前发生的事"这种情形。

---

2. **写入机制**

   **触发时机**：每次调用 `graphiti.add_episode(episode_body, reference_time)`（`graphiti.py:980`）时同步触发。

   **LLM 参与次数**：典型 **2~5+ 次**。下限是"抽实体 + 抽事实边"两次 LLM；每条新边若检测到矛盾或重复候选则额外追加一次；边越多、矛盾越多，调用越多。

   **写入管线**（`graphiti.py:980-1228`，串行执行）：
   1. 存 EpisodicNode（原文原样落库）
   2. 取最近 10 条 episode 作为 LLM 上下文
   3. **LLM 抽实体节点**（`extract_nodes`）
   4. 实体去重（三级成本分层）：精确名匹配 → MinHash/LSH 模糊匹配 → 未决者 LLM 仲裁
   5. **LLM 抽事实三元组 + 时间窗**（`extract_edges`，含相对时间解析为绝对时间）
   6. 事实去重 + 矛盾失效（`resolve_extracted_edges`）：每条新边做 2 次混合检索找候选，LLM 判断 `duplicate_facts` / `contradicted_facts` 双列表，确定性代码执行时间窗比较写失效时间戳（`edge_operations.py:538-573`）
   7. 更新实体 `summary`（小于 2000 字符直接拼接，否则 LLM 重写，`node_operations.py:833-910`）
   8. 批量落库，补 `fact_embedding`

   **过滤/筛选**：无。矛盾时打失效时间戳（`expired_at`/`invalid_at`），旧事实和新事实**同时保留在图里**，不删除。

---

3. **检索机制**

   入口 `graphiti.search(query)`（`graphiti.py:1527`），三路并行：

   - **向量相似搜索**：query 转向量 → 余弦相似度检索 `fact_embedding`
   - **BM25 全文检索**：在 `fact` 字段文本上做 BM25
   - **BFS 图遍历**（可选）：从已知实体出发扩展邻域

   **融合**：RRF（Reciprocal Rank Fusion，倒数排名求和，`search_utils.py:1780-1795`），不需要三路分数可比，鲁棒性强。

   **重要细节：默认不过滤失效边**。`SearchFilters` 时间过滤字段默认全是 `None`（`search_filters.py:62-65`），两年前被标记失效的旧事实默认也会出现在检索结果里，附带 `invalid_at` 时间戳，让下游 LLM 自己决定哪条有效。调用方需自己构造 `DateFilter` 才能"只看当前有效事实"。

   **重排**：可选 reranker，实际实现是对每个候选独立发 LLM 请求问"相关吗（True/False）"，拿 True 的 logprob 当分数——**不是真正的 cross-encoder**（真正的 cross-encoder 把 query 和 passage 拼在一起输入）。

   **检索端 LLM 参与次数**：RRF 本身 **0 次 LLM**；启用 reranker 则 N 次（每个候选 1 次）。

   **结果数量控制**：`limit` 参数；候选池由三路 top-k 决定。

---

4. **注入 prompt**

   `search_results_to_context_string`（`search_helpers.py:27-72`）格式化为：

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

   **插入位置**：由调用方决定（库本身只返回字符串），通常放在 user 消息的 context 段。

   **截断/优先级控制**：由检索侧 `limit` 参数控制数量，无动态优先级。图不做过滤，把有效和失效事实都给 LLM 看。

---

5. **记忆管理**

   **去重**：三级分层——精确名匹配 → MinHash/LSH 模糊 → LLM 仲裁（仅实体去重）；事实边去重：LLM 判断 `duplicate_facts` 列表，重复边直接丢弃（`IS_DUPLICATE_OF` 溯源边在主写入路径被丢弃，`graphiti.py:1131`）。

   **矛盾处理**：**最完整的开源实现**。LLM 返回 `duplicate_facts` + `contradicted_facts` 双列表（同一旧边可同时出现在两个列表）；确定性代码比较 `valid_at`：若旧事实早于新事实，给旧边写 `invalid_at = 新事实.valid_at`、`expired_at = now`；若旧事实反而更晚，则新边当场标失效。旧事实**永远不删除**，历史完整保留（`edge_operations.py:538-573`）。

   **遗忘/淘汰**：**完全无**。无 TTL、无 decay、无容量上限、无自动删除。图只增不减（手动 `remove_episode` 除外）。搜索全库 `decay/forget/prune/evict/ttl`，只在注释和 prompt 文案里有 "forget" 字样，无任何机制实现。

   **整合/抽象**：实体 summary 会随 episode 处理演化（小于 2000 字符直接拼接，超限 LLM 重写）；社区子图（全局主题聚类摘要）默认关闭（`update_communities=False`，`graphiti.py:989`），手动触发为全量重建（先删后建）。

---

6. **其他设计**

   - **实体去重三级成本分层**：精确名匹配（O(1)）→ MinHash/LSH 模糊哈希（处理拼写变体）→ LLM 仲裁（只在模糊情形才调用），将 LLM 成本控制在必要最小集（`dedup_helpers.py:220-279`）。
   - **矛盾处理语义/时序解耦**：LLM 只做语义判断（"这两条是否矛盾"），时间窗比较和时间戳写入全部用确定性代码，防止 LLM 在日期计算上出错（`edge_operations.py:538-573`）。
   - **每条新边 embedding 至少被重复计算 3 次**（落库前算一次，去重检索时再 embed 一次，失效候选检索时又 embed 一次，`edge_operations.py:363,392-418`），无向量缓存复用，浪费成本。
   - **社区摘要两两归并**：N 成员社区做类归并排序，O(N) 次 LLM，信息在多层合并中损耗不可控（`community_operations.py:174-213`）。

---

7. **核心创新点**

   **论文宣称的最核心贡献**：把"时序信息"系统性引入知识图谱记忆——**双时序模型**（T + T' 两条时间线）+ 矛盾处理机制（失效时间戳替代删除）。这是当前所有开源 agent 记忆项目里**最完整的矛盾处理实现**。

   **与同类方案的本质区别**：
   - vs Mem0/SimpleMem：Mem0/SimpleMem 存事实句，无法追踪"这件事从什么时候开始、什么时候失效"；Zep 用四时间戳精确记录事实生命周期
   - vs A-MEM：A-MEM 有记忆间链接（links），但无时序信息；Zep 的"链接"是图的结构性关系（节点-边），每条关系本身携带时间窗
   - vs MemGPT：MemGPT 的 Core Memory 是 LLM 自主覆写，旧信息被丢失；Zep 的失效机制保全历史，可查任意时间点状态（虽然接口未封装好）

---

8. **论文 vs 代码差异**

   - **"可查任意时间点历史状态"打折扣**：论文宣称 bi-temporal 模型支持时间点查询，实际没有封装好的"时间点查询"接口；`SearchFilters` 时间过滤字段默认 `None`，默认检索连过期边都会返回。调用方需自己构造 `DateFilter`（`search_filters.py:62-65`）。
   - **社区子图动态更新打折扣**：论文强调动态更新，代码里增量更新路径粗糙（把新实体 summary 和社区 summary 拼给 LLM 重写一次，不重新聚类）且默认不启用；全量重建是全删再建（`add_episode` 默认 `update_communities=False`，`graphiti.py:989`）。
   - **cross-encoder reranker 表述不准确**：论文说"cross-encoder reranker"，实际是对每个候选独立发 LLM True/False 请求（`graphiti.py:1527` 附近），代价和原理均与真正的 cross-encoder 不同。
   - **IS_DUPLICATE_OF 溯源边缺失**：论文描述了去重溯源边，主写入路径里该边直接被丢弃（`graphiti.py:1131`），疑似留给闭源商业部分。
   - **`node_distance_reranker` 退化为二值判断**：注释写"按最短路径排序"，实际 Cypher 只检查是否 1-hop 相连（命中得 1 分否则 ∞），不是路径长度（`search_utils.py:1816-1845`）。

---

9. **实验**

   **数据集**：DMR（Deep Memory Retrieval，500 段对话，每段最多 60 条消息）；LongMemEval（平均 115,000 词，跨会话信息整合、时间推理等复杂任务）。

   **基线**：递归摘要、对话摘要、MemGPT、全文上下文。

   **核心结论**：

   DMR（DMR 太简单，全文直接塞也能拿 94.4%，参考价值有限）：
   | 方法 | 分数 |
   |------|------|
   | MemGPT | 93.4% |
   | 全文上下文（gpt-4-turbo） | 94.4% |
   | Zep（gpt-4-turbo） | 94.8% |
   | 全文上下文（gpt-4o-mini） | 98.0% |
   | **Zep（gpt-4o-mini）** | **98.2%** |

   LongMemEval（更能说明问题）：
   | 方法 | 模型 | 准确率 | 延迟 | 平均上下文词数 |
   |------|------|--------|------|--------------|
   | 全文上下文 | gpt-4o-mini | 55.4% | 31.3s | 115k |
   | **Zep** | **gpt-4o-mini** | **63.8%** | **3.20s** | **1.6k** |
   | 全文上下文 | gpt-4o | 60.2% | 28.9s | 115k |
   | **Zep** | **gpt-4o** | **71.2%** | **2.58s** | **1.6k** |

   - 准确率提升 15-18.5%，多会话整合、偏好、时间推理改善最明显——正是 Zep 设计目标
   - 延迟降低约 90%（上下文从 115k 降到 1.6k 词）
   - **唯一变差**：single-session-assistant 下降 9-18%（细节在结构化抽取时可能丢失）

---

### G-Memory

**仓库**：bingreeky/GMemory，commit 7b581c5  
**论文**：arXiv:2506.07398，G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems

---

1. **记忆形式与粒度**

   G-Memory 的"一条记忆"是**任务级记忆（MASMessage）**，覆盖一次完整的多 agent 协作任务（`mas/memory/common.py:136-191`）：

   | 字段 | 含义 |
   |------|------|
   | `task_main` | 任务名称/主题（如 "put a clean egg in microwave"） |
   | `task_description` | 完整任务描述（含环境状态） |
   | `task_trajectory` | 所有 agent 发言和环境反馈的完整文字对话（展平文本） |
   | `label` | 任务成功（True）/ 失败（False） |
   | `key_steps` | LLM 写入时从轨迹提炼的关键步骤（如"1. 找蛋 2. 清洁蛋 3. 放入微波炉"） |
   | `fail_reason` | 仅失败任务：LLM 诊断的失败原因 |
   | `chain_of_states` | 每个环境步对应一张 `nx.DiGraph`，节点是 `AgentMessage`，边是时序关系 |

   **三层组织结构**：
   - **Interaction Graph（交互图）**：`chain_of_states` 序列，每步一张有向图，捕获 agent 间发言和行动的完整图结构。**写入有，检索后从不消费图结构**（见第 8 条）。
   - **Query Graph（查询图）**：networkx 图，节点是历史任务（task_main 为 ID），相似度 ≥ 0.7 的任务连边，权重=相似度（`GMemory.py:374-402`）。支持 k-hop 邻域扩展检索。
   - **Insight Graph（洞察图）**：实际是扁平的 `list[dict]`（`GMemory.py:477`），每条 insight：`{rule, score, positive_correlation_tasks, negative_correlation_tasks}`，**节点间无边**（见第 8 条）。

   **粒度**：任务级，远粗于其他方案的事实句或 QA 对。一条记忆可覆盖几十甚至上百条对话。通过 `key_steps` 字段在写入时压缩，检索时返回压缩版而非原始轨迹。

   **与其他粒度方案的本质区别**：（1）其他方案面向单 agent，G-Memory 专为 MAS 设计，记忆单元是一次多 agent 协作任务（单 agent 记忆直接移植会因对话量大 10-100 倍而失效）；（2）有成败标签（`label`），失败的经历也是可利用的信息；（3）`key_steps` 在写入时一次性计算好，检索时零成本复用，对比 Zep/Mem0 每次检索都需要实时处理。

---

2. **写入机制**

   **触发时机**：任务完成时调用 `save_task_context(label=True/False)`（`autogen.py:200 → memory_base.py:59-66`），不在任务执行中途写入。

   **LLM 参与次数**：
   - 成功任务：**1 次**（LLM 提炼 `key_steps`，`GMemory.py:244-281`）
   - 失败任务：**2 次**（+LLM 诊断 `fail_reason`）
   - 每满 5 任务触发 `finetune_insights`：约 **5-15 次 LLM**（成功/失败对比 critique + 成功批量 critique）
   - 每满 20 任务触发 `merge_insights`：FINCH 聚类后 LLM 批量合并，约 **N/10 次 LLM**

   **写入主流程**（`GMemory.add_memory`，`GMemory.py:76-111`）：
   1. **轨迹稀疏化**：删除 `reward < 0` 的环境步（中间走错的步骤）；数字泛化（`re.sub(r'\d+', '', trajectory)`，将"desk 1"→"desk "，防 LLM 过拟合到具体编号）
   2. LLM 提炼 `key_steps`，失败任务额外诊断 `fail_reason`
   3. 任务图加节点/连边：`task_description` embedding 后与已有节点计算相似度，≥ 0.7 连边（`GMemory.py:374-402`）
   4. Chroma 入库：整个 `MASMessage` 序列化为 JSON 字符串 + `task_description` embedding
   5. 周期性触发 `finetune_insights`（每 5 任务）和 `merge_insights`（每 20 任务）

   **过滤/筛选**：insight 写入时子串查重防止完全重复的规则入库（`GMemory.py:808-878`）；任务轨迹无去重，直接追加。

---

3. **检索机制**

   入口 `retrieve_memory`（`GMemory.py:189-241`），每个任务开始前调用（`autogen.py:108`），分三步：

   **第一步：粗检索**（`_retrieve_memory_raw`，`GMemory.py:113-187`）：
   - Chroma 向量检索当前任务 embedding 的 top-k 历史任务（k=1 或 2）
   - 在任务图上做 1-hop 邻域扩展：找到与 top-k 节点有边相连的所有任务节点
   - 结果按 `label` 分成成功和失败两组
   - **已知 bug**（`GMemory.py:150-156`）：候选不足时，普通向量检索结果**覆盖**（而非补充）图扩展结果，核心 k-hop 机制被静默旁路为纯向量检索

   **第二步：LLM 精排**（`GMemory.py:220-228`）：对每条候选成功轨迹，LLM 打 1-10 分（"这条历史轨迹对当前任务有多大参考价值？"），取 top-1（默认）。失败轨迹和 insight 不精排，直接截断。每次检索约 **2×successful_topk 次 LLM**（默认 2 次）。

   **第三步：Insight 检索**（`GMemory.py:490-506`）：向量检索找 4 个相关成功任务 + 2 个相关失败任务，统计每条 insight 的 `positive_correlation_tasks` 里有多少命中（**字符串精确匹配**，`GMemory.py:637`，不是语义相似度），取 top-k。

   **重排**：LLM 对成功轨迹精排（1 次/候选），insight 无重排。

   **结果数量控制**：`successful_topk`（默认 1）、`failed_topk`（默认 0，失败轨迹实际不进 prompt）、`insight_topk` 参数。

---

4. **注入 prompt**

   `format_task_prompt_with_insights`（`format.py:42-58`）拼成三段式：

   ```
   【Your Own Past Successes】
   Task: put a clean egg in microwave  (Status: Success)
   Key Steps: 1. Go to fridge and take egg  2. Find sink and clean egg  3. Put egg in microwave
   Detailed Trajectory: [压缩版轨迹全文]

   【Failure Lessons】
   （默认 failed_topk=0，此段通常不出现）

   【Key Insights】
   - Verify the state of objects before and after each action...
   - After cleaning an item, place it immediately to the designated location...
   ```

   **插入位置**：每个任务第一步（solver agent 开始推理前）拼入 user prompt（`autogen.py:135-140`）。

   **截断/优先级控制**：各 top-k 参数控制；失败轨迹默认 `failed_topk=0` 不进 prompt（三种 MAS 拓扑代码均用 `_` 忽略失败轨迹返回值）。

---

5. **记忆管理**

   **去重**：insight 写入时子串查重（防完全重复规则）；任务轨迹在 Chroma 中直接追加，无相似度去重。

   **矛盾处理**：insight 有 ADD/EDIT/REMOVE/AGREE 四类操作（`finetune_insights`），但这是"经验演化"而非"逻辑矛盾检测"，不专门处理语义矛盾。

   **遗忘/淘汰**：
   - **Insight 信用分机制**（核心设计）：每条 insight 带 `score`，成功使用 +1，失败使用 -2；`score ≤ 0` 时自动删除（`GMemory.py:584-586`）。失败惩罚大于成功奖励的不对称设计有道理（一次失败的危害大于一次成功的收益）。
   - **Merge 重置问题**：每满 20 任务 `merge_insights` 清空重建 insight 库，所有 `score` 重置为 2（`GMemory.py:508-549`），积累的信用分历史归零，与"长期演化"叙事相悖。
   - 任务轨迹（Chroma）无自动淘汰。

   **整合/抽象**：
   - 每 5 任务：`finetune_insights` 做成功/失败对比 critique，LLM 输出 ADD/EDIT/REMOVE/AGREE 操作更新 insight 库
   - 每 20 任务：`merge_insights` 用 FINCH 聚类自动发现任务族群，同簇 insight 用 LLM 批量合并精简（每 10 条合并为 1 条），清空重建

---

6. **其他设计**

   - **轨迹稀疏化三件套**：删负 reward 步 + 数字泛化 + 写入时一次性提炼 `key_steps`，检索时零成本复用。数字泛化让规律可迁移（"desk 1"和"desk 5"在规律层面等价）。
   - **成败 backward 闭环**：任务结束后 `backward` 更新本次使用的 insight 的信用分（`GMemory.py:292-297`），把"这条经验到底有没有用"做成可量化的在线反馈回路。
   - **分层触发维护节奏**：每任务轻量写入（1-2 次 LLM）→ 每 5 任务提炼 insight（5-15 次 LLM）→ 每 20 任务聚类合并（N/10 次 LLM），把昂贵的整合操作摊薄到周期任务，控制写放大。

---

7. **核心创新点**

   **论文宣称的最核心贡献**：**三层层级记忆（Interaction/Query/Insight）+ 双向检索**，第一个专门为多 agent 系统（MAS）设计的记忆架构。单 agent 记忆系统直接移植到 MAS 会因为对话量大 10-100 倍（AutoGen 在 ALFWorld 平均消耗 4.3×10⁶ token）而不可行。

   **与同类方案的本质区别**：
   - vs 所有单 agent 方案（Mem0/SimpleMem/Zep 等）：记忆单元是多 agent 任务轨迹（任务级），而非单条消息或事实（消息级）；有成败标签，失败经历也可被利用
   - vs MemGPT：MemGPT 的 Archival 是 LLM 手动存文本，G-Memory 是任务结束时系统自动写入 + 周期性自动整理
   - **Insight 信用分机制**是独特设计：经验的有效性通过成败结果量化追踪，无效规律自然淘汰

---

8. **论文 vs 代码差异**

   - **"Interaction Graph 捕获 agent 间协作结构"——存了但不用**：`chain_of_states` 确实写进 `MASMessage` 并落库（`common.py:75-87,179`），但检索回来后代码只读 `task_trajectory`（文字版）和 `key_steps`（`autogen.py:115-117`），图节点结构从未被消费。"能捕获协作关系"在写入侧是真的，在读取侧是假的。
   - **"Insight Graph"——代码里没有图**：论文图示里 insight 节点之间有 contextualizes 关系边（论文 Equation 10），代码是扁平 `list[dict]`（`GMemory.py:477`），insight 间无任何边，关联全靠 `positive_correlation_tasks` 字符串精确匹配隐式表达。
   - **"Agent-specific memory"——默认关闭**：`--use_projector` 默认 False（`GMemory.py:304-350`），默认所有角色拿同一份 insight，不分角色定制。
   - **失败轨迹"参与检索"——实际不进 prompt**：三种 MAS 拓扑代码均 `successful_trajectories, _, insights = retrieve_memory(...)`，用 `_` 忽略失败轨迹；`--failed_topk` 默认 0。失败轨迹仅在离线 `finetune_insights` 对比 critique 里被引用。
   - **DyLAN 路径 insight 反馈失效**：`dylan.py:259-264` 任务结束只调 `save_task_context`，无 `backward`，insight 信用分永远不更新（`dylan.py:259-264`）。
   - **k-hop 扩展经常被旁路**：候选不足时 `GMemory.py:150-153` 用纯向量检索**覆盖**图扩展结果，加上后续循环 `if doc not in true_tasks_doc` 永假 bug（`GMemory.py:154-164`），很多实际运行退化为纯向量检索。

---

9. **实验**

   **数据集**：ALFWorld（文字版家庭环境）、SciWorld（文字版科学实验）、PDDL（策略游戏/搭积木）、HotpotQA（多跳问答）、FEVER（事实核查）。5 个基准，结合 3 种 MAS 框架（AutoGen/DyLAN/MacNet）和 3 种 LLM（GPT-4o-mini/Qwen-2.5-7b/Qwen-2.5-14b）。

   **基线**：MetaGPT-M、各 MAS 框架无记忆版本、其他 MAS 记忆方案。

   **核心结论**：G-Memory 在所有配置上均超过全部对照系统：

   | 任务 | 框架 + LLM | 无记忆基线 | +G-Memory | 提升 |
   |------|-----------|-----------|----------|------|
   | ALFWorld | MacNet + Qwen-2.5-14b | 58.21% | **79.10%** | +20.89% |
   | HotpotQA + FEVER 均值 | — | — | — | **+10.12%** |

   - Token 消耗：G-Memory 额外约 1.4×10⁶，MetaGPT-M 额外约 2.2×10⁶，但 MetaGPT-M 提升幅度远不如 G-Memory
   - 消融实验：洞察层和交互层缺一不可（各缺少 3-4%）；1-hop 邻域扩展最优，2-3 hop 引入噪音导致性能下降

   **值得注意的细节**：Insight 查找靠 `positive_correlation_tasks` 字符串精确匹配（`GMemory.py:637`），任务名称有微小变化就失效；`merge_insights` 每 20 任务清空重建，与"长期演化"叙事相悖；k-hop 扩展在代码 bug 下经常退化为纯向量检索，实验中"图扩展"的真实贡献难以还原。

---

### MIRIX

**仓库**：Mirix-AI/MIRIX，commit 905984e  
**论文**：arXiv:2507.07957，MIRIX: Multi-Agent Memory System for LLM-Based Agents

---

1. **记忆形式与粒度**

   MIRIX 将记忆分为**六类**，每类有独立的结构化 schema 和专属 Memory Manager：

   | 记忆类型 | 存储内容 | 粒度 |
   |---------|---------|------|
   | **Core Memory** | agent persona + 用户长期偏好/身份信息 | 最粗，用户画像级 |
   | **Episodic Memory** | 带时间戳的事件（"2025-03-05 10:15 用户在看某论文"） | 事件级 |
   | **Semantic Memory** | 实体/关系/概念（"John 是用户朋友，住 San Francisco"） | 三元组级 |
   | **Procedural Memory** | 操作步骤（"如何通过 OpenTable 订餐"） | 步骤级 |
   | **Resource Memory** | 用户正在用的文档/文件/媒体内容 | 文档/片段级 |
   | **Knowledge Vault** | 需逐字保留的敏感信息（地址/手机号/API key） | 字段级精确值 |

   路由规则写在 prompt 文件中（`mirix/prompts/system/base/meta_memory_agent.txt:6-112`），由 Meta Memory Manager（LLM）判断输入应交给哪些子 manager。

   **粒度**：随记忆类型变化。同一句话"我电话是 1234-5678，明天上午 10 点去见 Alice"会分拆：Knowledge Vault 存电话号，Episodic 存明天见 Alice 的事件，Semantic 存"Alice 是用户联系人"这条关系。

   **与其他粒度方案的本质区别**：其他方案用统一粒度存所有信息（Mem0 全是事实句，MemGPT 全是自由文本），MIRIX 让不同性质的信息进不同"抽屉"；检索时可按记忆类型过滤，注入时按 XML 标签分类，让模型感知"这条来自事件记忆还是敏感信息库"。

---

2. **写入机制**

   **触发时机**：`POST /memory/add` 进 Kafka/内存队列，**异步处理**，不阻塞用户（`rest_api.py:2005`，`queue_util.py:76`）。

   **LLM 参与次数**：典型 **3-9 次**：
   - topic 抽取：**1 次 LLM**（`agent.py:2098`，强制调 `update_topic`）
   - meta 代理决策：**1-2 次 LLM**（含 function chaining）
   - 被选中的每类记忆子代理：**各 1 次 LLM**
   - 落库条目：每类 1-2 次 embedding（如 episodic 的 summary + details 各算一次，`episodic_memory_manager.py:557-566`）

   **写入主流程**（以 AutoGen 拓扑为例）：
   1. 消息进队列
   2. meta 代理 step 0：LLM 抽 topic（`_extract_topics_from_messages`，`agent.py:2098`）
   3. 用 topic 执行 Active Retrieval，重建 system prompt（`build_system_prompt_with_memories`，`agent.py:1726`）
   4. meta 代理看"新输入 + 已检索记忆"，调 `trigger_memory_update(memory_types=[...])`，用 `asyncio.gather` 并行运行被选中的子代理（`memory_tools.py:1187-1188`）
   5. 各子代理执行一次工具调用写 DB（如 `episodic_memory_insert`、`semantic_memory_update`）
   6. 子代理清空对话历史，只保留 system 消息（`agent.py:1252-1283`）——状态全在 DB，代理无状态

   **过滤/筛选**：多数 insert 工具拉全量后字段精确比对（完全相等才跳过，`memory_tools.py:625-647`），只防逐字重复，无语义去重。

---

3. **检索机制**

   **Active Retrieval（默认自动，每轮触发）**（`agent.py:1726`）：
   - 先抽当前 topic（LLM 1 次）
   - 用 topic 检索六类记忆，每类最多 top-10（`constants.py:87`）
   - **主链路 `search_method` 硬编码 `"bm25"`**（`agent.py:1754`），embedding 分支在默认路径不可达

   **"BM25"实际是 PostgreSQL 全文检索**：`to_tsvector + ts_rank_cd`（`episodic_memory_manager.py:1094-1118`），AND 模式失败时回退 OR；真正的 `BM25Okapi` 只在 SQLite 回退分支存在（`episodic_memory_manager.py:948-971`）。

   **显式工具检索**：`search_in_memory(memory_type, query, search_method)`（`base.py:84`），支持 bm25/embedding/string_match，供 agent 或外部 API 调用。

   **重排**：主链路无重排。

   **检索端 LLM 参与次数**：topic 抽取 **1 次 LLM**；检索本身 0 次 LLM。

   **结果数量控制**：每类 top-10，Core Memory 始终全量可见。

---

4. **注入 prompt**

   `build_system_prompt`（`agent.py:1966-2052`）把各类检索结果按 XML 标签块拼进 system prompt：

   ```
   <episodic_memory>
   2025-03-05 10:15: User was reading a paper on memory systems
   ...
   </episodic_memory>
   <knowledge_vault>
   Phone: 1234-5678
   ...
   </knowledge_vault>
   <semantic_memory>
   Alice is user's colleague at XYZ company
   ...
   </semantic_memory>
   ```

   **插入位置**：system prompt（每轮对话前自动重建，全量替换）。

   **截断/优先级控制**：每类 top-10，无动态优先级；Core Memory 始终纳入，不参与 top-k 竞争。无 token 预算控制。

---

5. **记忆管理**

   **去重**：精确字段完全相等才跳过（`memory_tools.py:625-647`），无语义去重。`semantic_memory_update` 是删旧插新（`memory_tools.py:685-744`），不做精细版本合并。

   **矛盾处理**：无统一矛盾检测器，靠子代理 LLM 在一次工具调用里自判断。Core Memory 超 90% 时返回错误提示，让 LLM 自己决定如何 rewrite（`memory_tools.py:57-67`），不是确定性压缩。

   **遗忘/淘汰**：六类长期记忆**无自动 TTL/衰减/容量淘汰**。`raw_memory` 有 14 天 TTL 清理脚本（`jobs/cleanup_raw_memories.py:20`），但 raw memory 不是六类长期记忆，且需外部 cron 或手动触发。

   **整合/抽象**：Reflexion 代理（`app_constants.py:28`，`WITH_REFLEXION_AGENT=False`）理论上可做全库去重和用户行为模式归纳，但**默认关闭且无调度触发**；Background 代理（`app_constants.py:31`）默认关闭且实现极薄。

---

6. **其他设计**

   - **多模态支持**：桌面端截图流通过 `TemporaryMessageAccumulator` 攒满 20 条触发记忆更新（`temporary_message_accumulator.py:446-501`），是论文 ScreenshotVQA 实验的实现基础。
   - **子代理无状态化**：记忆代理每次更新完即清空对话历史（`agent.py:1252-1283`），状态全在 PostgreSQL/pgvector/Redis，代理可水平扩展。
   - **XML 标签分类注入**：检索结果按来源类型标注，让模型感知"这条是事件记忆还是敏感数据库"，减少混用风险。
   - **写入全异步**：Kafka/内存队列解耦，不阻塞用户前端响应。

---

7. **核心创新点**

   **论文宣称的最核心贡献**：**六类记忆分工（Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault）+ Meta Memory Manager 路由 + Active Retrieval 自动注入 system prompt**，把"记忆"从扁平向量库提升为按信息性质分类管理的结构化系统。

   多模态价值证明：ScreenshotVQA 实验表明，把 GB 级截图抽成结构化记忆（MB 级），准确率反而更高（0.595 vs SigLIP RAG 0.441），存储减少 99.9%。

   **与同类方案的本质区别**：
   - vs Mem0/SimpleMem/Zep：这些方案用统一格式存所有信息，MIRIX 按性质分类；检索结果带来源标签，模型可感知信息类型
   - vs MemGPT：MemGPT 靠 LLM 自主决定存什么/检索什么，MIRIX 系统自动路由写入 + Active Retrieval 自动检索，LLM 只做分类和内容判断
   - vs G-Memory：G-Memory 是任务级记忆（为 MAS 设计），MIRIX 是对话级多类型记忆（为个人助手设计）

---

8. **论文 vs 代码差异**

   - **"多种检索函数智能选择"——默认只走 bm25**：`agent.py:1754` 硬编码 `search_method="bm25"`，embedding 分支在默认 Active Retrieval 路径不可达；向量检索主要通过显式工具或 HTTP API 使用。
   - **"BM25"名不副实**：PostgreSQL 路径实际是 `ts_rank_cd`（`episodic_memory_manager.py:1094`），不是标准 BM25；真正的 `BM25Okapi` 只在 SQLite 回退分支。
   - **Reflexion 代理默认是摆设**：`WITH_REFLEXION_AGENT=False`（`app_constants.py:28`），无默认定时触发，全仓库无调度代码。
   - **Background 代理基本是 stub**：`app_constants.py:31` 默认关闭，实现非常薄。
   - **prompt 与代码常数不一致**：episodic 代理 prompt 说展示最多 50 条，代码常数 top-10（`constants.py:87`）；`CALL_MEMORY_AGENT_IN_PARALLEL` 定义后无实际引用（`constants.py:237`）。
   - **六类记忆间一致性管理缺失**：同一信息可能同时进 Episodic 和 Semantic，后续如何同步更新无明确机制。

---

9. **实验**

   **数据集**：ScreenshotVQA（自建，3 名博士生 5349-18178 张屏幕截图，87 个手动构造的问题）；LOCOMO（长对话记忆，每个 conversation 约 600 轮 26000 tokens，含 single-hop/multi-hop/temporal/open-domain，论文排除 adversarial 类）。

   **基线**：ScreenshotVQA：Gemini 长上下文、SigLIP@50+Gemini 图像 RAG；LOCOMO：Zep、LangMem、Mem0、Full-Context。

   **核心结论**：

   ScreenshotVQA：
   | 方法 | Overall Acc | Storage |
   |------|------------|---------|
   | Gemini 长上下文 | 0.1166 | 236.70MB |
   | SigLIP@50 + Gemini | 0.4410 | 15.07GB |
   | **MIRIX** | **0.5950** | **15.89MB** |

   LOCOMO（LLM-as-Judge 总分）：
   | 方法 | Single-Hop | Multi-Hop | Temporal | Overall |
   |------|-----------|----------|---------|---------|
   | Zep | 79.43 | 69.16 | 83.33 | 79.09 |
   | Mem0 | 62.41 | 57.32 | 66.47 | 62.47 |
   | **MIRIX** | **85.11** | **83.70** | **88.39** | **85.38** |
   | Full-Context | 88.53 | 77.70 | 92.70 | 87.52 |

   - multi-hop：MIRIX 83.70 超过 Full-Context 77.70，论文解释为写入时已整合分散信息
   - ScreenshotVQA 准确率高 35%（vs SigLIP），存储减少 99.9%（vs SigLIP）

   **值得注意的细节**：ScreenshotVQA 只有 3 用户 87 题，问题由用户自己构造，规模偏小；LOCOMO 排除 adversarial 问题；写入成本 3-9 次 LLM/次，论文未与准确率并列给出成本/延迟对比；Reflexion 等维护机制默认关闭，评测实际测的是"路由写入 + Active Retrieval"的基础路径。

---

## 批次三横向对比（Zep / G-Memory / MIRIX）

| 维度 | Zep（graphiti） | G-Memory | MIRIX |
|------|----------------|----------|-------|
| **记忆单元** | 事实边（实体间带时序的关系陈述）+ 实体节点 + 情节节点 | 任务级轨迹（MASMessage，含轨迹全文/key_steps/成败标签/状态图序列） | 六类结构化记忆（Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault） |
| **记忆粒度** | 最细（关系三元组级）；一条对话可产生多条事实边 | 最粗（任务级，一条记忆覆盖整次多 agent 任务） | 因类而异（事件级/字段级/步骤级）；同一输入分拆到多类 |
| **写入时 LLM 次数** | 2~5+ 次/条（抽实体+抽边+矛盾判断；边越多越多） | 1~2 次/任务（提炼 key_steps±fail_reason）；每 5/20 任务额外批量维护 | 3~9 次/条（topic 抽取+meta 决策+各子代理各 1 次） |
| **写入是否即时** | 即时同步 | 任务完成后触发（非对话中途） | 异步入队（不阻塞用户） |
| **写入是否结构化** | 是，LLM 抽三元组+时间窗 | 部分（LLM 提炼 key_steps，轨迹原样存） | 是，各类 manager 有独立 schema |
| **检索方式** | 向量+BM25+BFS，RRF 融合 | Chroma 向量+1-hop 图扩展+LLM 精排（成功轨迹） | Active Retrieval 默认 bm25（硬编码），可选 embedding/string_match |
| **检索端 LLM** | 0 次（RRF 确定性）；reranker 可选（每候选 1 次） | 2 次（每候选成功轨迹 1 次打分） | 1 次（topic 抽取） |
| **默认检索过滤** | 不过滤失效边（需调用方自构造 DateFilter） | 按 label 分成功/失败两组 | 按 memory_type 分类检索 |
| **去重** | 实体：精确→MinHash→LLM；事实边：LLM 判 duplicate_facts | insight 子串查重；轨迹无去重 | 精确字段完全相等才跳过（无语义去重） |
| **矛盾处理** | **最完整**：LLM 语义判断 + 确定性时序裁决，失效时间戳替代删除，历史完整保留 | 无显式矛盾检测（insight 有 EDIT/REMOVE 但非针对矛盾） | 无统一检测器，靠子代理 LLM 自判断 |
| **遗忘/淘汰** | **完全无**（图只增不减） | insight 信用分（score≤0 删除）；merge_insights 每 20 任务重置 score | 六类长期记忆无自动淘汰；raw_memory 14 天 TTL 需外部 cron |
| **整合/抽象** | 实体 summary 演化（超 2000 字拼接触发 LLM 重写）；社区摘要默认关闭 | finetune_insights（每 5 任务）+ merge_insights/FINCH 聚类（每 20 任务） | Reflexion 代理默认关闭；无生产调度 |
| **记忆间链接** | 图边（实体-实体关系边，每边带时间窗）；三层图（情节/语义/社区）结构完整 | Query Graph 任务节点连边（相似度权重）；Insight 关联靠字符串精确匹配（非图边） | 无记忆间链接；六类独立存储，跨类一致性未统一管理 |
| **时序处理** | **最强**：双时序模型四时间戳，可追踪事实的现实有效期（valid_at/invalid_at）和入库时间（created_at/expired_at） | 任务有时间戳，key_steps 无时序注解 | Episodic Memory 有时间戳，其他类型笔记未记录 |
| **多 agent 支持** | 单 agent 设计，无多 agent 路由 | **专为 MAS 设计**，任务级轨迹天然多 agent；有 Query Graph 跨任务连接 | Meta Memory Manager 可视为多 agent 写入路由，但读取侧仍是单路径 |
| **持久化** | Neo4j 图数据库（主）+ 可选其他图 DB | Chroma 向量库（轨迹）+ networkx 内存图（Query Graph）+ JSON 文件（insights） | PostgreSQL + pgvector + Redis 缓存；可选 SQLite 回退 |
| **论文 vs 代码最大落差** | 时间点查询未封装好；社区动态更新默认关闭；cross-encoder 实为 LLM True/False；IS_DUPLICATE_OF 边丢弃 | Interaction Graph 存了但检索不消费图结构；Insight Graph 无边；k-hop 常被旁路；DyLAN 路径 backward 失效 | Active Retrieval 硬编码 bm25 不用 embedding；Reflexion/Background 默认关闭；"BM25"实为 ts_rank_cd |

**批次三共同特征（图/多表结构为核心）**：
- 三个方案都以图或多表/多类结构为核心，节点/边/关系是一等公民，而非纯平铺向量库
- **Zep 的图结构最完整**：三层子图（情节/语义/社区）+ 实体节点 + 事实边，双时序模型是当前最系统的时序设计；但图只增不减是明显缺陷
- **G-Memory 的图用于路由而非记忆内容**：Query Graph 帮助跨任务找相关历史，而非直接存记忆内容；insight 信用分机制是最有新意的遗忘设计，但 merge_insights 每 20 任务清零消解了长期演化价值
- **MIRIX 的"图"是类型分工**：六类记忆本质是六张独立表，不是图数据结构意义上的图（节点-边）；最大贡献是多模态场景的工程架构，和 ScreenshotVQA 的实验设计
- **共同研究空白**：矛盾处理只有 Zep 做到，其他两家基本缺失；遗忘机制三家都很弱（Zep 完全无，G-Memory 的 score 每 20 任务归零，MIRIX 无自动淘汰）；图扩展带来的"跨记忆关联"在检索时的真实收益均未被完整验证（Zep BFS 可选、G-Memory k-hop 常被旁路、MIRIX 六类无跨类链接）

---

### MemEvolve

**仓库**：bingreeky/MemEvolve，commit 6035d56（代码全部在 `Flash-Searcher-main/` 子目录下）  
**论文**：MemEvolve: Meta-Evolution of Agent Memory Systems，arXiv:2512.18746，ICML'26，OPPO AI + LV-NUS Lab

---

1. **记忆形式与粒度**

   MemEvolve 进化出了两个主力系统，记忆格式不同：

   **Lightweight Memory（双层，代码主力）**

   长期记忆（`storage/lightweight_memory/longterm_memory.json`）分两类：

   | 类别 | 字段 | 含义 |
   |------|------|------|
   | `strategic` | `content` | 高层规划决策："何时选什么方法、如何分解复杂问题" |
   | `strategic` | `tags` | 分类标签列表 |
   | `strategic` | `usage_count` | LLM 选中使用次数 |
   | `strategic` | `success_count` | 使用后任务成功次数 |
   | `strategic` | `signature` | 内容 sha256 摘要，用于精确去重 |
   | `operational` | 同上结构 | 具体工具用法和出错处理技巧 |

   每类最多 30 条 + 20 条缓冲区，共 50 条上限。代码内置 5 条 strategic + 2 条 operational 冷启动默认记忆，空库时注入。

   短期记忆（`self.shortterm_memory`，内存驻留）：任务执行中每步 LLM 抽取的关键事实清单，任务结束后清空。

   **Cerebra Memory（知识图谱节点）**

   节点存为 `storage/cerebra_fusion_memory/cf_database.json`，字段：`id`、`content`、`node_type`（如 "insight"）、`edges`（含 `target_id/edge_type/weight/usage_count/success_count`）、`usage_count`、`success_count`、`task_type`。两节点 embedding 相似度 ≥ 0.75 时自动建双向边。另有**工具记忆**：成功轨迹中的操作模式被抽成参数化 Python 函数，存入 `tools_storage.py`，检索到后直接注册为可调用工具。

   **粒度**：经验/策略级，不是对话事实，不是原始轨迹，而是从成功执行中提炼出的"可复用模式"（可迁移规律，不记录"北京=39.9°N"这类任务特定数据）。

   **与其他粒度方案的本质区别**：
   - vs Mem0/SimpleMem：存的不是对话事实句，而是跨任务可迁移的策略规律
   - vs G-Memory/LatentMem：G-Memory 存完整轨迹，MemEvolve 只存提炼后的 insight；LatentMem 用向量压缩轨迹，MemEvolve 用自然语言描述策略
   - 最根本区别：被进化的不是记忆内容，而是记忆系统本身的 Python 代码

---

2. **写入机制**

   **触发时机**：任务完成后触发（非即时），且**只处理成功轨迹**（`is_correct=True`，`lightweight_memory_provider.py:332-333`；`cerebra_fusion_memory_provider.py:1022-1023`）。成功判定由 LLM judge 对比 gold answer 完成（`run_flash_searcher_mm_gaia.py:305-332`），因此写入只在有标注数据的评测场景下有效。

   **LLM 参与次数**：
   - 任务内短期记忆：每执行步骤 1 次 LLM（抽取当前步关键事实，`agents.py:783`）
   - 任务结束写入：1 次 LLM（`_extract_memories`，`lightweight_memory_provider.py:993`），提取最多 2 条 strategic + 2 条 operational；超阈值时额外 1 次 LLM 剪枝（`_intelligent_prune_memories`）

   **提取过程**：
   ```
   完整执行轨迹（step-by-step 记录）
       ↓ _is_trajectory_success 门控（仅成功）
       ↓ LLM 调用 _extract_memories：提取"可复用模式"，强调不记任务特定数据
         → JSON: {"strategic": [...], "operational": [...]}
       ↓ sha256 去重（签名命中则跳过）
       ↓ 追加进 longterm_memory.json
       ↓ 若总数超 50 条 → LLM 剪枝（优先保留 usage_count/success_count 高的条目）
   ```

   **过滤/筛选**：sha256 精确去重（字符串哈希，非语义去重）；失败轨迹完全不写入，无"从失败中学习"路径。

---

3. **检索机制**

   检索分两个时机（`MemoryStatus.BEGIN` 和 `IN`）：

   **BEGIN（任务开始规划时）**：若 `enable_longterm_provision=True`（**默认 False！**，`EvolveLab/config.py:52`），把长期库全部（≤60 条）内容交给 LLM，LLM 选 top-k 最相关并合成 4-5 句 guidance，以 `————Memory System Guidance————` 包裹插入 prompt（`agents.py:414`）。默认配置下此路径完全跳过。

   **IN（每个执行步骤结束后）**：固定每步 1 次 LLM 抽取短期关键事实；每隔 5 步（`shortterm_provision_interval`）格式化为清单注入 prompt（`agents.py:783`，`lightweight_memory_provider.py:584-715`）。

   **Cerebra 检索**：TF-IDF（权重 0.2）+ embedding 语义（权重 0.8）混合检索；1 步图扩散（分 = 原始分 × 边权重 × 0.7）；阈值 0.22 过滤；top-3 节点；LLM 合成压成 ≤350 字 guidance。

   **召回后有无重排**：Lightweight 无重排；Cerebra 有阈值过滤。

   **检索端 LLM 参与次数**：BEGIN 1 次（仅 enable_longterm_provision=True 时）；IN 每步 1 次短期抽取；Cerebra 另有 1 次合成压缩。

   **结果数量控制**：长期库 ≤60 条（`enable_longterm_provision`）；短期清单逐步积累无硬截断。

---

4. **注入 prompt**

   **BEGIN（长期记忆，仅开启时）**：
   ```
   ————Memory System Guidance————
   Anti-ambiguity: Consider first locating the canonical Wikipedia article...
   Tool-use Suggestion: Based on similar tasks, use the MediaWiki API...
   ————————————————————————————
   ```
   作为 user message 插入（`agents.py:414`）。

   **IN（短期记忆，每 5 步）**：
   ```
   - Step 3: Located Wikipedia article 'Outer Wilds' at https://en.wikipedia.org/wiki/Outer_Wilds
   - Step 4: Confirmed release date listed as May 2019 (Windows/Xbox One)
   ```
   格式化为带步骤编号的项目符号清单，注入 prompt 上下文段。

   **截断/优先级控制**：无显式截断；由 top-k 配置参数和 ≤60 条上限间接控制。

---

5. **记忆管理**

   **去重**：sha256 精确匹配（`signature` 字段，`lightweight_memory_provider.py:993` 附近）；字面完全相同才跳过，语义近似的不同表述会重复存入。

   **矛盾处理**：**无显式机制**。由于只存成功轨迹，相互矛盾的两条 insight（如"先规划再行动"vs"直接行动"）均可能入库并存。剪枝时靠 usage_count/success_count 自然淘汰低效条目。

   **遗忘/淘汰**：LLM 智能剪枝（超 50 条触发），优先保留"被使用且有效"的条目（usage_count × success_rate），不依赖时间衰减。外层进化每轮默认 `clear_storage_per_round=True`（`auto_evolver.py:68`），整库清空，为公平对比架构。

   **整合/抽象**：Cerebra 有 `_consolidate_memory` 但需累计 50 个成功任务触发（`config.py:69`），而默认每轮 20 任务且轮后清空存储，**实际等同于死代码**。Lightweight 无整合机制。

---

6. **其他设计**

   - **EvolveLab 统一框架**：任何记忆系统都继承 `BaseMemoryProvider`（`EvolveLab/base_memory.py:10`），实现 `provide_memory/take_in_memory/initialize` 三个方法；动态注册靠 `PROVIDER_MAPPING`（`memory_types.py:48-63`）枚举-类名映射，进化新系统插一行即可自动加载和评测。12 个已有系统按统一接口重实现，是记忆系统横向对比的现成平台。
   - **四阶段进化流水线**：Analyze（`AnalysisAgent` 最多 20 步调查，三工具：TrajectoryViewer/StepViewer/MemoryDatabaseViewer，按 PROVIDE/TAKE-IN/MANAGEMENT 三维度输出缺陷报告）→ Generate（单次 LLM 调用输出完整 Python 文件，max_tokens=60000，temperature 随"创造性预算"动态调整：`0.3 + 0.9 × creativity`，`phase_generator.py:254`）→ Create（正则注释标记行插桩到 `memory_types.py` 和 `config.py`，`memory_creator.py:111-275`）→ Validate（AST 静态检查 + 真实 import + 可选 mini-swe-agent 自动修复，最多 3 次，`phase_validator.py:197-214`）。
   - **锦标赛选择**：初赛 20 任务并行跑所有候选，top-t 进决赛再测 40 任务，按 accuracy 降序 + tokens 升序字典序排序（`auto_evolver.py:763-782`）选出亲本。
   - **usage_count/success_count 轻量效用反馈**：记忆被 LLM 选中即计 usage，任务判对后给该记忆加 success_count，剪枝时以"用了有效"为第一优先级，成本极低（只加计数）。
   - **诊断 prompt 的约束工程**：`analysis_prompt.yaml:96-130` 明确禁止"加关键词硬匹配"类可在小数据集提分但不可泛化的建议，约束进化搜索空间避免过拟合。

---

7. **核心创新点**

   **作者宣称的最核心贡献**：**把"记忆系统的 Python 代码"本身当作进化对象（meta-evolution）**——不优化单条记忆内容，而是让另一个 LLM 重新设计整套 Encode→Store→Retrieve→Manage 管线的代码，通过评测选优。

   与同类方案的本质区别：
   - vs ExpeL/Voyager/G-Memory：这些系统优化记忆内容（提炼 insight、规则），架构固定由人工设计；MemEvolve 优化的是架构本身
   - vs AutoML/NAS：MemEvolve 的搜索空间是自然语言 + Python 代码，由 LLM 直接生成候选，而非预定义的超参数网格
   - EvolveLab 作为平台贡献：13 个系统统一接口是独立价值，可直接复用于其他记忆系统研究的基准评测

---

8. **论文 vs 代码差异**

   - **最重要偏差：长期记忆默认"只写不读"**：论文 Figure 7 展示长期记忆 guidance 注入效果，但默认 `enable_longterm_provision: False`（`EvolveLab/config.py:52`；`lightweight_memory_provider.py:113`）。默认跑论文命令只有任务内短期记忆生效，**跨任务长期记忆完全不起作用**，严重影响论文数字的解读。
   - **"dual-evolution 联合演化"是分时而非共时**：论文概念图暗示记忆内容和架构同时进化，实际每轮进化开始清空记忆库（`auto_evolver.py:68`），两个循环串行。
   - **只从成功轨迹学习**：论文提示词模板写"from all trajectories"，但两个主力系统都只处理 `is_correct=True` 的轨迹（`lightweight_memory_provider.py:332-333`；`cerebra_fusion_memory_provider.py:1022-1023`），失败信息完全丢弃。
   - **Cerebra 整合机制是死代码**：`_consolidate_memory` 需累计 50 成功任务，默认每轮 20 任务且清空存储，从不触发（`config.py:69`）。
   - **phase_validator.py:32 的 docstring 自相矛盾**："without automatic fixes"，但默认 `enable_auto_fix=True`（:42），失败时确实调 mini-swe-agent 自动修代码。
   - **Pareto 多目标选择默认关闭**：README 提及，实际 `use_pareto_selection=False`（`auto_evolver.py:67`），默认字典序排序。

---

9. **实验**

   **数据集**：GAIA（通用任务问答）、WebWalkerQA（网页浏览）、xBench（专业领域深度调研）、TaskCraft（合成工具使用任务）。

   **基线**：ExpeL、DILU、Cheatsheet、无记忆基线，以及各数据集专用系统（如 Flash-Searcher、SmolAgent）。

   **核心结论**（论文 Table 3）：

   | 场景 | 基础系统 | 无记忆 | MemEvolve | 提升 |
   |------|---------|--------|-----------|------|
   | GAIA | Flash-Searcher + GPT-5-Mini | 69.09% | **73.33%** | +4.24% |
   | WebWalkerQA | SmolAgent | 基线 | **最高 +17.06%** | — |
   | 三数据集均值 | — | — | **+3.54%–5%** | 稳定提升 |

   - 跨任务迁移：TaskCraft 进化出的系统直接迁移到 WebWalkerQA 和 xBench，仍有 +2.4%–+9.09%，说明进化出的是通用架构原则而非数据集技巧
   - 跨模型迁移：用 GPT-5-Mini 进化出的系统直接套用到 Kimi K2 和 DeepSeek V3.2 也有提升
   - 进化轨迹分析（Figure 6）：系统自发发现三条原则——主动 LLM 参与检索 > 死板规则、多层级记忆组织 > 单一扁平、内嵌工具记忆在工具密集型任务上更有效
   - API 调用成本与无记忆基线相当，说明进化后的架构在效用和效率上均优
   - 进化轮次：3 轮；初赛每轮 20 任务，决赛 40 任务

   **值得注意的细节**：实验使用有 gold answer 的评测数据集，成功判定依赖 LLM judge，这意味着实验设置不完全反映无 ground truth 的真实部署场景；Pareto 多目标选择默认关闭，论文效果在字典序（accuracy 优先）下取得。

---

### LatentMem

**仓库**：KANABOON1/LatentMem，commit 7173f64  
**论文**：LatentMem: Customizing Latent Memory for Multi-Agent Systems，arXiv:2602.03036  
**基础模型**：Qwen3-4B-Instruct-2507

---

1. **记忆形式与粒度**

   一条记忆是**完整的成功任务执行轨迹**（`Trajectory` 对象），字段（`latentmem/utils/message.py:135-141`）：

   | 字段 | 类型 | 含义 |
   |------|------|------|
   | `task_init_description` | str | 任务描述（同时作为向量检索的索引键） |
   | `trajectory` | list[MessageGraph] | 多个交互步骤的时间序列 |
   | `label` | bool | True=成功，False=失败 |
   | `extra_fields` | dict | 扩展字段（检索后临时存储，不持久化） |

   每个 `MessageGraph` 步骤字段（`message.py:39-92`）：`state`（输入状态）、`action`（MAS 输出决策）、`observation`（环境反馈）、`mas_message_graph`（MAS 内部各 agent 间消息传递 DAG，每节点含 system_prompt + user_prompt + `response`）。

   **存入 Chroma 的具体形态**：`page_content`（仅 `task_init_description`，供 embedding 向量化检索）+ `metadata`（轨迹 JSON 全文 + label）。

   **粒度**：任务级，是本批次乃至所有已调研系统中粒度最粗的记忆单元。一条记忆覆盖整次多步骤多 agent 的完整交互。**写入时零 LLM 变换**，原始轨迹直接序列化。

   **与其他粒度方案的本质区别**：
   - vs Mem0/SimpleMem：完全不提炼事实句，不做代词消解，不做时间绝对化
   - vs G-Memory：同样存轨迹，但 G-Memory 额外用 LLM 提炼 key_steps；LatentMem 轨迹原样入库
   - 最根本区别：记忆的最终注入形式**不是文本**，而是通过 Composer 压缩成的 **8 个隐向量**（`[8, hidden_size]`），直接拼接到 agent 的输入 embedding 序列末尾，绕过 token 预算限制

---

2. **写入机制**

   **触发时机**：`bootstrap_data` 阶段（`runner.py:100-142`），MAS 跑完训练集任务后批量触发，非推断时在线写入。

   **LLM 参与次数**：**0 次**。只调用 embedding 模型一次，把 `task_init_description` 转向量入库。

   **提取过程**（`runner.py:129-131`，`metagpt.py:28-39`）：
   ```python
   if trajectory.label == True:
       self.memory_mas.centralized_memory.add_memory(trajectory)
   # 内部：轨迹 to_serializable() → JSON → Document → Chroma.add()
   # page_content = task_init_description（embedding）
   # metadata = {trajectory: JSON全文, label: true}
   ```

   **过滤/筛选**：仅通过 `label == True` 门控过滤；成功判定依赖 LLM judge 对比 gold answer（与 MemEvolve 相同）。**无语义去重，无摘要，无结构化提炼。**

   **代价**：每条记忆体积大（完整轨迹 JSON），检索后需由 Composer 压缩；好处是写入成本极低、信息无损失。

---

3. **检索机制**

   **默认路径（metagpt 模式，所有发布配置均使用）**：

   纯向量相似搜索（`metagpt.py:42-61`），`sentence-transformers/all-MiniLM-L6-v2` 向量化新任务描述 → Chroma `similarity_search_with_score` → 取 1 条最近邻 `label=True` 轨迹（`pos_shots_num=1`，`neg_shots_num=0`）。

   **检索端 LLM 参与次数：0 次**（metagpt 模式）。

   **召回后有无重排**：无（metagpt 模式）。

   **可选路径（ExperienceBank，rag_mode="latentmem"，默认不启用）**：
   - 先向量检索 2 倍候选量（`experience.py:106-146`）
   - LLM 对每条候选给出 0–10 相关性评分，重排后取 top-k（1 次 LLM/候选，max_new_tokens=20）
   - 角色感知历史提取（`experience.py:148-162`）：按当前 agent role 从轨迹 DAG 里抽取该角色的历史发言序列，传给 Composer

   **结果数量控制**：`pos_shots_num=1`（默认），fixed 取 1 条。

---

4. **注入 prompt**

   LatentMem 的记忆注入机制与所有其他系统根本不同——**记忆不以文本形式出现在 prompt 里**：

   - agent prompt 中 `{memory_content}` 占位符填入 `<|Memory_Empty|>` 空标记（`agent.py:351-353`），**不占任何实质 token**
   - 检索到的轨迹文本传给 **Composer**，Composer 经前向计算产出 8 个隐向量
   - 通过线性投影层调整维度后，**直接与 agent 的 text embeddings 在 embedding 层拼接**（`agent.py:248-259`）：
     ```python
     torch.cat([text_embeddings, latent_emb], dim=1)
     # text_embeddings: [T_text, hidden_size]
     # latent_emb:      [8,      hidden_size]
     # 拼接后:          [T_text+8, hidden_size]
     ```
   - 这个矩阵作为 `inputs_embeds` 送进 agent LLM 生成输出

   **额外 token 数**：恒为 **8**，无论原始轨迹长度是 100 token 还是 10000 token。

   **截断/优先级控制**：天然无需截断，8 个位置是硬性固定。

---

5. **记忆管理**

   **去重**：**无**。同一任务执行多次若均成功，可被重复写入多条轨迹。

   **矛盾处理**：**无**。只存成功轨迹，但同一问题的不同成功路径（如"先搜索 Wikipedia"vs"先搜索 Bing"）可并存。

   **遗忘/淘汰**：**完全无**。Chroma 只增不减，无 TTL、无容量上限、无 LRU。记忆库随训练集规模线性增长，无任何自动清理机制。

   **整合/抽象**：**无**。单条轨迹从不被合并、摘要或抽象成更高层规则。Composer 在推断时压缩，但压缩的输出（8 个隐向量）不持久化，不积累。

   **记忆管理的本质**：写成功则入库，检索最近邻，其余均不做。记忆质量的"管理"完全转移到 Composer 的训练上——通过 LMPO 让 Composer 自动学会从粗糙轨迹里提炼有用信号。

---

6. **其他设计**

   - **LMPO 训练（核心工程贡献）**：以任务成败为 RL 奖励（`trajectory.label → 0/1`），只训练 Composer（LoRA + 8 个 `query_latents` + 线性投影层），agent LLM 全程冻结（`autogen_main.py:47-48`）。梯度路径：latent 向量拼在 `inputs_embeds` → agent 复现自身历史回复的 per-token log probability → advantage 加权 → 反向传播回 Composer。两阶段训练：SFT 预热（直接监督）→ LMPO 策略优化（RL，`runner.py:144-160`，`grpo_trainer.py`）。
   - **8 个可学习 query_latents**（`composer.py:43-46`，`nn.Parameter(8, hidden_size)`）：压缩的"瓶颈"，初始随机，通过 LMPO 被端到端优化为对任务有效的记忆表示。
   - **Composer 结构**（`composer.py:56-91`）：与 agent 同规模的 Qwen3-4B + LoRA（r=16，q_proj/v_proj）+ 8 个 query_latents。输入：任务描述 + 轨迹文本 + 角色。输出：最后 8 个位置的隐藏状态（不生成任何文字）。
   - **统一基线框架**：同一 MAS + 同一冻结 LLM，通过切换 RAG 后端复刻 MetaGPT/Voyager/Generative/GMemory/OAgent 五种记忆方式，控制变量对比不同记忆机制。

---

7. **核心创新点**

   **作者宣称的最核心贡献**：**以隐向量替代文本记忆（latent memory），从根本上绕开 token 预算限制**，同时用 LMPO 端到端训练压缩器，使其自动学会"压缩什么才有助于完成任务"。

   三个子贡献：
   1. **零 token 开销记忆**：记忆恒占 8 个向量位置，不随轨迹长度变化，无截断损失
   2. **LMPO**：以 agent 任务成败为奖励信号，通过 latent 向量 → per-token logp 这条可微路径，把"记忆有没有帮助"这个不可微评估转化为可微优化
   3. **冻结 agent，只训练 Composer**：Composer 可与任意冻结 agent 组合，且 agent 不需要支持 fine-tuning

   **与同类方案的本质区别**：
   - vs Mem0/SimpleMem/A-MEM：这些方案都把记忆存成文本 token，受 context window 约束；LatentMem 的记忆在 token 层面不可见
   - vs G-Memory/MemEvolve：均存文本记忆或策略文字，均需 token 开销；LatentMem 是唯一把记忆压缩到向量空间的方案
   - 研究定位最接近"外部记忆 + 参数知识"的融合中间地带——记忆不在 prompt 里，也不在 agent 参数里，而在 Composer 学到的压缩表示里

---

8. **论文 vs 代码差异**

   - **ExperienceBank 被误作主路径描述**：论文宣传的 role-aware 检索和 LLM 重排属于 ExperienceBank（`rag.mode: latentmem`），但所有发布的四个数据集 yaml 均配置 `mode: metagpt`（纯向量搜索，无 LLM 重排，无角色感知）。**默认推断管线里 role-aware 完全不起作用**，影响对"多 agent 专属设计"这一卖点的判断。
   - **训练脚本入口命名不一致**：README 写 `bash scripts/lmpo_train.sh`，脚本目录实际文件为 `lmpo.sh`（轻微问题）。
   - **lmpo.sh 无法直接复现**：脚本传 `run.mode grpo`，但 `runner.py` 只接受 `data/sft/lmpo/eval`（`runner.py:285-294`）；`train()` 注解写 `Literal["sft", "grpo"]` 但分支判 `"lmpo"`（`runner.py:186, 204`）——GRPO 改名 LMPO 时代码未全量同步，**发布的训练脚本直接跑会 ValueError**。
   - **yaml 的 lmpo 超参块是死配置**：`popqa.yaml:87-117` 的 `run.lmpo` 配置从未被读取，runner 实际读 `run_cfg.grpo`（`runner.py:76-77`）。真正有效的超参必须通过命令行 `run.grpo.*` 传入。
   - **LMPO 实际是 vanilla 策略梯度**：`beta: 0.0`（KL 正则不起作用），`old_per_token_logps = per_token_logps.detach()`（`grpo_trainer.py:281`）使 importance sampling ratio 恒为 1、clip 操作失效。论文若宣称 PPO 式 clip，代码实际只是最基础的策略梯度。
   - **`abstract` 分支是死代码**：`_construct_text_memory` 里的 `pos_shots[0].extra_fields.get("abstract")` 判断（`latentmem.py:123`）全仓库无任何地方写入 `abstract` 字段，此分支永不触发。
   - **负例记忆主管线下无效**：配置 `neg_shots_num=0`，runner 只写成功轨迹，库里无 `label=False` 条目，`NEG_SHOTS_TEMPLATE` 永不触发。
   - **Composer prompt 的"8 token"指令是心理暗示**：`EXTRACT_LATENT_PROMPT` 写"Using no more than 8 tokens"，但 Composer 根本不生成文字，只取 query_latents 位置的 hidden state；该指令只是引导模型关注压缩，与实际计算过程无对应。

---

9. **实验**

   **数据集**：PopQA、TriviaQA、StrategyQA（知识问答）；KodCode（编码任务）。

   **基础模型**：Qwen3-4B-Instruct-2507。

   **基线**：无记忆（MetaGPT 基线）、文本摘要式记忆（Voyager）、LLM 生成记忆（Generative）、规则演化记忆（GMemory、OAgent）。

   **核心结论**：论文称 LatentMem 在上述数据集上准确率超过所有文本记忆基线，同时 prompt 中记忆部分占用 token 接近零。**具体准确率数字笔记未记录**（论文 Table 结果未在笔记中逐一抄录）。

   **实验设置关键细节**：
   - 所有基线在完全相同的 MAS 框架（AutoGen + 同一冻结 Qwen3-4B）上运行，仅切换 RAG 后端，变量控制严格
   - 训练时需要 gold answer（bootstrap_data 阶段 LLM judge），因此实验是 closed-loop 评测，不代表无标注数据的真实部署场景
   - 推断时建议直接使用 HuggingFace 发布的预训练 Composer 和轨迹库（`eval_hf.sh`），可信度高于自行复现训练（训练脚本有上述 bug）
   - 消融方向：论文比较了是否有 Composer、是否有 LMPO 训练、是否有 role-aware（ExperienceBank 模式）的对比，但代码默认配置只跑最基础的 metagpt+Composer 路径

---

## 批次四横向对比（MemEvolve / LatentMem）

| 维度 | MemEvolve（Lightweight） | LatentMem |
|------|--------------------------|-----------|
| **记忆单元** | 经验/策略条目（strategic/operational insight，自然语言）+ 短期任务内事实清单 | 完整成功任务轨迹（原始 JSON，含 MAS 内部消息 DAG） |
| **记忆粒度** | 中等（insight 是单句规律，一条任务提炼最多 2+2 条）| 最粗（任务级，一条记忆=一次完整多步骤 MAS 执行） |
| **写入时 LLM 次数** | 每步 1 次（短期抽取）+ 任务结束 1 次（长期提炼）+ 剪枝时可选 1 次 | **0 次**（只调 embedding 模型） |
| **写入是否即时** | 非即时，任务完成后触发，仅成功轨迹 | 非即时，bootstrap 阶段批量写入，仅成功轨迹 |
| **写入是否结构化** | 是（strategic/operational 两类分开，sha256 精确去重） | 否（原始轨迹序列化，无任何变换） |
| **检索方式** | BEGIN：LLM 从全库选最相关合成 guidance（默认 **关闭**）；IN：短期记忆每 5 步注入 | 纯向量相似搜索（Chroma，默认 1 条最近邻），无重排 |
| **检索端 LLM** | BEGIN 1 次（仅开启时）；IN 每步 1 次（短期抽取） | **0 次**（metagpt 模式） |
| **注入 prompt 的位置** | BEGIN：user message；IN：执行上下文 | **不注入文本**，以 8 个隐向量拼在 inputs_embeds 末尾 |
| **记忆 token 开销** | 数十至数百 token（guidance + 短期清单） | **恒为 8 个向量位置，对 token 计数为零** |
| **去重** | sha256 精确匹配（字面完全相同才跳过） | 无 |
| **矛盾处理** | 无显式检测；靠 usage_count/success_count 剪枝间接淘汰低效条目 | 无 |
| **遗忘/淘汰** | LLM 智能剪枝（>50 条触发，效用优先）；外层进化每轮清空整库 | **无任何淘汰机制**，Chroma 只增不减 |
| **整合/抽象** | 无（Cerebra 有 consolidate 但实际是死代码） | 无（Composer 在推断时压缩但不持久化） |
| **训练需求** | 无需训练（prompt engineering + eval loop） | 需要训练 Composer（SFT + LMPO RL，Qwen3-4B LoRA） |
| **多 agent 支持** | 单 agent（评测用 Flash-Searcher/SmolAgent） | 专为 MAS 设计，中心记忆单实例供所有 agent 共用 |
| **持久化** | JSON 文件（Lightweight）/ JSON + networkx（Cerebra） | Chroma 向量库 + HuggingFace 发布的预训练 Composer 权重 |
| **论文 vs 代码最大落差** | 长期记忆默认只写不读（enable_longterm_provision=False）；"dual-evolution 共时"实为串行 | ExperienceBank/role-aware 是主路径描述但默认不启用；lmpo.sh 训练脚本无法直接复现（GRPO→LMPO 改名未同步） |
| **核心研究问题** | 如何让记忆架构本身自适应演化（meta-learning on memory system code） | 如何绕过 token 限制把记忆压缩为向量（latent-space memory for MAS） |

**批次四共同特征（从任务轨迹中学习策略，而非对话事实）**：
- **记忆单元是经验而非事实**：Mem0/SimpleMem 存"用户住北京"这类对话事实；MemEvolve 存"当任务无进展时尝试第三方源"这类可迁移策略；LatentMem 存整条成功执行轨迹供 Composer 提炼。记忆的语义从"用户信息"转向"任务经验"。
- **写入有成功门控**：两个系统都只写成功轨迹（需要 gold answer 判定），这与 Mem0/SimpleMem 的"写入任何对话"根本不同，也意味着两者在无标注数据的开放环境中的写入行为尚未被验证。
- **检索端 LLM 的分歧**：MemEvolve 重度使用 LLM（每步抽取 + BEGIN 综合），LatentMem 默认不使用 LLM（纯向量），两者代表了"检索侧 LLM 成本"的两个极端。
- **记忆注入的创新方向**：MemEvolve 代表"让记忆系统架构可进化"；LatentMem 代表"让记忆突破文本 token 的形式限制"——两者都在质疑"记忆必须是 prompt 里的一段文字"这个隐含假设，但路径完全不同。
- **共同局限**：去重和矛盾处理均缺失；遗忘机制均简陋（MemEvolve 靠剪枝，LatentMem 无机制）；成功判定依赖 gold answer，限制了在无监督场景的适用性；两者的论文与代码均有重要出入（长期记忆默认关闭、训练脚本 bug），直接复现需仔细核查。

---

### Light-Mem

**仓库**：zjunlp/LightMem，commit 579ee76  
**论文**：LightMem: Lightweight and Efficient Memory-Augmented Generation，ICLR 2026，arXiv:2510.18866

---

1. **记忆形式与粒度**

   一条记忆是 `MemoryEntry` 结构化记录（`memory/utils.py:13-32`）：

   | 字段 | 含义 | 实际状态 |
   |------|------|---------|
   | `id` | UUID | 生效 |
   | `time_stamp` | ISO 格式时间戳（如 "2023-05-20T00:44:00.000"）| 生效 |
   | `float_time_stamp` | 时间戳浮点版本，方便数值比较 | 生效 |
   | `weekday` | 星期几（如 "Sat"）| 生效 |
   | `memory` | 核心事实句（"User is planning a trip to Tokyo next month."）| 生效 |
   | `topic_id` | 话题段编号（写入时归入哪个话题） | 生效 |
   | `speaker_id` | 说话人 ID | 生效 |
   | `update_queue` | 睡眠期维护用：指向可能覆盖本条的更新者 | 生效 |
   | `consolidated` | 是否已被睡眠期整合 | 生效 |
   | `original_memory`、`compressed_memory`、`category/subcategory/memory_class/bam_tags` | 定义了但实际为空 | 空字段 |

   **粒度**：事实句级。一条记忆是改写成第三人称的独立事实（"User started a new job at Google last week as a software engineer."），不是原始消息、摘要段落或 QA 对。

   **与其他粒度方案的本质区别**：（1）写入前经 LLMLingua-2 预压缩（本地 BERT 模型，非 API），是 13 个方案中唯一在 LLM 处理之前做文本降噪的；（2）有 `topic_id` 追踪话题归属，话题切分复用压缩模型的注意力矩阵（"一模两用，注意力免费"），不额外调用 API；（3）写入端在线追加、维护推到离线睡眠期，记忆管理与在线响应完全解耦。

---

2. **写入机制**

   **触发时机**：**非即时**，两级缓冲批处理。感觉缓冲（`SenMemBufferManager`）积累到 512 token 上限时触发话题切分；短期缓冲（`ShortMemBufferManager`）积累的 topic 段超过阈值 `th`（默认 512 token）时触发一次 LLM 调用（`short_term_memory.py:36-57`）。

   **LLM 参与次数**：每次触发 **1 次 LLM 调用**（`factory/memory_manager/openai.py:143-206`，flat 模式），一次处理多个 topic 段，摊到每条对话远低于 1 次。本地 embedding 模型（all-MiniLM-L6-v2）不走 API。

   **提取过程（五阶段）**：
   - 阶段 1：**时间戳归一化**（`lightmem.py:276`），规范化为 ISO 格式，同 session 内消息按 500ms 递增生成独立时间戳
   - 阶段 2：**LLMLingua-2 预压缩**（`lightmem.py:278-298`），本地 BERT 分类器对每个 token 打分，按保留率保留最重要词（如"I'm planning a trip to Tokyo next month"→"planning trip Tokyo next month"）
   - 阶段 3：**话题切分**（`sensory_memory.py:43-113`）：LLMLingua-2 第 8-11 层注意力矩阵检测粗边界 + all-MiniLM-L6-v2 余弦相似度检测细边界，位置差 ≤3 轮取交集，得 topic 段
   - 阶段 4：**LLM 批量抽取**（`prompts.py:1-52`），把所有 topic 段拼成一个 prompt，LLM 返回 `{source_id, fact}` JSON 数组（只从 user 消息抽取，默认 `messages_use="user_only"`）
   - 阶段 5：**入库**（`lightmem.py:363-443`），每条 fact 转成 MemoryEntry，embed 后插入 Qdrant，纯追加不做任何在线去重

   **过滤/筛选**：无显式过滤。LLM 可跳过"Hi"/"lol"/"thanks"类无信息量内容，但无相似度阈值或去重检查。

---

3. **检索机制**

   入口 `LightMemory.retrieve(query, limit=10)`（`lightmem.py:644-707`），全程 **0 次 LLM 调用**：
   - query 文本转 all-MiniLM-L6-v2 向量
   - Qdrant 余弦相似度 top-k 检索
   - 每条记忆格式化为 `"时间戳 星期 记忆文本"` 字符串

   **召回后有无重排**：无。

   **检索端 LLM 参与次数**：**0 次**。

   **结果数量控制**：`limit` 参数（默认 10）。

   **BM25 检索**：`factory/retriever/contextretriever/bm25.py` 是**空文件**，`examples/run_lightmem_bm25.py` 也是空文件，不可用。

---

4. **注入 prompt**

   检索结果换行拼接（`experiments/longmemeval/run_lightmem_qwen.py:198-205`）：

   ```
   Please answer the question based on the following memories:
   2023-05-20T00:44:00.000 Sat User is planning a trip to Tokyo next month.
   2023-05-20T00:44:00.000 Sat User plans to visit Shibuya and Shinjuku in Tokyo.
   2023-05-20T00:44:01.000 Sat User has a budget of around 3000 dollars for the Tokyo trip.
   Question: What are the user's travel plans?
   ```

   **插入位置**：由调用方决定，通常为 QA prompt 的记忆上下文段，不固定位置。

   **截断/优先级控制**：无；由 `limit` 参数决定数量，无动态优先级。

---

5. **记忆管理**

   **去重**：**无**。写入纯追加，不做任何在线相似度检查，语义重复的事实可重复存入。

   **矛盾处理**：推到睡眠期离线处理（`lightmem.py:539-642`）。`UPDATE_PROMPT`（`prompts.py:334-406`）三规则：新旧信息矛盾→删旧（`delete`）；新信息补充旧信息→合并改写旧记忆的 `memory` 字段（`update`）；无关→忽略（`ignore`）。**关键 bug**：`update` 操作只改 payload 里的文本，Qdrant 里的向量不重算（`lightmem.py:617-621`），导致文本与检索索引失配。

   **遗忘/淘汰**：仅通过睡眠期的 `delete` 操作实现（矛盾时删旧）。`hit_time` 字段定义了访问次数统计（`utils.py:29`），但系统里无任何地方在检索后递增它，**无基于访问频率的遗忘**。

   **整合/抽象**：睡眠期 `update` 操作可合并补充性信息（改写旧记忆文本），但向量索引不同步（同上 bug）。

---

6. **其他设计**

   - **"一模两用，注意力免费"**：压缩和话题切分共用同一个 LLMLingua-2 BERT 模型（`lightmem.py:169`），注意力矩阵是压缩的副产物，拿来做粗边界检测零额外推理成本。
   - **写读维护三路径完全解耦**：在线写（Qdrant 追加）、在线读（向量检索）、离线维护（睡眠期 LLM 仲裁）完全独立，互不阻塞。
   - **内建分项计费器**（`lightmem.py:143-160`）：把 summary/update/embedding 的 token 和调用次数分开统计，效率数字由框架本身产出，可直接用于实验报告。
   - **睡眠期并行处理**（`lightmem.py:626-627`）：每条记忆的更新独立互不依赖，线程池并行处理，控制批量更新耗时。

---

7. **核心创新点**

   **作者宣称的最核心贡献**：**三阶段类人记忆架构**（对应 Atkinson-Shiffrin 感觉/短期/长期记忆模型）——预压缩降噪（感觉缓冲）→ 攒批触发抽取（短期缓冲）→ 睡眠期离线整合（长期维护），把在线 LLM 调用成本降到极低。

   **与同类方案的本质区别**：
   - vs Mem0：Mem0 每次 add 即时调一次 LLM；LightMem 用本地 BERT 预处理 + 缓冲攒批，摊到每条消息的 LLM 成本趋近于零
   - vs SimpleMem：都用缓冲批量触发，但 LightMem 额外用 LLMLingua-2 做写入前降噪，且话题切分不需要额外 LLM 调用
   - vs A-MEM：A-MEM 每条消息最多 2 次 LLM，LightMem 摊后远低于 1 次，且在线追加零仲裁延迟
   - 核心设计哲学：矛盾处理推迟到离线，在线路径保持最低成本

---

8. **论文 vs 代码差异**

   - **`update="online"` 静默失效**：`online_update()` 直接 `return None`（`lightmem.py:394-395`），设置该参数后记忆根本不入库，也不报错，用户无感知丢数据。
   - **`graph_mem=True` 必然报错**：`memory/graph.py` 全文只有一行 `class GraphMem:` 没有类体，README 仍宣称"支持图记忆"（`README.md:521`）。
   - **KV cache 是摆设**：配置文件有 `kv_cache` 字段（`configs/base.py:98-105`），全库无任何地方读取或使用。
   - **BM25 检索是空壳**：`factory/retriever/contextretriever/bm25.py` 是空文件，对应 `examples/run_lightmem_bm25.py` 也是空文件。
   - **`text_summary=False` 会崩溃**：触发 NameError（`lightmem.py:344-363`）。
   - **`topic_segment=False` 静默丢数据**：函数提前 return，不存任何记忆（`lightmem.py:300-309`，代码自带 TODO 注释）。
   - **离线更新改文本后不重算向量**：action=="update" 时只改 Qdrant payload 的 `memory` 字段，向量仍是旧的（`lightmem.py:617-621`），越更新检索越不准。
   - **`hit_time` 字段从不更新**（`utils.py:29`）：定义了访问次数统计，但无任何地方在检索后递增，无法支撑基于访问频率的遗忘机制。

---

9. **实验**

   **数据集**：LongMemEval（500 道题，模拟 ~115k token 超长对话）；LoCoMo（现实长对话评测）。

   **基线**：A-MEM、Mem0、MemoryBank、MemGPT 等六个方案。

   **核心结论**：

   | 基准 | 指标 | 结论 |
   |------|------|------|
   | LongMemEval | 准确率 | 比最强基线（A-MEM）高 **2%-7.7%** |
   | LongMemEval | 总 token（含离线更新） | 少 **10×-38×** |
   | LongMemEval | API 调用次数（总） | 少 **3.6×-30×** |
   | LongMemEval | 在线成本（仅写入期） | token 省 **105×**，API 少 **159×** |
   | LoCoMo | 准确率 | 高 **6%-29%** |
   | LoCoMo | 总 token | 少 **3×-21×**，API 少 **13×-55×** |

   **实验设置值得注意**：效率数字由内建计费器（`lightmem.py:143-160`）直接产出，可信度高。论文区分"在线成本"和"总成本"两个口径，总成本含离线更新；与其他系统的效率对比基于总成本口径。话题切分准确率约 80%（论文 §5.5），不如 LLM 切分稳定，但无额外 API 成本。

---

### MemoryBank

**仓库**：zhongwanjun/MemoryBank-SiliconFriend，commit cf61c41  
**论文**：MemoryBank: Enhancing Large Language Models with Long-Term Memory，arXiv:2305.10250

---

1. **记忆形式与粒度**

   MemoryBank 无"记忆类"抽象对象，**全部状态是一个 JSON 文件**，分三层：

   **原始对话（最细粒度）**：每轮 `{query, response}` 对，纯原始文本，无 embedding，按日期组织。写入时 0 次 LLM 调用，直接 append（`utils/memory_utils.py:72-87`）。

   **每日摘要层（中粒度）**：LLM 生成的两类摘要（`memory_bank/summarize_memory.py:109-147`）：
   - `summary.{date}.content`：当日事件摘要（"Gary 分享了压力缓解方法，还讨论了电影推荐"）
   - `personality.{date}.content`：当日性格分析（"Gary 看起来承受工作压力，处事果断直接"）

   **全局画像层（最粗粒度）**：
   - `overall_personality`：全局性格描述（每次摘要时全量重生成）
   - `overall_history`：全局事件摘要（生成但从不进 prompt，见第 8 条）

   **FAISS 中的检索单元**：每轮对话一个 Document + 每日摘要一个 Document，带 `{date, type}` metadata（`local_doc_qa.py:25-61`）。对话和摘要混在同一个 FAISS 索引里。

   **粒度**：对话轮级（最细）到日级摘要到全局画像（最粗），三层叠加。

   **与其他粒度方案的本质区别**：（1）不提炼事实句，保留原始 QA 对全文（Mem0/SimpleMem 要求代词消解和时间绝对化，MemoryBank 无此变换）；（2）分层记忆（对话/摘要/画像）是 13 个方案中最早的分层设计，后来 MemoryOS/MemGPT/MemOS 均有类似分层；（3）全局画像（`overall_personality`）同时作为"背景知识"和"回复策略指令"进 prompt，是伴侣类应用的独特设计，后来方案无此专门设计。

---

2. **写入机制**

   **触发时机**：
   - 原始对话：每轮对话后**立即自动追加**到 JSON（`utils/memory_utils.py:72-87`），此时 FAISS 索引不更新
   - 摘要（每日事件 + 性格分析 + 全局）：**手动触发**（app 界面按钮 `app_demo.py:258,379` 或 CLI 登录确认 `cli_demo.py:179-180`），非自动
   - FAISS 索引：**每次登录全量重建**（`memory_utils.py:13-41`，`shutil.rmtree` 删旧索引再从零建）

   **LLM 参与次数**：
   - 对话写入：**0 次**
   - 摘要触发时：**2 × 新日期数 + 2**（每个新日期调 2 次 gpt-3.5-turbo 生成当日事件和性格；全局摘要每次全重生成 2 次）；已摘要日期跳过（增量处理）

   **提取过程**：对话 → JSON 追加（无 LLM）→ 摘要（手动触发，LLM 批量生成）→ 登录时读 JSON 全量建 FAISS 索引 → 检索时 top-k 检索

   **过滤/筛选**：无。

---

3. **检索机制**

   入口 `search_memory`（`local_doc_qa.py:263-288`），**0 次 LLM 调用**：
   - 用用户输入原文（无任何改写）计算 HuggingFace 嵌入向量（英文 MiniLM-L6，中文 text2vec，`configs/model_config.py:20-21`）
   - FAISS top-k 检索（k=2，`sys_args.py:5`）
   - **同日扩展补丁**（`local_doc_qa.py:135-178`）：命中某条后沿时间序翻同日相邻对话，拼接到 200 字符为止；这是 monkey-patch（直接替换 langchain FAISS 对象的方法），论文未提及

   **召回后有无重排**：无（同日扩展是扩充而非重排）。

   **检索端 LLM 参与次数**：**0 次**。

   **结果数量控制**：top-k=2（默认），实际通过同日扩展可拼入更多内容，但上限 200 字符。

---

4. **注入 prompt**

   检索结果和 `overall_personality` 填入 meta_prompt 模板（`prompt_utils.py:14-23`）：

   ```
   你将扮演 Gary 的 AI 伴侣。
   你想起的最相关[回忆]是：
   "[User]: I've been feeling stressed... [AI]: There are many ways: exercise, music...
   记忆日期：2023-05-03"
   Gary 的性格及你的回复策略：decisive and straightforward, open to practical advice...
   以下是多轮对话：
   [User]: 你之前给我推荐的缓解压力的方法是什么？
   ```

   **插入位置**：prompt 开头（system 级位置）。

   **截断/优先级控制**：无；top-2 + 200 字符同日扩展共同决定数量上限。

---

5. **记忆管理**

   **去重**：**无**。原始对话直接 append，同一事实可重复记录。

   **矛盾处理**：**无**。矛盾事实并存；全局摘要每次全量重生成可能隐式消解（LLM 被指令"整合"时会自然合并），但无显式冲突检测。

   **遗忘/淘汰**：艾宾浩斯遗忘曲线（`R = e^(−t/S)`）：
   - **默认关闭**（`sys_args.py:10` 的 `enable_forget_mechanism` 默认 False）
   - 启动脚本明确传 False（`launch_belle_cmd.sh:9`）
   - CLI 分支 ImportError（`cli_demo.py:53` import 的 `forget_memory_new` 文件在仓库里不存在）
   - **公式实现有 bug**（`forget_memory.py:36`）：`math.exp(-t / 5*S)` 在 Python 优先级下等于 `exp(-t·S/5)`，S 在分子——S 越大忘得越快，与论文逻辑完全相反

   被检索命中的记忆可强化：`memory_strength += 1`，`last_recall_date` 更新，立即落盘（`forget_memory.py:63-71`），仅在遗忘模式开启时生效。

   **整合/抽象**：每日摘要 → 全局摘要（有实际 LLM 调用）；已摘要日期跳过的增量策略有效控制成本。

---

6. **其他设计**

   - **登录全量重建 FAISS**（`memory_utils.py:27`，`shutil.rmtree`）：每次登录不管有无新数据都重建，历史越长登录越慢；但嵌入用本地 HuggingFace 模型，不花 API 费用。
   - **同日扩展补丁**：命中某条 chunk 后翻同日相邻 chunk 拼到 200 字符，用启发式方法代替真正的语义段落边界（`local_doc_qa.py:135-178`）。
   - **记忆强化元数据**（遗忘模式下）：`memory_strength`（检索命中次数）和 `last_recall_date` 两个字段实现"use it or lose it"闭环——概念上是 13 个方案中最早的"使用频率影响遗忘"设计，后续 G-Memory 的 insight 信用分机制与此思路一脉相承。
   - **事实记忆与用户画像分轨**：`summary`（发生了什么）和 `personality`（用户是什么人）分开生成、分开进 prompt，是伴侣类场景区别于纯知识检索的关键设计。

---

7. **核心创新点**

   **作者宣称的最核心贡献**：**外挂长期记忆 + 艾宾浩斯遗忘曲线** ——给 LLM 外挂记忆模块，用遗忘曲线模拟人类记忆强化/衰减，构建伴侣类 AI（SiliconFriend）。

   **与同类方案的本质区别**：（1）2023 年早期方案，无事实提炼，存原始 QA 对；（2）引入遗忘概念是 13 个方案中最早的（虽然实现有 bug 且默认关闭）；（3）全局画像（`overall_personality`）双用途设计——既是背景知识也是回复策略指令，后来方案大多无此设计；（4）架构极简：全部状态是一个 JSON 文件，无额外依赖，方便复现。

---

8. **论文 vs 代码差异**

   - **艾宾浩斯遗忘曲线是摆设**：默认关闭（`sys_args.py:10`）；启动脚本明确传 False（`launch_belle_cmd.sh:9`）；CLI 分支 ImportError（`cli_demo.py:53`）；公式实现括号缺失 bug（`forget_memory.py:36`），S 在分子、遗忘方向与论文逻辑相反。**实验数字在遗忘机制关闭下取得**，论文宣称的"遗忘与巩固闭环"在代码里基本是展示性存在。
   - **全局事件摘要进 prompt 是摆设**：`overall_history` 被生成（`summarize_memory.py:141`）并包装（`prompt_utils.py:117`），但所有 meta_prompt 模板无 `{history_summary}` 占位符（`prompt_utils.py:14-23`），Python `str.format()` 静默忽略多余参数，内容从未被任何 LLM 看到。只有 `overall_personality`（全局性格）真正进 prompt。
   - **"持续自动演化"是手动触发**：README 写"continually evolve through memory updates"，实际摘要和性格分析需用户主动点按钮；新对话在下次登录重建 FAISS 前无法被检索。

---

9. **实验**

   **数据集**：自建（15 个虚拟用户，每人积累 10 天对话记忆，194 个记忆探针问题；评估维度：记忆检索准确率/回答正确性/上下文连贯性）。另有定性分析（展示 SiliconFriend 实际对话样例）。

   **基线**：无同类方案可比（2023 年早期工作，当时无其他长期记忆系统）；三个底座模型（ChatGPT / ChatGLM / BELLE）相互对比。

   **核心结论（Table 2）**：

   | 底座模型 | 检索准确率（英文） | 回答正确性（英文） | 连贯性（英文） |
   |---------|--------------|--------------|-----------|
   | ChatGPT | 0.76 | **0.716** | **0.912** |
   | ChatGLM | 0.86 | 0.628 | 0.710 |
   | BELLE | 0.80 | 0.568 | 0.700 |

   三个版本在检索准确率上相近（0.76-0.86），说明 MemoryBank 的检索机制对不同底座模型均有效；回答质量差异主要来自底座模型能力而非记忆机制。

   **实验设置值得注意**：无真正的 ablation study（遗忘机制关闭，无与其他方案的定量对比）；评测为自建题，可能被优化针对；2023 年工作，LoCoMo/LongMemEval 等标准 benchmark 还未发布，实验结果无法与后续方案横向比较。

---

## 二、横向对比

> 覆盖全部 13 个项目：Mem0、SimpleMem、A-MEM、MemoryOS、MemGPT、MemOS、Zep、G-Memory、MIRIX、MemEvolve、LatentMem、LightMem、MemoryBank

### 记忆粒度

13 个方案可归为 5 类，区别在于"一条记忆对应什么自然单元"：

**1. 关系三元组级（Zep）**
粒度最细。一条记忆是实体间的关系断言（"Alice works at Acme Corp as a senior engineer"），带四时间戳双时序模型。一条对话可产生多条事实边，显式图结构（主语实体 - 关系 - 宾语实体）。

**2. 事实句级（Mem0、SimpleMem、MemOS、LightMem）**
从对话中 LLM 提炼的自包含事实陈述句，核心变换是代词消解（"我"→"Alice"）和时间绝对化（"上周"→"2025-06-08"）。SimpleMem 和 MemOS 还附加显式结构化元数据字段（timestamp、persons、tags 等），LightMem 加入 topic_id 追踪话题归属。

**3. 对话轮/QA 对级（MemoryBank、MemoryOS-STM、A-MEM）**
保留原始对话文本，不做事实提炼。MemoryBank 存 `{query, response}` 对；MemoryOS 的 STM 存 `{user_input, agent_response, timestamp}`；A-MEM 存原始对话文本但附 LLM 生成的 keywords/context/tags 增强 embedding。代词和相对时间均未消解，粒度较粗，但保留了上下文语境。

**4. 任务级轨迹（G-Memory、LatentMem）**
一条记忆覆盖整次多步骤多 agent 任务。G-Memory 额外用 LLM 提炼 key_steps（`GMemory.py:244-281`）；LatentMem 轨迹原样序列化入库，在推断时由 Composer 压缩成 8 个隐向量注入 agent——是 13 个方案中唯一把记忆压缩到向量空间的。

**5. 策略/经验级（MemEvolve）**
从成功轨迹提炼可跨任务迁移的规律（"当任务无进展时尝试第三方信息源"），记忆的语义从"用户信息"转向"任务经验"，不记录任务特定数据。

**混合分层方案**：MemGPT（Core 粗粒自由文本 + Archival 由 LLM 决定粒度 + Recall 原始消息）、MemoryOS（STM 对话轮 → MTM 对话链 → LPM 事实句+90 维画像）、MemOS（统一事实句但按 WorkingMemory/LongTermMemory/UserMemory 分桶容量）、MIRIX（六类独立粒度，同一输入分拆到多类）。

---

### 写入成本（每轮 LLM 次数）

| 档次 | 方案 | 每条消息/任务的 LLM 次数 |
|------|------|----------------------|
| **0 次** | LatentMem | 0（仅 embedding） |
| | MemoryBank（对话写入） | 0（摘要手动触发，另算） |
| **<1 次（攒批摊薄）** | SimpleMem | ~1/38（每 40 条触发 1 次，`config_default.py:60`） |
| | LightMem | 远低于 1（缓冲满才触发，内建计费器可量化，`lightmem.py:143-160`） |
| **≈1 次** | Mem0 | 1（每次 add，`main.py:751`） |
| | MemOS fine 模式 | 1/窗口（每 1024 token 滑动窗口 1 次） |
| | G-Memory | 1-2/任务（提炼 key_steps ± fail_reason） |
| **2-3 次** | A-MEM | 最多 2（analyze_content + process_memory） |
| | MemoryOS（STM 已满） | 3/条（连续性 + 链摘要 + 多主题，`utils.py:359-386`） |
| | MemoryBank（摘要触发） | 2×新日期数 + 2（全局摘要全重生成） |
| **3-9 次** | MIRIX | 3-9（topic 抽取 + meta 决策 + 各子代理各 1 次） |
| | Zep | 2-5+（抽实体 + 抽边 + 矛盾判断，边越多越多） |
| **多步不确定** | MemGPT（Sleeptime，默认关闭） | ≥1 步（默认路径 0 次额外） |
| | MemEvolve（AnalysisAgent） | 最多 20 步（仅进化阶段） |

**极端对比**：SimpleMem 实验显示 A-MEM 构建速度 5140.5 s/样本，SimpleMem 92.6 s，差距 55 倍——根本原因是 A-MEM 每条消息最多 2 次 LLM，SimpleMem 每 40 条触发 1 次，批量摊薄效果显著。

---

### 检索方式

| 类型 | 代表方案 | 说明 |
|------|---------|------|
| **纯向量** | MemoryBank（FAISS）、LatentMem（Chroma，注入为隐向量） | 最简单，无 LLM 介入 |
| **声称混合实为纯向量** | A-MEM（BM25 import 但从未调用，`memory_system.py:9`） | 工程实现与接口描述不符 |
| **向量 + BM25/全文混合** | Mem0（向量+BM25+实体boost，`scoring.py:60-139`）、Zep（向量+BM25+BFS，RRF 融合）、SimpleMem（向量+BM25+SQL 符号层三路，`hybrid_retriever.py`）、G-Memory（Chroma 向量 + 1-hop 图扩展） | Mem0 和 Zep 在专有名词/精确匹配上显著优于纯向量 |
| **精确元数据匹配 + 向量** | MemOS（图节点 key 精确匹配 + tags 匹配 + 向量并行）、MIRIX（默认硬编码 PostgreSQL 全文检索，`agent.py:1754`） | 高精度已知字段检索 |
| **LLM 介入检索** | G-Memory（LLM 对成功轨迹候选打 1-10 分精排，`GMemory.py:220-228`）、MemEvolve（BEGIN 时 LLM 从全库合成 guidance，默认关闭）、SimpleMem（规划 + 反思，2-4 次 LLM） | 检索质量高但成本显著上升 |
| **分层并行** | MemoryOS（STM 全量 + MTM 两级 FAISS + LPM 向量，三路并行 0 次 LLM，`retriever.py:110`） | 多层同时检索，结果拼接注入 |
| **LLM 自主调工具** | MemGPT（LLM 决定何时调 `archival_memory_search`，`agent_manager.py:2416`） | 检索时机完全由 LLM 判断，灵活但不可控 |

---

### 记忆管理是否真实生效

**去重**：
- **真实生效**：Mem0（MD5 精确，`main.py:825-829`）、Zep（实体三级分层+LLM 判重，`dedup_helpers.py:220-279`）、MemEvolve（sha256 精确，`lightweight_memory_provider.py`）
- **摆设/无**：SimpleMem（无）、A-MEM（无）、MemoryOS（同 page 多主题重复插入）、MemoryBank（无）、LightMem（无）、LatentMem（无）

**矛盾处理**：
- **真实生效**：**仅 Zep**（LLM 语义判断 + 确定性时序裁决 + 失效时间戳，`edge_operations.py:538-573`）
- **有 bug / 推迟但未完成**：LightMem（改文本不重算向量，`lightmem.py:617-621`）
- **默认关闭**：MemOS（`reorganize=False`，`handler.py:30`）、MemGPT（靠 LLM 自觉）
- **完全无**：Mem0、SimpleMem、A-MEM、MemoryOS、G-Memory、MIRIX、MemEvolve、LatentMem、MemoryBank

**遗忘/淘汰**：
- **真实生效**：MemoryOS（STM FIFO + MTM LFU + LPM deque，三层均有实际容量管控）、MemOS（WorkingMemory FIFO 裁 20 条，同步路径）、G-Memory（insight 信用分 score≤0 删除，`GMemory.py:584-586`，但每 20 任务重置消解长期效果）
- **有机制但默认关闭/有 bug**：MemoryBank（遗忘公式括号缺失 bug，默认关闭，`forget_memory.py:36`）、SimpleMem（`consolidation_triggered=False` 硬编码，`cross/session_manager.py:394`）、MemGPT（Sleeptime 默认关闭，`schemas/agent.py:318`）、MemOS（LTM 80% 阈值仅异步路径生效；`_update_usage_history` 整体注释掉）、LightMem（睡眠期需手动触发）
- **完全无淘汰**：Mem0、Zep（图只增不减）、A-MEM、MIRIX、LatentMem

---

### 实验评测口径

**共同 benchmark LoCoMo**（但数字不可横向比较）：

| 方案 | 报告数字 | 底座模型 | 评分口径 |
|------|---------|---------|---------|
| Mem0（v3） | F1=91.6 | GPT-4o | 标准 F1 |
| SimpleMem | F1=43.24 | GPT-4.1-mini | 标准 F1 |
| MemoryOS | F1=36.23 | GPT-4o-mini | 标准 F1 |
| MemOS | +38.97%（相对提升） | GPT-4o-mini | 相对提升率 |
| MIRIX | Overall=85.38 | 笔记未记录 | LLM-as-Judge 0-100 分 |
| LightMem | 高基线 6%-29% | GPT-4.1-mini+Qwen | 相对提升率 |

底座模型不同（GPT-4o vs GPT-4.1-mini vs GPT-4o-mini）、评分口径不同（F1 vs LLM-as-Judge vs 相对提升率）、子集选取不同（MemoryOS 用 LoCoMo 全集，MIRIX 排除 adversarial，SimpleMem 用 LoCoMo 4 类，Mem0 也是全集）——**以上数字无法横向比较**。

**共同 benchmark LongMemEval**（同样不可横向比较，底座模型和口径各异）：Mem0（94.8 F1）、Zep（63.8% Acc，gpt-4o）、MemOS（+40.43%）、LightMem（比基线高 2%-7.7%）。

**专用 benchmark（不与对话型共用）**：G-Memory/MemEvolve/LatentMem 用任务型（ALFWorld/GAIA/PopQA 等）；MemoryBank 用自建数据集（194 题）；MemGPT 用 MSC（2023 年早期数据）。

**横向比较的根本障碍**：无一篇论文在完全统一实验设置（同底座模型、同评分口径、同数据集全集、同 baseline）下对比超过 5 个方案。

---

## 三、共同启发

> 每条均有 3 个以上项目支撑

**1. 批量缓冲写入可以在不降低效果的前提下显著降低成本。**
SimpleMem（每 40 条触发 1 次，LoCoMo F1 比 Mem0 高 9 点，token 消耗降至 531 vs 973）、LightMem（在线成本比 Mem0 低 105×，LongMemEval 准确率反而更高）、G-Memory（任务结束后写入，避免中途 LLM 频繁调用）均证明写入不必即时触发。原因是批量处理让 LLM 一次看到更多连贯上下文，提取质量反而更高。

**2. 在线写入性能与记忆质量可以彻底解耦。**
LightMem（在线纯追加 + 离线睡眠期 LLM 仲裁）、MemOS（fast 模式先存原文 + 后台精化升级，`simple_struct.py:347`）、MemGPT（在线 compaction 用小模型 + 离线 Sleeptime 语义整合）、MIRIX（异步入队不阻塞用户，`queue_util.py:76`）均采用此策略。共同结论：记忆质量提升靠的是更多计算，而这些计算不必占用用户等待时间。

**3. 写入时事实提炼（而非保留原始文本）系统性提升检索准确率。**
Mem0（事实句级，LoCoMo 91.6）、SimpleMem（事实句+结构化元数据，LoCoMo 43.24 vs MemoryBank 的 6.84）、MemOS（事实句+图节点，LoCoMo+38.97%）均明显优于保留原始 QA 对的 MemoryBank（LoCoMo 6.84）和原始对话文本的 A-MEM（构建后检索准确率低）。原因：代词消解和时间绝对化让记忆在离开原始上下文后仍可被精确检索。

**4. 显式时间戳字段是时序推理任务的必要条件。**
SimpleMem（Temporal F1 去掉语义压缩后从 58.62 降至 25.40，`-56.7%`，即显式 timestamp 贡献约 33 个 F1 点）、MemoryOS（Temporal 子任务提升最大，约 +119% vs A-Mem，对话链机制和时间戳共同贡献）、Zep（四时间戳双时序模型专门为时序一致性设计）、LightMem（time_stamp/weekday 字段直接用于时序问答）——凡是有显式时间字段的方案在 Temporal 子任务上均优于无时间字段的方案。

**5. 检索端多路信号融合弥补纯向量在精确匹配上的系统性盲点。**
Mem0 三信号（向量+BM25+实体 boost）在 LoCoMo 上相比 v2（二段式 ADD/UPDATE）提升约 20 点；SimpleMem 三路（向量+BM25+SQL 符号层）在多跳推理 F1 上优于单路基线；Zep 的 RRF 融合在专有名词、公司名等精确匹配场景显著优于纯余弦相似度。三个方案一致证明：纯语义向量在专有名词、时间词、精确属性查询上存在系统性检索漏召，多路融合是直接补救方案。

**6. "论文宣称的高级功能"在开源代码里普遍默认关闭或空转，实验数字反映的是简化路径。**
13 个方案中至少 10 个有此问题：SimpleMem 的 synthesis（未实现，`memory_builder.py:29` 注释方向相反）、A-MEM 的 Note Construction（论文复现仓 NameError，`memory_layer.py:380-389`）、MemOS 的图谱关系推理（全注释，`relation_reason_detector.py:49-80`）和参数记忆（占位符）、MemEvolve 的长期记忆默认只写不读（`config.py:52`）、MemoryBank 的遗忘机制（默认关闭，公式有 bug）、MIRIX 的 Reflexion 代理（默认关闭，`app_constants.py:28`）、LatentMem 的角色感知检索（默认不启用）。这是领域性规律而非个别现象。

---

## 四、存在的问题

> 每条点名至少 2 个项目并附代码证据

**1. 矛盾处理几乎全部缺失，13 个方案中只有 Zep 真正实现。**
Zep 通过 LLM 语义判断 + 确定性时序裁决 + 失效时间戳实现矛盾处理（`edge_operations.py:538-573`），旧事实标记 `invalid_at` 而非删除，历史完整保留。其余 12 个方案均无此能力：Mem0 采用 ADD-only，矛盾事实并存靠 BM25 排序间接抑制旧版本（代码注释承认"在极端矛盾场景下可能失效"，`main.py:751`）；MemOS 的矛盾检测默认完全不生效（`reorganize=False`，`handler.py:30`，`mem_reorganizer_on()` 是空 `pass`）；LightMem 把矛盾处理推到离线睡眠期，但 `update` 操作只改文本不重算向量（`lightmem.py:617-621`），处理后检索索引与内容失配；MemGPT 靠 LLM 自觉发现并修改 core memory，无系统保障（`core_tool_executor.py:346`）。在用户信息频繁更新（搬家、换工作、改计划）的场景下，除 Zep 外所有方案都会积累矛盾记忆，且无自动修正路径。

**2. "论文核心宣称的功能"在代码中大面积默认关闭、空转或有 bug，实验复现必须逐项核查。**
SimpleMem：论文 Section 2.2 核心贡献"在线语义合成"在代码里根本不存在（`memory_builder.py:29` 注释明确把"合成→更少条目"改成"生成足够多条目"，方向完全相反）；消融实验"w/o Online Synthesis 导致 Multi-hop F1 下降 31.3%"（论文 Table 5）无法证明该功能有效，因为该功能从未实现。A-MEM：论文复现仓 Note Construction 永远失效（`memory_layer.py:380-389`，`re.sub` 无 `import re` → NameError → 静默返回空元数据），**论文实验在该 bug 下跑出来，相当于纯文本存储条件下测试**。MemOS：参数记忆（LoRA）是占位符（`lora.py:37-41`，`dump()` 只写 `b"Placeholder"`）；图谱关系推理（INFERS/FOLLOWS 等边类型）被三引号注释掉（`relation_reason_detector.py:49-80`），恒返回空字典。这些功能被列为论文三大支柱，但代码中不存在。

**3. 遗忘/淘汰机制普遍简陋，无一实现基于重要性的持续自动遗忘。**
当前遗忘机制分两类：容量上限 FIFO/LFU（MemoryOS 的 STM/MTM，`mid_term.py:71-101`；MemOS 的 WorkingMemory，`manager.py:527`）或手动/外部触发清理（LightMem 睡眠期需手动触发；MemoryBank 摘要需用户点按钮）。G-Memory 的 insight 信用分（`score≤0` 删除，`GMemory.py:584-586`）是最接近"基于效用的遗忘"的设计，但每 20 任务 `merge_insights` 清空重建（`GMemory.py:508-549`），信用分历史归零，与"长期演化"叙事相悖。Zep 完全无遗忘（图只增不减，全库搜索无任何 decay/prune/TTL 机制）；Mem0 无 TTL 无上限无自动删除；A-MEM 的 `retrieval_count`/`last_accessed` 初始化后从不更新（`memory_system.py:77-80`），无法支撑任何频率遗忘。没有任何方案实现了"按访问频率 + 时间衰减 + 语义重要性三维度自动遗忘"。

**4. 实验评测口径不统一，跨论文数字不可横向比较。**
同是 LoCoMo 基准：Mem0 报 F1=91.6（GPT-4o，v3 管线）；SimpleMem 报 F1=43.24（GPT-4.1-mini）；MemoryOS 报 F1=36.23（GPT-4o-mini）；MIRIX 报 Overall=85.38（LLM-as-Judge 0-100 分，不是 F1）——底座模型差 2-3 代，评分口径根本不同，子集选取各异（MemoryBank 只用 LoCoMo 子集，MIRIX 排除 adversarial 类）。LongMemEval 上同样：Mem0 报 F1=94.8，Zep 报准确率 63.8%（均为 gpt-4o），MemOS 报相对提升 +40.43%，LightMem 报"比最强基线高 2%-7.7%"——口径四种，无法从论文数字得出"哪个系统更好"的结论。这一问题在任务型 benchmark 和对话型 benchmark 间尤为严重（G-Memory/MemEvolve/LatentMem 只用任务型，无法与对话型方案比较），且无任何论文在完全统一设置下对比超过 5 个方案。
