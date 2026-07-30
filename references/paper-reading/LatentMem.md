# LatentMem

**论文**：《LatentMem: Customizing Latent Memory for Multi-Agent Systems》，arXiv:2602.03036  
**仓库**：https://github.com/KANABOON1/LatentMem（commit: 7173f64，以下行号仅在此版本有效）  
**代码基础模型**：Qwen3-4B-Instruct-2507（Qwen 系列的 40 亿参数指令模型）  
**Stars**：46（2026-06-11）

---

## 一、为什么会有这篇论文？

### 背景：AI agent 的"短期记忆"问题

大语言模型（LLM）天生没有长期记忆——每次对话开始时，模型只能看到当前对话窗口里的内容，上一次说了什么它完全不记得。要让 agent 能"记住"历史经验，通常的做法是维护一个外部"记忆库"：把历史对话、过去的任务经验写成文字，需要时从库里取出来，作为文本放进 prompt（发给模型的请求）里让模型参考。

这种方法叫 **RAG（Retrieval-Augmented Generation，检索增强生成）**：先从库里检索相关历史，再把检索结果拼进 prompt，让模型"看到"这些补充信息后再生成回答。逻辑简单直接，工程上也很成熟。

### 新挑战：多智能体系统中文本记忆代价成倍增加

LatentMem 瞄准的场景是**多智能体系统（MAS，Multi-Agent System）**——不是单个 AI，而是多个 AI agent 组成团队协作完成任务，比如一个"assistant"负责推理、一个"user proxy"负责与外部环境（搜索引擎、代码执行器）交互。

文本记忆在 MAS 里面临一个具体痛点：**每个 agent 都要在自己的 prompt 里放一段历史记忆文字**，而大语言模型能处理的文字长度有上限（"上下文窗口"，以 token 为单位，token 大约是一个词或一个中文字），记忆占得越多，当前任务可用的空间就越少。agent 数量一多，这个问题成倍放大。

此外，多轮任务里每次都要把完整历史文字重新送给模型，计算成本也不低。

### LatentMem 的核心想法：把记忆变成数字向量，绕过 token 限制

LatentMem 提出：**不把记忆转成文字放进 prompt，而是训练一个"记忆压缩器"（Composer，代码里也叫 weaver）把历史经验压缩成 8 个"隐向量"（latent vector），直接拼接到 agent 接收的数字表示序列末尾**，绕过文本 token 的限制。

**隐向量是什么？** 大语言模型内部不是直接操作文字的，它先把每个词转换成一组数字（"嵌入向量"，embedding），然后在这些数字上做计算。LatentMem 的思路是：既然模型内部用的是数字，何不直接把记忆做成 8 个数字向量，跳过"先变成文字、模型再把文字转回数字"这个中间步骤？这样 prompt 里记忆部分的 token 消耗为零，记忆内容以数字向量形式"悄悄插入"模型的内部表示里。

**关键难题**：这 8 个向量到底该编码什么内容？手工设计很难。LatentMem 用强化学习训练这个压缩器，以 agent 是否完成任务为奖励信号，让压缩器自己学会"压缩什么才有用"。这套训练方法叫 **LMPO（Latent Memory Policy Optimization，隐记忆策略优化）**。

### 实验证明了什么

论文在四类任务上与多种记忆基线对比（数据集：PopQA、TriviaQA、StrategyQA 等问答任务，KodCode 等编码任务；基础模型：Qwen3-4B）。对比基线包括：不用记忆（MetaGPT 基线模式）、文本摘要式记忆（Voyager）、LLM 生成记忆（Generative）、规则演化记忆（GMemory、OAgent）。论文宣称 LatentMem 在准确率上超过上述所有文本记忆基线，同时 prompt 中记忆部分占用的 token 接近零。

---

## 二、记忆条目设计：一条记忆长什么样？

### 存什么：完整的"成功轨迹"，不做摘要

LatentMem 记忆库里存的不是事实句、不是 QA 对、也不是摘要，而是**一次完整任务执行过程的全程记录**，叫做 `Trajectory`（轨迹）。且只存**任务成功**的轨迹（`runner.py:129-131`，`if trajectory.label == True: add_memory(trajectory)`）。

一条 `Trajectory` 的字段（[latentmem/utils/message.py:135-141](research/LatentMem/latentmem/utils/message.py#L135-L141)）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `task_init_description` | str | 任务描述，即初始问题/目标（同时作为向量检索的索引键） |
| `trajectory` | list[MessageGraph] | 多个交互步骤的时间序列 |
| `label` | bool | True=成功，False=失败 |
| `extra_fields` | dict | 扩展字段（检索后临时存储，不持久化） |

每个 `MessageGraph` 代表 MAS 与环境完成一轮交互（[message.py:39-92](research/LatentMem/latentmem/utils/message.py#L39-L92)）：

- `state`：这一步的输入状态
- `action`：MAS 的整体输出（agent 决定执行什么动作）
- `observation`：环境的反馈（执行动作后环境返回什么）
- `mas_message_graph`：一个有向图（DAG，有向无环图），记录 MAS 内部各 agent 之间的消息传递关系

每个 agent 在这一步的具体发言存在图节点里（`MessageNode`），包含 system_prompt、user_prompt 和 `response`。

### 写入前经历了什么变换：几乎没有

这是 LatentMem 与其他方法最明显的区别——**原始轨迹直接序列化存入数据库，不做 LLM 摘要、不提炼规则、不改写**。

- 整条轨迹调用 `to_serializable()` 转成 JSON 字典，存进 Chroma 的 `metadata` 字段
- `task_init_description` 单独作为 `page_content` 被 embedding 模型转成向量入库（[experience.py:43-50](research/LatentMem/latentmem/mas_core/memory/backbone/experience.py#L43-L50) 或 metagpt.py 里的同结构代码）
- **写入时 LLM 调用次数：0**；只调用 embedding 模型一次

**对比**：Voyager 写入时用 LLM 生成 3 句话的摘要作为索引描述；GMemory 每积累 20 条就触发 LLM 提炼"规则"，每轮迭代多次对比和总结调用。

**好处**：写入成本低，没有信息损失（完整轨迹都保留了）。  
**代价**：每条记忆体积大，需要在检索之后再由 Composer 压缩。

**向量数据库（Chroma）是什么？** 一种特殊数据库，不靠关键词匹配，而是把文字先转成"嵌入向量"（embedding，一组表示语义的数字），通过计算向量间的数学距离找到语义相近的内容。即使查询和文档没有共同关键词，只要意思相近就能被找到。

**Embedding 是什么？** 把一段文字转化成固定长度数字向量的过程，由专门训练的小模型完成（这里用 `sentence-transformers/all-MiniLM-L6-v2`）。语义相近的句子转换后得到数学上也相近的向量。

### 一条轨迹在库里的具体形态

以 PopQA（知识问答）为例，一条存入 Chroma 的记录大致如下：

```
Document {
  page_content: "Who was the first president of the United States?"
  
  metadata: {
    task_init_description: "Who was the first president of the United States?",
    label: true,
    trajectory: "[
      {\"state\": \"Who was the first president...\",
       \"action\": \"Search('first president United States')\",
       \"observation\": \"George Washington (1732–1799) was the first President...\",
       \"mas_message_graph_data\": \"{...assistant节点和user_proxy节点的发言...}\"},
      {\"state\": \"Based on search results, answer.\",
       \"action\": \"George Washington\",
       \"observation\": \"Correct!\",
       \"mas_message_graph_data\": \"{...}\"}
    ]"
  }
}
```

`page_content` 是向量检索的索引依据，`metadata.trajectory` 是整条交互序列的 JSON 字符串。

---

## 三、系统完整运转流程

### 整体架构总览

整个系统是一个 `BaseMemoryMAS`（`AutoGenMemoryMAS`），内部有两个 agent（assistant + user proxy），共享一个 `LatentMem` 中心记忆对象。`LatentMem` 由两部分组成：

- **RAG 后端**：负责存储和检索轨迹，默认用 `MetaGPT`（简单向量相似搜索）
- **Composer（Weaver）**：负责把检索到的文本轨迹压缩成 8 个隐向量

**默认配置（四个数据集的 yaml 都一样）**：  
`rag.mode: metagpt`（纯向量相似搜索）、`latents_len: 8`（压缩成 8 个向量）、`pos_shots_num: 1`（取 1 条成功轨迹）、`neg_shots_num: 0`（不用失败案例）

### 阶段 0：数据采集——建立经验库

运行 `bash scripts/data.sh`，触发 `runner.py` 的 `bootstrap_data` 函数（[runner.py:100-142](research/LatentMem/latentmem/runner.py#L100-L142)）：

MAS 在训练集上逐批运行任务，对每批任务的执行过程打标签（成功/失败），**只把 `label==True` 的轨迹写入记忆库**：

```python
if trajectory.label == True:
    self.memory_mas.centralized_memory.add_memory(trajectory)
```

写入由 `MetaGPT.add`（[metagpt.py:28-39](research/LatentMem/latentmem/mas_core/memory/backbone/metagpt.py#L28-L39)）完成：轨迹 JSON 序列化 → 创建 Document → 存入 Chroma。

### 阶段 1：检索——向量相似搜索找最近经验

当 agent 接到新任务、准备生成回复前，调用 `_trigger_memory(task_description)`（[agent.py:341-361](research/LatentMem/latentmem/utils/agent.py#L341-L361)），触发检索：

1. 把当前任务描述通过 embedding 模型转成向量
2. 在 Chroma 里搜索向量距离最近的 1 条 `label=True` 的轨迹（[metagpt.py:42-61](research/LatentMem/latentmem/mas_core/memory/backbone/metagpt.py#L42-L61)）
3. 反序列化 metadata，还原成 `Trajectory` 对象

**这步没有 LLM 调用，只有 embedding 计算和向量距离比较。**

注意：MetaGPT RAG 不做角色感知（role-aware）提取。`LatentMem.retrieve_memory` 里的 `_concat_agents_response` 会尝试从轨迹里取当前角色的历史响应，但 MetaGPT 模式下 `extra_fields` 没有设置这些内容，最终返回空字符串，后续 Composer 使用不含角色历史的简化 prompt。

### 阶段 2：压缩——Composer 把轨迹文本变成 8 个向量

检索到轨迹后，系统把它传给 **Composer**（[latentmem.py:143-163](research/LatentMem/latentmem/mas_core/memory/latentmem.py#L143-L163)，[composer.py:56-91](research/LatentMem/latentmem/mas_core/memory/composer.py#L56-L91)）进行压缩。

**Composer 是什么？** 一个和 agent 同规模的语言模型（Qwen3-4B），加了：
- **LoRA 适配器**：只更新 attention 层中极少量的参数（r=16，目标模块 q_proj、v_proj）
- **8 个可学习的查询向量**（`query_latents`，形状 `[8, hidden_size]`，初始随机，通过训练优化）
- **线性投影层**（`weaver2agent_proj`）：把 Composer 隐层维度映射到 agent 的隐层维度

**LoRA（Low-Rank Adaptation，低秩适配）是什么？** 直接微调一个 40 亿参数模型需要更新所有参数，成本极高。LoRA 的办法是：在模型的每个注意力层旁边插入两个很小的矩阵（参数量可以缩减 99%），训练时只更新这些小矩阵，原始参数冻结不动。效果接近全量微调，但训练成本低得多。

**Composer 的工作过程**（[composer.py:56-91](research/LatentMem/latentmem/mas_core/memory/composer.py#L56-L91)）：

1. 系统把"当前任务 + 检索到的轨迹文本 + agent 角色"填入 prompt 模板（[prompt.py:25-34](research/LatentMem/latentmem/mas_core/memory/prompt.py#L25-L34)）：

   ```
   # Current Task
   When was Alexander Graham Bell born?
   
   # Examples of previous successful tasks related to this task
   ## Your Own Past Successes (Execution Patterns)
   <|im_start|>user
   Who invented the telephone?<|im_end|>
   <|im_start|>assistant
   Search('telephone inventor')<|im_end|>
   <|im_start|>user
   Alexander Graham Bell invented the telephone...<|im_end|>
   <|im_start|>assistant
   Alexander Graham Bell<|im_end|>
   label: True
   
   You are acting as a assistant. Using no more than 8 tokens, extract the most
   relevant information from the memory above that will help you accomplish the
   current task effectively.
   ```

2. 这段文字（可能上百个 token）被 tokenizer 转成数字 ID，再通过 embedding 层变成向量序列（形状 `[seq_len, hidden_size]`）

3. 在这个向量序列**末尾拼上 8 个 `query_latents`**（`composer.py:78`，形状变成 `[seq_len+8, hidden_size]`）

4. 整段序列做一次前向计算（模型推断），取**最后 8 个位置**（即 query_latents 所在位置）的最后一层隐藏状态（`composer.py:88-89`），得到形状 `[8, hidden_size]` 的张量

5. 通过线性投影层调整维度，匹配 agent 模型的隐藏层大小（`latentmem.py:161`）

**直觉类比**：想象一个学生读完一篇参考文章后，写出 8 个"关键词"概括精髓。Composer 读完历史轨迹，用 8 个数字向量概括"这次成功经验的核心要素"。只不过这 8 个"关键词"不是人类能直读的文字，而是模型内部表示空间里的语义向量。

**注意**：prompt 里的 "Using no more than 8 tokens" 只是给模型的"心理暗示"，Composer 实际上不生成任何文字，只通过内部计算产出 8 个隐藏状态向量。

### 阶段 3：注入——把 8 个向量直接插进 agent 的输入

agent 生成回复时（[agent.py:244-259](research/LatentMem/latentmem/utils/agent.py#L244-L259)）：

1. 当前 prompt（系统提示 + 任务描述）通过 tokenizer 和 embedding 层转成向量序列（text embeddings）
2. **在 text embeddings 末尾拼接 8 个 latent 向量**（`torch.cat([text_embeddings, latent_emb], dim=1)`）
3. 把这个拼接后的向量序列（不是 token ID，而是 embedding 矩阵）送进 agent LLM 直接生成输出

关键点：prompt 里记忆内容的占位符 `{memory_content}` 此时填的是空标记 `<|Memory_Empty|>`，不占任何实质 token（[agent.py:351-353](research/LatentMem/latentmem/utils/agent.py#L341-L361)）。**记忆以向量形式"悄悄插入"，对 token 计数没有影响。**

无论历史轨迹有多长（100 token 还是 1000 token），最终记忆部分给 agent 带来的额外长度始终是 8 个向量位置。

### 阶段 4：训练——用任务成败教会 Composer 怎么压缩（LMPO）

Composer 一开始不知道该把什么信息编码进 8 个向量。训练分两步：

**第一步：SFT 预热**（Supervised Fine-Tuning，监督微调）

把采集阶段成功轨迹里的每条 agent 发言作为训练数据，直接用监督学习让 Composer 初步学会理解任务结构（[runner.py:144-160](research/LatentMem/latentmem/runner.py#L144-L160)）。相当于"先背参考答案，建立基础能力"。

**第二步：LMPO 策略优化**（核心）

本质是强化学习——不告诉 Composer"应该输出什么"，只告诉它"这次任务做成了还是没做成"，让它自己探索什么样的压缩策略有效（[latentmem/trainer/grpo_trainer.py](research/LatentMem/latentmem/trainer/grpo_trainer.py)）：

1. **采样**：对同一个任务，让 MAS（使用当前 Composer 产出的 latent 记忆）运行 4 次（`num_generations=4`）
2. **打分**：每次运行结束后取 `trajectory.label`（True=1, False=0）作为奖励（[grpo_trainer.py:231-232](research/LatentMem/latentmem/trainer/grpo_trainer.py#L231-L232)）
3. **归一化**：4 次奖励减去其均值 → advantage（优势值）。若 4 次都成功（平均=1），advantage 为 0；若 3 成 1 败，成功的 advantage=0.25，失败的=-0.75
4. **计算梯度**：重新带梯度地跑一次 Composer → 生成 latent → agent 看到 latent → 算 agent 复现自身历史回复的概率（per-token log probability）→ 用 advantage 加权 → 反向传播
5. **更新**：梯度只流向 Composer 的 LoRA 参数、query_latents、投影层；agent LLM 全程冻结不更新（[autogen_main.py:47-48](research/LatentMem/latentmem/mas_core/structures/autogen/autogen_main.py#L47-L48)）

**强化学习的直觉**：Composer 是"记忆整理员"，agent 是"考生"。不能直接告诉整理员"这段记忆有没有用"，但能看到考生用了这段记忆后的考试成绩。考得好 → 正反馈，教整理员以后多产这类向量；考得差 → 负反馈，引导整理员调整压缩策略。

---

## 四、一个具体例子走一遍

假设任务：`"Alexander Graham Bell 是哪年出生的？"`（PopQA 风格知识问答，agent 可以调用搜索工具）

**阶段 0：库里已有的轨迹**

训练集期间，有一次任务 `"Who invented the telephone?"` 执行成功，被写入 Chroma：
- page_content（向量索引）: `"Who invented the telephone?"`  
- metadata: 整条交互轨迹 JSON（搜索动作 → 搜索结果 → 最终回答 "Alexander Graham Bell"）+ `label: true`

**推断阶段**：

1. **embedding**：新任务 `"When was Alexander Graham Bell born?"` 被转成向量 `[v₁, v₂, ..., v₃₈₄]`

2. **向量检索**：计算与库中所有 `label=True` 轨迹的 page_content 向量的距离，发现 `"Who invented the telephone?"` 的向量距离最近（两者都涉及 Bell）→ 取出这条轨迹

3. **拼装 Composer prompt**：把当前任务 + 取出的轨迹文本填入 `EXTRACT_LATENT_PROMPT`（参见上面阶段2的示例，约 150 tokens）

4. **Composer 前向**：
   - 150 tokens 的文本 → embedding 层 → `[150, 2048]` 的向量矩阵
   - 末尾拼上 8 个 query_latents → `[158, 2048]`
   - 一次前向 → 取最后 8 个位置的隐藏状态 → `[8, 2048]`
   - 线性投影 → `[8, 2048]`（如果 agent 也是 2048 维就不变）

5. **agent prompt 构建**：
   - 文字 prompt（只有当前任务描述，无记忆文字）→ 通过 embedding 层 → `[T_text, 2048]`
   - 末尾拼上 8 个 latent 向量 → `[T_text + 8, 2048]`
   - 这个矩阵送进 agent LLM 生成

6. **agent 生成**：输出 `Search('Alexander Graham Bell birth year')` → 搜索工具返回 `"Born: March 3, 1847"` → agent 再次生成 → 最终回答 `"1847"`

整个过程，prompt 里记忆部分的文字是空的 `<|Memory_Empty|>`，记忆通过 8 个数字向量无形地影响 agent 的生成过程。

---

## 五、可选扩展：ExperienceBank（rag_mode="latentmem"）

除了默认的 MetaGPT RAG，代码里还实现了一个更复杂的 `ExperienceBank`（[experience.py](research/LatentMem/latentmem/mas_core/memory/backbone/experience.py)），通过 `rag.mode: latentmem` 启用。默认配置**不使用**此模式，所有发布的四个数据集 yaml 都是 `mode: metagpt`。

ExperienceBank 在 MetaGPT 基础上增加了两个功能：

**1. LLM 重排（Generative Reranking）**（[experience.py:106-146](research/LatentMem/latentmem/mas_core/memory/backbone/experience.py#L106-L146)）：  
先用向量相似搜索取 2 倍候选数量，再把每条候选轨迹和当前任务一起发给 LLM，让它给出 0-10 的相关性分数，按分数重排后取 top-k。

评分 prompt 示例：
```
System: 你是一个评估文本相关性的 agent。
User:
1. 过去的案例：[轨迹文本]
2. 当前任务：When was Alexander Graham Bell born?
评估过去案例对解决当前任务有多大帮助。用 1-10 分打分，只输出数字。
Score:
```
LLM 输出 `8` → 解析为 8.0，其他候选可能得 2、5 分 → 取最高分的候选。

**2. 角色感知历史提取（Role-Aware Context Extraction）**（[experience.py:148-162](research/LatentMem/latentmem/mas_core/memory/backbone/experience.py#L148-L162)）：  
从检索到的轨迹 `MessageGraph` DAG 里，按当前 agent 的 role（如 "assistant"），把该角色在历史轨迹每一步的实际回复抽取出来，作为额外上下文传给 Composer。  
这使得 Composer prompt 变成更完整的 `EXTRACT_LATENT_PROMPT_FULL`，包含"该角色上次的完整发言序列"（[prompt.py:36-47](research/LatentMem/latentmem/mas_core/memory/prompt.py#L36-L47)）。

**代价**：检索时多一次 LLM 批量调用（所有候选各一次，max_new_tokens=20）。

---

## 六、论文宣称 vs 代码实际

**核心机制真实存在且一致**：latent 向量替代文本记忆（[agent.py:248-259](research/LatentMem/latentmem/utils/agent.py#L248-L259)）、token-efficient（[agent.py:351-353](research/LatentMem/latentmem/utils/agent.py#L341-L361)）、LMPO 训练流程（[grpo_trainer.py](research/LatentMem/latentmem/trainer/grpo_trainer.py)）、HuggingFace 上发布了训好的模型和轨迹库（README.md:24-25）均真实可用。

**需要注意的出入**：

1. **原始笔记将 ExperienceBank 误作主路径描述**：之前的理解把 `ExperienceBank`（rag_mode="latentmem"）描述为 LatentMem 的核心 RAG 后端，包括 LLM 重排、角色感知提取等。但实际上，所有发布配置都使用 `mode: metagpt`（简单向量搜索），ExperienceBank 是可选扩展模式，在默认推断管线里不启用。**这影响如何理解"role-aware"这一论文卖点**：role-aware 在 ExperienceBank 模式下才真正起作用。

2. **训练入口脚本命名不一致**：README 写 `bash scripts/lmpo_train.sh`，脚本目录下实际文件叫 `lmpo.sh`（轻微问题，可能改名未同步文档）。

3. **lmpo.sh 按当前代码跑会报错**：脚本传 `run.mode grpo`，但 `runner.py` 只接受 `data/sft/lmpo/eval`，传 `grpo` 直接 ValueError（[runner.py:285-294](research/LatentMem/latentmem/runner.py#L285-L294)）；`train()` 函数注解写 `Literal["sft", "grpo"]` 但分支判 `"lmpo"`（[runner.py:186, 204](research/LatentMem/latentmem/runner.py#L186)）——GRPO 改名 LMPO 时代码没全量同步，**发布的训练脚本无法直接复现**。

4. **yaml 里 lmpo 超参块是摆设**：`popqa.yaml:87-117` 定义了完整的 `run.lmpo` 配置，但 runner 实际读 `run_cfg.grpo`（[runner.py:76-77](research/LatentMem/latentmem/runner.py#L76-L77)）。该配置块从未被读取，真正起作用的超参要通过命令行 `run.grpo.*` 传入。

5. **LMPO 实际上是 vanilla policy gradient**：`beta: 0.0`（KL 正则不起作用），`old_per_token_logps = per_token_logps.detach()`（[grpo_trainer.py:281](research/LatentMem/latentmem/trainer/grpo_trainer.py#L281)）使 importance sampling ratio 恒为 1、clip 不起作用。若论文宣称 PPO 式 clip，代码实际只是最基础的 vanilla 策略梯度。

6. **"abstract" 分支是死代码**：`_construct_text_memory` 里的 `pos_shots[0].extra_fields.get("abstract")` 判断（[latentmem.py:123](research/LatentMem/latentmem/mas_core/memory/latentmem.py#L123)）全仓库无任何地方写入 `abstract` 字段，永远走不到这个分支。

7. **负例记忆在主管线下无效**：配置 `neg_shots_num=0`，runner 只写成功轨迹，库里根本没有 `label=False` 的条目，`NEG_SHOTS_TEMPLATE` 永不触发。

8. **EXTRACT_LATENT_PROMPT 的指令是"暗示"不是实际操作**：提示写"使用不超过 k 个 token 提取信息"，但 Composer 根本不生成文字，只取查询向量位置的 hidden state。这行指令只是引导模型关注压缩，与计算过程无对应。

**对引用和复现的影响**：可以引用 latent 向量替代文本记忆的思路和 LMPO 训练范式。复现实验建议直接下载 HuggingFace 上发布的模型和轨迹库跑 `eval_hf.sh`，可信度高于自己跑训练脚本。若要复现训练，需手动修正 lmpo.sh 中的 `run.mode grpo` → `run.mode lmpo`，并通过命令行传入 LMPO 超参。

---

## 七、关键代码位置

| 机制环节 | 文件路径 | 类/函数 | 行号 | 一句话作用 |
|---|---|---|---|---|
| 记忆数据结构 | latentmem/mas_core/base_centralized_memory.py | Memory | 8-12 | text_memory + latent_memory 双通道 |
| 轨迹/消息结构 | latentmem/utils/message.py | Trajectory / MessageGraph / MessageNode | 23-206 | MAS 交互全程记录 |
| 中心记忆主类 | latentmem/mas_core/memory/latentmem.py | LatentMem | 17-163 | RAG 后端 + Composer 组合 |
| 默认RAG后端 | latentmem/mas_core/memory/backbone/metagpt.py | MetaGPT | — | 纯向量相似搜索，无 LLM 重排 |
| 写入触发 | latentmem/runner.py | bootstrap_data | 129-131 | 仅 label==True 成功轨迹入库 |
| 记忆触发检索 | latentmem/utils/agent.py | _trigger_memory | 341-361 | invoke 时调记忆，得 text+latent |
| 向量检索（默认） | backbone/metagpt.py | retrieve | 41-75 | similarity_search_with_score |
| 文本记忆拼装 | latentmem/mas_core/memory/latentmem.py | _construct_text_memory | 110-141 | 轨迹格式化成文字供 Composer 读 |
| latent 压缩 | latentmem/mas_core/memory/latentmem.py | _construct_latent_memory | 143-163 | 调 Composer 产出 8 个向量 |
| Composer 核心 | latentmem/mas_core/memory/composer.py | text_to_latent | 56-91 | 文本嵌入尾拼查询向量取 hidden |
| 可学习查询向量 | latentmem/mas_core/memory/composer.py | query_latents | 43-46 | nn.Parameter(8, hidden_size) |
| latent 注入 agent | latentmem/utils/agent.py | invoke | 248-259 | torch.cat(text_emb, latent_emb) |
| 训练时重算 latent | latentmem/utils/agent.py | forward / _recover_memory | 77-104, 363-375 | 带梯度重建 latent 算 logp |
| LMPO rollout | latentmem/trainer/grpo_trainer.py | _generate_and_score_completions | 203-258 | label 作 reward，组内算 advantage |
| LMPO loss | latentmem/trainer/grpo_trainer.py | _compute_loss | 260-304 | 策略梯度，默认 beta=0 无 KL |
| 冻结 agent | latentmem/mas_core/structures/autogen/autogen_main.py | fix_model_parameters | 47-48 | agent LLM 不参与训练 |
| 可选：LLM重排 | backbone/experience.py | _generative_retrieve | 106-146 | ExperienceBank 模式才用 |
| 可选：角色提取 | backbone/experience.py | _extract_agent_context | 148-162 | ExperienceBank 模式才用 |

---

## 八、值得借鉴的地方

1. **"只优化记忆生成器，不优化 agent"这个问题切分干净**：冻结 agent LLM，以任务成败为奖励，只训练 Composer。这意味着 Composer 可以和任意冻结的 agent 配合，且 agent 不需要支持 fine-tuning。对"如何端到端优化记忆质量而不改变 agent 能力"是一个可直接复用的范式。

2. **梯度如何穿透不可微的记忆质量评估**：latent 向量拼在 inputs_embeds 上，通过"agent 复现自身响应的 per-token log probability"作为可微中间变量，把 advantage 信号传回 Composer（[grpo_trainer.py:260-304](research/LatentMem/latentmem/trainer/grpo_trainer.py#L260-L304) + [agent.py:77-104](research/LatentMem/latentmem/utils/agent.py#L77-L104)）。这是把"记忆有没有帮助"这个不可微的评估变成可微优化的工程关键点。

3. **统一基线框架设计**：同一个 MAS + 同一个冻结 LLM，通过切换 RAG 后端复刻了 MetaGPT/Voyager/Generative/GMemory/OAgent 五种记忆方式，在完全受控的变量下对比不同记忆机制。这套评测框架的设计思路值得参考。

4. **role-aware 的实现思路**（ExperienceBank 模式）：中心记忆只存一份轨迹，检索时按 agent role 现场抽取该角色的历史响应再压缩，"一份存储、多角色视图消费"比每个 agent 各存一份省空间，也更容易保持一致性。
