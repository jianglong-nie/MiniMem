# G-Memory

- 仓库：https://github.com/bingreeky/GMemory
- commit：7b581c5（本文所有行号仅在该 commit 下有效）
- stars：245（2026-06-11 查询）
- 论文：G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems（arXiv:2506.07398）

---

## 一、论文的故事

### 问题是什么

现在很流行让多个 AI 智能体（agent）协作完成任务——一个负责思考、一个负责执行、一个负责核查，这叫多智能体系统（MAS，Multi-Agent System）。但这类系统有一个严重的缺陷：**它们没有记忆**，或者说记忆设计极为简陋。

更具体地说，有两类问题：

**问题一：已有 MAS 的跨任务记忆要么没有、要么太粗糙**。MetaGPT 只记录当前任务内部的对话，ChatDev 把历史结果压成一句话存起来，MacNet 只保留上一轮的最终答案——这些做法完全丢弃了多个 agent 协作时产生的中间过程，比如"哪个 agent 提出了正确的思路"、"哪一步出错导致任务失败"。

**问题二：直接把单 agent 的记忆机制移植过来也行不通**。单个 agent 解决一个任务，一般产生几千个 token 的对话。三个 agent 协作解同一个任务，对话量可达几十万 token（论文 Figure 1 实测 AutoGen 在 ALFWorld 上平均消耗 4.3×10⁶ token）。如果把整条协作轨迹原封不动地塞进记忆、再检索出来，LLM 根本消化不了。

### 核心想法

G-Memory 的回答是：**对记忆分三个层次组织，从粗到细，再双向检索**。

类比可以帮助理解这个想法：想象一家公司的知识管理系统。
- 最底层是每次会议的完整录音（细节、原始）；
- 中层是会议纪要的索引卡（知道开了哪些会、谁参加了、议题是什么）；
- 顶层是公司总结出来的操作规范和经验教训（"合同审核前必须XX"）。

G-Memory 的三层也是这个逻辑：
1. **Interaction Graph（交互图）**：每次 MAS 任务的完整对话轨迹——谁说了什么、做了什么行动、环境给了什么反馈。
2. **Query Graph（查询图）**：所有历史任务的"节点图"——每个任务是一个节点，语义相近的任务之间连边，这样检索时可以顺着边找到相关任务。
3. **Insight Graph（洞察图）**：从历史经验中提炼出的规律，比如"在家庭场景里搬物品之前先确认物品位置，因为它可能在意想不到的地方"。

检索时双向走：向上到洞察层拿抽象规则，向下到交互层拿压缩版的历史对话片段。这样既有经验总结（高层），又有具体参照（低层），还不会因为太多细节而淹没 LLM。

### 实验证明了什么

论文在 5 个基准测试上，结合 3 种 MAS 框架（AutoGen、DyLAN、MacNet）和 3 种 LLM（GPT-4o-mini、Qwen-2.5-7b、Qwen-2.5-14b）进行了测试。

5 个测试集分别是：
- **ALFWorld**：文字版家庭环境，agent 要操作对象完成目标（"把干净的蛋放进微波炉"）
- **SciWorld**：文字版科学实验环境
- **PDDL**：策略游戏（搭积木类）
- **HotpotQA**：多跳问答（"X 和 Y 是同一类职业吗？"）
- **FEVER**：事实核查

**核心结论**：

G-Memory 在所有测试配置上均超过了全部对照系统，且在多数情况下领先幅度是第二名的两倍以上。具体数字：
- 在体现记忆效果最显著的 ALFWorld 任务上，MacNet + Qwen-2.5-14b 加入 G-Memory 后成功率从 58.21% 涨到 79.10%，提升 **20.89%**；
- 知识问答类任务（HotpotQA + FEVER）平均提升 **10.12%**；
- Token 消耗比同类最佳对照方（MetaGPT-M）少得多：G-Memory 额外消耗约 1.4×10⁶ token，MetaGPT-M 额外消耗 2.2×10⁶ token，但 MetaGPT-M 的提升幅度远不如 G-Memory。

消融实验表明洞察层和交互层两者缺一不可：只用洞察层平均少 3~4%，只用交互层平均少 3~5%，两者合用效果最好。1-hop 的邻域扩展最优，扩太多（2~3 hop）反而引入噪音导致性能下降。

---

## 二、三层架构是什么，为什么这样设计

在讲代码之前，先把三层的概念说清楚——它们在论文和代码里有很大的差距，所以先理解"论文想要什么"，再看"代码实际做了什么"。

### 论文定义的三层

**第一层：Interaction Graph（交互图）**

每个历史任务对应一张有向图：节点是每一条 agent 发言（包括谁说的、说了什么），边表示"这条发言影响了下一条发言"（时间顺序）。检索时，G-Memory 不把整张图返回，而是用 LLM 从中提取"核心子图"——只保留对当前任务有价值的关键步骤。

**第二层：Query Graph（查询图）**

一张存所有历史任务的图：每个历史任务是一个节点，节点存任务描述、成败状态、指向对应交互图的指针。两个任务节点之间如果语义相近（embedding 相似度超过阈值），就连一条边。

为什么需要图而不是直接向量检索？原因是：向量相似度找到的是"语义表面相近"的任务，但有时候两个任务词汇完全不同却在策略上互相启发。图的邻域扩展（k-hop）能顺着"曾经互相参考"的历史关系找到这些间接相关的任务。

**第三层：Insight Graph（洞察图）**

存从历史任务中提炼出的抽象规律（insight），每条规律是一句话，比如"在行动前先验证目标对象的状态，避免无效操作"。每条 insight 带有"支撑任务集"——记录是从哪些历史任务中提炼出来的、在哪些任务上被使用过并验证有效。论文里这些 insight 节点之间也有边，但代码里并没有实现图结构（见后文"宣称 vs 实际"）。

### 直观类比

用一次真实的检索过程来说明三层怎么配合：

> 当前任务：「把一块干净的布放到台面上」（ALFWorld，来自论文 Figure 5）
>
> 1. **向量检索 + 图扩展**：从 Query Graph 找到最相近的历史任务节点，比如「把干净的蛋放进微波炉」（两者都需要先清洁对象）。再顺着图的边找到这个节点的邻居——可能有几个类似的"清洁后放置"任务。
> 2. **向下走（→ Interaction Graph）**：从找到的历史任务对应的交互图里，让 LLM 挑出"核心步骤"——比如"Solver Agent 先把蛋拿出来就想放进微波炉，被 Ground Agent 拦下来：你得先清洁它！"这段对话对当前任务直接有用。
> 3. **向上走（→ Insight Graph）**：找到在这些相关历史任务上被验证有效的 insight，比如"清洁对象后立即放到指定位置，不要中途放到别处"。
> 4. **进 prompt**：把压缩后的历史轨迹片段 + 关联 insight 拼到每个 agent 的 prompt 里，供本次任务参考。

---

## 三、记忆条目设计：一条记忆长什么样

代码里的记忆单元是任务级的，叫 `MASMessage`，定义在 [mas/memory/common.py:136-191](mas/memory/common.py)。

一条 `MASMessage` 包含这些字段：

```
MASMessage {
  task_main:        "put a clean egg in microwave"     # 任务主题/名称
  task_description: "你在一个房间里。环境包含：冰箱..."  # 完整任务描述
  task_trajectory:  "Solver: 我要去拿蛋\nExecutor: 执行：取蛋\n环境反馈：取到了一个蛋..."
                                                       # 多 agent 协作的完整文字对话
  label:            True / False                       # 任务成功=True，失败=False
  key_steps:        "1. 先找到蛋\n2. 清洁蛋\n3. 放入微波炉"
                                                       # LLM 从轨迹里提炼出的关键步骤（写入时算好）
  fail_reason:      "Agent 忘记在放入前清洁蛋"          # 仅失败任务有此字段
  chain_of_states:  [StateGraph1, StateGraph2, ...]    # 每一个环境步对应一张小图
                                                       # 节点是 AgentMessage，记录发言和行动
}
```

**`chain_of_states`** 是论文"交互图"的真实载体（[common.py:53-133](mas/memory/common.py)）。它是一串 `nx.DiGraph`（networkx 的有向图，一种 Python 图数据结构），每个环境步建一张图，节点是 `AgentMessage`（含说话 agent 的名字、内容、上下游边），边是时序关系。

> **向量/embedding 是什么**：把文字转成一串数字（比如 384 维），语义相近的文字转出来的数字串在数学上"距离"也近。这样就能用数学方法找"意思相似"的记忆，而不只是关键词匹配。

**写入前的变换**：并非原始的 `MASMessage` 直接入库，而是先经历"稀疏化"处理（[GMemory.py:244-281](mas/memory/mas_memory/GMemory.py)）：
1. 删掉所有 `reward < 0` 的环境步（中间走错的步骤）；
2. 把轨迹里的所有数字删掉（`re.sub(r'\d+', '', trajectory)` 把"desk 1"变成"desk "），避免 LLM 抽规律时过拟合到具体编号；
3. 用 LLM 读稀疏化后的轨迹、提炼 `key_steps`；
4. 失败任务额外让 LLM 诊断 `fail_reason`。

稀疏化的好处：`key_steps` 在写入时一次性算好，检索时直接用，不需要重复调用 LLM；数字泛化让学到的规律可以迁移到不同的具体实例（"desk 1"和"desk 5"在规律上是一样的）。

最终落地存储是这样的：
- **轨迹层**：`MASMessage` 序列化成 JSON 字符串，作为一个文档存入 [Chroma 向量数据库](https://www.trychroma.com/)（Chroma 是专门存储向量的数据库，能快速找出语义相近的文档）。task_description 同时被 embedding，作为检索的索引。
- **任务图层**：任务节点加入 networkx 图（`task_main` 字符串作为节点 ID），与 embedding 相似度 ≥ 0.7 的已有节点连边。
- **洞察层**：一个 JSON 文件 `insights.json`，是 `list[dict]` 格式，每条 insight 是：`{rule, score, positive_correlation_tasks, negative_correlation_tasks}`。

---

## 四、数据怎么流动：写入路径

以 AutoGen 拓扑为例（[tasks/mas_workflow/autogen/autogen.py](tasks/mas_workflow/autogen/autogen.py)），一条完整的写入流程：

**阶段 1：任务开始，建空容器**

```python
# autogen.py:102
init_task_context(task_description)
```
创建一个空的 `MASMessage`，里面只有任务描述，轨迹为空。

**阶段 2：执行过程中，实时追加**

每次 agent 发言：
```python
# autogen.py:184
add_agent_node(agent_name, message_content)  # 把发言写进当前 StateChain
```
每次环境响应：
```python
# autogen.py:192
move_memory_state(action, observation, reward)  # 记录动作、环境反馈、奖励分数
```

执行结束后，`MASMessage` 的 `chain_of_states` 里已经有了完整的多 agent 对话图序列，`task_trajectory` 里有展平的文字版轨迹。

**阶段 3：任务完成，打标签落库**

```python
# autogen.py:200 → memory_base.py:59-66
save_task_context(label=True/False, feedback="...")
```

这一步触发 `GMemory.add_memory`（[GMemory.py:76-111](mas/memory/mas_memory/GMemory.py)）：

1. **稀疏化 + 提炼**（[GMemory.py:244-281](mas/memory/mas_memory/GMemory.py)）：删负 reward 步 → 数字泛化 → LLM 提炼 key_steps（1 次 LLM 调用）→ 失败任务再诊断 fail_reason（+1 次 LLM 调用）。

2. **任务图建节点/连边**（[GMemory.py:374-402](mas/memory/mas_memory/GMemory.py)）：把 task_description embedding 后，找已有节点里相似度 ≥ 0.7 的，连边，权重 = 相似度。（1 次 embedding 调用 + 1 次向量查询）

3. **Chroma 入库**（[GMemory.py:95-101](mas/memory/mas_memory/GMemory.py)）：整个 `MASMessage` 序列化为 JSON 字符串，连同 task_description 的 embedding 一起存入 Chroma。（1 次 embedding）

4. **周期性触发洞察维护**（[GMemory.py:106-109](mas/memory/mas_memory/GMemory.py)）：
   - 每满 5 个任务：`finetune_insights`——随机抽 5 个历史任务，做成功/失败对比 + 成功批量 critique，LLM 输出 ADD/EDIT/REMOVE/AGREE 操作来更新 insight 库（约 5~15 次 LLM 调用）。
   - 每满 20 个任务：`merge_insights`——用 FINCH 聚类算法（一种不需要预先指定分几组、自动发现聚类数量的聚类方法）把全部任务 embedding 后聚类，同一簇内的相似 insight 用 LLM 合并精简。

**一条 insight 的样子**（在 finetune_insights 后写入 `insights.json`）：

```json
{
  "rule": "Verify the state of objects before and after each action, because agents may overlook intermediate states",
  "score": 3,
  "positive_correlation_tasks": ["put a clean egg in microwave", "clean the cloth and put it on countertop"],
  "negative_correlation_tasks": []
}
```

`score` 是这条规则的"信用分"：成功使用 +1，导致失败 -2，≤ 0 时被删除。这样无用的规律自然淘汰，有效的规律保留并积累分数。

---

## 五、数据怎么流动：检索路径

检索入口是 `retrieve_memory`（[GMemory.py:189-241](mas/memory/mas_memory/GMemory.py)），在每个任务开始前调用（[autogen.py:108](tasks/mas_workflow/autogen/autogen.py)）。

**第一步：粗检索（从任务图找候选）**

`_retrieve_memory_raw`（[GMemory.py:113-187](mas/memory/mas_memory/GMemory.py)）：

1. 把当前任务 embedding，在 Chroma 里找 top-k 个最相近的历史任务（k=1 或 2，论文中最优参数）。
2. 用这 k 个节点在任务图上做 1-hop 邻域扩展：找到与这些节点有边相连的所有任务节点——这些"邻居"可能语义上不如 top-k 那么像，但在解决策略上曾经互相参考过。
3. 对所有候选节点，去 Chroma 里按 task_main 名字反查文档，按 label 分成"成功"和"失败"两组。
4. 重新计算每个候选和当前任务的余弦相似度（两个向量方向的接近程度，1=完全一致，0=无关），按相似度过阈值排序。

> **注意：这里有个已知 bug**（[GMemory.py:150-156](mas/memory/mas_memory/GMemory.py)）：如果候选不够数，代码会用普通向量检索结果**覆盖**任务图扩展结果，而不是合并，导致很多时候"图扩展"这个核心机制实际上被旁路了，退化为纯向量检索。

**第二步：LLM 精排（对成功轨迹逐条打分）**

对每条候选"成功轨迹"，调用 LLM 打 1-10 分："这条历史轨迹对当前任务有多大参考价值？"（[GMemory.py:220-228](mas/memory/mas_memory/GMemory.py)，prompt 在 [prompt.py:240-246](mas/memory/mas_memory/prompt.py)）。按分数重排取 top-1（默认）。失败轨迹和 insight 不精排，直接截断取 top-k。

每次检索约消耗 LLM 调用：2×successful_topk 次（默认 successful_topk=1 → 2 次）。

**第三步：Insight 检索**

`query_insights_with_score`（[GMemory.py:490-506](mas/memory/mas_memory/GMemory.py)）：

1. 先向量检索找 4 个相关成功任务 + 2 个相关失败任务。
2. 统计每条 insight 的 `positive_correlation_tasks` 里有多少条出现在这 6 个相关任务中，作为"相关分"。
3. 取相关分最高的 top-k 条 insight 返回。

注意：这里判断 insight 是否相关，完全是靠任务名字符串的精确匹配（[GMemory.py:637](mas/memory/mas_memory/GMemory.py)），而不是看 insight 文本本身的语义。这意味着如果任务名称有微小变化，insight 就找不到了。

**第四步：组装进 prompt**

检索结果经 `format_task_prompt_with_insights`（[tasks/mas_workflow/format.py:42-58](tasks/mas_workflow/format.py)）拼成三段式用户 prompt：

```
【Your Own Past Successes】
Task: put a clean egg in microwave  (Status: Success)
Key Steps: 1. Go to fridge and take egg  2. Find sink and clean egg  3. Put egg in microwave
Detailed Trajectory: [压缩版轨迹全文]

【Failure Lessons】
（如果有失败轨迹）
Fail Reason: Agent attempted to place egg before cleaning

【Key Insights】
- Verify the state of objects before and after each action, because agents may overlook...
- After cleaning an item, place it immediately to the designated location...
```

这段内容在每个任务的第一步（solver agent 开始推理前）被拼入 user prompt（[autogen.py:135-140](tasks/mas_workflow/autogen/autogen.py)）。

**第五步：任务结束后更新 insight 的信用分**

`backward`（[autogen.py:201 → GMemory.py:292-297](mas/memory/mas_memory/GMemory.py)）：对本次用过的 insight（记录在 `insights_cache` 里），成功时每条 +1，失败时每条 -2，score ≤ 0 的 insight 被 `clear_insights` 删除。

---

## 六、记忆的长期演化

三种维护机制，触发频率不同：

**每任务触发（写入时）**：把新任务存进 Chroma + 任务图，按需求调 1~2 次 LLM 提炼 key_steps 和 fail_reason。轻量。

**每 5 任务触发（finetune_insights）**（[GMemory.py:647-751](mas/memory/mas_memory/GMemory.py)）：随机抽 5 个历史任务，做两类操作：
- 成功-失败对比 critique：把一对成功/失败任务喂给 LLM，让它发现"这次成功而上次失败是因为什么"，提炼 insight（ADD/EDIT/REMOVE/AGREE 操作）
- 成功批量 critique：把成功任务批量归纳共性规律

LLM 输出 ADD/EDIT/REMOVE/AGREE 操作，`_parse_rules` 正则解析，`_update_rules` 执行（[GMemory.py:792-878](mas/memory/mas_memory/GMemory.py)）：ADD 初始 2 分，AGREE/EDIT +1 分，REMOVE -1 分（库满时 -3），子串查重防止完全重复的 insight 入库。

**每 20 任务触发（merge_insights）**（[GMemory.py:508-549](mas/memory/mas_memory/GMemory.py)）：
- 把全部任务节点 embedding，用 FINCH 聚类自动发现任务族群；
- 同一族群里的 insight 用 LLM 批量合并精简，每 10 条规则合并成 1 条；
- **清空旧 insight 库重建**，score 全部重置为 2，negative_correlation_tasks 清空。

这里有一个设计上的矛盾：backward 积累下来的信用分会在每 20 任务时被 merge_insights 清零。也就是说，每隔 20 个任务，所有 insight 的"历史口碑"就归零一次，与论文"经验随成败长期演化"的叙事有些相悖。

---

## 七、论文宣称 vs 代码实际

下面这些差异对于引用或复现这篇论文非常重要：

**1. "Interaction Graph 捕获 agent 间协作结构"——存了但不用**

`add_agent_node` 确实把每条 AgentMessage 写进 StateChain 并序列化入库（[common.py:75-87, 179](mas/memory/common.py)）。但检索回来后，代码只从 `task_trajectory`（纯文字版轨迹）和 `key_steps` 读数据（[autogen.py:115-117](tasks/mas_workflow/autogen/autogen.py)），从不消费图节点级别的结构数据。论文强调"交互图能捕获 agent 间协作关系"——这在写入侧是真的，在读取侧是假的。

**2. "Insight Graph"——代码里没有图**

论文图示里 insight 节点之间有边（contextualizes 关系，论文 Equation 10），代码是一个扁平的 `list[dict]`（[GMemory.py:477](mas/memory/mas_memory/GMemory.py)），insight 之间没有任何边，关联全靠 `positive_correlation_tasks` 字符串精确匹配隐式表达。

**3. "Agent-specific memory"——默认关闭**

论文宣传的卖点之一：为不同角色的 agent 提供定制化记忆。代码里有 `project_insights`（[GMemory.py:304-350](mas/memory/mas_memory/GMemory.py)）实现了，但 `--use_projector` 默认 False，run 脚本也没传该 flag。默认情况下，所有角色拿的是同一份 insight。

**4. 失败轨迹"参与检索"——实际上不进 prompt**

三种 MAS 拓扑的代码都是 `successful_trajectories, _, insights = retrieve_memory(...)`，用 `_` 忽略了失败轨迹的返回值。`--failed_topk` 默认为 0。失败轨迹仅在离线的 `finetune_insights` 对比 critique 里用到，不直接进任务 prompt。

**5. DyLAN 拓扑下 Insight 反馈失效**

[dylan.py:259-264](tasks/mas_workflow/dylan/dylan.py) 任务结束只调 `save_task_context`，**没有 `backward`**，insight 分数永远不更新。DyLAN 路径上的信用分机制是死代码。

**6. k-hop 扩展经常被旁路**

当候选不足时，[GMemory.py:150-153](mas/memory/mas_memory/GMemory.py) 用普通向量检索结果覆盖任务图扩展结果（而不是补充）。加上后面合并循环里的 `if doc not in true_tasks_doc` 永假 bug（[GMemory.py:154-164](mas/memory/mas_memory/GMemory.py)），很多实际运行中"Query Graph + k-hop 扩展"就退化成了纯向量检索。

**与论文一致的部分**：三层写入/检索串联、LLM 双向遍历打分、ADD/EDIT/REMOVE/AGREE 式 insight 演化、成败 backward 加减分——这些核心宣称在 AutoGen/MacNet 路径上是真实运行的。

---

## 八、关键代码位置表

| 机制环节 | 文件路径 | 类/函数 | 行号 | 一句话作用 |
|---|---|---|---|---|
| 记忆单元定义 | mas/memory/common.py | MASMessage / StateChain | 136-191 / 53-133 | 任务级记忆 = 描述+轨迹+label+状态链图 |
| 试内记忆生命周期 | mas/memory/mas_memory/memory_base.py | MASMemoryBase.init/add_agent_node/move/save | 35-68 | 任务上下文构建与落库入口 |
| 写入主流程 | mas/memory/mas_memory/GMemory.py | GMemory.add_memory | 76-111 | 稀疏化→任务图加节点→Chroma 入库→周期维护 |
| 轨迹稀疏化 | mas/memory/mas_memory/GMemory.py | GMemory._extract_mas_message | 244-281 | 删负 reward 步、数字泛化、LLM 提炼 key_steps |
| 任务图建边 | mas/memory/mas_memory/GMemory.py | TaskLayer.add_task_node | 374-402 | 相似度 ≥ 0.7 连边，权重=相似度 |
| k-hop 检索 | mas/memory/mas_memory/GMemory.py | TaskLayer.retrieve_related_task | 404-423 | 向量 top-k + 1-hop 邻域扩展 |
| 检索主流程 | mas/memory/mas_memory/GMemory.py | GMemory.retrieve_memory / _retrieve_memory_raw | 189-241 / 113-187 | 粗检索 2× 候选 + LLM 逐条打分精排 |
| insight 检索 | mas/memory/mas_memory/GMemory.py | InsightsManager.query_insights_with_score | 490-506 | 按相关任务命中 positive_correlation_tasks 计票 |
| insight 蒸馏 | mas/memory/mas_memory/GMemory.py | InsightsManager.finetune_insights / _finetune_insights | 647-751 | 随机采样任务做对比/批量 critique |
| 规则操作执行 | mas/memory/mas_memory/GMemory.py | InsightsManager._update_rules | 808-878 | 查重、改分、增删规则 |
| insight 合并 | mas/memory/mas_memory/GMemory.py | InsightsManager.merge_insights / TaskLayer.cluster_tasks | 508-549 / 425-454 | FINCH 聚类后按簇 LLM 合并、重建 insight 库 |
| 成败反馈 | mas/memory/mas_memory/GMemory.py | GMemory.backward / InsightsManager.backward | 292-297 / 575-582 | 用过的 insight 成功 +1/失败 -2 |
| 遗忘 | mas/memory/mas_memory/GMemory.py | InsightsManager.clear_insights | 584-586 | 删除 score ≤ 0 的规则 |
| 进 prompt | tasks/mas_workflow/format.py | format_task_prompt_with_insights | 42-58 | 成功轨迹+key_steps+insight 拼三段式 prompt |
| MAS 调用闭环 | tasks/mas_workflow/autogen/autogen.py | AutoGen.schedule | 75-203 | retrieve→执行→save→backward |

---

## 九、研究价值判断

**值得借鉴的设计**：

- **轨迹稀疏化三件套**：删负 reward 步 + 数字泛化 + LLM 提炼 key_steps，三段式进 prompt（task description + key_steps + 详细轨迹）比直接塞原始轨迹干净得多，且 key_steps 只需写入时算一次、检索时零成本复用。

- **Insight 信用分机制**：每条 insight 带 score 和正/负支撑任务集，成功 +1/失败 -2、score ≤ 0 淘汰，把"这条经验到底有没有用"做成可量化的在线反馈回路。失败惩罚大于成功奖励的不对称设计也有道理（一次失败的危害大于一次成功的收益）。

- **分层触发的维护节奏**：每任务轻量写入 → 每 5 任务提炼 insight → 每 20 任务聚类合并，把昂贵的 LLM 整合操作摊薄到周期任务里，控制写放大。

**实现上的粗糙之处**（做研究时需注意）：

- `merge_insights` 每 20 任务清空重建全部 insight，score 一律重置，与长期演化叙事相悖。
- insight 与任务关联用 task_main 全文精确字符串匹配，脆弱，任务名称有微小变化就失效。
- `sort_and_filter_by_similarity` 对每个候选文档重新调 embedding，不复用 Chroma 已存向量，效率差。
- 多处存在无效代码：`sorted()` 不接收返回值（[GMemory.py:606-607](mas/memory/mas_memory/GMemory.py)）；LLM 打分用 `re.search(r'\d+')` 取第一个数字且不校验范围（[GMemory.py:227](mas/memory/mas_memory/GMemory.py)）。
- k-hop 图扩展经常被静默旁路为纯向量检索（[GMemory.py:150-156](mas/memory/mas_memory/GMemory.py) 的覆盖逻辑 + 后续循环永假 bug）。

**对研究方向的启示**：

G-Memory 的核心贡献是验证了"对 MAS 轨迹做层级抽象确实有效"，但实现上三层之间的真正互联（尤其 insight 图的边结构和交互图的节点级消费）并未落地。这是一个潜在的改进空间：如果真正利用 agent 级别的协作图结构（谁影响了谁、哪段对话最关键），检索精度应该能进一步提升。另外，insight 查找靠精确字符串匹配的问题也值得解决，换成向量相似度检索 insight 文本本身是直接的改进方向。
