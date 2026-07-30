# MemEvolve

- 仓库：https://github.com/bingreeky/MemEvolve
- commit：6035d56（本文所有行号仅在该 commit 下有效）
- stars：241（2026-06-11 查询）
- 对应论文：MemEvolve: Meta-Evolution of Agent Memory Systems（arXiv 2512.18746，ICML'26，OPPO AI + LV-NUS Lab）
- 代码全部在 `Flash-Searcher-main/` 子目录下，本文相对路径均以该目录为根

---

## 一、论文讲了什么

### 背景：agent 记忆系统的演化困境

AI agent（自主智能体）在完成任务时需要"记忆"——就像人做题需要积累经验一样。早期的记忆系统只是把历史轨迹原封不动存下来（相当于学生把做过的题全部复印进笔记本）；后来更先进的系统会从轨迹中提取"心得"、"技巧"、"可复用工具"（相当于学生整理错题、提炼解题模板）。

但这些系统有个共同的根本问题：**记忆的存法是预先设计好、一成不变的**。研究者提前决定"记忆应该是这种形式、用这种方式检索"，然后就固定下来了。

论文用一个人类学习的类比来说明这个问题：
- **普通学生**：见过的题直接硬背，不管做对做错（= 无记忆/只存原始轨迹）
- **优秀学生**：会总结错误、提炼规律（= ExpeL、G-Memory 等现有系统）
- **最顶尖的学生**：会根据科目调整学习方法——背语文选择记原文，学数学则抽象解题框架（= MemEvolve 的目标）

关键洞见是：**没有一种记忆架构对所有任务都最优**。擅长网页浏览任务的记忆系统，在数学推理任务上可能毫无用处。反过来亦然。因此，让记忆系统本身能根据任务动态自适应，是比优化记忆内容更根本的突破。

### 核心提问

> 如何让记忆系统不仅能辅助 agent 积累经验，还能**进化自身的架构**，从而适应不同的任务域？

### 提出的方法：MemEvolve

MemEvolve 的核心思路是**把"记忆系统的 Python 代码"本身当作进化的对象**。它不优化单条记忆的内容，而是分析 agent 的失败案例，让另一个 LLM 重新设计整套记忆系统的代码，然后通过评测选出更好的版本。

具体来说，MemEvolve 由两个嵌套的进化循环组成：

- **内层循环（Experience Evolution，经验进化）**：agent 用当前的记忆系统跑任务、积累经验，记忆库在这个过程中不断填充。这和所有现有系统做的事一样。
- **外层循环（Architecture Evolution，架构进化）**：对多套候选记忆系统评测后，用"诊断-设计"流程生成下一代记忆系统的代码——改的不是记忆内容，而是存记忆、查记忆的 Python 逻辑本身。

### 统一框架：EvolveLab

为了让这套进化有清晰的约束边界，论文把任何记忆系统都抽象成四个模块（Encode→Store→Retrieve→Manage），并将 12 个已有的代表性记忆系统（包括 ExpeL、Voyager、SkillWeaver 等）全部按照这个统一接口重新实现，形成可对比的实验平台 **EvolveLab**。

这样做的好处是：LLM 在设计新的记忆系统时，有明确的"搜索空间边界"——只能修改这四个模块的具体实现，不能乱来；同时 12 个基线可以在完全一致的评测环境下横向对比。

### 实验结果

在 GAIA（通用任务问答）、WebWalkerQA（网页浏览）、xBench（专业领域深度调研）、TaskCraft（合成工具使用任务）四个基准测试上：

- MemEvolve 给 Flash-Searcher + GPT-5-Mini 在 GAIA 上从 69.09% 提升到 73.33%（+4.24%）；给 SmolAgent 在 WebWalkerQA 上最高提升 17.06%
- 在 TaskCraft 上进化出来的记忆系统，**直接迁移**到 WebWalkerQA 和 xBench 仍然有效（+2.4% 到 +9.09%），说明进化出的不是针对单个数据集的技巧，而是更通用的记忆架构原则
- 跨模型迁移：用 GPT-5-Mini 进化出来的记忆系统，直接套到 Kimi K2 和 DeepSeek V3.2 上也有提升
- 进化轨迹分析（Figure 6 in paper）显示，系统自发发现了几个设计原则：更多 LLM 参与的"主动"检索优于死板规则、多层级记忆组织优于单一扁平存储、内嵌工具记忆比纯文本记忆在工具密集型任务上更有效

**与手工设计基线的对比**（Table 3）：现有系统（ExpeL、DILU、Cheatsheet 等）在不同数据集上表现不一致，有时比不加记忆更差；MemEvolve 在三个数据集上全部稳定提升 3.54%–5%，且 API 调用成本与不加记忆的基线相当。

---

## 二、记忆条目设计：一条记忆长什么样

MemEvolve 最终的主力进化产物有三个：**Lightweight Memory**、**Riva**、**Cerebra**，各自有不同的记忆格式。

### Lightweight Memory（代码主力，最轻量）

进化自"从简单 few-shot 轨迹存储"起步，最终演变成一个**双层记忆系统**：

**长期记忆**（跨任务持久）：存为 `storage/lightweight_memory/longterm_memory.json`，包含两类条目：

```json
{
  "strategic": [
    {
      "content": "When a task is making no progress, proactively try alternative sources—sometimes third-party sites are more effective than the primary one.",
      "tags": ["strategy", "web_search", "fallback"],
      "usage_count": 7,
      "success_count": 5,
      "signature": "a3f8e1c2..."
    }
  ],
  "operational": [
    {
      "content": "When web_search fails, try splitting the query into shorter, more specific search terms and retry.",
      "tags": ["web_search", "error_handling"],
      "usage_count": 12,
      "success_count": 9,
      "signature": "b7d2f..."
    }
  ]
}
```

- `strategic`（战略记忆）：关于**何时选择什么方法**、如何分解复杂问题——对应高层规划决策
- `operational`（操作记忆）：关于**工具怎么用、出错怎么处理**——对应具体执行技巧
- `usage_count`：这条记忆被 LLM 选中使用了多少次
- `success_count`：使用这条记忆的任务最终成功了多少次
- `signature`：内容的 sha256 摘要，用于精确去重（不是向量相似度，是字符串哈希）

每类最多 30 条（+ 20 条缓冲区）。条目只有成功任务才写入。

**短期记忆**（任务内临时）：存在内存里（`self.shortterm_memory`），是当前任务执行过程中积累的关键事实清单，任务结束时清空。

**冷启动记忆**（hardcoded）：代码里内置了 5 条 strategic + 2 条 operational 的"默认经验"（如"要直接行动而非过度规划"），在长期库为空时注入，避免冷启动时没有任何引导。

### Cerebra Memory（图结构，更复杂）

记忆条目是**知识图谱的节点**，存为 `storage/cerebra_fusion_memory/cf_database.json`：

```json
{
  "id": "node_042",
  "content": "Semantic search with hybrid TF-IDF + embedding works better for vague queries than keyword search alone.",
  "node_type": "insight",
  "edges": [
    {"target_id": "node_039", "edge_type": "SIMILAR_CONCEPT", "weight": 0.82, "usage_count": 3, "success_count": 2}
  ],
  "usage_count": 5,
  "success_count": 4,
  "task_type": "web_research"
}
```

节点之间建语义边（两个节点 embedding 相似度 ≥0.75 时自动建双向边），检索时可以沿边"扩散"到相邻节点，一定程度上实现知识关联。

此外 Cerebra 还有**工具记忆**：把成功轨迹里的操作模式抽成参数化 Python 函数（如 `def search_wikipedia(title: str) -> str`），存在 `tools_storage.py` 里，检索到后直接注册为可调用工具。

### 写入前经历了什么变换

以 Lightweight 为例，一条任务轨迹到成为长期记忆条目，经过以下阶段：

```
任务轨迹（完整 step-by-step 执行记录）
    ↓ 只有成功轨迹才进入下一步（_is_trajectory_success 判断）
    ↓ LLM 调用（_extract_memories 函数）：
        提示词要求提取"可复用模式"（不是任务特定数据）
        → 输出 JSON: {"strategic": [...], "operational": [...]}
    ↓ sha256 去重（避免重复存入相同内容）
    ↓ 追加进 longterm_memory.json
    ↓ 若超过 30+20=50 条，触发 LLM 剪枝（_intelligent_prune_memories）
```

这里有两个重要的设计选择：
1. **只从成功轨迹抽取**：避免错误经验污染记忆库。代价是失败中学习的信息完全丢失（这与 ExpeL 等系统不同）
2. **提示词强调可复用**：提示词明确要求"不要记录'北京=39.9°N'这类任务特定数据"，只要规律性技巧

---

## 三、源码实现：系统如何运转

### 整体架构

代码分两层：

```
Flash-Searcher-main/
├── EvolveLab/               ← 内层：统一记忆接口 + 13 个记忆系统实现
│   ├── base_memory.py       ← BaseMemoryProvider 抽象基类
│   ├── memory_types.py      ← 数据结构定义 + 动态加载映射表
│   ├── config.py            ← 各记忆系统配置参数
│   └── providers/           ← 13 个具体实现（11 baseline + 2 进化产物）
├── MemEvolve/               ← 外层：进化循环
│   ├── core/auto_evolver.py ← 主进化控制器
│   └── phases/              ← 四阶段：分析→生成→落盘→验证
└── FlashOAgents/agents.py   ← 接入记忆系统的 agent 框架
```

### EvolveLab：统一接口

任何记忆系统都必须继承 `BaseMemoryProvider`（[EvolveLab/base_memory.py:10](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/base_memory.py#L10)），实现三个方法：

- `provide_memory(request: MemoryRequest) → MemoryResponse`：检索记忆
- `take_in_memory(trajectory_data: TrajectoryData) → (bool, str)`：写入记忆
- `initialize() → bool`：初始化（加载已有记忆库）

核心数据结构（[EvolveLab/memory_types.py:72-106](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/memory_types.py#L72)）：
- `MemoryRequest`：检索时传入，包含 `query`（任务问题）、`context`（当前执行上下文）、`status`（`BEGIN` 或 `IN` 两种时机）
- `MemoryItem`：单条记忆，包含 `id`、`content`（文本或可执行代码）、`metadata`、`score`（检索相关度）
- `TrajectoryData`：写入时传入，包含 `query`（任务问题）、`trajectory`（完整执行步骤列表）、`metadata`（含 `is_correct` 判断结果）

`MemoryStatus.BEGIN` 和 `IN` 是两个触发时机：BEGIN 在任务开始规划时，IN 在执行的每个步骤时。不同时机注入不同类型的记忆。

动态加载靠 `PROVIDER_MAPPING`（[memory_types.py:48-63](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/memory_types.py#L48)）：枚举 → (类名, 模块名)。进化系统新增一个 provider 时，就往这个映射表里插一条，代码自动 `import` 并实例化。

### 内层：一条数据从进入到检索的完整路径

以 `lightweight_memory` 为例，假设 agent 正在处理一个 GAIA 任务：

**1. 任务开始（BEGIN 阶段）**

[FlashOAgents/agents.py:414](../../../research/MemEvolve/Flash-Searcher-main/FlashOAgents/agents.py#L414) 调用 `provide_memory(BEGIN)`。

若配置 `enable_longterm_provision=True`（默认 False！），则：
- 把长期记忆库里全部最多 60 条记忆的内容传给 LLM
- LLM 从中选出 top-k 条最相关的，并合成一段 4-5 句话的 guidance
- guidance 以 `————Memory System Guidance————` 包裹，作为 user message 插入 prompt

guidance 示例（来自 paper Figure 7）：
> "Anti-ambiguity: Consider first locating the canonical Wikipedia article with targeted site:wikipedia.org queries... Tool-use Suggestion: Based on similar tasks, use the MediaWiki API/history endpoints to list revisions and apply a cutoff..."

**若 `enable_longterm_provision=False`（默认），长期记忆的写入工作都在白白做，BEGIN 阶段直接跳过**（这是重要的论文-代码偏差，详见第四节）。

**2. 执行中（IN 阶段）**

每个执行步骤结束后，[agents.py:783](../../../research/MemEvolve/Flash-Searcher-main/FlashOAgents/agents.py#L783) 调用 `provide_memory(IN)`。

内部逻辑（[lightweight_memory_provider.py:584-715](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/providers/lightweight_memory_provider.py#L584)）：
- 每步都调用 LLM 抽取"当前步骤的关键事实"（例如：本步发现了 Wikipedia 文章标题、确认了发布日期是 2019 年 5 月），追加到短期记忆列表
- 每隔 5 步（`shortterm_provision_interval`），把短期记忆列表格式化成清单，注入 prompt

注入后 prompt 里的短期记忆大约长这样：
> - Step 3: Located Wikipedia article 'Outer Wilds' at https://en.wikipedia.org/wiki/Outer_Wilds
> - Step 4: Confirmed release date listed as May 2019 (Windows/Xbox One)

**3. 任务结束（写入阶段）**

[run_flash_searcher_mm_gaia.py:319-338](../../../research/MemEvolve/Flash-Searcher-main/run_flash_searcher_mm_gaia.py#L319) 调用 `take_in_memory(trajectory_data)`。

成功与否的判断来自 LLM judge 对比 gold answer（:305-332）——这意味着写入是有 gold answer 做门控的，只在有标注数据的评测场景下才能工作。

若成功，调用 `_extract_memories`（[lightweight_memory_provider.py:993](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/providers/lightweight_memory_provider.py#L993)），提示词要求提取最多 2 条 strategic + 2 条 operational insights；结果去重后追加进 JSON 文件。超过阈值则触发 LLM 剪枝。

### 外层：四阶段进化循环

外层进化由 `AutoEvolver.run()`（[MemEvolve/core/auto_evolver.py:788](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/core/auto_evolver.py#L788)）控制，每轮迭代五步：

**Step 1：收集轨迹**
当前最优记忆系统跑 20 个任务，记录执行日志（哪些步骤成功、失败、用了什么记忆）。

**Step 2：诊断-设计-落盘-验证（针对每个候选系统并行执行）**

- **Analyze（诊断）**：`AnalysisAgent`（[MemEvolve/phases/phase_analyzer.py:51](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/phase_analyzer.py#L51)）是一个 agentic 的分析器，拥有三个工具：TrajectoryViewer（读任务执行日志）、StepViewer（看具体失败的步骤）、MemoryDatabaseViewer（看当前记忆库内容）。它最多 20 步地"调查"，产出一份结构化的缺陷报告，内容按 PROVIDE（检索问题）/ TAKE-IN（写入问题）/ MANAGEMENT（维护问题）三个维度组织。

  例如：*"retrieval 总是返回太多无关的长期记忆，导致 agent 在规划时注意力分散；建议增加 LLM 二次过滤"*

- **Generate（生成）**：单次 LLM 调用，max_tokens=60000，输入是当前 provider 的全部源码 + 缺陷报告，输出是**一份完整的新 provider Python 代码**（不只是改动，是完整文件），同时输出配置参数和枚举名称。temperature 参数会随"创造性预算"动态调整（[phase_generator.py:254](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/phase_generator.py#L254)，公式：`0.3 + 0.9 × creativity`）。

- **Create（落盘）**：把生成的代码写成 `.py` 文件放入 `EvolveLab/providers/`，然后用正则表达式定位注释标记行（如 `# add new memory type upside this line`），直接在 `memory_types.py` 和 `config.py` 的源文件里插入新枚举条目和配置（[memory_creator.py:111-275](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/memory_creator.py#L111)）。

  这是 monkey-patch 式的源码注入——没有 AST 操作，靠字符串查找替换，脆弱但有效。

- **Validate（验证）**：先做 AST 静态检查（必须继承 `BaseMemoryProvider`、实现三个方法）；再在隔离环境里真实 `import` 并模拟调用 provide/take_in；若失败则触发 mini-swe-agent 自动修代码，最多 3 次（[phase_validator.py:197-214](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/phase_validator.py#L197)）。

**Step 3：初赛**
当前最优系统 + 所有新候选系统在同一批任务上并行跑，收集 accuracy / token 成本 / 延迟。

**Step 4：决赛**
初赛 Top-t 晋级，加测（之前任务 + 新采样任务）后再比一轮。

**Step 5：换底**
按 accuracy 降序、tokens 升序排序（[auto_evolver.py:763-782](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/core/auto_evolver.py#L763)）选出胜者，作为下一轮进化的起点（"亲本"）。

论文实验跑了 3 轮迭代，每轮初赛 20 个任务（40 新 + 20 复用），决赛 40 个任务。

**重要细节：每轮清空记忆库**

默认 `clear_storage_per_round=True`（[auto_evolver.py:68](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/core/auto_evolver.py#L68)），每轮开始把上轮积累的记忆内容全部重命名清空，让每套候选系统从空库起步——这样评测的是架构而不是记忆积累量的差异（公平对比）。

### Cerebra：相比 Lightweight 的不同

Cerebra 是另一个进化产物，检索逻辑更复杂：
1. 先用 LLM 把任务 query 改写成更抽象的 `retrieval_focus`
2. 用 TF-IDF（占 0.2 权重）+ 向量语义检索（占 0.8 权重）混合检索。向量检索（embedding）是把文本转成数字向量，两段文本意思越相近，向量越接近，就能检索出来——这里用的是本地的 `sentence-transformers/all-MiniLM-L6-v2` 模型，不调 API。TF-IDF 是基于关键词频率的经典文本匹配方法
3. 沿知识图谱的语义边做一跳扩展，传播分 = 原始分 × 边权重 × 0.7
4. 阈值 0.22 过滤，取 top-3 节点后，再用 LLM 合成压成 ≤350 字的 guidance

---

## 四、论文宣称 vs 代码实际

这里列出阅读时需要注意的几个偏差：

**1. 最重要偏差：长期记忆默认是"只写不读"**

论文图 7 展示了 Lightweight 提供长期记忆 guidance 给 agent 的例子，给人感觉这是默认行为。但代码里默认配置 `enable_longterm_provision: False`（[EvolveLab/config.py:52](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/config.py#L52)；[lightweight_memory_provider.py:113](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/providers/lightweight_memory_provider.py#L113)）。也就是说，默认情况下长期记忆只会在每个任务结束后写入、做剪枝，但从不被读出来注入 prompt。**你跑论文的默认命令，实际上只有任务内短期记忆在生效，跨任务记忆完全不起作用。** 这会严重影响你如何解读论文的实验数字。

**2. "dual-evolution 联合演化"在默认路径下是分时而非共时**

论文概念图给人的印象是记忆内容和架构同时在进化。但实际上每轮进化开始都清空记忆库，进化的只有代码（架构），记忆内容只在每轮内的 20 个任务里短暂积累。两个循环是串行（先内层积累，再外层进化），不是并行共同演化。

**3. 只从成功轨迹学习**

论文提示词模板写 "from all trajectories"，但两个主力进化系统都只处理 `is_correct=True` 的轨迹（[lightweight_memory_provider.py:332-333](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/providers/lightweight_memory_provider.py#L332)；[cerebra_fusion_memory_provider.py:1022-1023](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/providers/cerebra_fusion_memory_provider.py#L1022)）。失败轨迹的信息被完全丢弃。

**4. Cerebra 的记忆整合在默认参数下几乎不会触发**

整合需要累计 50 个成功任务（[config.py:69](../../../research/MemEvolve/Flash-Searcher-main/EvolveLab/config.py#L69)），但默认每轮只有 20 个任务且每轮清空存储，实际上 `_consolidate_memory` 基本是死代码。

**5. 验证阶段文档自相矛盾**

`phase_validator.py:32` 的 docstring 写 "without automatic fixes"，但默认 `enable_auto_fix=True`（:42），失败时确实会调 mini-swe-agent 自动修代码。

**6. Pareto 多目标选择默认关闭**

README 提到 Pareto 选择，实际默认 `use_pareto_selection=False`（[auto_evolver.py:67](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/core/auto_evolver.py#L67)），默认就是 accuracy 第一优先、tokens 次之的字典序排序。

**与论文一致的核心部分**：四阶段 Analyze→Generate→Create→Validate 流水线、锦标赛选择机制、断点续跑、EvolveLab 的 13 个系统统一接口——这些都真实存在且完整。README 自己也承认 "This is not the final release"。

---

## 五、值得注意的几个设计选择

**1. EvolveLab 这套脚手架本身的价值**

统一接口 + 动态注册 + 隔离环境验证，使得"LLM 写一个新记忆系统并立即可评测"成为闭环。这套平台有 13 个系统的统一实现，想做记忆系统横向对比实验可以直接借用，不用自己复现每篇论文的代码。

**2. usage_count / success_count 的轻量效用反馈**

记忆被 LLM 选中即计 usage，任务判对后再给这条记忆加 success_count。剪枝时以"被用过且用了有效"为第一优先级，而不是时间衰减或 LRU。这个机制成本极低（只加计数），但比纯时间衰减更贴近任务效用。

**3. 诊断 prompt 的约束工程**

外层进化的分析 prompt（`analysis_prompt.yaml:96-130`）明确禁止一些"投机取巧"的建议，比如"加关键词硬匹配"——这类建议在小数据集上能提分，但不可泛化。这种对 LLM 设计空间的显式约束，避免了进化系统发现表面有效但脆弱的方案。

**4. 代码实现的脆弱之处（不建议直接用于生产）**

- 用正则解析 LLM 生成的 Markdown、用注释标记行做源码插桩（[phase_generator.py:317-365](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/phase_generator.py#L317)，[memory_creator.py:133-180](../../../research/MemEvolve/Flash-Searcher-main/MemEvolve/phases/memory_creator.py#L133)）：无 AST 操作，无事务回滚，并发不安全
- lightweight 的记忆 ID 用列表下标（`"strategic_3"`，:526-537）：剪枝后下标移位，可能记错 success_count 到其他条目
- cerebra 每写入一次都全量重建 TF-IDF + embedding 索引（:1221, :413-433），O(n²) 的边查找，数据量稍大即不可用
