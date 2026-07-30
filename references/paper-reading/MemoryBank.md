# MemoryBank

- 仓库：https://github.com/zhongwanjun/MemoryBank-SiliconFriend
- commit：cf61c41（本文所有行号仅在该 commit 下有效）
- stars：432（2026-06-11 查询）
- 论文：MemoryBank: Enhancing Large Language Models with Long-Term Memory（arXiv:2305.10250）

---

## 这篇论文为什么出现

2023 年初，ChatGPT、ChatGLM 这类大语言模型（LLM）已经能聊出非常流畅的对话，但它们有一个根本性限制：**每次对话都是全新的开始**。这是因为 LLM 在生成回复时，只能"看到"你当前这次对话里填进去的文字，超出这个范围的历史就完全看不见了。这个范围有多大？当时大约是几千到一万多个词（称为"上下文窗口"），相当于几轮对话的量级。

这对随便聊聊没什么影响，但一旦场景需要长期互动——比如你把 AI 当作每天陪你聊天的伴侣，或者作为记录你日常状态的心理咨询助手——这个问题就非常明显了：它永远不记得你昨天说过什么，每次见面都像陌生人。

中山大学的 Wanjun Zhong 等人针对这个场景提出了 **MemoryBank**：给 LLM 外挂一个长期记忆模块，让它能跨越多次会话记住你说过的事、总结你的个性，以及模拟人类的"遗忘规律"来决定哪些记忆应该被保留或淡忘。他们同时用这套机制构建了一个伴侣聊天机器人 **SiliconFriend**，用来验证方案的实际效果。

---

## 论文提出的方案：三个支柱

MemoryBank 由三个组件组成（论文 §2，Figure 1）：

### 1. 记忆存储（Memory Storage）

存什么？存三种东西，按层次叠加：

- **原始对话**：每一轮"用户说了什么 / AI 回了什么"都按日期存下来，带时间戳。
- **每日事件摘要**：让 LLM 把当天所有对话压缩成一段"今天发生了什么事"的摘要，再把多天的摘要进一步合并成一个跨天的全局摘要（Global Summary）。
- **用户画像**：让 LLM 分析"从对话里看出这个用户是什么性格、情绪状态如何"，同样先生成每日分析，再汇总成全局性格描述（Global Personality）。

这个层次设计的逻辑是：原始对话保留细节（但多了会占空间），摘要压缩信息便于快速检索，性格描述则给 AI 提供"用户是什么样的人"这条元信息。

### 2. 记忆检索（Memory Retrieval）

用一个叫 **FAISS 向量检索**的技术来做。要理解这个，需要先理解两个概念：

**嵌入（Embedding）**：是一种把文字转成数字向量的技术。你可以把每段文字想象成在一个多维空间里的一个点，语义相近的文字对应的点会靠近。例如"我在学 Python"和"我想学编程"对应的点会比较近，而"今天天气不好"和"我在学 Python"对应的点就远了。

**FAISS**：是 Facebook 开发的一个专门用于"在大量向量里快速找出最相近的几个"的工具库。类比就是：你有一个巨大的图书馆，FAISS 帮你在几毫秒内找出和你想找的内容最相关的几本书。

MemoryBank 的检索流程：事先把每轮对话和每日摘要都通过嵌入模型转成向量，存进 FAISS 索引。用户每次说话，就拿用户这句话的向量去 FAISS 里搜最相近的几条记忆，取出原文，拼进 AI 的对话 prompt 里——这样 AI 在回复时就能"看见"这些过去的记忆。

### 3. 记忆更新/遗忘机制（Memory Updating）

论文的亮点之一：受**艾宾浩斯遗忘曲线**启发的记忆衰减机制。

什么是艾宾浩斯遗忘曲线？这是 19 世纪德国心理学家赫尔曼·艾宾浩斯发现的：人类的记忆会随时间指数级衰减——刚学完记得最清楚，一天后忘掉大半，但如果反复复习，遗忘速度会变慢，记忆可以被"巩固"。

论文的数学模型（§2.3）：
```
R = e^(−t/S)
```
其中 R 是保留概率（0到1，越高越难忘），t 是距上次复习的天数，S 是记忆强度（初始为1，被检索到一次就 +1，t 重置为0）。直觉：越是经常被"回想起"的记忆（比如某个经常被提到的事件），S 越大，分母越大，e 的指数越接近0，所以 R 越接近1——即越不容易忘。

---

## 实验验证了什么

论文做了定性和定量两种分析（§4）。

**定量分析**（关键数字）：模拟了 15 个不同性格的虚拟用户，每人积累 10 天的对话记忆，然后提 194 个记忆探针问题（例如："我上次推荐给你的书叫什么？"），评估 SiliconFriend 是否能正确召回记忆并回答。

评估维度：
1. 记忆检索准确率（能不能找到相关记忆）
2. 回答正确性（答案对不对）
3. 上下文连贯性（回答是否自然）
4. 三个版本（ChatGPT/ChatGLM/BELLE）的相对排名

主要结论（Table 2）：
- SiliconFriend ChatGPT 在回答正确性（英文 0.716）和连贯性（英文 0.912）上最好，整体最优；
- 三个版本在**记忆检索准确率**上都相近（0.76-0.86），说明 MemoryBank 的检索机制对不同底座模型都有效；
- ChatGLM 和 BELLE 中文表现略好，ChatGPT 和 ChatGLM 英文表现略好；
- ChatGLM/BELLE 的回答质量不如 ChatGPT，作者认为主要因为底座模型能力差距，而非记忆机制的问题。

**定性分析**：展示了实际对话例子，SiliconFriend 能做到：（1）对用户情绪给出共情式回应；（2）在几天后被问到时还能准确回忆之前推荐的书名和代码；（3）根据对用户性格的总结，给出个性化建议（如"你喜欢探索新文化，可以去参加烹饪课"）。

论文作者想证明的核心点：给 LLM 外挂记忆模块是可行的，不依赖底座模型架构修改，且能显著提升长期陪伴场景的用户体验。艾宾浩斯遗忘机制则是作者认为未来 AI 要更像人的重要方向（虽然代码实现有问题，后文会讲）。

---

## 一条记忆在代码里长什么样

这里用一个具体例子，走完"一条对话怎么变成一条记忆"的全过程。

**场景**：用户 Gary 在 2023-05-03 和 AI 聊了压力管理和电影推荐。

**Step 1：原始对话保存进 JSON**

每轮对话在 `utils/memory_utils.py:save_local_memory` 追加到 JSON 文件里（`memories/update_memory_0512_eng.json`）：

```json
{
  "Gary": {
    "history": {
      "2023-05-03": [
        {
          "query": "I've been feeling stressed lately. Do you have good ways to relieve stress?",
          "response": "There are many ways: moderate exercise, listening to music, reading, talking to friends..."
        },
        {
          "query": "What movies would you recommend?",
          "response": "I'd recommend 'The Shawshank Redemption'..."
        }
      ]
    },
    "summary": {},
    "personality": {},
    "overall_history": "",
    "overall_personality": ""
  }
}
```

**此刻的记忆条目**：两个字段 `query` + `response`，纯原始文本，没有 embedding，没有任何额外处理。

**Step 2：触发摘要（手动，非自动）**

用户在 app 界面点"总结记忆"按钮，或者第二天登录时选 yes，系统调 `memory_bank/summarize_memory.py:summarize_memory` 用 gpt-3.5-turbo 生成：

```json
"summary": {
  "2023-05-03": {
    "content": "Gary shared stress relief methods including exercise, music, and reading. Also discussed movie recommendations."
  }
},
"personality": {
  "2023-05-03": {
    "content": "Gary appears to be someone experiencing work pressure. He is decisive and straightforward, and open to practical advice."
  }
},
"overall_personality": "Gary is decisive and straightforward, enjoys racing, chess, and painting. He is helpful and responds well to practical, actionable suggestions."
```

**Step 3：重建 FAISS 索引（下次登录时触发）**

用户下次登录，系统在 `utils/memory_utils.py:enter_name` 删掉旧索引目录，重新建索引。`memory_bank/memory_retrieval/local_doc_qa.py:JsonMemoryLoader.load` 遍历 JSON，把每一轮对话和每日摘要分别转成 langchain 的 Document 对象：

```python
# 一轮对话 → 一个 Document
Document(
    page_content="[User]: I've been feeling stressed lately... [AI]: There are many ways: exercise, music...",
    metadata={"date": "2023-05-03", "type": "conversation"}
)

# 当日摘要 → 一个 Document
Document(
    page_content="Gary shared stress relief methods including exercise, music, and reading. Also discussed movies.",
    metadata={"date": "2023-05-03", "type": "summary"}
)
```

然后用本地 HuggingFace 嵌入模型（英文用 MiniLM-L6，中文用 text2vec，`configs/model_config.py:20-21`）把每个 Document 的文本转成向量，全部存进 FAISS 索引。

**Step 4：Gary 七天后问问题**

Gary 在 2023-05-10 问："你之前给我推荐的缓解压力的方法是什么？"

- 这句话经过同样的嵌入模型变成向量
- 去 FAISS 索引里找最接近的 k=2 条记忆（默认值在 `utils/sys_args.py:5`）
- 找到了 2023-05-03 的那条对话 Document
- 还有一个"同日扩展"逻辑（`local_doc_qa.py:135-178`）：命中某条后，会把同一天的相邻对话也拼进来，直到凑满 200 字符——目的是还原上下文，避免把一段连续对话切断

**Step 5：拼进 prompt**

检索结果和 `overall_personality` 填进 meta_prompt 模板（`prompt_utils.py:14-23`），大概是：

```
你将扮演 Gary 的 AI 伴侣。
你想起的最相关[回忆]是：
"[User]: I've been feeling stressed... [AI]: exercise, music, reading...
记忆日期：2023-05-03"
Gary 的性格及你的回复策略：decisive and straightforward, open to practical advice...
以下是多轮对话：
[User]: 你之前给我推荐的缓解压力的方法是什么？
[AI]: ...（模型在这里生成回复）
```

**从输入到 prompt 的变换总结**：

| 阶段 | 数据形态 | 何时发生 |
|------|---------|---------|
| 原始对话 | `{query, response}` 对 | 每轮对话后，自动写入 JSON |
| 事件/性格摘要 | LLM 生成的摘要字符串 | 用户手动触发或登录时选 yes |
| FAISS 向量 | 每段文本对应一个浮点数向量 | 每次登录重建，全量处理 |
| 检索结果 | 原始文本（非向量），按日期分组 | 每轮对话检索一次 |
| 进入 prompt | 回忆文本 + 日期 + 全局性格，填模板 | 每轮生成回复前 |

如果开启了遗忘机制，每条对话还额外带两个字段：`memory_strength`（强度，初始为1）和 `last_recall_date`（上次被检索的日期），检索命中后会自动 +1 并落盘。

---

## 源码如何运转

整个项目没有"记忆类"这个抽象对象，**全部状态就是一个 JSON 文件**。两条主线：ChatGLM/BELLE 版（重点分析）和 ChatGPT 版（用 llama_index，逻辑类似，不做详细分析）。

### 写入路径：从对话到 JSON

每轮对话结束后，`utils/memory_utils.py:save_local_memory`（行 72-87）做一件很简单的事：把 `{query, response}` 对 append 进当天日期对应的列表，然后**把整个 JSON 文件重写一遍**。

为什么整文件重写？这是最简单的实现方式，不需要处理文件并发问题，代价是用户对话越多，每次写入越慢。此时 FAISS 索引不更新——新对话无法被检索。

### 摘要路径：LLM 批量压缩

当用户触发摘要（app 界面的按钮 `app_demo.py:258,379`，或 CLI 登录确认 `cli_demo.py:179-180`），`memory_bank/summarize_memory.py:summarize_memory`（行 109-147）：

1. 遍历所有日期，跳过已有 summary 的日期（`his_flag/person_flag` 控制，行 128-129）——增量处理
2. 对每个新日期：调两次 gpt-3.5-turbo，分别生成**当日事件摘要**（行 133）和**当日性格分析**（行 136）
3. 完成后再调两次：全量重生成 `overall_history`（行 141）和 `overall_personality`（行 142）
4. 全部写回 JSON

LLM 调用成本：2 × 新日期数 + 2（全局摘要每次全重生成）。

### 建索引路径：登录时全量重建

`utils/memory_utils.py:enter_name`（行 13-41）是整个系统最重要的"初始化"函数：

1. 删掉旧索引目录（`shutil.rmtree`，行 27）
2. 调 `init_memory_vector_store`（`local_doc_qa.py:196-255`）：读 JSON，用 `JsonMemoryLoader.load`（行 25-61）把每轮对话和每日摘要各转成一个带日期 metadata 的 Document，全量做嵌入，建 FAISS 索引，保存到磁盘
3. 加载用户配置和记忆对象

这意味着：每次用户登录，不管有没有新数据，都要把全部历史重新算一遍嵌入——计算量随历史长度线性增长。但嵌入用的是本地 HuggingFace 模型（不花 API 费用），速度尚可。

### 检索路径：每轮 top-k

用户每说一句话，`build_prompt_with_search_memory_chatglm_app`（`prompt_utils.py:101-137`）调 `search_memory`（`local_doc_qa.py:263-288`）：

1. 用用户输入原文（不做任何改写）做 query，算嵌入向量
2. FAISS top-k 检索（k=2，`sys_args.py:5`）
3. **同日扩展补丁**（`local_doc_qa.py:135-178`）：找到某条 chunk 后，沿着时间序向前向后翻同一天的相邻 chunk，拼接到 200 字符为止——这是个 monkey-patch，直接替换了 langchain FAISS 对象上的方法，目的是"如果这段对话和你相关，那当天前后的对话也可能相关"，类似于把被切开的段落拼回去
4. 结果按日期分组，拼成回忆文本 `related_memory_content` 和日期串 `memo_dates`

同时，`overall_personality` 从 JSON 直接读（`prompt_utils.py:123`），包装成 `personality`。

这三者填进 meta_prompt 模板，交给 LLM 生成回复。

### 遗忘路径：默认关闭

`memory_bank/memory_retrieval/forget_memory.py` 的 `MemoryForgetterLoader` 版本通过 `--enable_forget_mechanism` 开关（`sys_args.py:10`）控制是否 import。

它在 `initial_load_forget_and_save`（行 83-148）里对所有对话条目算保留概率，用 `random.random()` 随机决定是否删除，删除后直接覆写原始 JSON——是真删、不可恢复的。

被检索命中的记忆通过 `update_memory_when_searched`（行 63-71）强化：`memory_strength += 1`，`last_recall_date` 更新，立即落盘（行 353-354）。

---

## 论文宣称 vs 代码实际

以下几点偏差对理解和引用这篇论文很重要：

### 1. 艾宾浩斯遗忘曲线：论文重点宣传，代码近似摆设

论文 §2.3 花了大篇幅描述遗忘曲线，把它作为"更像人类记忆"的核心卖点之一。

代码现实：
- **默认关闭**：`sys_args.py:10` 的 `enable_forget_mechanism` 默认 False
- **启动脚本无一开启**：`launch_belle_cmd.sh:9` 显式传 False，`launch_chatglm_app.sh` 根本不传该参数
- **CLI 分支 import 失败**：`cli_demo.py:53` 开遗忘时 import 的是 `forget_memory_new`，但这个文件在仓库里不存在，直接 ImportError
- **公式写反**：论文公式是 `R = e^(−t/S)`，S 在分母，S 越大衰减越慢（记忆越强久越不容易忘）。代码 `forget_memory.py:36` 写的是 `math.exp(-t / 5*S)`，Python 运算优先级下等于 `exp((-t/5) * S) = exp(-t·S/5)`，S 在分子指数上，**S 越大忘得越快**——与论文的 docstring 和逻辑都相反

**结论**：遗忘机制的删除代码本身是存在的，强化逻辑也写了，但默认路径绕开它，实际演示实验（quantitative analysis）用的是无遗忘版本，论文说的"遗忘与巩固闭环"在代码里基本是展示性存在。如果你要引用这篇论文的遗忘机制作为对比基准，需要知道实验数据并没有打开这个开关。

### 2. 全局事件摘要进 prompt：论文描述与代码实现不符

论文 §3 写道，检索后会组织"relevant memory、global user portrait、global event summary"进 prompt。

代码实际：`overall_history` 被生成（`summarize_memory.py:141`），也被包装进 `history_summary` 变量（`prompt_utils.py:117`）传给 `.format()`，但所有 meta_prompt 模板（`prompt_utils.py:14-23`）里都没有 `{history_summary}` 这个占位符。Python 的 `str.format()` 会静默忽略多余的参数，不报错。

所以 `overall_history` 唯一的实际作用是决定用"老用户模板"还是"新用户模板"（`prompt_utils.py:132`）——它存在，它被生成，但它的内容从未被任何 LLM 看到。只有 `overall_personality`（全局性格）是真正进 prompt 的。

### 3. "持续演化"：论文宣称是自动的，代码是手动触发

README 说"continually evolve through memory updates"，暗示记忆会自动更新演化。

代码里，摘要和性格分析需要用户主动点按钮或在登录时手动确认，新对话在下次登录重建索引之前也无法被检索到。并不是"持续"。

### 4. 与实现一致的宣称

- 基于 FAISS 的向量检索：真实，代码完整实现
- 用户画像（overall_personality）进 prompt：真实，且是主要的个性化来源
- 每日事件摘要分层结构：真实，已摘要日期跳过节省成本
- 被检索到的记忆得到强化（遗忘版）：代码写了，逻辑正确，只是默认不启用

---

## 这套设计的问题和值得借鉴之处

### 设计问题

**全量重建索引**：每次登录 rmtree 删掉旧索引再从零建，整文件重写每轮对话——历史越积越长，登录延迟越高。2023 年的脚本项目，没有考虑增量更新。

**检索粒度粗糙**：用户输入原文直接做 query，没有查询改写，也没有按类型分库检索（对话和摘要混在一个 FAISS 索引里）。同日扩展逻辑是个 monkey-patch 补丁，用"拼到 200 字符"的启发式方法代替真正的语义段落边界。

**大段复制**：`forget_memory.py` 和 `local_doc_qa.py` 是两份几乎相同的 `LocalMemoryRetrieval`，改一处忘一处，`cli_demo.py:53` 的 `forget_memory_new` 幽灵模块即此类产物。代码里 `for k in range(...)` 的循环变量 `k` 还覆盖了形参 top-k 的 `k`（`local_doc_qa.py:138,151`）。

**遗忘触发边界**：登录时对**所有用户**遗忘，不是只对当前登录用户（过滤代码被注释掉，`forget_memory.py:88-89`）；用 `random.random()` 导致行为不可复现；如果某日 summary 在 JSON 里不存在而对话被全删，会触发 KeyError（行 131）。

### 值得借鉴之处

**记忆强度 + 最近召回日期的元数据设计**：每条记忆只需两个额外字段（`memory_strength`、`last_recall_date`）就能实现"use it or lose it"（常用不忘、久不用则忘）的闭环。这是后来很多项目"访问频次打分"思路的早期形态。公式修一下括号即可用。

**日期锚定 + 检索后按日期分组**：把记忆的日期作为 metadata 保留，prompt 里明确给出回忆的日期（`memo_dates`），让模型可以直接回答"你在 X 月 X 日做了什么"，对时序类 benchmark 非常重要。

**事实记忆与用户画像分轨**：`summary`（发生了什么）和 `personality`（用户是什么人）分开生成，分开进 prompt。`personality` 进 prompt 后既作为"背景知识"也作为"回复策略指令"，比只存事实的系统多了一层用法——这是伴侣类应用区别于纯知识检索的关键设计。

**按日期增量摘要**：已摘要的日期跳过（`his_flag/person_flag`），只处理新日期，控制了 LLM 调用成本；全局摘要每次全重生成，成本固定不随历史增长。这个权衡是个合理的工程选择。

---

## 关键代码位置

| 机制环节 | 文件路径 | 函数/行号 | 对应论文概念 |
|---------|---------|---------|------------|
| 每轮写入 | utils/memory_utils.py:72-87 | save_local_memory | §2.1 对话存储 |
| 每日/全局摘要 | memory_bank/summarize_memory.py:109-147 | summarize_memory | §2.1 Hierarchical Event Summary + Personality |
| 登录建索引 | utils/memory_utils.py:13-41 | enter_name | §2.2 预编码记忆到向量 |
| 记忆→Document | memory_bank/memory_retrieval/local_doc_qa.py:25-61 | JsonMemoryLoader.load | §2.2 每轮/每日摘要 → memory piece |
| 建 FAISS 索引 | memory_bank/memory_retrieval/local_doc_qa.py:196-255 | init_memory_vector_store | §2.2 FAISS 索引 M |
| 检索 | memory_bank/memory_retrieval/local_doc_qa.py:263-288 | search_memory | §2.2 top-k 检索 |
| 同日扩展补丁 | memory_bank/memory_retrieval/local_doc_qa.py:135-178 | similarity_search_with_score_by_vector | 代码自创，论文未提 |
| 拼 prompt | utils/prompt_utils.py:101-137 | build_prompt_with_search_memory_chatglm_app | §3 prompt 组装 |
| meta_prompt 模板 | utils/prompt_utils.py:13-24 | generate_meta_prompt_dict_chatglm_app | §3 memory augmented prompt |
| 遗忘公式（有 bug） | memory_bank/memory_retrieval/forget_memory.py:20-36 | forgetting_curve | §2.3 R=e^(-t/S)，括号缺失 |
| 执行遗忘 | memory_bank/memory_retrieval/forget_memory.py:83-148 | initial_load_forget_and_save | §2.3 遗忘并删除记忆 |
| 记忆强化 | memory_bank/memory_retrieval/forget_memory.py:63-71 | update_memory_when_searched | §2.3 Spacing Effect |
| 遗忘开关 | utils/sys_args.py:10 | DataArguments.enable_forget_mechanism | 默认 False，实验未开启 |
