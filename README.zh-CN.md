# MiniMem

[English](README.md) | [简体中文](README.zh-CN.md)

MiniMem 是一个刻意做小的基线系统，用于在 LoCoMo 风格的问答任务上评测对话记忆。

它实现了一条透明的流水线：

```text
对话
    -> LLM 事实抽取
    -> embedding 余弦相似度
    -> 直接 top-k 检索
    -> LLM 回答
    -> token F1、BLEU-1 与 LLM Judge
```

## 评测结果

以下参考数字来自每个 benchmark 的一次完整运行，配置为
`deepseek-v4-flash`（关闭思考模式）、`all-MiniLM-L6-v2` 向量模型、
`TOP_K = 15`。三个 benchmark 共用同一套分词器（LoCoMo-Refined 的
打分器），因此词面指标的计算方式完全一致；但由于问题风格和标准答案
格式不同，绝对数值仍不能跨数据集直接比较。词面 F1 和 BLEU-1 仅作
合理性检查；Judge 列是 LLM 评审的准确率，使用同一个 DeepSeek 模型运行。
LongMemEval 使用论文官方的评审 prompt 原文，两个 LoCoMo 变体使用本仓库
自己的 CORRECT/WRONG prompt（见各自的 `evaluate_answer.py`），因此
Judge 数字不能与已发表的结果直接对比。

**LoCoMo**（1,540 道非对抗性问题）：

| 类别 | 题数 | F1 | BLEU-1 | Judge |
| --- | ---: | ---: | ---: | ---: |
| Single-hop | 841 | 38.20 | 31.74 | 63.97% |
| Multi-hop | 282 | 30.06 | 23.20 | 46.45% |
| Open-domain | 96 | 18.74 | 12.49 | 31.25% |
| Temporal | 321 | 32.62 | 27.69 | 60.12% |
| **Overall** | **1,540** | **34.33** | **28.13** | **57.92%** |

**LoCoMo-Refined**（1,382 道问题）：

| 类别 | 题数 | F1 | BLEU-1 | Judge |
| --- | ---: | ---: | ---: | ---: |
| Single-hop | 802 | 41.15 | 34.28 | 66.46% |
| Multi-hop | 213 | 33.43 | 26.07 | 61.03% |
| Open-domain | 68 | 35.58 | 27.35 | 57.35% |
| Temporal | 299 | 38.33 | 33.46 | 57.86% |
| **Overall** | **1,382** | **39.07** | **32.50** | **63.31%** |

**LongMemEval-Oracle**（500 道问题；Judge 是官方指标——preference 类
问题的标准答案是一份评分细则，30 道 abstention 问题考察的是拒答，
这两类的词面指标没有参考意义）：

| 题型 | 题数 | F1 | BLEU-1 | Judge |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 65.14 | 54.44 | 94.29% |
| single-session-assistant | 56 | 59.84 | 51.85 | 75.00% |
| single-session-preference | 30 | 9.23 | 2.46 | 36.67% |
| multi-session | 133 | 53.21 | 47.25 | 73.68% |
| knowledge-update | 78 | 51.29 | 41.56 | 79.49% |
| temporal-reasoning | 133 | 42.33 | 28.03 | 68.42% |
| **Overall** | **500** | **49.79** | **40.08** | **74.00%** |

abstention 子集（30 题，已计入上表各行）在专用的拒答评审 prompt 下
得分 96.67%。

## 开源记忆系统研究

在界定 MiniMem 的范围时，我们调研了以下开源论文与项目。

| 论文 / 项目 | 主要方向 | 论文 | 代码 |
| --- | --- | --- | --- |
| Mem0 | 面向生产环境的长期记忆抽取与检索。 | [PDF](https://arxiv.org/pdf/2504.19413) | [GitHub](https://github.com/mem0ai/mem0) |
| Zep | 基于 Graphiti 的时序知识图谱记忆。 | [PDF](https://arxiv.org/pdf/2501.13956) | [GitHub](https://github.com/getzep/graphiti) |
| MemGPT | 受操作系统启发的上下文与记忆管理。 | [PDF](https://arxiv.org/pdf/2310.08560) | [GitHub](https://github.com/letta-ai/letta) |
| MemOS | 参数记忆、激活记忆与明文记忆的统一管理。 | [PDF](https://arxiv.org/pdf/2505.22101) | [GitHub](https://github.com/MemTensor/MemOS) |
| MIRIX | 跨六种记忆类型的多智能体协同。 | [PDF](https://arxiv.org/pdf/2507.07957) | [GitHub](https://github.com/Mirix-AI/MIRIX) |
| SimpleMem | 结构化压缩、语义合成与意图感知检索。 | [PDF](https://arxiv.org/pdf/2601.02553) | [GitHub](https://github.com/aiming-lab/SimpleMem) |
| LangMem | 面向 LangChain 与 LangGraph 智能体的长期记忆 SDK。 | — | [GitHub](https://github.com/langchain-ai/langmem) |
| MemoryOS | 分层的短期、中期与长期个人记忆。 | [PDF](https://arxiv.org/pdf/2506.06326) | [GitHub](https://github.com/BAI-LAB/MemoryOS) |
| A-MEM | 受卡片盒笔记法启发、带笔记链接的智能体记忆。 | [PDF](https://arxiv.org/pdf/2502.12110) | [GitHub](https://github.com/agiresearch/A-mem) |
| LightMem | 低成本更新的轻量多阶段记忆。 | [PDF](https://arxiv.org/pdf/2510.18866) | [GitHub](https://github.com/zjunlp/LightMem) |
| MemoryBank | 带艾宾浩斯遗忘模型的长期对话记忆。 | [PDF](https://arxiv.org/pdf/2305.10250) | [GitHub](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) |
| G-Memory | 面向多智能体协作的层级图记忆。 | [PDF](https://arxiv.org/pdf/2506.07398) | [GitHub](https://github.com/bingreeky/GMemory) |
| MemEvolve | 模块化记忆架构的元进化。 | [PDF](https://arxiv.org/pdf/2512.18746) | [GitHub](https://github.com/bingreeky/MemEvolve) |
| Nemori | 受事件分割启发的自组织记忆。 | [PDF](https://arxiv.org/pdf/2508.03341) | [GitHub](https://github.com/nemori-ai/nemori) |
| SuperLocalMemory | 本地优先、带投毒防御的多智能体记忆。 | [PDF](https://arxiv.org/pdf/2603.02240) | [GitHub](https://github.com/qualixar/superlocalmemory) |
| SuperLocalMemory V3 | 基于信息几何的零 LLM 企业级记忆。 | [PDF](https://arxiv.org/pdf/2603.14588) | [GitHub](https://github.com/qualixar/superlocalmemory) |
| LatentMem | 面向多智能体系统的角色化潜在记忆组合。 | [PDF](https://arxiv.org/pdf/2602.03036) | [GitHub](https://github.com/KANABOON1/LatentMem) |
| MemP | 程序性记忆的构建、检索与更新。 | [PDF](https://arxiv.org/pdf/2508.06433) | [GitHub](https://github.com/zjunlp/MemP) |
| MemOCR | 面向长程推理的版面感知视觉记忆。 | [PDF](https://arxiv.org/pdf/2601.21468) | [GitHub](https://github.com/meituan/MemOCR) |
| E-mem | 多智能体情景上下文重建。 | [PDF](https://arxiv.org/pdf/2601.21714) | [GitHub](https://github.com/dog-last/E-mem) |
| JitRL | 基于检索经验的测试时持续学习。 | [PDF](https://arxiv.org/pdf/2601.18510) | [GitHub](https://github.com/liushiliushi/JitRL) |
| BudgetMem | 跨记忆预算档位的查询感知路由。 | [PDF](https://arxiv.org/pdf/2602.06025) | [GitHub](https://github.com/ViktorAxelsen/BudgetMem) |
| RF-Mem | 基于熟悉度与回忆双通路的自适应检索。 | [PDF](https://arxiv.org/pdf/2603.09250) | [GitHub](https://github.com/Zhang-Yingyi/ICLR2026_RF-Mem) |
| MAD-M² | 面向多智能体辩论的记忆掩码。 | [PDF](https://arxiv.org/pdf/2603.20215) | [GitHub](https://github.com/HongduanTian/MAD-MM) |
| 分布式图记忆研究 | 向量记忆与图记忆的成本与精度对比。 | [PDF](https://arxiv.org/pdf/2601.07978) | [GitHub](https://github.com/wolffbe/dmas-long-context-memory) |

研究索引、详细阅读笔记与范围限制见
[references/README.md](references/README.md)。

## 环境要求

- Python 3.10 或更新版本
- 一个 OpenAI 兼容的 chat completion API
- 首次运行需要联网下载默认的向量模型

所有仓库命令都在 MiniMem 根目录下执行。安装依赖：

```bash
python -m pip install -r requirements.txt
```

复制环境变量模板并填入你的 API 配置：

```bash
cp .env.example .env
```

必填变量：

```text
LLM_API_KEY
LLM_MODEL_ID
```

`LLM_BASE_URL` 可选，可以指向任何兼容的 API 端点。
`LLM_THINKING_TYPE` 可选；在大规模运行前，将它设为 `disabled` 可以
关闭 DeepSeek 等推理模型的思考模式。

## 快速开始

自带的示例包含一段合成对话和一道问题：

```bash
python -m examples.quickstart
```

正常情况下它会进行一次 LLM 调用构建记忆、一次 LLM 调用回答问题。
模型输出格式非法或 API 瞬时故障可能触发重试。命令会打印抽取出的
记忆、检索到的 top-k 事实、预测答案、标准答案和 token 级 F1，并在
每个 API 阶段前打印提示，让你清楚何时在调用远端模型。

合成输入存放在
[`examples/tiny_conversation.json`](examples/tiny_conversation.json)。

## 完整 LoCoMo 评测

MiniMem 内置了来自
[LoCoMo 官方仓库](https://github.com/snap-research/locomo)的
`locomo10.json`，位于：

```text
benchmarks/locomo/data/locomo10.json
```

内置数据集按
[CC BY-NC 4.0](https://github.com/snap-research/locomo/blob/main/LICENSE.txt)
分发。署名与许可细节见
[`benchmarks/locomo/data/README.md`](benchmarks/locomo/data/README.md)。

完整评测分为三个显式阶段。

> **API 成本警告：** 当前的 LoCoMo 文件包含 272 个对话 session 和
> 1,540 道非对抗性问题。一次完整运行通常需要 272 次记忆构建调用和
> 1,540 次回答调用，重试可能带来更多调用。开始前请确认你的模型价格
> 与限流。

### 一次跑完所有阶段

依次运行记忆构建、问题回答和评测：

```bash
python -m benchmarks.locomo.run_all
```

任何一个阶段失败时脚本会立即停止。三个阶段也可以按下述方式分开运行。

### 1. 构建记忆

```bash
python -m benchmarks.locomo.construct_memory
```

为全部十段对话的每个 session 构建记忆，并写入：

```text
benchmarks/locomo/memories/
├── conv_0.json
├── conv_1.json
├── ...
└── conv_9.json
```

session 以 `MAX_WORKERS = 8` 并发处理。

### 2. 回答问题

```bash
python -m benchmarks.locomo.answer_question
```

加载已保存的记忆，回答所有非对抗性问题，并写入：

```text
benchmarks/locomo/predictions/
├── conv_0.json
├── conv_1.json
├── ...
└── conv_9.json
```

问题以 `MAX_WORKERS = 8` 并发处理。每条预测包含问题序号、类别、
标准答案、预测答案、检索到的 top-k 记忆，以及逐题的 `token_cost`
（对检索记忆、prompt 和答案的 tiktoken 计数）。

### 3. 评测答案

```bash
python -m benchmarks.locomo.evaluate_answer
```

加载所有预测文件，计算总体和分类别的 token F1 与 BLEU-1，运行
LLM judge（每条预测一次调用；在 `evaluate_answer.py` 顶部将
`RUN_LLM_JUDGE` 设为 `False` 可以只跑免费的词面指标），打印汇总表格，
并保存：

```text
benchmarks/locomo/results/summary.json
benchmarks/locomo/results/judgments.jsonl
```

报告的分数使用 `0–100` 量表。汇总里还包含总题数和
`No information available.` 回答的数量。

如果你的 API 限流较低，运行前请修改 `construct_memory.py` 和
`answer_question.py` 顶部的 `MAX_WORKERS`。

## LoCoMo-Refined 评测

MiniMem 同样内置了
[LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) 数据集
（同样的十段对话，1,382 道重新校准过的问题），位于
`benchmarks/locomo_refined/data/`，按 CC BY-NC 4.0 分发。流水线与
LoCoMo 的三个阶段一致：

```bash
python -m benchmarks.locomo_refined.run_all
```

一次完整运行通常需要 272 次记忆构建调用和 1,382 次回答调用。预测
记录与 LoCoMo 相同的字段，以 `qa_id` 为键。评测会报告本地词面 F1 和
BLEU-1 作为合理性检查，运行与 LoCoMo 相同的 LLM judge（由
`RUN_LLM_JUDGE` 开关控制），并写出
`benchmarks/locomo_refined/results/predictions.jsonl`——官方
LLM-judge 评测框架所需的提交文件。

## LongMemEval-Oracle 评测

MiniMem 还支持 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
的 oracle 切分（500 道基于用户-助手聊天历史的问题；oracle 切分只保留
证据 session）。该数据集不随仓库分发：请从官方发布下载
`longmemeval_oracle` 并放到：

```text
benchmarks/longmemeval/data/longmemeval_oracle
```

与 LoCoMo 不同，LongMemEval 的每道问题自带独立的对话干草堆，因此
流水线为每道问题单独构建一个记忆库：

```bash
python -m benchmarks.longmemeval.run_all
```

一次完整运行需要 948 次记忆构建调用（每个干草堆 session 一次）和
500 次回答调用。每个阶段并发处理所有问题，全部完成后按题号顺序写出
一个 JSONL 文件。预测记录与其他 benchmark 相同的逐题 `token_cost`。

评测会报告词面 F1 和 BLEU-1 作为本地合理性检查，并默认运行
LongMemEval 的官方指标：按题型区分 prompt 的 LLM judge（从官方仓库
原样移植，含专用的拒答 prompt），每题一次调用。在
`evaluate_answer.py` 顶部将 `RUN_OFFICIAL_JUDGE` 设为 `False` 可以
只跑免费的词面指标。

## 项目结构

```text
minimem/
  llm.py          OpenAI 兼容的 chat 客户端
  base.py         记忆条目结构
  construct.py    LLM 事实抽取
  retrieve.py     直接 embedding top-k 检索与问答
benchmarks/locomo/
  run_all.py           依次运行三个阶段
  construct_memory.py  构建所有对话记忆
  answer_question.py  回答所有非对抗性问题
  evaluate_answer.py  报告总体与分类别指标
benchmarks/locomo_refined/
  与 LoCoMo 相同的三阶段流水线，外加官方提交文件导出
benchmarks/longmemeval/
  与 LoCoMo 相同的三阶段流水线，每道问题一个独立记忆库
examples/
  quickstart.py   两次调用的合成示例
pyproject.toml    包元数据与依赖约束
```

## 许可证

MiniMem 的源代码和合成示例数据以
[MIT License](LICENSE) 发布。内置的 LoCoMo 与 LoCoMo-Refined 数据集
单独以 CC BY-NC 4.0 分发；见
[benchmarks/locomo/data/README.md](benchmarks/locomo/data/README.md) 和
[benchmarks/locomo_refined/data/README.md](benchmarks/locomo_refined/data/README.md)。
LongMemEval 数据集不在本仓库中再分发；请按其自身许可条款从
[官方仓库](https://github.com/xiaowu0162/LongMemEval)获取。
