# MiniMem

MiniMem is a deliberately small baseline for evaluating conversational memory
on LoCoMo-style question answering.

It implements one transparent pipeline:

```text
conversation
    -> LLM fact extraction
    -> embedding cosine similarity
    -> direct top-k retrieval
    -> LLM answer
    -> token F1 and BLEU-1
```

MiniMem is intended for education, debugging, and lightweight evaluation. It
does not include memory updating, clustering, keyword search, reranking,
graph retrieval, or token-budget optimization.

## Open-source memory research

The following open-source papers and projects were reviewed while defining the
scope of MiniMem.

| Paper or project | Main focus | Paper | Code |
| --- | --- | --- | --- |
| Mem0 | Production-oriented long-term memory extraction and retrieval. | [PDF](https://arxiv.org/pdf/2504.19413) | [GitHub](https://github.com/mem0ai/mem0) |
| Zep | Temporal knowledge-graph memory built on Graphiti. | [PDF](https://arxiv.org/pdf/2501.13956) | [GitHub](https://github.com/getzep/graphiti) |
| MemGPT | Operating-system-inspired context and memory management. | [PDF](https://arxiv.org/pdf/2310.08560) | [GitHub](https://github.com/letta-ai/letta) |
| MemOS | Unified management of parametric, activation, and plaintext memory. | [PDF](https://arxiv.org/pdf/2505.22101) | [GitHub](https://github.com/MemTensor/MemOS) |
| MIRIX | Multi-agent coordination across six memory types. | [PDF](https://arxiv.org/pdf/2507.07957) | [GitHub](https://github.com/Mirix-AI/MIRIX) |
| SimpleMem | Structured compression, semantic synthesis, and intent-aware retrieval. | [PDF](https://arxiv.org/pdf/2601.02553) | [GitHub](https://github.com/aiming-lab/SimpleMem) |
| LangMem | Long-term memory SDK for LangChain and LangGraph agents. | — | [GitHub](https://github.com/langchain-ai/langmem) |
| MemoryOS | Hierarchical short-, mid-, and long-term personal memory. | [PDF](https://arxiv.org/pdf/2506.06326) | [GitHub](https://github.com/BAI-LAB/MemoryOS) |
| A-MEM | Zettelkasten-inspired agentic memory with linked notes. | [PDF](https://arxiv.org/pdf/2502.12110) | [GitHub](https://github.com/agiresearch/A-mem) |
| LightMem | Lightweight multi-stage memory with low-cost updates. | [PDF](https://arxiv.org/pdf/2510.18866) | [GitHub](https://github.com/zjunlp/LightMem) |
| MemoryBank | Long-term conversational memory with an Ebbinghaus forgetting model. | [PDF](https://arxiv.org/pdf/2305.10250) | [GitHub](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) |
| G-Memory | Hierarchical graph memory for multi-agent collaboration. | [PDF](https://arxiv.org/pdf/2506.07398) | [GitHub](https://github.com/bingreeky/GMemory) |
| MemEvolve | Meta-evolution of modular memory architectures. | [PDF](https://arxiv.org/pdf/2512.18746) | [GitHub](https://github.com/bingreeky/MemEvolve) |
| Nemori | Self-organizing memory inspired by event segmentation. | [PDF](https://arxiv.org/pdf/2508.03341) | [GitHub](https://github.com/nemori-ai/nemori) |
| SuperLocalMemory | Local-first multi-agent memory with poisoning defense. | [PDF](https://arxiv.org/pdf/2603.02240) | [GitHub](https://github.com/qualixar/superlocalmemory) |
| SuperLocalMemory V3 | Zero-LLM enterprise memory using information geometry. | [PDF](https://arxiv.org/pdf/2603.14588) | [GitHub](https://github.com/qualixar/superlocalmemory) |
| LatentMem | Role-specific latent memory composition for multi-agent systems. | [PDF](https://arxiv.org/pdf/2602.03036) | [GitHub](https://github.com/KANABOON1/LatentMem) |
| MemP | Construction, retrieval, and updating of procedural memory. | [PDF](https://arxiv.org/pdf/2508.06433) | [GitHub](https://github.com/zjunlp/MemP) |
| MemOCR | Layout-aware visual memory for long-horizon reasoning. | [PDF](https://arxiv.org/pdf/2601.21468) | [GitHub](https://github.com/meituan/MemOCR) |
| E-mem | Multi-agent episodic context reconstruction. | [PDF](https://arxiv.org/pdf/2601.21714) | [GitHub](https://github.com/dog-last/E-mem) |
| JitRL | Test-time continual learning from retrieved experience. | [PDF](https://arxiv.org/pdf/2601.18510) | [GitHub](https://github.com/liushiliushi/JitRL) |
| BudgetMem | Query-aware routing across memory budget tiers. | [PDF](https://arxiv.org/pdf/2602.06025) | [GitHub](https://github.com/ViktorAxelsen/BudgetMem) |
| RF-Mem | Adaptive retrieval using familiarity and recollection paths. | [PDF](https://arxiv.org/pdf/2603.09250) | [GitHub](https://github.com/Zhang-Yingyi/ICLR2026_RF-Mem) |
| MAD-M² | Memory masking for multi-agent debate. | [PDF](https://arxiv.org/pdf/2603.20215) | [GitHub](https://github.com/HongduanTian/MAD-MM) |
| Distributed graph memory study | Cost and accuracy comparison of vector and graph memory. | [PDF](https://arxiv.org/pdf/2601.07978) | [GitHub](https://github.com/wolffbe/dmas-long-context-memory) |

See [references/README.md](references/README.md) for the research index,
detailed reading notes, and scope limitations.

## Requirements

- Python 3.10 or newer
- An OpenAI-compatible chat completion API
- Internet access on the first run to download the default embedding model

Run all repository commands from the MiniMem root directory. Install MiniMem
and its dependencies in editable mode:

```bash
python -m pip install -r requirements.txt
```

Copy the environment template and add your API settings:

```bash
cp .env.example .env
```

Required variables:

```text
LLM_API_KEY
LLM_MODEL_ID
```

`LLM_BASE_URL` is optional and can point to any compatible API endpoint.
`LLM_THINKING_TYPE` is optional; set it to `disabled` to turn off the thinking
mode of reasoning models such as DeepSeek before a large run.

## Quick start

The bundled example contains one synthetic conversation and one question:

```bash
python -m examples.quickstart
```

This normally performs one LLM call to construct memory and one LLM call to
answer the question. Invalid model output or transient API failures may cause
retries. The command prints the extracted memories, retrieved top-k facts,
predicted answer, gold answer, and token-level F1. It also prints a message
before each API stage so it is clear when the remote model is being called.

The synthetic input is stored in
[`examples/tiny_conversation.json`](examples/tiny_conversation.json).

## Full LoCoMo evaluation

MiniMem includes `locomo10.json` from the
[official LoCoMo repository](https://github.com/snap-research/locomo) at:

```text
benchmarks/locomo/data/locomo10.json
```

The bundled dataset is distributed under
[CC BY-NC 4.0](https://github.com/snap-research/locomo/blob/main/LICENSE.txt).
See [`benchmarks/locomo/data/README.md`](benchmarks/locomo/data/README.md)
for attribution and license details.

The full evaluation has three explicit stages.

> **API cost warning:** the current LoCoMo file contains 272 conversation
> sessions and 1,540 non-adversarial questions. A full run therefore normally
> makes 272 memory-construction calls and 1,540 answer calls. Retries may add
> more calls. Check your model pricing and rate limits before starting.

### Run all stages

To run memory construction, question answering, and evaluation in sequence:

```bash
python -m benchmarks.locomo.run_all
```

The script stops immediately if any stage fails. The three stages can still be
run separately as described below.

### 1. Construct memory

```bash
python -m benchmarks.locomo.construct_memory
```

This builds memory for every session in all ten conversations and writes:

```text
benchmarks/locomo/memories/
├── conv_0.json
├── conv_1.json
├── ...
└── conv_9.json
```

Sessions are processed concurrently with `MAX_WORKERS = 8`.

### 2. Answer questions

```bash
python -m benchmarks.locomo.answer_question
```

This loads the saved memories, answers every non-adversarial question, and
writes:

```text
benchmarks/locomo/predictions/
├── conv_0.json
├── conv_1.json
├── ...
└── conv_9.json
```

Questions are processed concurrently with `MAX_WORKERS = 8`. Each prediction
contains the question index, category, gold answer, predicted answer, and
retrieved top-k memories.

### 3. Evaluate answers

```bash
python -m benchmarks.locomo.evaluate_answer
```

This loads all prediction files, computes overall and category-level
token F1, prints a summary table, and saves:

```text
benchmarks/locomo/results/summary.json
```

Reported F1 values use a `0–100` scale. The summary also includes the total
question count and the number of `No information available.` answers.

If your API has a lower rate limit, change `MAX_WORKERS` near the top of
`construct_memory.py` and `answer_question.py` before running.

## LoCoMo-Refined evaluation

MiniMem also bundles the
[LoCoMo-Refined](https://github.com/mem-eval-suite/LoCoMo_refined) dataset
(the same ten conversations with 1,382 recalibrated questions) at
`benchmarks/locomo_refined/data/`, distributed under CC BY-NC 4.0. The
pipeline mirrors the three LoCoMo stages:

```bash
python -m benchmarks.locomo_refined.run_all
```

A full run normally makes 272 memory-construction calls and 1,382 answer
calls. Each prediction additionally records a per-question `token_cost`
(tiktoken counts of the retrieved memories, prompt, and answer). Evaluation
reports local lexical F1 and BLEU-1 as a sanity check and writes
`benchmarks/locomo_refined/results/predictions.jsonl`, the submission file
for the official LLM-judge harness.

## LongMemEval-Oracle evaluation

MiniMem also supports the oracle split of
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (500 questions over
user–assistant chat histories; the oracle split keeps only the evidence
sessions). The dataset is not bundled: download `longmemeval_oracle` from the
official release and place it at:

```text
benchmarks/longmemeval/data/longmemeval_oracle
```

Unlike LoCoMo, every LongMemEval question ships its own haystack, so the
pipeline builds one memory store per question:

```bash
python -m benchmarks.longmemeval.run_all
```

A full run makes 948 memory-construction calls (one per haystack session) and
500 answer calls. Each stage appends one JSONL line per question as it
finishes and skips already-completed questions on restart, so an interrupted
run can simply be re-run. Predictions record the same per-question
`token_cost` as LoCoMo-Refined.

Evaluation is free by default: it reports lexical F1 and BLEU-1 as a local
sanity check. The official LongMemEval metric is an LLM judge with one prompt
per question type (ported verbatim from the official repository, including
the dedicated abstention prompt); set `RUN_OFFICIAL_JUDGE = True` near the
top of `evaluate_answer.py` to run it (one call per question).

## Results

Reference numbers from one full run of each benchmark with
`deepseek-v4-flash` (thinking disabled), `all-MiniLM-L6-v2` embeddings, and
`TOP_K = 15`. All three benchmarks share the same tokenizer (the
LoCoMo-Refined scorer), so lexical scores are computed identically; absolute
numbers are still not comparable across datasets because question styles and
gold-answer formats differ. Lexical F1 and BLEU-1 are sanity-check metrics
only. For LongMemEval the official metric is the LLM judge, run here with the
same DeepSeek model rather than the GPT-4o judge used in the paper, so the
numbers are not directly comparable with published results.

**LoCoMo** (1,540 non-adversarial questions):

| Category | Questions | F1 | BLEU-1 |
| --- | ---: | ---: | ---: |
| Single-hop | 841 | 37.22 | 31.30 |
| Multi-hop | 282 | 26.66 | 20.07 |
| Open-domain | 96 | 22.77 | 16.55 |
| Temporal | 321 | 35.30 | 30.32 |
| **Overall** | **1,540** | **33.98** | **28.12** |

**LoCoMo-Refined** (1,382 questions):

| Category | Questions | F1 | BLEU-1 |
| --- | ---: | ---: | ---: |
| Single-hop | 802 | 41.15 | 34.28 |
| Multi-hop | 213 | 33.43 | 26.07 |
| Open-domain | 68 | 35.58 | 27.35 |
| Temporal | 299 | 38.33 | 33.46 |
| **Overall** | **1,382** | **39.07** | **32.50** |

**LongMemEval-Oracle** (500 questions; Judge is the official metric — the
lexical columns are meaningless for preference questions, whose gold answer
is a rubric, and for the 30 abstention questions):

| Question type | Questions | F1 | BLEU-1 | Judge |
| --- | ---: | ---: | ---: | ---: |
| single-session-user | 70 | 65.14 | 54.44 | 94.29% |
| single-session-assistant | 56 | 59.84 | 51.85 | 75.00% |
| single-session-preference | 30 | 9.23 | 2.46 | 36.67% |
| multi-session | 133 | 53.21 | 47.25 | 73.68% |
| knowledge-update | 78 | 51.29 | 41.56 | 79.49% |
| temporal-reasoning | 133 | 42.33 | 28.03 | 68.42% |
| **Overall** | **500** | **49.79** | **40.08** | **74.00%** |

The abstention subset (30 questions, included in the rows above) scores
96.67% under the dedicated abstention judge prompt.

## Project structure

```text
minimem/
  llm.py          OpenAI-compatible chat client
  base.py         Memory item structure
  construct.py    LLM fact extraction
  retrieve.py     Direct embedding top-k retrieval and question answering
benchmarks/locomo/
  run_all.py           Run all three stages in sequence
  construct_memory.py  Build all conversation memories
  answer_question.py  Answer all non-adversarial questions
  evaluate_answer.py  Report overall and category-level F1
benchmarks/locomo_refined/
  Same three-stage pipeline for LoCoMo-Refined, plus official submission export
benchmarks/longmemeval/
  Same three-stage pipeline for LongMemEval-Oracle, one memory store per question
examples/
  quickstart.py   Two-call synthetic example
pyproject.toml    Package metadata and dependency constraints
```

## License

MiniMem source code and synthetic example data are released under the
[MIT License](LICENSE). The bundled LoCoMo and LoCoMo-Refined datasets are
released separately under CC BY-NC 4.0; see
[benchmarks/locomo/data/README.md](benchmarks/locomo/data/README.md) and
[benchmarks/locomo_refined/data/README.md](benchmarks/locomo_refined/data/README.md).
The LongMemEval dataset is not redistributed here; obtain it from the
[official repository](https://github.com/xiaowu0162/LongMemEval) under its own
license terms.
