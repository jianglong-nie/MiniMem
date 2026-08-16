# LoCoMo-Refined

> A stricter and cleaner recalibration of LoCoMo for long-conversation memory evaluation.

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/mem-eval-suite/LoCoMo_refined/releases/tag/v1.0.0)
[![Dataset](https://img.shields.io/badge/Dataset-LoCoMo--Refined-blue)](data/raw/locomo_refined.json)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](#quick-start)
[![Judge](https://img.shields.io/badge/Judge-Qwen3--14B-purple)](src/llm_judge.py)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE.txt)

`LoCoMo Refined` recalibrates the original LoCoMo benchmark for long-conversation memory evaluation. It focuses on whether an agent can accurately recall time, events, relationships, and preferences after very long dialogues.

We found that the original evaluation could over-credit answers that were close in topic but wrong in detail, especially around time, missing facts, and unsupported additions. LoCoMo-Refined tightens the judge and audits the QA set so scores better reflect real memory reliability.

- 📦 Dataset: `data/raw/locomo_refined.json` · `data/public/questions.jsonl`
- ⚖️ Official judge: `Qwen/Qwen3-14B`
- 🚀 Evaluation: `scripts/run_eval.sh`

---

## News

- [Coming soon] 🔥 Technical report.
- [2026-04-26] 🏆 New SOTA! MemoraX AI reached **82.65%** on LoCoMo-Refined.
- [2026-04-14] 🎉 Dataset release — LoCoMo-Refined is now open for long-conversation memory benchmarking.

---

## Overview

### At a Glance

| Item | Value |
|---|---:|
| Questions | 1,382 |
| Revised QA samples | 337 |
| Human annotators | 5 |
| Human-alignment samples | 300 |
| Original judge agreement | 43.67% |
| Refined judge agreement | **86.33%** |
| Official judge model | `Qwen/Qwen3-14B` |

### What LoCoMo-Refined Changes

This release focuses on two changes: a stricter LLM judge and a cleaner QA set.

| Component | Original LoCoMo | LoCoMo-Refined |
|---|---|---|
| QA quality | Contains noisy or ambiguous samples | 337 samples revised after AI screening and human audit, covering issues such as ambiguous wording, reversed subject-object relationships, and time information inconsistent with the original conversations |
| Judge behavior | Boundaries are too loose for memory evaluation | Stricter judge with clearer correctness rules |
| Temporal answers | Can gloss over vague conversion or unsupported extra detail | Requires strict temporal granularity alignment |
| List / set answers | Can allow partial or over-extended answers to pass | Requires required information to be covered without unsupported additions |
| Evaluation | Original setup | Unified scripts for lexical and LLM-judge metrics |

#### 1. ⚖️ A stricter judger

The refined judge is built around one principle:

> **Inclusive without contradiction, complete without overreach.**

That means the answer must cover the required information, avoid unsupported additions, and preserve strict temporal granularity. The full prompt is in `src/llm_judge.py`; the original judge is kept for comparison.

#### 2. 🧹 A cleaner dataset

We used AI-assisted screening plus review from 5 human annotators to revise **337** samples with logical or factual issues, including ambiguous wording, reversed subject-object relationships, and inconsistent time information.

The public dataset is available at `./data/raw/locomo_refined.json` (1,382 questions). The QA schema uses:

- `answer`: list of acceptable gold answers. Each item in the list is a complete correct answer candidate. If a question requires multiple facts, those facts should appear together inside one answer string; they are not split across list items as a required set.

### Evaluation Principle

> **Inclusive without contradiction, complete without overreach.**

A prediction is considered correct only if it:

1. includes all required information from the gold answer;
2. does not contradict the gold answer;
3. does not introduce unsupported extra details;
4. preserves the correct temporal granularity;
5. handles list-style answers without missing required items or adding unsupported ones.

### What We Hope This Benchmark Solves

LoCoMo-Refined is designed to make memory scores more meaningful by exposing time drift, missing facts, redundant details, and unsupported claims that a looser judge may miss.

### 📊 Evaluation Results

We re-scored the same system predictions with the LoCoMo-Refined judge. The drop is the absolute decrease from the original LoCoMo judge in percentage points.

| System | LoCoMo-Refined score | Drop vs. original judge |
|---|---:|---:|
| MemoraX AI | 82.65% | N/A |
| EverMemOS | 58.25% | 22.07% |
| MemOS | 63.60% | 17.30% |
| MemPalace | 58.68% | 15.78% |
| Mem0 | 48.91% | 15.56% |

---

## 🚀 Quick Start

### 1. Environment setup

Requirements:

- Python `3.11+`
- The `openai` and `tenacity` packages can be installed

Create a Python 3.11 environment:

```bash
cd /path/to/LoCoMo_refined
conda create -n locomo-refined python=3.11 -y
conda activate locomo-refined
pip install openai tenacity
export LOCOMO_PYTHON_BIN="$(which python)"
```

Or use an existing environment:

```bash
cd /path/to/LoCoMo_refined
conda activate <your-env-name>
python --version
python -m pip show openai tenacity
export LOCOMO_PYTHON_BIN="$(which python)"
```

### 2. Prepare the prediction file

By default, the evaluator reads `./outputs/predictions.jsonl`. Each line should contain:

```jsonl
{"qa_id":"conv-26#q0000","predicted_answer":"7 May 2023"}
{"qa_id":"conv-26#q0001","predicted_answer":"2022"}
```

The `qa_id` values should match `./data/public/questions.jsonl`.

### 3. Run lexical evaluation

```bash
./scripts/run_eval.sh --metrics f1 bleu
```

The evaluation outputs are written by default to:

- `./outputs/predictions_scored.jsonl`
- `./outputs/predictions_scored_summary.json`
- `./outputs/predictions_scored_summary.md`

### 4. Run LLM Judge evaluation

Configure the evaluator:

```bash
export EVALUATOR_MODEL=qwen3-14b
# Optional: set this if you use a custom OpenAI-compatible endpoint
export EVALUATOR_API_BASE=https://your-endpoint/v1
# Optional: set API key if your endpoint requires authentication
export EVALUATOR_API_KEY=your_api_key
```

Then run:

```bash
./scripts/run_eval.sh --metrics llm f1 bleu --llm-judge refined
```

LoCoMo-Refined's official judge LLM is **Qwen3-14B**. Non-Qwen models trigger a warning and require confirmation.

---

## Reference

### ⚙️ Judge Configuration

| Item | Value |
|---|---|
| Judge model | `Qwen/Qwen3-14B` |
| Temperature | `0.0` |
| Thinking mode | Disabled when supported |
| Judge prompt | `src/llm_judge.py` |
| Runtime | `src/llm_judge_runtime.py` |
| Default judge | `refined` |
| Alternative judge | `original` |

Accepted model aliases for `EVALUATOR_MODEL`:

```text
qwen3-14b
qwen3_14b
Qwen/Qwen3-14B
qwen/qwen3-14b
dashscope/qwen3-14b   (vendor-prefixed variants ending with the above)
```

If a non-Qwen model is specified, the script will warn and require manual confirmation before continuing.

### Prediction Format

| Field | Type | Description |
|---|---|---|
| `qa_id` | string | Question ID; should match `data/public/questions.jsonl` |
| `predicted_answer` | string | Model-generated answer |

```jsonl
{"qa_id":"conv-26#q0000","predicted_answer":"7 May 2023"}
```

### Dataset Schema

`data/public/questions.jsonl`:

| Field | Type | Description |
|---|---|---|
| `qa_id` | string | Unique question ID (`<sample_id>#<index>`) |
| `sample_id` | string | Source conversation ID |
| `conversation_idx` | int | Source conversation index |
| `qa_index` | int | Question index within the source conversation |
| `speaker_a` | string | First participant in the conversation |
| `speaker_b` | string | Second participant in the conversation |
| `question` | string | Memory question |
| `answer` | list[string] | Acceptable complete gold answer candidates; any one candidate can be matched |
| `category` | string | Question category |
| `is_multi_modality` | bool | Whether the question involves an image |
| `evidence` | list[string] | Supporting dialogue turn IDs |
| `evidence_messages` | list[object] | Resolved supporting turns when the evidence IDs can be matched |

`data/public/conversations.jsonl`:

| Field | Type | Description |
|---|---|---|
| `sample_id` | string | Conversation ID |
| `conversation_idx` | int | Conversation index |
| `speaker_a` | string | First participant |
| `speaker_b` | string | Second participant |
| `session_count` | int | Number of sessions |
| `message_count` | int | Number of text messages |
| `multimodal_message_count` | int | Number of messages with image/caption/query context |
| `image_count` | int | Number of referenced images |
| `sessions` | list[object] | Session objects, each with `session_index`, `date_time`, and `messages` |
| `conversation_history_text` | string | Flattened text-only conversation history |
| `conversation_history_multimodal_text` | string | Flattened conversation history including image, caption, and query context |

Messages inside `sessions[].messages` include `dia_id`, `speaker`, `role`, `text`, `images`, `blip_caption`, `query`, and `has_multimodal_context`.

### 📊 Judge Validation

On 300 manually annotated samples, `Qwen/Qwen3-14B + the refined prompt` reached **86.33%** agreement with human annotations, compared with **43.67%** for the original LoCoMo setup.

| Judge Setup | Model | Human Agreement |
|---|---|---:|
| Original LoCoMo judge | GPT-4o-mini | 43.67% |
| LoCoMo-Refined judge | Qwen/Qwen3-14B | **86.33%** |

This suggests that the refined judge moves the decision boundary closer to human consensus rather than simply making the benchmark harsher.

### Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── locomo_refined.json        # Full annotated dataset (1,382 questions)
│   └── public/
│       ├── questions.jsonl            # Public QA format
│       ├── conversations.jsonl        # Conversation context
│       ├── manifest.json
│       └── submission_template.jsonl  # Blank prediction template
├── scripts/
│   ├── run_eval.sh                    # Main evaluation entry point
│   ├── build_predictions.py
│   ├── env.sh
│   └── export_dataset.sh
├── src/
│   ├── llm_judge.py                   # Judge prompt definitions
│   ├── llm_judge_runtime.py           # Judge inference logic
│   ├── evaluate.py
│   ├── bleu_f1.py
│   ├── export.py
│   └── summarize.py
├── outputs/                           # Evaluation outputs
├── README.md
├── LICENSE.txt
└── NOTICE
```

---

## License and Citation

### License and Attribution

LoCoMo-Refined is released under **CC BY-NC 4.0**. This benchmark modifies the original LoCoMo benchmark; see `NOTICE` for attribution and modification details.

### Citation

```bibtex
@misc{locomo_refined_2026,
  author = {Mem-eval-suite Team},
  title  = {LoCoMo-Refined: Recalibrating LoCoMo for Long-Conversation Memory Evaluation},
  year   = {2026},
  url    = {https://github.com/mem-eval-suite/LoCoMo_refined}
}
```
