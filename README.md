# West African Machine Translation with AfriCOMET-Guided RLHF

**A machine translation system for West African languages. It teaches a small language model to translate, then improves it using feedback from a quality-scoring model built for African languages. Work is still ongoing.**

| | |
|---|---|
| **Base model** | [`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) |
| **Published checkpoint** | [`BeardedMonster/gemma-3-270m-translate-it`](https://huggingface.co/BeardedMonster/gemma-3-270m-translate-it) |
| **Training data** | [`Aletheia-ng/tds-sft`](https://huggingface.co/datasets/Aletheia-ng/tds-sft) |
| **Reward model** | [`masakhane/africomet-stl`](https://huggingface.co/masakhane/africomet-stl) |
| **Evaluation benchmark** | [`masakhane/mafand`](https://huggingface.co/datasets/masakhane/mafand) (validation split) |
| **Languages** | Hausa, Igbo, Yoruba, Wolof, Ewe, Fon, Twi (+ Nigerian Pidgin in training data) |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Approach](#2-approach)
3. [How It Works](#3-how-it-works)
4. [Evaluation](#4-evaluation)
5. [Related Academic Work](#5-related-academic-work)
6. [Quick Start](#6-quick-start)
7. [Multi-GPU Training](#7-multi-gpu-training)
8. [Project Layout](#8-project-layout)
9. [Tech Stack](#9-tech-stack)

---

## 1. Problem Statement

Most translation tools work well for languages like English, French, and German — but **West African languages are underserved**. Hausa, Igbo, Yoruba, Wolof, and others have far less training data, and general-purpose AI models often produce weak translations for them.

Three challenges make this hard:

1. **Not enough data** — There are far fewer high-quality translation examples for these languages than for major world languages.
2. **Hard to measure quality** — Standard translation scores do not always reflect how good a translation actually sounds to speakers of the language.
3. **Training cost** — Improving a model on millions of examples, then refining it with quality feedback, takes real compute — even with a small model.

**What this project does:**

- Trains **[Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it)**, a compact Google language model, on [`Aletheia-ng/tds-sft`](https://huggingface.co/datasets/Aletheia-ng/tds-sft) (~11M translation examples).
- Runs a second training stage that **rewards better translations** using **AfriCOMET**, a quality scorer built for African languages.
- Supports **single-GPU or multi-GPU training** so the same pipeline works on a laptop GPU or a small cluster.
- Includes **benchmarking** on [MaFAND](https://huggingface.co/datasets/masakhane/mafand) and a **live demo** (web UI on Vercel, API on Modal).

---

## 2. Approach

### 2.1 Two-step training

Instead of jumping straight to reinforcement learning, we train in two stages:

| Step | What it does | Why |
|------|--------------|-----|
| **Step 1 — Supervised fine-tuning (SFT)** | Shows the model many example translations and teaches it to follow a fixed prompt format. | Gives the model a solid starting point before optimization. |
| **Step 2 — RLHF** | Generates translations, scores them with AfriCOMET, and nudges the model toward higher-scoring outputs. | Improves quality beyond what examples alone can teach. |

Each stage can be run and evaluated on its own. Step 2 is optional if compute is limited.

### 2.2 AfriCOMET as the quality judge

During step 2, **AfriCOMET** ([`masakhane/africomet-stl`](https://huggingface.co/masakhane/africomet-stl)) rates each translation. The model is updated to produce outputs that score higher.

- AfriCOMET is **trained on African language pairs**, so it is a better judge than generic translation scorers for this use case.
- It is used as an **external scorer** — we read its score and update the model, without rebuilding the scorer itself.
- The trade-off: scoring adds time per training step, which we reduce via batching.

### 2.3 Keeping the model stable

During step 2, the model is penalized if it drifts too far from the step-1 checkpoint. That prevents it from chasing high scores in ways that break basic translation ability.

### 2.4 Focused learning during step 1

The model is trained only on the **translation output**, not on repeating the instruction template. That keeps training focused on actual translation quality.

### 2.5 Why Gemma 3 270M?

We use **[google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)** — a small, instruction-ready model from Google:

| | |
|---|---|
| Size | ~270M parameters (~540 MB) |
| Context | Up to 32K tokens |
| Vocabulary | 262K tokens (helps rare and language-specific words) |
| Hardware | Full training fits on a single consumer GPU (~2–4 GB VRAM) |
| Pretrained base | [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) |

Small enough to train affordably; large enough to learn useful translation patterns.

### 2.6 Multi-GPU support

Training can run on **one GPU** or scale to **multiple GPUs** using PyTorch **DDP** (split work across copies of the model) or **FSDP** (split the model itself across GPUs). For this model size, multi-GPU is mainly about **training faster**, not fitting the model in memory.

### 2.7 Configuration

All settings live in a single `.env` file, loaded by `src/config.py`. No hidden config scattered across the codebase.

---

## 3. How It Works

### 3.1 Pipeline

```
Training data (tds-sft)
        │
        ▼
   Step 1: SFT  ──►  checkpoint saved
        │
        ▼
   Step 2: RLHF  ◄──  AfriCOMET scores each translation
        │
        ▼
   Published model (Hugging Face)
        │
        ├─►  Benchmark (MaFAND)
        └─►  Live demo (web app + API)
```

### 3.2 Code structure

| Folder / file | What it does |
|---------------|--------------|
| `src/config.py` | Reads settings from `.env` |
| `src/data.py` | Loads and formats training data |
| `src/prompts.py` | Shared prompt format for training and inference |
| `src/model_utils.py` | Loads the language model and tokenizer |
| `src/distributed.py` | Multi-GPU setup (DDP and FSDP) |
| `src/sft_train.py` | Step 1 training loop |
| `src/rlhf_train.py` | Step 2 training loop |
| `src/reward.py` | Calls AfriCOMET to score translations |
| `src/inference.py` | Runs translation on a single sentence |
| `scripts/` | Command-line entry points to train, evaluate, and translate |
| `app/backend/` | GPU API for the live demo (Modal) |
| `app/frontend/` | Web UI for the live demo (Vercel) |

---

## 4. Evaluation

### 4.1 Benchmark

We evaluate on **[MaFAND](https://huggingface.co/datasets/masakhane/mafand)** — a dataset of professional English ↔ African language translations.

Default language pairs: `en-hau`, `en-ibo`, `en-yor`, `en-wol`, `en-ewe`, `en-fon`, `en-twi`.

### 4.2 Metrics

| Metric | What it measures |
|--------|------------------|
| **BLEU** | How closely the output matches the reference wording |
| **chrF** | Character-level similarity (better for languages with rich word forms) |
| **AfriCOMET** | Learned quality score for African languages (optional) |

### 4.3 Running evaluation

```bash
# Full benchmark (7 language pairs)
python scripts/eval.py --model-path ./outputs/rlhf

# Subset + AfriCOMET + JSON export
python scripts/eval.py \
  --model-path ./outputs/rlhf \
  --langs en-hau,en-ibo,en-yor \
  --use-africomet \
  --output-file results.json

# Quick smoke test
python scripts/eval.py --model-path ./outputs/sft --max-samples 100
```

---

## 5. Related Academic Work

This project builds on and connects to the following research lines:

### Low-resource African MT

- **Masakhane** — Pan-African NLP community; MaFAND benchmark and AfriCOMET metric.  
  *Dossou et al., "AfriCOMET: Automatic Evaluation of Machine Translation for African Languages"*

### Metric-based MT evaluation

- **COMET** — Cross-lingual MT evaluation with pre-trained multilingual encoders.  
  *Rei et al., "COMET: A Neural Framework for MT Evaluation" (EMNLP 2020)*
- **AfriCOMET-STL** — COMET fine-tuned on African language pairs; used as RL reward here.

### RLHF for language models

- **InstructGPT / RLHF pipeline** — SFT followed by reward-model-guided policy optimization.  
  *Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (NeurIPS 2022)*
- **REINFORCE for NLG** — Policy gradient fine-tuning of seq2seq and causal LMs.  
  *Rennie et al., "Self-Critical Sequence Training" (CVPR 2017)* — baseline for RL MT.

### Base model

- **Gemma 3 270M** — Compact instruction-tuned model for efficient task-specific fine-tuning.  
  *Google DeepMind, [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it) · [Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)*

---

## 6. Quick Start

### 6.1 Install

```bash
git clone https://github.com/pauljeffrey/westafrican-mt-rlhf.git
cd west-african-mt-rlhf
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env             # set HF_TOKEN, adjust paths
```

### 6.2 Launch commands

#### Single-GPU training

```bash
# Stage 1 — SFT
python scripts/run_sft.py

# Stage 2 — RLHF (requires SFT checkpoint)
python scripts/run_rlhf.py

# Inference
python scripts/translate.py \
  --instruction "Translate the sentence below to Hausa:" \
  --input "The market opens early every morning."
```

#### Multi-GPU training

Set `DIST_STRATEGY=ddp` or `DIST_STRATEGY=fsdp` in `.env`, then:

```bash
# 2 GPUs — DDP
torchrun --nproc_per_node=2 scripts/run_sft.py
torchrun --nproc_per_node=2 scripts/run_rlhf.py

# 4 GPUs — FSDP
torchrun --nproc_per_node=4 scripts/run_sft.py
torchrun --nproc_per_node=4 scripts/run_rlhf.py
```

---

## 7. Multi-GPU Training

When using more than one GPU, set `DIST_STRATEGY` in `.env`:

| Strategy | Plain English | Best for |
|----------|---------------|----------|
| `none` | Single GPU, no distribution | Local development |
| `ddp` | Each GPU holds a full copy of the model; data is split across GPUs | Faster training when the model fits on one GPU |
| `fsdp` | The model is split across GPUs | Larger batch sizes or very tight GPU memory |

FSDP sharding options (`FSDP_SHARDING` in `.env`):

| Option | What it does |
|--------|--------------|
| `FULL_SHARD` | Split model weights, gradients, and optimizer state (default) |
| `HYBRID_SHARD` | Full split within a machine, replicate across machines |
| `SHARD_GRAD_OP` | Keep full weights on each GPU; split only gradients and optimizer state |

---

## 8. Project Layout

```
west-african-mt-rlhf/
├── README.md
├── requirements.txt
├── .env.example
├── configs/default.yaml
├── scripts/          # CLI: train, evaluate, translate
├── src/              # Core training and inference code
└── app/
    ├── backend/      # Modal GPU API
    └── frontend/     # Vercel web demo
```

---

## 9. Tech Stack

| Tool | Role |
|------|------|
| **PyTorch** | Model training and inference |
| **Hugging Face Transformers** | Loads Gemma 3 and the tokenizer |
| **AfriCOMET (unbabel-comet)** | Scores translation quality during RLHF |
| **Hugging Face Datasets** | Streams training and benchmark data |
| **sacrebleu** | BLEU and chrF evaluation metrics |
| **Modal** | Hosts the GPU API for the live demo |
| **Vercel + Next.js** | Hosts the web demo |
