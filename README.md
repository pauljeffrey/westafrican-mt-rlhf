# West African Machine Translation with AfriCOMET-Guided RLHF

**A distributed fine-tuning pipeline for low-resource West African languages using supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and Fully Sharded Data Parallel (FSDP) training. Work is still ongoing.**

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
2. [Decision and Design Choices](#2-decision-and-design-choices)
3. [Architectural Choices](#3-architectural-choices)
4. [Evaluation](#4-evaluation)
5. [Related Academic Work](#5-related-academic-work)
6. [Quick Start](#6-quick-start)
7. [Distributed Training (FSDP)](#7-distributed-training-fsdp)
8. [Project Layout](#8-project-layout)
9. [Tech Stack](#9-tech-stack)

---

## 1. Problem Statement

Machine translation for **West African languages** remains severely under-served relative to high-resource pairs (e.g., en–de, en–fr). Three compounding factors drive this gap:

1. **Data scarcity** — Parallel corpora for Hausa, Igbo, Yoruba, Wolof, and related languages are orders of magnitude smaller than for European or East Asian languages. General-purpose LLMs exhibit high perplexity and suboptimal COMET scores on these pairs despite strong English performance.

2. **Evaluation mismatch** — Standard MT metrics (BLEU, chrF) correlate imperfectly with human judgments on morphologically rich, low-resource African languages. Generic COMET models trained predominantly on European data underperform on these language families.

3. **Compute constraints** — Large-scale SFT over millions of translation examples and RLHF with per-step AfriCOMET scoring remain expensive even with a compact base model. We use **Gemma 3 270M** (~270M parameters, ~540 MB in bf16) so full fine-tuning fits on a single consumer GPU, while **FSDP** scales throughput and effective batch size across multiple devices when needed.

**This project addresses all three concerns** by:

- Fine-tuning **[Gemma 3 270M IT](https://huggingface.co/google/gemma-3-270m-it)** on [`Aletheia-ng/tds-sft`](https://huggingface.co/datasets/Aletheia-ng/tds-sft), a large-scale West African translation SFT dataset (~11M instruction–input–response triples).
- Applying a lightweight **REINFORCE-style RLHF stage** with **AfriCOMET** — a COMET variant trained specifically on African languages — as the reward signal.
- Providing **FSDP + optional tensor parallelism** so the same pipeline scales from a single GPU (full fine-tune of 270M) to multi-node clusters without code changes.
- Shipping a **reproducible evaluation harness** against the [MaFAND](https://huggingface.co/datasets/masakhane/mafand) benchmark and a public demo (Vercel frontend + Modal backend).

---

## 2. Decision and Design Choices

### 2.1 Two-stage training over end-to-end RL

We adopt **SFT → RLHF** rather than pure RL from a base checkpoint:

| Rationale | Detail |
|-----------|--------|
| Sample efficiency | SFT establishes a reasonable translation prior in ~1 epoch, reducing the RL exploration space. |
| Stability | Policy-gradient RL from a cold-start base model on low-resource MT may produce high-variance, often degenerate outputs. |
| Modularity | SFT and RL checkpoints can be evaluated independently; RL can be skipped if compute is limited. |

### 2.2 AfriCOMET as a black-box reward

We use **AfriCOMET** ([`masakhane/africomet-stl`](https://huggingface.co/masakhane/africomet-stl)) as a non-differentiable reward rather than training a separate reward head or using DPO/IPO:

- AfriCOMET is **pre-trained on African language pairs** and correlates better with human judgments than generic COMET-XL on Hausa, Igbo, and Yoruba.
- A black-box scorer avoids backpropagating through the COMET encoder, simplifying the RL loop and decoupling reward-model versioning from policy training.
- The trade-off is higher per-step latency (CPU/GPU COMET inference) mitigated by batching and rank-0-only scoring in distributed mode.

### 2.3 REINFORCE + KL penalty over PPO

Full PPO (clipped surrogate objective, value head, GAE) adds significant complexity for marginal gain on a fixed MT dataset. We implement:

$$\mathcal{L} = -\mathbb{E}\left[r \cdot \log \pi_\theta(y|x)\right] + \beta \cdot D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$$

where $r$ is the (optionally z-score normalized) AfriCOMET score, $\pi_{\mathrm{ref}}$ is the frozen SFT checkpoint, and $\beta$ is `RL_KL_COEF`. This is sufficient to nudge the policy toward higher-quality translations while preventing catastrophic forgetting.

### 2.4 Completion-only SFT loss

Translation quality is sensitive to prompt formatting. We train only on tokens **after** `### Translation:\n` using TRL's `DataCollatorForCompletionOnlyLM`, preventing the model from wasting capacity re-learning the instruction template.

### 2.5 Gemma 3 270M as the base model

We use **[google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)** — Google's compact instruction-tuned Gemma 3 variant designed for task-specific fine-tuning:

| Property | Value |
|----------|-------|
| Total parameters | ~270M (170M embedding + ~100M transformer blocks) |
| Architecture | 18 layers, hidden size 640, 262K vocabulary, 32K context |
| Base (pretrained) | [`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) |
| bf16 weight memory | ~540 MB |
| Typical full fine-tune (1× GPU) | ~2–4 GB peak VRAM (batch 4, seq ~768) |
| Typical inference | < 1 GB VRAM |

The large vocabulary (262K tokens) helps rare and language-specific tokens — relevant for West African scripts and transliterations — while keeping the transformer stack small enough for accessible training hardware.

### 2.6 FSDP over DDP for multi-GPU training

**Fully Sharded Data Parallel (FSDP)** shards model parameters, gradients, and optimizer states across GPUs. For Gemma 3 270M, single-GPU full fine-tuning is already feasible; FSDP is primarily used for **throughput scaling** and **larger effective batch sizes**, not because the model fails to fit in memory:

- Gemma 3 270M in bf16: **~540 MB** per full replica.
- FSDP 4-way shard: **~135 MB** parameters per GPU (+ activations).
- FSDP integrates natively with HuggingFace `Trainer` / TRL `SFTTrainer`.
- Optional **tensor parallelism** (`FSDP_TP_SIZE > 1`) splits linear layers across GPUs; auto-wrap uses `Gemma3DecoderLayer`.

### 2.7 Environment-driven configuration

All hyperparameters live in `.env` → `src/config.py`. No YAML loader in the training path — single source of truth, easy to override in CI/cluster job definitions.

---

## 3. Architectural Choices

### 3.1 System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│  Aletheia-ng/tds-sft  ──►  SFT (TRL + FSDP)  ──►  outputs/sft/   │
│                                      │                              │
│                                      ▼                              │
│              AfriCOMET reward ◄──  RLHF (REINFORCE + FSDP)         │
│                                      │                              │
│                                      ▼                              │
│                          BeardedMonster/gemma-3-270m-translate-it   │
└─────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
  scripts/eval.py                      app/backend/ (Modal GPU)
  masakhane/mafand                     app/frontend/ (Vercel UI)
  BLEU · chrF · AfriCOMET
```

### 3.2 Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `src/config.py` | Frozen `Settings` dataclass from `.env` |
| `src/data.py` | Stream and format HF dataset rows for SFT/RL |
| `src/prompts.py` | Shared prompt template + output extraction |
| `src/model_utils.py` | Tokenizer/model loading; defers device placement under FSDP |
| `src/distributed.py` | **FSDP wrap, tensor parallel, checkpoint gather, broadcast** |
| `src/sft_train.py` | TRL `SFTTrainer` with optional FSDP config |
| `src/rlhf_train.py` | Custom RL loop with FSDP-wrapped policy + reference |
| `src/reward.py` | AfriCOMET batch scorer |
| `src/inference.py` | Single-sentence greedy decode |
| `scripts/` | Thin CLI entry points (`run_sft`, `run_rlhf`, `eval`, `translate`) |

---

## 4. Evaluation

### 4.1 Benchmark

We evaluate on **[MaFAND](https://huggingface.co/datasets/masakhane/mafand)** (`validation` split) — a multi-way parallel corpus covering English ↔ African languages with professional translations.

Default language pairs: `en-hau`, `en-ibo`, `en-yor`, `en-wol`, `en-ewe`, `en-fon`, `en-twi`.

### 4.2 Metrics

| Metric | Purpose |
|--------|---------|
| **BLEU** | N-gram overlap with reference (standard MT baseline) |
| **chrF** | Character n-gram F-score; more robust for morphologically rich languages |
| **AfriCOMET** | Learned metric trained on African languages; optional (`--use-africomet`) |

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

### 4.4 Recommended reporting

For reproducible results, report:

1. Checkpoint path (SFT vs RLHF)
2. `MAX_TRAIN_SAMPLES` / `MAX_RL_SAMPLES` used
3. MaFAND validation split (never train on MaFAND)
4. Per-language BLEU, chrF, and AfriCOMET with macro-average across pairs

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

### Distributed training

- **FSDP** — Fully Sharded Data Parallel; parameter sharding across GPUs.  
  *Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (VLDB 2023)*
- **Tensor parallelism** — Layer-wise split of weight matrices.

### Base model

- **Gemma 3 270M** — Compact instruction-tuned model for efficient task-specific fine-tuning.  
  *Google DeepMind, [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m-it) · [Developers Blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)*

---

## 6. Quick Start

### 6.1 Install

```bash
git clone <repo-url> west-african-mt-rlhf
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

```bash
# 2-GPU data parallel (throughput scaling; 270M fits on 1 GPU)
torchrun --nproc_per_node=2 scripts/run_sft.py
torchrun --nproc_per_node=2 scripts/run_rlhf.py

# 4-GPU FSDP full shard (larger effective batch)
torchrun --nproc_per_node=4 scripts/run_sft.py
torchrun --nproc_per_node=4 scripts/run_rlhf.py

# 8-GPU: 4 data-parallel groups × 2-way tensor parallel
# FSDP_TP_SIZE=2 in .env
torchrun --nproc_per_node=8 scripts/run_sft.py
```

---

## 7. Distributed Training (FSDP)

### 7.1 Sharding strategies

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `FULL_SHARD` | Shard params, grads, optimizer states | Default; best memory efficiency |
| `HYBRID_SHARD` | Full shard intra-node, replicate inter-node | Multi-node clusters |
| `SHARD_GRAD_OP` | Shard grads/optimizer only | When params fit per-GPU but grads don't |

---

## 8. Project Layout

```
west-african-mt-rlhf/
├── README.md
├── requirements.txt
├── .env.example
├── configs/default.yaml          # reference defaults (informational)
├── scripts/
│   ├── run_sft.py                # SFT entry point (+ distributed init)
│   ├── run_rlhf.py               # RLHF entry point (+ distributed init)
│   ├── translate.py              # single-sentence inference CLI
│   └── eval.py                   # MaFAND benchmark evaluation
├── src/
│   ├── config.py                 # Settings from .env
│   ├── data.py                   # dataset streaming + formatting
│   ├── prompts.py                # prompt template + extraction
│   ├── model_utils.py            # model/tokenizer loading
│   ├── distributed.py            # FSDP + TP + checkpoint I/O
│   ├── sft_train.py              # SFT training
│   ├── rlhf_train.py             # RLHF training
│   ├── reward.py                 # AfriCOMET wrapper
│   └── inference.py              # generation helper
├── app/
│   ├── backend/                  # Modal serverless GPU API
│   └── frontend/                 # Next.js demo (Vercel)
```

---

## 9. Tech Stack

| Library | Role |
|---------|------|
| PyTorch ≥ 2.1 | Training, FSDP, generation |
| Transformers ≥ 4.50 | Causal LM (Gemma 3), tokenizer, TP integration |
| TRL ≥ 0.9.6 | `SFTTrainer`, completion-only collator |
| PEFT ≥ 0.12 | Optional LoRA |
| unbabel-comet ≥ 2.2 | AfriCOMET reward |
| sacrebleu ≥ 2.4 | BLEU / chrF evaluation |
| Datasets ≥ 2.19 | HF streaming |
| Accelerate ≥ 0.33 | Mixed precision, FSDP trainer backend |
