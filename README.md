# West African Machine Translation with AfriCOMET-Guided RLHF

**A distributed fine-tuning pipeline for low-resource West African languages using supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF), and Fully Sharded Data Parallel (FSDP) training.**

| | |
|---|---|
| **Base model** | `google/gemma-2-2b-it` (configurable) |
| **Published checkpoint** | [`BeardedMonster/gemma-270m-translate-it`](https://huggingface.co/BeardedMonster/gemma-270m-translate-it) |
| **Training data** | [`Aletheia-ng/tds-sft`](https://huggingface.co/datasets/Aletheia-ng/tds-sft) |
| **Reward model** | [`masakhane/africomet-stl`](https://huggingface.co/masakhane/africomet-stl) |
| **Evaluation benchmark** | [`masakhane/mafand`](https://huggingface.co/datasets/masakhane/mafand) (validation split) |
| **Languages** | Hausa, Igbo, Yoruba, Wolof, Ewe, Fon, Twi (+ Nigerian Pidgin in training data) |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Decision and Design Choices](#2-decision-and-design-choices)
3. [Architectural Choices](#3-architectural-choices)
4. [Engineering Bottlenecks](#4-engineering-bottlenecks)
5. [Trade-offs](#5-trade-offs)
6. [Evaluation](#6-evaluation)
7. [Edge Cases and Considerations](#7-edge-cases-and-considerations)
8. [Related Academic Work](#8-related-academic-work)
9. [Quick Start](#9-quick-start)
10. [Distributed Training (FSDP)](#10-distributed-training-fsdp)
11. [Project Layout](#11-project-layout)
12. [Deployment](#12-deployment)

---

## 1. Problem Statement

Machine translation for **West African languages** remains severely under-served relative to high-resource pairs (e.g., en–de, en–fr). Three compounding factors drive this gap:

1. **Data scarcity** — Parallel corpora for Hausa, Igbo, Yoruba, Wolof, and related languages are orders of magnitude smaller than for European or East Asian languages. General-purpose LLMs exhibit high perplexity and poor COMET scores on these pairs despite strong English performance.

2. **Evaluation mismatch** — Standard MT metrics (BLEU, chrF) correlate imperfectly with human judgments on morphologically rich, low-resource African languages. Generic COMET models trained predominantly on European data underperform on these language families.

3. **Compute constraints** — Fine-tuning even a 2B-parameter instruction-tuned model on millions of translation examples requires multi-GPU infrastructure that most research groups and practitioners in the region lack access to.

**This project addresses all three concerns** by:

- Fine-tuning an instruction-tuned causal LM on `Aletheia-ng/tds-sft`, a large-scale West African translation SFT dataset (~11M instruction–input–response triples).
- Applying a lightweight **REINFORCE-style RLHF stage** with **AfriCOMET** — a COMET variant trained specifically on African languages — as the reward signal.
- Providing **FSDP + optional tensor parallelism** so the same pipeline scales from a single GPU (with LoRA) to multi-node clusters without code changes.
- Shipping a ** reproducible evaluation harness** against the MaFAND benchmark and a public demo (Vercel frontend + Modal backend).

---

## 2. Decision and Design Choices

### 2.1 Two-stage training over end-to-end RL

We adopt **SFT → RLHF** rather than pure RL from a base checkpoint:

| Rationale | Detail |
|-----------|--------|
| Sample efficiency | SFT establishes a reasonable translation prior in ~1 epoch, reducing the RL exploration space. |
| Stability | Policy-gradient RL from a cold-start base model on low-resource MT produces high-variance, often degenerate outputs. |
| Modularity | SFT and RL checkpoints can be evaluated independently; RL can be skipped if compute is limited. |

### 2.2 AfriCOMET as a black-box reward

We use **AfriCOMET** (`masakhane/africomet-stl`) as a non-differentiable reward rather than training a separate reward head or using DPO/IPO:

- AfriCOMET is **pre-trained on African language pairs** and correlates better with human judgments than generic COMET-XL on Hausa, Igbo, and Yoruba.
- A black-box scorer avoids backpropagating through the COMET encoder, simplifying the RL loop and decoupling reward-model versioning from policy training.
- The trade-off is higher per-step latency (CPU/GPU COMET inference) — mitigated by batching and rank-0-only scoring in distributed mode.

### 2.3 REINFORCE + KL penalty over PPO

Full PPO (clipped surrogate objective, value head, GAE) adds significant complexity for marginal gain on a fixed MT dataset. We implement:

$$\mathcal{L} = -\mathbb{E}\left[r \cdot \log \pi_\theta(y|x)\right] + \beta \cdot D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$$

where $r$ is the (optionally z-score normalized) AfriCOMET score, $\pi_{\mathrm{ref}}$ is the frozen SFT checkpoint, and $\beta$ is `RL_KL_COEF`. This is sufficient to nudge the policy toward higher-quality translations while preventing catastrophic forgetting.

### 2.4 Completion-only SFT loss

Translation quality is sensitive to prompt formatting. We train only on tokens **after** `### Translation:\n` using TRL's `DataCollatorForCompletionOnlyLM`, preventing the model from wasting capacity re-learning the instruction template.

### 2.5 FSDP over DDP for multi-GPU training

**Fully Sharded Data Parallel (FSDP)** shards model parameters, gradients, and optimizer states across GPUs — enabling larger effective batch sizes and models that do not fit on a single device. We chose FSDP over vanilla DDP because:

- Gemma-2-2B in bf16 requires ~4 GB per replica; FSDP reduces per-GPU memory to ~1 GB + activations at 4-way shard.
- FSDP integrates natively with HuggingFace `Trainer` / TRL `SFTTrainer`.
- Optional **tensor parallelism** (`FSDP_TP_SIZE > 1`) splits individual linear layers across GPUs within a node, further reducing per-device memory for long sequences.

### 2.6 Environment-driven configuration

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
│                          BeardedMonster/gemma-270m-translate-it     │
└─────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
  scripts/eval.py                      backend/ (Modal GPU)
  masakhane/mafand                     frontend/ (Vercel UI)
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

### 3.3 Distributed architecture

```
                    torchrun --nproc_per_node=N
                              │
              ┌───────────────┴───────────────┐
              │     setup_distributed()       │
              │  WORLD_SIZE=N, TP_SIZE=T      │
              │  DP_SIZE = N / T              │
              └───────────────┬───────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    DeviceMesh           DeviceMesh           DeviceMesh
    (dp=4, tp=2)         rank 0..7            ...
         │                    │
    apply_tensor_parallel   wrap_fsdp (policy)
    (attention/MLP shards)  wrap_fsdp (reference, frozen)
         │                    │
    FSDP shard on dp dim   DistributedSampler (data parallel)
```

**Key invariant:** `WORLD_SIZE` must be divisible by `FSDP_TP_SIZE`. Each tensor-parallel group shares layers; each data-parallel group receives a disjoint batch shard.

### 3.4 Prompt contract

All training, inference, and evaluation share one template:

```
### Instruction:
Translate the sentence below to Hausa:

### Source:
The meeting was held on Friday.

### Translation:
```

The model generates tokens after `### Translation:`. SFT loss is masked to the completion; inference strips the prompt prefix.

---

## 4. Engineering Bottlenecks

| Bottleneck | Where it appears | Mitigation in this repo |
|------------|------------------|-------------------------|
| **Dataset size (~11M rows)** | SFT data loading | Streaming (`DATASET_STREAMING=true`) + `MAX_TRAIN_SAMPLES` cap |
| **AfriCOMET inference latency** | RLHF inner loop | Batch scoring (`REWARD_BATCH_SIZE`); rank-0-only scoring + `broadcast_tensor` under FSDP |
| **FSDP `generate()` overhead** | RLHF sampling step | `summon_full_params` context — trades memory for correct autoregressive decoding |
| **Checkpoint gather** | End of training / Hub push | `FullStateDictConfig(rank0_only=True)` — all ranks participate, only rank 0 writes |
| **Cold start (Modal backend)** | Demo API | Container idle timeout (5 min); optional `keep_warm=1` |
| **Gemma license gating** | Model download | Requires accepted HF license + `HF_TOKEN` |
| **Reward–policy device mismatch** | Multi-GPU RL | AfriCOMET on `REWARD_GPUS`; rewards broadcast to all policy ranks |

---

## 5. Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| REINFORCE vs PPO | Simple, debuggable, no value head | Higher gradient variance; slower convergence |
| AfriCOMET black-box vs differentiable reward | Correct African-language metric; easy to swap versions | ~100–300 ms/batch scoring overhead |
| FSDP vs LoRA-only | Full-weight updates; multi-GPU scale | Requires `torchrun`; more complex checkpointing |
| LoRA + FSDP (`use_orig_params=True`) | Train 2B model on consumer GPUs | Slightly lower throughput vs full fine-tune |
| Streaming dataset | No 11M-row local download | Non-deterministic epoch boundaries when capped |
| Completion-only SFT loss | Better translation quality per step | Requires exact prompt template match at inference |
| Z-score reward normalization | Stabilizes RL across batches | Distorts absolute reward scale; batch-size dependent |
| Single published checkpoint | Simple demo/deploy story | SFT-only vs RLHF quality difference not exposed in UI |

---

## 6. Evaluation

### 6.1 Benchmark

We evaluate on **[MaFAND](https://huggingface.co/datasets/masakhane/mafand)** (`validation` split) — a multi-way parallel corpus covering English ↔ African languages with professional translations.

Default language pairs: `en-hau`, `en-ibo`, `en-yor`, `en-wol`, `en-ewe`, `en-fon`, `en-twi`.

### 6.2 Metrics

| Metric | Purpose |
|--------|---------|
| **BLEU** | N-gram overlap with reference (standard MT baseline) |
| **chrF** | Character n-gram F-score; more robust for morphologically rich languages |
| **AfriCOMET** | Learned metric trained on African languages; optional (`--use-africomet`) |

### 6.3 Running evaluation

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

### 6.4 Recommended reporting

For reproducible results, report:

1. Checkpoint path (SFT vs RLHF)
2. `MAX_TRAIN_SAMPLES` / `MAX_RL_SAMPLES` used
3. MaFAND validation split (never train on MaFAND)
4. Per-language BLEU, chrF, and AfriCOMET with macro-average across pairs

---

## 7. Edge Cases and Considerations

### Training

- **`FSDP_ENABLED=true` without `torchrun`** — Falls back to single-process mode; FSDP config is not applied. Always launch with `torchrun --nproc_per_node=N`.
- **`WORLD_SIZE % FSDP_TP_SIZE ≠ 0`** — Raises at startup. Example: 8 GPUs with `FSDP_TP_SIZE=2` → 4 data-parallel groups of 2 tensor-parallel ranks.
- **LoRA + FSDP** — Set `USE_LORA=true`; FSDP uses `use_orig_params=True` automatically.
- **Missing SFT checkpoint for RL** — `train_rlhf()` raises `FileNotFoundError` with an actionable message.
- **Empty translation after RL** — Usually indicates prompt template mismatch; verify `### Translation:\n` suffix in `src/prompts.py`.
- **AfriCOMET language coverage** — AfriCOMET-STL covers ~20 languages. Pairs outside its training distribution may receive miscalibrated scores; use `masakhane/africomet-stl-1.1` for broader coverage.

### Distributed RLHF

- **Reward scoring on rank 0 only** — Non-main ranks receive broadcast reward tensors; policy gradients remain consistent across ranks.
- **`summon_full_params` during generate** — Required for FSDP-wrapped models; increases peak memory during the sampling phase.
- **Hub push under FSDP** — All ranks participate in weight gather; only rank 0 uploads (see `push_to_hub`).

### Inference / Demo

- **Modal cold start** — First request after idle may take 30–60 s while the model downloads and loads.
- **Frontend without `NEXT_PUBLIC_API_URL`** — Returns a configuration error, not a silent failure.

### Data

- **Instruction field variance in tds-sft** — Some rows use free-form instructions; the model learns to follow varied phrasings. Evaluation uses a fixed instruction template per language.
- **Code-switching in source text** — Common in West African social media text; not explicitly filtered.

---

## 8. Related Academic Work

This project builds on and connects to the following research lines:

### Low-resource African MT

- **Masakhane** — Pan-African NLP community; MaFAND benchmark and AfriCOMET metric.  
  *Dossou et al., "AfriCOMET: Automatic Evaluation of Machine Translation for African Languages"*
- **TDS-SFT dataset** — Large-scale instruction SFT corpus for African languages.  
  [`Aletheia-ng/tds-sft`](https://huggingface.co/datasets/Aletheia-ng/tds-sft)

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
  *Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)*

### Base model

- **Gemma 2** — Google's open instruction-tuned family.  
  *Gemma Team, "Gemma 2: Improving Open Language Models at a Practical Size" (2024)*

---

## 9. Quick Start

### 9.1 Install

```bash
git clone <repo-url> west-african-mt-rlhf
cd west-african-mt-rlhf
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env             # set HF_TOKEN, adjust paths
```

Accept the Gemma license on Hugging Face before downloading `google/gemma-2-2b-it`.

### 9.2 Single-GPU training

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

### 9.3 Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_TOKEN` | — | Hugging Face API token |
| `HF_PUSH_REPO_ID` | `BeardedMonster/gemma-270m-translate-it` | Push checkpoint after training |
| `MODEL_NAME_OR_PATH` | `google/gemma-2-2b-it` | Base causal LM |
| `MAX_TRAIN_SAMPLES` | `50000` | Cap SFT examples |
| `MAX_RL_SAMPLES` | `10000` | Cap RL examples |
| `USE_LORA` | `false` | LoRA adapters (single-GPU friendly) |
| `FSDP_ENABLED` | `false` | Enable multi-GPU FSDP |
| `FSDP_TP_SIZE` | `1` | Tensor parallel degree |
| `FSDP_SHARDING` | `FULL_SHARD` | `FULL_SHARD`, `HYBRID_SHARD`, `SHARD_GRAD_OP` |
| `RL_KL_COEF` | `0.1` | KL penalty weight |

---

## 10. Distributed Training (FSDP)

### 10.1 Enable in `.env`

```env
FSDP_ENABLED=true
FSDP_TP_SIZE=1          # set to 2 for tensor parallelism within a node
FSDP_SHARDING=FULL_SHARD
FSDP_CPU_OFFLOAD=false
# FSDP_TRANSFORMER_LAYER_CLS=GemmaDecoderLayer   # auto-detected if blank
```

### 10.2 Launch commands

```bash
# 4-GPU data parallel (FSDP full shard)
torchrun --nproc_per_node=4 scripts/run_sft.py
torchrun --nproc_per_node=4 scripts/run_rlhf.py

# 8-GPU: 4 data-parallel groups × 2-way tensor parallel
# FSDP_TP_SIZE=2 in .env
torchrun --nproc_per_node=8 scripts/run_sft.py
```

### 10.3 Sharding strategies

| Strategy | Behavior | When to use |
|----------|----------|-------------|
| `FULL_SHARD` | Shard params, grads, optimizer states | Default; best memory efficiency |
| `HYBRID_SHARD` | Full shard intra-node, replicate inter-node | Multi-node clusters |
| `SHARD_GRAD_OP` | Shard grads/optimizer only | When params fit per-GPU but grads don't |

### 10.4 How it works (implementation)

All distributed logic lives in **`src/distributed.py`** — a single module reused by both training stages:

```
setup_distributed()     → init process group, build DeviceMesh (dp × tp)
apply_tensor_parallel() → shard linear layers on tp mesh (if TP > 1)
wrap_fsdp()             → FSDP auto-wrap per transformer layer (RLHF)
fsdp_trainer_config()   → HuggingFace Trainer FSDP kwargs (SFT)
summon_full_params()    → safe generate() under FSDP (RLHF)
save_fsdp_model()       → rank-0 full state dict gather
broadcast_tensor()      → sync AfriCOMET rewards across ranks
```

SFT delegates FSDP wrapping to TRL `SFTTrainer`. RLHF wraps policy and reference models explicitly and uses `DistributedSampler` for data parallelism.

---

## 11. Project Layout

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
├── backend/                      # Modal serverless GPU API
└── frontend/                     # Next.js demo (Vercel)
```

---

## 12. Deployment

### Backend — Modal

```bash
cd backend && pip install -r requirements.txt
modal setup
modal deploy backend/app.py    # copy the printed URL
```

Serves `BeardedMonster/gemma-270m-translate-it` on a T4 GPU. See `backend/README.md`.

### Frontend — Vercel

```bash
cd frontend && npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL
npm run dev                        # http://localhost:3000
```

Set `NEXT_PUBLIC_API_URL` in Vercel project settings, root directory `frontend/`.

---

## Tech Stack

| Library | Role |
|---------|------|
| PyTorch ≥ 2.1 | Training, FSDP, generation |
| Transformers ≥ 4.44 | Causal LM, tokenizer, TP integration |
| TRL ≥ 0.9.6 | `SFTTrainer`, completion-only collator |
| PEFT ≥ 0.12 | Optional LoRA |
| unbabel-comet ≥ 2.2 | AfriCOMET reward |
| sacrebleu ≥ 2.4 | BLEU / chrF evaluation |
| Datasets ≥ 2.19 | HF streaming |
| Accelerate ≥ 0.33 | Mixed precision, FSDP trainer backend |

---

## License and Attribution

- **Gemma 2** — subject to [Google Gemma license](https://ai.google.dev/gemma/terms).
- **AfriCOMET / MaFAND** — Masakhane community resources; cite accordingly in academic use.
- **tds-sft** — Aletheia-ng dataset; verify license before redistribution.

When using this work academically, please cite the Masakhane, COMET/AfriCOMET, and Gemma papers listed in [Section 8](#8-related-academic-work).
