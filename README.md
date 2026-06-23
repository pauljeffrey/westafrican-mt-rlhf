# West African MT + AfriCOMET RLHF

Fine-tune a Hugging Face causal language model for **West African machine translation** using a two-stage pipeline:

1. **Supervised Fine-Tuning (SFT)** — trains on `Aletheia-ng/tds-sft`, an instruction-following dataset of translation examples (`instruction` + `input` → `response`).
2. **RLHF** — improves translations via a REINFORCE-style policy gradient using [AfriCOMET](https://huggingface.co/masakhane/africomet-stl) as a black-box reward signal, with a KL penalty against a frozen reference model to prevent drift.

The default base model is **`google/gemma-2-2b-it`**. Targeted languages include Hausa, Igbo, Yoruba, Wolof, Ewe, Fon, Twi, and Nigerian Pidgin.

---

## Pipeline

```
HF Dataset (Aletheia-ng/tds-sft)
         │
         ▼
   run_sft.py  ──►  outputs/sft/      (TRL SFTTrainer, completion-only loss)
         │
         ▼
  run_rlhf.py  ──►  outputs/rlhf/     (REINFORCE + KL penalty)
         ▲
 AfriCOMET reward  (masakhane/africomet-stl)
         │
         ▼
  translate.py                         (greedy decode from RL checkpoint)
         │
         ▼
    eval.py  ──►  masakhane/mafand     (BLEU · chrF · AfriCOMET per language)
```

**RL loss:** `-(reward × log_prob).mean() + kl_coef × KL(policy ∥ reference)`

AfriCOMET is called as a black-box scorer — no differentiable reward head required.

---

## Project Layout

```
west-african-mt-rlhf/
├── README.md
├── pyproject.toml          # package metadata + core dependencies
├── requirements.txt        # full pip install list
├── .env.example            # configuration template
├── configs/
│   └── default.yaml        # reference defaults (informational; .env takes precedence)
├── scripts/
│   ├── run_sft.py          # Stage 1: supervised fine-tuning
│   ├── run_rlhf.py         # Stage 2: AfriCOMET-guided RLHF
│   ├── translate.py        # inference CLI
│   └── eval.py             # benchmark evaluation (masakhane/mafand)
├── src/
│   ├── __init__.py
│   ├── config.py           # loads .env into a frozen Settings dataclass
│   ├── data.py             # streams and formats Aletheia-ng/tds-sft
│   ├── prompts.py          # prompt templates + output extraction
│   ├── model_utils.py      # tokenizer/model loading, optional LoRA
│   ├── reward.py           # AfriCOMET wrapper (batch scoring)
│   ├── sft_train.py        # SFT training logic (TRL SFTTrainer)
│   ├── rlhf_train.py       # custom RL training loop
│   └── inference.py        # translate() generation helper
├── backend/                # Modal serverless inference API
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── frontend/               # Next.js demo UI (Vercel)
    ├── src/app/            # App Router pages
    ├── src/lib/            # API client + language data
    ├── vercel.json
    └── .env.local.example
```

Generated at runtime (gitignored): `outputs/sft/`, `outputs/rlhf/`, `.cache/`, `wandb/`

---

## Setup

```bash
cd west-african-mt-rlhf
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
copy .env.example .env          # then fill in HF_TOKEN and adjust paths
```

Accept the Gemma license on Hugging Face before using `google/gemma-2-2b-it`.

---

## Configuration

All settings live in `.env` (copy from `.env.example`). The file is loaded by `src/config.py` into a frozen `Settings` dataclass at import time. `configs/default.yaml` documents the same defaults but is not loaded by any code.

| Category | Variable | Default | Purpose |
|----------|----------|---------|---------|
| **Auth** | `HF_TOKEN` | — | Hugging Face API token |
| | `HF_PUSH_REPO_ID` | `BeardedMonster/gemma-270m-translate-it` | Push trained model to this HF repo (leave blank to skip) |
| **Model** | `MODEL_NAME_OR_PATH` | `google/gemma-2-2b-it` | Base causal LM |
| | `TRUST_REMOTE_CODE` | `false` | Allow custom model code |
| **Reward** | `REWARD_MODEL_ID` | `masakhane/africomet-stl` | AfriCOMET checkpoint |
| | `REWARD_BATCH_SIZE` | `8` | Scoring batch size |
| | `REWARD_GPUS` | `0` | GPU index for AfriCOMET |
| **Dataset** | `DATASET_ID` | `Aletheia-ng/tds-sft` | SFT training dataset |
| | `MAX_TRAIN_SAMPLES` | `50000` | Cap SFT examples |
| | `MAX_RL_SAMPLES` | `10000` | Cap RL examples |
| | `DATASET_STREAMING` | `true` | Stream without full download |
| **Paths** | `SFT_OUTPUT_DIR` | `./outputs/sft` | SFT checkpoint (input to RL) |
| | `RL_OUTPUT_DIR` | `./outputs/rlhf` | Final model checkpoint |
| **Generation** | `MAX_NEW_TOKENS` | `128` | Max tokens to generate |
| | `MAX_SOURCE_LEN` | `256` | Source truncation length |
| **SFT** | `SFT_LR` | `2e-5` | Learning rate |
| | `SFT_EPOCHS` | `1` | Training epochs |
| | `SFT_BF16` | `true` | Mixed precision (requires CUDA) |
| **RLHF** | `RL_KL_COEF` | `0.1` | KL penalty weight |
| | `RL_TEMPERATURE` | `0.7` | Sampling temperature |
| | `RL_REWARD_NORMALIZE` | `true` | Z-score normalize rewards |
| | `RL_MAX_STEPS` | `1000` | Hard step cap |
| **LoRA** | `USE_LORA` | `false` | Enable LoRA (recommended for single GPU) |
| | `LORA_R` | `16` | LoRA rank |
| | `LORA_ALPHA` | `32` | LoRA scaling |
| **Logging** | `WANDB_ENABLED` | `false` | Enable Weights & Biases |
| | `WANDB_PROJECT` | `west-african-mt-rlhf` | W&B project name |

---

## Training

```bash
# Stage 1 — supervised fine-tuning
python scripts/run_sft.py

# Stage 2 — AfriCOMET-guided RLHF (requires SFT checkpoint at SFT_OUTPUT_DIR)
python scripts/run_rlhf.py
```

**SFT details:**
- Concatenates `prompt + response` into a single `text` field
- Uses `DataCollatorForCompletionOnlyLM` so loss is applied only after `### Translation:\n`
- Saves model + tokenizer to `SFT_OUTPUT_DIR`

**RLHF details:**
- Loads both policy and frozen reference from the SFT checkpoint
- Samples translations with temperature / top-p
- Scores each translation with AfriCOMET (optionally batch-normalized)
- Runs AdamW with linear warmup; respects `RL_MAX_STEPS` as a hard cap

---

## Inference

```bash
python scripts/translate.py ^
  --instruction "Translate the sentence below to Hausa:" ^
  --input "One of the earliest geographers to study this relationship was Friedrich Ratzel."

# Use a specific checkpoint instead of the default RL output
python scripts/translate.py ^
  --instruction "Translate the sentence below to Igbo:" ^
  --input "The market opens at dawn." ^
  --model-path ./outputs/sft
```

---

## Evaluation

Evaluate any checkpoint against the [masakhane/mafand](https://huggingface.co/datasets/masakhane/mafand) benchmark on the `validation` split. Metrics reported: **BLEU**, **chrF**, and optionally **AfriCOMET**.

```bash
# Default: evaluate all 7 language pairs
python scripts/eval.py --model-path ./outputs/rlhf

# Evaluate a subset of languages
python scripts/eval.py --model-path ./outputs/rlhf --langs en-hau,en-ibo,en-yor

# Include AfriCOMET scoring (slower; requires GPU)
python scripts/eval.py --model-path ./outputs/rlhf --use-africomet

# Limit examples per language for a quick check
python scripts/eval.py --model-path ./outputs/rlhf --max-samples 200

# Save full results to JSON
python scripts/eval.py --model-path ./outputs/rlhf --output-file results.json
```

Supported language pairs: `en-ibo`, `en-hau`, `en-yor`, `en-wol`, `en-ewe`, `en-fon`, `en-twi`

---

## Prompt Format

All scripts share the same structured prompt template:

```
### Instruction:
Translate the sentence below to Hausa:

### Source:
The meeting was held on Friday.

### Translation:
```

The model generates the text after `### Translation:`. The SFT trainer applies loss only to the completion; inference strips everything before the translation.

---

## Tech Stack

| Library | Role |
|---------|------|
| PyTorch ≥ 2.1 | Training, generation, RL loss |
| Transformers ≥ 4.44 | Causal LM + tokenizer |
| TRL ≥ 0.9.6 | `SFTTrainer`, completion-only collator |
| PEFT ≥ 0.12 | Optional LoRA fine-tuning |
| Datasets ≥ 2.19 | HF dataset streaming |
| unbabel-comet ≥ 2.2 | AfriCOMET reward model |
| sacrebleu ≥ 2.4 | BLEU and chrF evaluation metrics |
| bitsandbytes | Quantization (non-macOS) |
| wandb | Optional experiment tracking |

---

## Deployment

### Backend — Modal (serverless GPU)

The `backend/` folder contains a Modal app that loads `BeardedMonster/gemma-270m-translate-it` on a T4 GPU and exposes a REST API.

```bash
cd backend
pip install -r requirements.txt
modal setup                         # authenticate once
modal deploy backend/app.py         # prints your permanent API URL
```

See `backend/README.md` for full instructions, including private model tokens and GPU configuration.

### Frontend — Vercel

The `frontend/` folder is a standalone Next.js app with a language selector, sample examples, and a translation panel.

```bash
cd frontend
npm install
cp .env.local.example .env.local    # add NEXT_PUBLIC_API_URL from Modal deploy
npm run dev                         # http://localhost:3000
```

**Deploying to Vercel:**

1. Push `frontend/` to a GitHub repo (or connect the monorepo with root directory set to `frontend/`).
2. Add the environment variable `NEXT_PUBLIC_API_URL` in the Vercel project settings.
3. Vercel auto-builds on every push.

Alternatively, use the Vercel CLI:

```bash
cd frontend
npm i -g vercel
vercel --prod
```

---

## Notes

- AfriCOMET-STL covers ~20 languages. For broader coverage use `masakhane/africomet-stl-1.1` in `.env`.
- The SFT dataset has ~11M rows; keep `MAX_*_SAMPLES` modest until you scale hardware.
- LoRA (`USE_LORA=true`) is recommended when training Gemma-2-2B on a single GPU.
- `configs/default.yaml` exists as a reference document but is not read by any Python code — `.env` is the live configuration.
