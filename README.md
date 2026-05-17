# West African MT + AfriCOMET RLHF

Fine-tune a Hugging Face causal LM on [Aletheia-ng/tds-sft](https://huggingface.co/datasets/Aletheia-ng/tds-sft), then improve translations with reinforcement learning using [AfriCOMET](https://huggingface.co/masakhane/africomet-stl) as the reward model.

## Pipeline

1. **SFT** — supervised fine-tuning on `instruction` + `input` → `response`
2. **RLHF** — policy gradient with AfriCOMET scores and KL penalty vs a frozen reference

## Setup

```bash
cd west-african-mt-rlhf
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # set HF_TOKEN and paths
```

Accept the Gemma license on Hugging Face if using `google/gemma-2-2b-it`.

## Configuration

All settings live in `.env` (see `.env.example`). Key variables:

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Hugging Face API token |
| `MODEL_NAME_OR_PATH` | Base model (e.g. `google/gemma-2-2b-it`) |
| `REWARD_MODEL_ID` | AfriCOMET checkpoint |
| `DATASET_ID` | `Aletheia-ng/tds-sft` |
| `MAX_TRAIN_SAMPLES` / `MAX_RL_SAMPLES` | Subsample large dataset |
| `SFT_OUTPUT_DIR` | Checkpoint for RL stage |
| `USE_LORA` | Enable LoRA for smaller GPUs |

## Training

```bash
# Stage 1: supervised fine-tuning
python scripts/run_sft.py

# Stage 2: AfriCOMET-guided RLHF (requires SFT checkpoint)
python scripts/run_rlhf.py
```

## Inference

```bash
python scripts/translate.py ^
  --instruction "Translate the sentence below to Hausa:" ^
  --input "One of the earliest geographers to study this relationship was Friedrich Ratzel."
```

## Notes

- AfriCOMET-STL is most reliable for its ~20 covered languages (Hausa, Igbo, Yoruba, Nigerian Pidgin, etc.). For broader coverage use `masakhane/africomet-stl-1.1` in `.env`.
- The dataset has ~11M rows; keep `MAX_*_SAMPLES` modest until you scale hardware.
- RL uses a lightweight policy-gradient + KL setup so AfriCOMET can be called directly (no differentiable reward head required).

## Project layout

```
src/
  config.py      # loads .env
  data.py        # dataset + formatting
  prompts.py     # prompt templates
  reward.py      # AfriCOMET wrapper
  sft_train.py   # SFT trainer
  rlhf_train.py  # RL trainer
  inference.py   # generation helper
scripts/
  run_sft.py
  run_rlhf.py
  translate.py
```
