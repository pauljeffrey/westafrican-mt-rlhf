# Backend — Modal Serverless Inference

Serves `BeardedMonster/gemma-3-270m-translate-it` on a T4 GPU via [Modal](https://modal.com). The container idles for 5 minutes after the last request and spins back up on demand.

## API

### `POST /translate`
```json
// Request
{ "text": "The market opens early.", "target_lang": "hau" }

// Response
{ "translation": "Kasuwa ta buɗe da wuri.", "target_lang": "hau", "lang_name": "Hausa" }
```

Supported `target_lang` codes: `hau` · `ibo` · `yor` · `wol` · `ewe` · `fon` · `twi`

### `GET /languages`
Returns the list of supported language codes and names.

### `GET /health`
Returns `{ "status": "ok", "model": "..." }`.

---

## Setup

```bash
cd backend
pip install -r requirements.txt
modal setup          # authenticates your Modal account
```

If the model is private, create a Modal secret named `huggingface-secret` with a `HF_TOKEN` key:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
```

---

## Local development

```bash
modal serve backend/app.py
```

Modal prints a temporary URL (e.g. `https://your-name--west-african-mt-serve.modal.run`).  
Use that as `NEXT_PUBLIC_API_URL` in the frontend during development.

---

## Deploy

```bash
modal deploy backend/app.py
```

Modal prints the permanent deployment URL — copy it into the frontend's `NEXT_PUBLIC_API_URL` environment variable on Vercel.

---

## Notes

- The `TranslationModel` class loads the model once per container via `@modal.enter()`.  
  Subsequent requests to the same container reuse the loaded weights (no cold-start overhead).
- Cold start (~30–60 s on first request) downloads the model from HuggingFace Hub.  
  Set `keep_warm=1` in `@app.cls(...)` to keep one container always hot (adds cost).
- Change `gpu="T4"` to `gpu="A10G"` for faster throughput with batched requests.
