"""Modal serverless inference app for West African MT.

Deploy:
    modal deploy backend/app.py

Run locally:
    modal serve backend/app.py

After deploying, Modal prints the web endpoint URL — add it to the
frontend as NEXT_PUBLIC_API_URL.
"""

import modal
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modal app & image
# ---------------------------------------------------------------------------

app = modal.App("west-african-mt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "transformers>=4.44.0",
        "accelerate>=0.33.0",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "BeardedMonster/gemma-270m-translate-it"

SUPPORTED_LANGS: dict[str, str] = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "yor": "Yoruba",
    "wol": "Wolof",
    "ewe": "Ewe",
    "fon": "Fon",
    "twi": "Twi",
}

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n"
    "### Source:\n{source}\n\n"
    "### Translation:\n"
)

# ---------------------------------------------------------------------------
# Model class — loads once per container, reused across requests
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="T4",
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
class TranslationModel:
    @modal.enter()
    def load(self) -> None:
        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        self.model.eval()

    @modal.method()
    def translate(self, text: str, target_lang: str) -> str:
        import torch

        lang_name = SUPPORTED_LANGS.get(target_lang, target_lang.capitalize())
        instruction = f"Translate the sentence below to {lang_name}:"
        prompt = PROMPT_TEMPLATE.format(instruction=instruction, source=text.strip())

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True)

        if "### Translation:\n" in decoded:
            result = decoded.split("### Translation:\n", 1)[-1]
            result = result.split("\n###")[0]
            return result.strip()
        return decoded.strip()


# ---------------------------------------------------------------------------
# FastAPI ASGI web app
# ---------------------------------------------------------------------------

_model = TranslationModel()

web_app = modal.fastapi_app()


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


class TranslateResponse(BaseModel):
    translation: str
    target_lang: str
    lang_name: str


@web_app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    from fastapi import HTTPException

    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    if req.target_lang not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{req.target_lang}'. "
            f"Supported: {list(SUPPORTED_LANGS.keys())}",
        )

    translation = await _model.translate.remote.aio(req.text, req.target_lang)
    return TranslateResponse(
        translation=translation,
        target_lang=req.target_lang,
        lang_name=SUPPORTED_LANGS[req.target_lang],
    )


@web_app.get("/languages")
def languages():
    return {"languages": [{"code": k, "name": v} for k, v in SUPPORTED_LANGS.items()]}


@web_app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.function(image=image)
@modal.asgi_app()
def serve():
    from fastapi.middleware.cors import CORSMiddleware

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    return web_app
