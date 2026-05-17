RESPONSE_TEMPLATE = "### Translation:\n"


def build_prompt(instruction: str, source: str) -> str:
    instruction = (instruction or "").strip()
    source = (source or "").strip()
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Source:\n"
        f"{source}\n\n"
        f"{RESPONSE_TEMPLATE}"
    )


def extract_translation(generated: str, prompt: str) -> str:
    """Strip prompt echo and keep only the model translation."""
    text = generated
    if text.startswith(prompt):
        text = text[len(prompt) :]
    if RESPONSE_TEMPLATE in text:
        text = text.split(RESPONSE_TEMPLATE, 1)[-1]
    return text.strip().split("\n###")[0].strip()
