def build_prompt(instruction: str, source: str) -> str:
    instruction = (instruction or "").strip()
    source = (source or "").strip()
    return f"{instruction}\n{source}".strip()


def extract_translation(generated: str, prompt: str) -> str:
    """Strip prompt echo and keep only the model translation."""
    text = generated
    if text.startswith(prompt):
        text = text[len(prompt) :]
    return text.strip()
