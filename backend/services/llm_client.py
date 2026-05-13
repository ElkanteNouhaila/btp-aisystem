from openai import OpenAI
from config import HF_TOKEN, HF_MODEL

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


def _extract_message_text(message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return " ".join(parts).strip()
    return ""


def generate_answer(prompt):
    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=512
    )

    choice = response.choices[0]
    stripped = _extract_message_text(choice.message)
    if stripped:
        return stripped

    fallback_text = getattr(choice, "text", None)
    if isinstance(fallback_text, str) and fallback_text.strip():
        return fallback_text.strip()

    finish = getattr(choice, "finish_reason", None)
    raise RuntimeError(
        f"Model returned no text (finish_reason={finish!r}). "
        "Try a shorter context, larger max_tokens, or another HF_MODEL."
    )