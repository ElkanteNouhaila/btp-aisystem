import unicodedata
import re

# CLEAN TEXT
# =========================
def clean_text(text):
    if not text:
        return ""

    # Normalize unicode (fix accents)
    text = unicodedata.normalize("NFKC", text)

    # Remove line breaks
    text = text.replace("\n", " ")

    # ❗ Fix spaced letters ONLY when needed (less aggressive)
    text = re.sub(r'(?:(?<=\s)|^)([A-Za-zÀ-ÿ])(?:\s+[A-Za-zÀ-ÿ]){2,}',
                  lambda m: m.group(0).replace(" ", ""), text)

    # Fix glued words (camelCase → split)
    text = re.sub(r"([a-zà-ÿ])([A-ZÀ-Ÿ])", r"\1 \2", text)

    # Fix tech words
    text = re.sub(r"\b(Java|Type|Node|React)\s+(Script)\b", r"\1\2", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()