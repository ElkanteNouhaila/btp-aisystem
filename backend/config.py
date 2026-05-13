import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/gemma-4-31B-it")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is missing from .env")