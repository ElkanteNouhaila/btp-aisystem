import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.text_cleaning import clean_text
from services.embeddings import embeddings
from services.pinecone_db import index

NAMESPACE = "documents"


# -----------------------------
# Load email from file (optional use)
# -----------------------------
def load_email_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# -----------------------------
# PROCESS EMAIL FROM FILE (optional)
# -----------------------------
def process_email(file_path: str, doc_id: str):
    text = load_email_text(file_path)
    text = clean_text(text)

    return {
        "text": text,
        "doc_id": doc_id,
        "source": os.path.basename(file_path),
        "type": "email"
    }


# -----------------------------
# PROCESS EMAIL FROM GMAIL TEXT (MAIN PIPELINE)
# -----------------------------
def process_email_from_text(
    text: str,
    doc_id: str,
    email_date: str = "",
    message_id: str = "",
    sender: str = "",
    subject: str = ""
):
    text = clean_text(text)

    if not text or len(text.strip()) < 10:
        return {
            "doc_id": doc_id,
            "chunks": 0
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    chunks = splitter.split_text(text)

    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        return {
            "doc_id": doc_id,
            "chunks": 0
        }

    vectors = []

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_documents([chunk])[0]

        vectors.append({
            "id": f"{doc_id}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "doc_id": doc_id,
                "source": "gmail_sync",
                "type": "email",
                "chunk_index": i,
                "date": email_date,
                "message_id": message_id,
                "sender": sender,
                "subject": subject
            }
        })

    if vectors:
        index.upsert(
            vectors=vectors,
            namespace=NAMESPACE
        )

    return {
        "doc_id": doc_id,
        "chunks": len(vectors)
    }