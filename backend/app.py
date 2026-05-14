from flask import Flask, request, jsonify
from services.ingestion import extract_text_from_pdf
from services.embeddings import embeddings
from services.pinecone_db import index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from apscheduler.schedulers.background import BackgroundScheduler
from services.llm_client import generate_answer
from utils.text_cleaning import clean_text
from services.retrieval import retrieve_context
from services.email_ingestion import process_email, process_email_from_text
from services.gmail_service import fetch_emails
from flask_cors import CORS
import os
import uuid


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "data/raw_docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NAMESPACE = "documents"
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "600"))
EMBED_BATCH_SIZE = 32

# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "BTP AI System is running"


# =========================
# ASK ROUTE (RAG)
# =========================
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}

    question = data.get("question")
    doc_id = data.get("doc_id")
    doc_type = data.get("type")
    source = data.get("source")

    if isinstance(doc_id, str):
        doc_id = doc_id.strip() or None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # =========================
    # FILTERS
    # =========================
    filters = {}

    if doc_type:
        filters["type"] = doc_type

    if source:
        filters["source"] = source

    # =========================
    # RETRIEVAL
    # =========================
    context, sources = retrieve_context(
        question=question,
        doc_id=doc_id,
        doc_type=doc_type,
        source=source
    )

    if not context or not context.strip():
        return jsonify({
            "question": question,
            "answer": "No document text found.",
            "sources": sources,
            "retrieval_empty": True,
            "doc_id": doc_id
        }), 200

    # =========================
    # PROMPT
    # =========================
    prompt = f"""
You are an AI assistant that answers questions using the provided document.

Instructions:
- Give a short explanation based only on the document
- Use ONLY the document
- Do NOT add extra details
If the information appears partially in the document,
answer using the closest matching content.
Only say "Information not found in the document"
if the information is completely absent.

Question:
{question}

Document:
{context}

Answer:
"""

    # =========================
    # LLM
    # =========================
    try:
        answer = generate_answer(prompt)

    except Exception as e:
        print("LLM ERROR:", e)

        return jsonify({
            "error": str(e),
            "answer": "",
            "sources": sources
        }), 502

    return jsonify({
        "question": question,
        "answer": answer.strip(),
        "sources": sources,
        "doc_id": doc_id
    })

# =========================
# UPLOAD ROUTE
# =========================
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    doc_id = str(uuid.uuid4())

    text = extract_text_from_pdf(filepath)
    text = clean_text(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "? ", "! ", ";"]
    )

    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if len(c.strip()) > 30]

    if MAX_CHUNKS_PER_DOC > 0:
        chunks = chunks[:MAX_CHUNKS_PER_DOC]

    vectors = []

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        batch_vectors = embeddings.embed_documents(batch)

        for offset, (chunk, vector) in enumerate(zip(batch, batch_vectors)):
            i = start + offset

            vectors.append({
                "id": f"{doc_id}_{i}",
                "values": vector,
                "metadata": {
                    "text": chunk,
                    "source": file.filename,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "type": "pdf"
                }
            })

    index.upsert(vectors=vectors, namespace=NAMESPACE)

    return jsonify({
        "message": "file indexed in Pinecone",
        "doc_id": doc_id,
        "chunks_indexed": len(vectors)
    })

@app.route("/upload-email", methods=["POST"])
def upload_email():
    if "file" not in request.files:
        return jsonify({"error": "No email uploaded"}), 400

    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    doc_id = str(uuid.uuid4())

    # Step 1: extract + clean
    email_data = process_email(filepath, doc_id)
    text = clean_text(email_data["text"])

    # Step 2: chunk email properly
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_text(text)

    # Safety filter (avoid empty chunks)
    chunks = [c for c in chunks if c and len(c.strip()) > 20]

    vectors = []

    # Step 3: embed + prepare vectors
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk)

        vectors.append({
            "id": f"{doc_id}_{i}",
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": file.filename,
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "type": "email"
            }
        })

    # Step 4: store in Pinecone
    index.upsert(vectors=vectors, namespace=NAMESPACE)

    # Debug logs
    print("EMAIL VECTORS UPLOADED:", len(vectors))
    if vectors:
        print("SAMPLE METADATA:", vectors[0]["metadata"])

    return jsonify({
        "message": "email indexed",
        "doc_id": doc_id,
        "chunks_indexed": len(vectors)
    })

@app.route("/sync-gmail", methods=["GET"])
def sync_gmail():
    print("STEP 1 - Starting Gmail sync")

    emails = fetch_emails()

    print("STEP 2 - Emails fetched:", len(emails))

    for email in emails:
        print("STEP 3 - Processing email")

        # -------------------------
        # CHECK IF EMAIL EXISTS
        # -------------------------
        existing = index.query(
            vector=[0] * 384,
            top_k=1,
            include_metadata=True,
            namespace="documents",
            filter={
                "message_id": email["message_id"]
            }
        )

        already_exists = len(existing["matches"]) > 0

        if already_exists:
            print("EMAIL ALREADY INDEXED -> SKIPPED")
            continue

        # -------------------------
        # NEW EMAIL -> INDEX IT
        # -------------------------
        doc_id = str(uuid.uuid4())

    result = process_email_from_text(
        text=email["text"],
        doc_id=doc_id,
        email_date=email["date"],
        message_id=email["message_id"],
        sender=email["from"],
        subject=email["subject"]
    )

    print("STEP 4 - Indexed:", result)
# @app.route("/docs", methods=["GET"])
# def list_docs():
#     stats = index.describe_index_stats()

#     namespaces = stats.get("namespaces", {})
#     doc_list = []

#     for ns, data in namespaces.items():
#         doc_list.append({
#             "namespace": ns,
#             "vector_count": data.get("vector_count", 0)
#         })

#     return jsonify(doc_list)
    
@app.route("/docs", methods=["GET"])
def list_docs():
    results = index.query(
        vector=[0] * 768,
        top_k=100,
        include_metadata=True,
        namespace=NAMESPACE
    )

    seen = {}

    for match in results["matches"]:
        meta = match.get("metadata", {})

        doc_id = meta.get("doc_id")
        source = meta.get("source")

        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "doc_id": doc_id,
                "filename": source
            }

    return jsonify(list(seen.values()))



def sync_gmail_background():
    print("AUTO SYNC STARTED")

    emails = fetch_emails()

    for email in emails:
        text = email.get("text", "")

        if not text.strip():
            continue

        doc_id = str(uuid.uuid4())

        process_email_from_text(text, doc_id)

    print("AUTO SYNC FINISHED")


scheduler = BackgroundScheduler()

scheduler.add_job(
    sync_gmail_background,
    "interval",
    minutes=1
)

scheduler.start()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)