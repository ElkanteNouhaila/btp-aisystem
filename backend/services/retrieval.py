import re
from typing import List, Optional, Tuple

from services.embeddings import embeddings
from services.pinecone_db import index
from utils.text_cleaning import clean_text

NAMESPACE = "documents"
MIN_SCORE = 0.15
STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "at", "by",
    "is", "are", "was", "were", "be", "this", "that", "it", "as", "with", "from",
}


def _pinecone_doc_filter(doc_id: Optional[str]) -> Optional[dict]:
    if doc_id is None:
        return None
    trimmed = doc_id.strip()
    if not trimmed:
        return None

    if trimmed.lower() in {"documents", "all", "all documents"}:
        return None
    return {"doc_id": {"$eq": trimmed}}


def _question_terms(question: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (question or "").lower()))


def _signal_terms(question: str) -> set:
    return {t for t in _question_terms(question) if t not in STOPWORDS and len(t) > 2}


def _exact_phrases(question: str) -> List[str]:
    # Quoted snippets in user queries are strong retrieval hints.
    return [p.lower() for p in re.findall(r'"([^"]+)"', question or "") if p.strip()]


def _as_metadata_dict(meta) -> dict:
    """Pinecone may return metadata as dict, pydantic model, or mapping-like object."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    model_dump = getattr(meta, "model_dump", None)
    if callable(model_dump):
        try:
            d = model_dump()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    try:
        return dict(meta)
    except Exception:
        pass
    if hasattr(meta, "items"):
        try:
            return {str(k): v for k, v in meta.items()}
        except Exception:
            pass
    return {}


def _match_metadata_dict(match) -> dict:
    raw = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", None)
    return _as_metadata_dict(raw)


def _chunk_text_from_metadata(meta: dict) -> str:
    t = meta.get("text")
    if t is None:
        return ""
    return str(t).strip()


def _iter_matches(results):
    if getattr(results, "matches", None) is not None:
        return list(results.matches)
    if isinstance(results, dict):
        return list(results.get("matches") or [])
    return []


def _score(m) -> float:
    if isinstance(m, dict):
        return float(m.get("score") or 0)
    return float(getattr(m, "score", None) or 0)


def _chunk_index(match, meta: dict) -> int:
    idx = meta.get("chunk_index")
    if isinstance(idx, int):
        return idx
    match_id = match.get("id") if isinstance(match, dict) else getattr(match, "id", "")
    if isinstance(match_id, str):
        m = re.search(r"_(\d+)$", match_id)
        if m:
            return int(m.group(1))
    return -1


def _dedupe_chunks_preserve_order(chunks: List[str]) -> List[str]:
    """Drop identical chunk bodies (Pinecone often returns near-duplicate matches)."""
    seen: set = set()
    out: List[str] = []
    for c in chunks:
        key = re.sub(r"\s+", " ", c.strip()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _collect_scored_chunks(matches_raw: list, min_score: float, question: str) -> List[Tuple[str, float, int]]:
    terms = _question_terms(question)
    signal_terms = _signal_terms(question)
    phrases = _exact_phrases(question)
    chunks: List[Tuple[str, float, int]] = []
    for m in matches_raw:
        meta = _match_metadata_dict(m)
        raw = _chunk_text_from_metadata(meta)
        if not raw:
            continue
        score = _score(m)
        if score <= min_score:
            continue
        cleaned = clean_text(raw)
        text = cleaned if cleaned.strip() else raw
        chunk_terms = _question_terms(text)
        overlap = len(terms & chunk_terms)
        signal_overlap = len(signal_terms & chunk_terms)
        lower_text = text.lower()
        phrase_hits = sum(1 for p in phrases if p in lower_text)
        # Pinecone score is primary; lexical and phrase bonuses help clause-level precision.
        rank_score = score + (overlap * 0.01) + (signal_overlap * 0.03) + (phrase_hits * 0.12)
        chunks.append((text, rank_score, _chunk_index(m, meta)))

    if not chunks:
        for m in matches_raw:
            meta = _match_metadata_dict(m)
            raw = _chunk_text_from_metadata(meta)
            if not raw:
                continue
            score = _score(m)
            cleaned = clean_text(raw)
            text = cleaned if cleaned.strip() else raw
            chunk_terms = _question_terms(text)
            overlap = len(terms & chunk_terms)
            signal_overlap = len(signal_terms & chunk_terms)
            lower_text = text.lower()
            phrase_hits = sum(1 for p in phrases if p in lower_text)
            rank_score = score + (overlap * 0.01) + (signal_overlap * 0.03) + (phrase_hits * 0.12)
            chunks.append((text, rank_score, _chunk_index(m, meta)))

    chunks.sort(key=lambda x: x[1], reverse=True)
    return chunks


def _pick_diverse(scored: List[Tuple[str, float, int]], max_items: int = 5) -> List[str]:
    selected: List[str] = []
    selected_idx: List[int] = []
    for text, _, idx in scored:
        # Prefer coverage across the document over adjacent near-duplicate chunks.
        if idx >= 0 and any(abs(idx - sidx) <= 1 for sidx in selected_idx):
            continue
        selected.append(text)
        if idx >= 0:
            selected_idx.append(idx)
        if len(selected) >= max_items:
            break
    if len(selected) < max_items:
        for text, _, _ in scored:
            if text in selected:
                continue
            selected.append(text)
            if len(selected) >= max_items:
                break
    return selected


def _query_index(query_vector: list, top_k: int, filt: Optional[dict]):
    q_kwargs = {
        "vector": query_vector,
        "top_k": top_k,
        "include_metadata": True,
        "namespace": NAMESPACE,
    }
    if filt is not None:
        q_kwargs["filter"] = filt
    return index.query(**q_kwargs)



def retrieve_context(
    question: str,
    top_k: int = 18,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
    source: Optional[str] = None
) -> Tuple[str, List[str]]:

    # =========================
    # EMBEDDING
    # =========================
    query_vector = embeddings.embed_query(question)

    # =========================
    # BUILD FILTER
    # =========================
    filt = {}

    if doc_id:
        filt["doc_id"] = doc_id

    if doc_type:
        filt["type"] = doc_type

    if source:
        filt["source"] = source

    if not filt:
        filt = None

    # =========================
    # QUERY PINECONE
    # =========================
    results = _query_index(
        query_vector=query_vector,
        top_k=top_k,
        filt=filt
    )

    matches_raw = _iter_matches(results)

    chunks_scored = _collect_scored_chunks(
        matches_raw,
        MIN_SCORE,
        question
    )

    chunks = [c for c, _, _ in chunks_scored]

    # =========================
    # FALLBACK
    # =========================
    if not chunks and filt is not None:

        print("FILTERED SEARCH EMPTY -> FALLBACK")

        results = _query_index(
            query_vector=query_vector,
            top_k=top_k,
            filt=None
        )

        matches_raw = _iter_matches(results)

        chunks_scored = _collect_scored_chunks(
            matches_raw,
            MIN_SCORE,
            question
        )

        chunks = [c for c, _, _ in chunks_scored]

    # =========================
    # DEDUPE
    # =========================
    chunks = _dedupe_chunks_preserve_order(chunks)

    # =========================
    # RERANK
    # =========================
    reranked_scored = []

    for c in chunks:
        for text, score, idx in chunks_scored:

            if text == c:
                reranked_scored.append(
                    (text, score, idx)
                )
                break

    # =========================
    # PICK BEST
    # =========================
    used = _pick_diverse(
        reranked_scored,
        max_items=8
    )

    context = "\n".join(used)

    print("RETRIEVAL FILTER:", filt)
    print("FINAL CHUNKS:", len(used))

    return context, used