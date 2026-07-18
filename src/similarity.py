"""
How a trace is SEGMENTED and COMPARED -- steps and embeddings together,
since R consumes them jointly and Phase-1 retrieval reuses embed() as-is.

Step definition (applied identically everywhere):
    steps = non-empty lines, EXCLUDING pure '#### N' answer-marker lines;
    a single long line falls back to sentence splitting.
Used by: components.R_score, the cascade's s, ST-pool curation later.

Embedding backend: MiniLM (sentence-transformers/all-MiniLM-L6-v2) as the
design doc specifies; TF-IDF cosine fallback if not installed so phase-0
sanity checks still run. backend_name() reports which is active -- pool
embeddings are only persisted under MiniLM (TF-IDF vectors are not
comparable across calls).
Used by: components.R_score now; Phase-1 retrieval stage 1 (query cosine)
and stage 3 (MMR diversity) later.
"""

from __future__ import annotations

import re

import numpy as np

# ============================================================================
# STEP SPLITTING
# ============================================================================

RE_ANSWER_LINE = re.compile(r"^\s*####")
RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_steps(trace: str) -> list[str]:
    lines = [ln.strip() for ln in trace.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not RE_ANSWER_LINE.match(ln)]
    if len(lines) == 1:
        sents = [s.strip() for s in RE_SENTENCE_SPLIT.split(lines[0]) if s.strip()]
        if len(sents) > 1:
            return sents
    return lines


def step_count(trace: str) -> int:
    """s -- model step count (feeds the cascade)."""
    return len(split_steps(trace))


# ============================================================================
# EMBEDDINGS
# ============================================================================

_ST_MODEL = None
_BACKEND: str | None = None


def _init_backend():
    global _ST_MODEL, _BACKEND
    if _BACKEND is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        _BACKEND = "minilm"
    except Exception:
        _BACKEND = "tfidf"


def backend_name() -> str:
    _init_backend()
    return _BACKEND


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (n, d)."""
    _init_backend()
    if _BACKEND == "minilm":
        return np.asarray(_ST_MODEL.encode(texts, normalize_embeddings=True))
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        mat = TfidfVectorizer().fit_transform(texts).toarray()
    except ValueError:                       # no alphanumeric tokens at all
        return np.eye(len(texts))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def pairwise_similarity(texts: list[str]) -> np.ndarray:
    """n x n cosine matrix."""
    if not texts:
        return np.zeros((0, 0))
    e = embed(texts)
    return e @ e.T