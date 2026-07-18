"""
Two responsibilities, deliberately together (the label IS the routing):

  1. classify()  -- the count-based label table
  2. TraceStore  -- persistent storage that routes by label:
        GOOD      -> store/pool.jsonl          (the Phase-1 exemplar pool;
                                                the ONLY way in)
        MEDIUM_*  -> store/medium_queue.jsonl  (retried FIRST, end-of-pass)
        BAD_*     -> store/bad_queue.jsonl     (retried after all MEDIUMs)

The label table:
    correct  misses=0 AND bad_eqs=0 AND coherent AND R>=0.6   -> GOOD
    correct  misses<=1 AND bad_eqs=0 AND coherent             -> MEDIUM_CORRECT
    correct  anything else                                    -> BAD_CORRECT
    wrong    misses<=1 AND bad_eqs<=1 AND coherent AND G>=0.5 -> MEDIUM_WRONG
    wrong    anything else                                    -> BAD_WRONG

GOOD = perfect reasoning and perfect answer. MEDIUM = one gap at most.
BAD = guesswork, answer-only, nonsense-that-landed-right, wrong answers
with unremarkable reasoning. coherent=None (no equations) counts as NOT
coherent -- "no visible work" is not "coherent work".

Retry rule -- universal, "retry once, graduate or die": every non-GOOD
trace is diagnosed by the cascade and retried exactly once, end-of-pass,
MEDIUM queue before BAD queue. Successful MEDIUM retries enter the pool
BEFORE any BAD retry runs -- easy recoveries finance hard ones.

Phase-1 hook: TraceStore.exemplars() returns the GOOD pool entries (with
question/trace/answer, complexity = s_hat, and a precomputed MiniLM
embedding when available) -- exactly what retrieval will consume.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from config import (GOOD_R_MIN, MEDIUM_MISSES_MAX, MEDIUM_WRONG_BAD_EQS_MAX,
                    G_MIN, STORE_ROOT, POOL_FILE, MEDIUM_QUEUE_FILE,
                    BAD_QUEUE_FILE)
from components import ComponentScores
from similarity import embed, backend_name


# ============================================================================
# LABELS
# ============================================================================

class Label(str, Enum):
    GOOD = "GOOD"
    MEDIUM_CORRECT = "MEDIUM_CORRECT"
    BAD_CORRECT = "BAD_CORRECT"
    MEDIUM_WRONG = "MEDIUM_WRONG"
    BAD_WRONG = "BAD_WRONG"


CONSEQUENCE = {
    Label.GOOD:           "enters exemplar pool (the only way in)",
    Label.MEDIUM_CORRECT: "diagnose -> typed polish retry (MEDIUM queue); pooled only if retry is GOOD",
    Label.BAD_CORRECT:    "diagnose -> typed polish retry (BAD queue; NR-correct = diagnosed guess)",
    Label.MEDIUM_WRONG:   "diagnose -> typed hint-free retry (MEDIUM queue; the core case)",
    Label.BAD_WRONG:      "diagnose -> typed hint-free retry (BAD queue, after all MEDIUM retries)",
}

# strict retry order: the MEDIUM queue drains before the BAD queue
RETRY_ORDER = [Label.MEDIUM_WRONG, Label.MEDIUM_CORRECT,
               Label.BAD_WRONG, Label.BAD_CORRECT]


def classify(c: ComponentScores) -> Label:
    coh = bool(c.coherent)        # None -> False

    if c.correct:
        if c.misses == 0 and c.bad_eqs == 0 and coh and c.R >= GOOD_R_MIN:
            return Label.GOOD
        if c.misses <= MEDIUM_MISSES_MAX and c.bad_eqs == 0 and coh:
            return Label.MEDIUM_CORRECT
        return Label.BAD_CORRECT

    # wrong answer: never pooled, at any quality
    if (c.misses <= MEDIUM_MISSES_MAX and c.bad_eqs <= MEDIUM_WRONG_BAD_EQS_MAX
            and coh and c.G >= G_MIN):
        return Label.MEDIUM_WRONG
    return Label.BAD_WRONG


def is_pool_admissible(label: Label) -> bool:
    """Invariant: the pool contains exclusively GOOD traces."""
    return label is Label.GOOD


def needs_diagnosis(label: Label) -> bool:
    """EVERY non-GOOD trace runs the cascade -- weak successes included."""
    return label is not Label.GOOD


# ============================================================================
# PERSISTENT STORE
# ============================================================================

class TraceStore:
    """Append-only JSONL storage routed by label, deduped by trace id.

    Layout under `root` (default: ./store):
        pool.jsonl          GOOD only -- the growing exemplar pool
        medium_queue.jsonl  MEDIUM_CORRECT + MEDIUM_WRONG -- first retry queue
        bad_queue.jsonl     BAD_CORRECT + BAD_WRONG -- second retry queue

    Every record carries the full (question, trace, gold, answer), the label,
    the cascade diagnosis (for non-GOOD: it selects the typed cure at retry
    time), all component signals, and complexity = s_hat. Pool records
    additionally carry a precomputed question embedding when the MiniLM
    backend is active (TF-IDF vectors are not comparable across calls, so
    under the fallback the embedding is left null and Phase 1 recomputes).
    """

    def __init__(self, root: Path = STORE_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "pool": self.root / POOL_FILE,
            "medium": self.root / MEDIUM_QUEUE_FILE,
            "bad": self.root / BAD_QUEUE_FILE,
        }

    # ------------------------------------------------------------- routing
    @staticmethod
    def _bucket(label: Label) -> str:
        if label is Label.GOOD:
            return "pool"
        if label in (Label.MEDIUM_CORRECT, Label.MEDIUM_WRONG):
            return "medium"
        return "bad"

    # ------------------------------------------------------------- writing
    def add(self, item_id: str, question: str, trace: str,
            gold_solution: str, gold_answer: float,
            label: Label, comps: ComponentScores,
            diagnosis: Optional[str] = None,
            diagnosis_reason: Optional[str] = None) -> Optional[Path]:
        """Persist one classified trace. Returns the file written, or None
        if this id is already stored in that bucket (idempotent re-runs)."""
        bucket = self._bucket(label)
        path = self.paths[bucket]
        if item_id in self._ids(path):
            return None

        record = {
            "id": item_id,
            "question": question,
            "trace": trace,
            "gold_solution": gold_solution,
            "gold_answer": gold_answer,
            "label": label.value,
            "diagnosis": diagnosis,            # None for GOOD
            "diagnosis_reason": diagnosis_reason,
            "signals": comps.signals(),
            "complexity": comps.n_checkpoints,  # s_hat, for Fu-style re-rank
            "stored_at": time.time(),
            "retried": False,                   # flipped by the retry phase
        }
        if bucket == "pool":
            record["embedding"] = (embed([question])[0].tolist()
                                   if backend_name() == "minilm" else None)

        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return path

    # ------------------------------------------------------------- reading
    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path) as f:
            return [json.loads(ln) for ln in f if ln.strip()]

    def _ids(self, path: Path) -> set[str]:
        return {r["id"] for r in self._load(path)}

    def load_pool(self) -> list[dict]:
        return self._load(self.paths["pool"])

    def load_queue(self, which: str) -> list[dict]:
        """which in {'medium', 'bad'} -- in stored (stream) order."""
        return self._load(self.paths[which])

    def counts(self) -> dict[str, int]:
        return {k: len(self._load(p)) for k, p in self.paths.items()}

    # ------------------------------------------------------- Phase-1 hook
    def exemplars(self) -> list[dict]:
        """The GOOD pool, ready for retrieval: each entry has question,
        trace, gold_answer, complexity (s_hat), and embedding (or None).
        This is the function Phase 1's retrieval imports."""
        return self.load_pool()