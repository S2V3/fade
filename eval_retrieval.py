"""
eval_retrieval.py -- MEASURE the retriever, offline, with NO GPU.

WHY THIS EXISTS
---------------
Retrieval is the load-bearing component of FADE. It chooses the exemplars for the
first attempt, the typed cure at retry, and (if FST ever ships) the preloaded cure.
If retrieval is weak the whole method is weak -- and until now nothing measured it.
Worse, the 1,500-problem run never exercised it at all (D15: strategy 2 rendered
fixed seeds and discarded the pool), so its quality has literally never been
observed.

This script fixes that. It needs no model and no GPU, so retrieval can be tuned in
minutes instead of GPU-hours.

THE METRIC, AND WHY IT IS FAIR
------------------------------
"Was this a good exemplar for this question?" needs a ground truth. We use:

    two problems MATCH if their GOLD solutions share reasoning structure
      -- same operator multiset (+,-,*,/ actually used), and
      -- checkpoint counts within +/- 1

Gold is used ONLY to score the retrieval, never to perform it. The retriever sees
the question string and the pool, exactly as at run time. This is the same
train/eval discipline the rest of the project uses: gold may grade, never inform.

precision@k = fraction of the k retrieved exemplars that match the query.

CAVEAT, stated plainly: structural match is a PROXY for "this exemplar helps".
The only true test is downstream accuracy. But a retriever that cannot beat random
on the proxy will not help downstream either, so this is a cheap necessary
condition -- and it discriminates between strategies for free.

USAGE
    python eval_retrieval.py --results results.jsonl
    python eval_retrieval.py --results results.jsonl --k 8 --queries 400
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

import extraction as X
from preprocessing import normalize_math

_OPS = "+-*/"


# ---------------------------------------------------------------- ground truth
def gold_signature(gold_solution: str) -> tuple:
    """(operator multiset, checkpoint count) of the REFERENCE solution.
    Evaluation-only -- never shown to the retriever."""
    g = normalize_math(gold_solution)
    cps = X.extract_gold_checkpoint_terms(g)
    ops = frozenset(o for c in cps for o in c.ops if o in _OPS)
    return ops, len(cps)


def structurally_matches(a: tuple, b: tuple) -> bool:
    ops_a, n_a = a
    ops_b, n_b = b
    if abs(n_a - n_b) > 1:
        return False
    if not ops_a and not ops_b:
        return True
    return bool(ops_a & ops_b) and len(ops_a ^ ops_b) <= 1


# ------------------------------------------------------- question-side features
_NUM = re.compile(r"\d+(?:\.\d+)?")
_CUES = {
    "per": r"\bper\b|each|every", "pct": r"%|percent", "money": r"\$|dollar|cost|price|pay",
    "time": r"hour|minute|day|week|month|year", "cmp": r"more than|less than|twice|half|double",
    "tot": r"total|altogether|in all|combined|left|remaining",
}


def question_signature(q: str) -> dict:
    """Gold-free structural view of a QUESTION -- what the retriever may use."""
    nums = _NUM.findall(q)
    sig = {"n_nums": len(nums), "n_words": len(q.split()),
           "n_distinct": len(set(nums))}
    for k, pat in _CUES.items():
        sig[k] = 1 if re.search(pat, q, re.I) else 0
    return sig


def struct_sim(a: dict, b: dict) -> float:
    """Similarity between two question signatures, in [0,1]."""
    cue = sum(1 for k in _CUES if a[k] == b[k]) / len(_CUES)
    dn = 1.0 - min(abs(a["n_nums"] - b["n_nums"]) / 6.0, 1.0)
    dw = 1.0 - min(abs(a["n_words"] - b["n_words"]) / 60.0, 1.0)
    return 0.5 * cue + 0.35 * dn + 0.15 * dw


# ------------------------------------------------------------------- embeddings
def build_tfidf(corpus: list[str]):
    """TF-IDF fitted ONCE over the whole corpus, so vectors are comparable.
    (similarity.embed refits per call, which is why config.REQUIRE_MINILM exists.)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    M = vec.fit_transform(corpus).toarray().astype("float32")
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def build_minilm(corpus: list[str]):
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(m.encode(corpus, normalize_embeddings=True, batch_size=64))


# ------------------------------------------------------------------ strategies
def retrieve(strategy, qi, pool_idx, E, qsig, depth, k, rng):
    """Return k pool indices for query index qi. E = embedding matrix."""
    if strategy == "random":
        return rng.sample(pool_idx, min(k, len(pool_idx)))

    sims = {j: float(E[qi] @ E[j]) for j in pool_idx}

    if strategy == "cosine":
        return sorted(pool_idx, key=lambda j: -sims[j])[:k]

    if strategy == "structural":
        return sorted(pool_idx, key=lambda j: -struct_sim(qsig[qi], qsig[j]))[:k]

    if strategy == "hybrid":
        return sorted(pool_idx,
                      key=lambda j: -(0.5 * sims[j] + 0.5 * struct_sim(qsig[qi], qsig[j])))[:k]

    # the two 3-stage variants: relevance shortlist -> complexity re-rank -> diversity
    short = sorted(pool_idx, key=lambda j: -sims[j])[:max(15, k)]
    mx = max((depth[strategy][j] for j in short), default=1) or 1
    scored = sorted(short, key=lambda j: -(0.6 * sims[j] + 0.4 * depth[strategy][j] / mx))
    out: list[int] = []
    for j in scored:
        if any(float(E[j] @ E[m]) > 0.85 for m in out):
            continue
        out.append(j)
        if len(out) >= k:
            break
    for j in scored:                                  # top up if diversity was strict
        if len(out) >= k:
            break
        if j not in out:
            out.append(j)
    return out[:k]


def main():
    ap = argparse.ArgumentParser(description="offline retrieval benchmark (no GPU)")
    ap.add_argument("--results", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--embed", choices=["tfidf", "minilm"], default="tfidf")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    rows = [r for r in rows if r.get("phase") == "pass1"]
    print(f"loaded {len(rows)} pass-1 records")

    qs = [r["question"] for r in rows]
    gsig = [gold_signature(r["gold_solution"]) for r in rows]
    qsig = [question_signature(q) for q in qs]

    # pool = the GOOD traces (what FADE would actually have); queries = the rest
    pool_idx = [i for i, r in enumerate(rows) if r["label"] == "GOOD"]
    other = [i for i in range(len(rows)) if i not in set(pool_idx)]
    rng = random.Random(a.seed)
    rng.shuffle(other)
    queries = other[:a.queries]
    print(f"pool = {len(pool_idx)} GOOD traces | queries = {len(queries)}")

    E = build_minilm(qs) if a.embed == "minilm" else build_tfidf(qs)
    print(f"embeddings: {a.embed}  shape {E.shape}")

    depth = {
        # what the CURRENT code ranks by (prose lines) vs what it SHOULD (D31)
        "3stage_lines":  {i: (rows[i]["signals"].get("n_steps") or 1) for i in range(len(rows))},
        "3stage_fixed":  {i: (rows[i]["signals"].get("n_comp_steps")
                              or rows[i]["signals"].get("n_checkpoints") or 1)
                          for i in range(len(rows))},
    }

    strategies = ["random", "cosine", "structural", "hybrid",
                  "3stage_lines", "3stage_fixed"]
    print(f"\n{'strategy':<16}{'precision@' + str(a.k):>14}{'vs random':>12}")
    print("-" * 42)
    base = None
    for s in strategies:
        hits = tot = 0
        for qi in queries:
            got = retrieve(s, qi, pool_idx, E, qsig, depth, a.k, rng)
            hits += sum(1 for j in got if structurally_matches(gsig[qi], gsig[j]))
            tot += len(got)
        p = hits / max(tot, 1)
        if base is None:
            base = p
        print(f"  {s:<14}{p:>14.3f}{(p - base) * 100:>+11.1f}pp")

    print("\nNOTE: structural match is a PROXY for exemplar usefulness. A strategy")
    print("that cannot beat random here will not help downstream; one that does")
    print("still has to prove itself on accuracy. Re-run with --embed minilm on")
    print("Kaggle to see the numbers you will actually deploy with.")


if __name__ == "__main__":
    main()