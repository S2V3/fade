"""
FST feature extraction -- GOLD-FREE by construction.

The single hard rule of Phase 3: a feature may look at the QUESTION only.
Never the gold solution, never the gold answer, never the model's trace, never
any signal derived from them (E, A, misses, s_hat/n_checkpoints, correctness).
If one gold-dependent feature sneaks in, the whole "predict from the question
alone, before the first attempt" claim collapses and the method is not
deployable. So this module takes a raw question string and nothing else --
there is no parameter through which gold could arrive.

`question_features(q)` returns a flat dict of named features (numeric or
categorical). `vectorize(list_of_dicts)` turns a list of them into an (X,
names) matrix for a classifier, one-hot-encoding the categorical ones.

The optional `surprisal` argument is the ONE place a model-derived number may
enter -- the frozen model's own token-level surprisal over the QUESTION (still
gold-free: it reads the question, not the answer). It is off by default because
it costs a forward pass; the gate runs fine on the cheap surface features
alone. Pass a precomputed value if you have one.

Reuses categorizer.categorize(question) with NO solution argument -- that path
estimates complexity/steps from the question only, so it stays gold-free. (With
a solution it would derive steps from the trace/gold; we never call it that
way here.)
"""
from __future__ import annotations

import re
from typing import Optional

from categorizer import QuestionCategorizer

_CAT = QuestionCategorizer()  # keyword-based; cheap; instantiated once

GOLD_FREE = True  # invariant asserted by construction (see module docstring)

# A number token in the question: integers, decimals, thousands-separated,
# simple fractions. Question text only.
_RE_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_RE_SENT = re.compile(r"[.!?]+\s+|\n+")
_RE_WORD = re.compile(r"[A-Za-z]+")

# cheap surface cues -- presence, not correctness (that would need gold)
_CUES = {
    "cue_percent": re.compile(r"%|percent|percentage", re.I),
    "cue_money":   re.compile(r"\$|dollar|cent|cost|price|pay|spend|buy|sell", re.I),
    "cue_ratio":   re.compile(r"\bratio\b|\bper\b|\brate\b|each|every", re.I),
    "cue_time":    re.compile(r"hour|minute|second|day|week|month|year|\bage\b", re.I),
    "cue_compare": re.compile(r"more than|less than|fewer|greater|difference|twice|half|double", re.I),
    "cue_total":   re.compile(r"total|altogether|in all|combined|sum|remaining|left", re.I),
}


def question_features(question: str, surprisal: Optional[float] = None) -> dict:
    """Gold-free feature dict for one question. `question` MUST be the raw
    question text and nothing else. `surprisal`, if given, is the frozen
    model's mean token surprisal over the QUESTION (still gold-free)."""
    if not isinstance(question, str):
        raise TypeError("question_features takes the question STRING only "
                        "(gold-free); got %r" % type(question))
    q = question

    nums = _RE_NUM.findall(q)
    sentences = [s for s in _RE_SENT.split(q) if s.strip()]
    words = _RE_WORD.findall(q)

    # categorizer, called question-only -> gold-free category/op/step estimate
    cat = _CAT.categorize(q)  # NOTE: no solution arg -> stays gold-free

    feats = {
        # categorical (one-hot at vectorize time)
        "category_type":  cat.get("category_type", "unknown"),
        "main_operation": cat.get("main_operation", "unknown"),
        "complexity":     cat.get("complexity", "unknown"),
        # numeric surface features
        "n_quantities":   float(len(nums)),
        "n_tokens":       float(len(words)),
        "n_chars":        float(len(q)),
        "n_sentences":    float(max(1, len(sentences))),
        "avg_sent_len":   float(len(words)) / float(max(1, len(sentences))),
        # gold-free step estimate (question-only; NOT the gold s_hat)
        "est_steps_qfree": float(cat.get("estimated_steps", 0)),
    }
    for name, rx in _CUES.items():
        feats[name] = 1.0 if rx.search(q) else 0.0

    if surprisal is not None:
        feats["surprisal"] = float(surprisal)

    return feats


# feature keys that are categorical (string-valued) -> one-hot at vectorize time
CATEGORICAL = ("category_type", "main_operation", "complexity")


def vectorize(rows: list[dict]):
    """Turn a list of feature dicts into (X, feature_names).

    Uses sklearn.DictVectorizer when available (clean one-hot); otherwise a
    small pure-python one-hot fallback so the gate can run in a bare env.
    Returns (X, names) where X is a list-of-lists (or numpy array via sklearn).
    """
    try:
        from sklearn.feature_extraction import DictVectorizer
        dv = DictVectorizer(sparse=False)
        # DictVectorizer one-hots any string value automatically
        X = dv.fit_transform(rows)
        return X, dv.get_feature_names_out().tolist()
    except Exception:
        # ---- pure-python fallback ----
        # collect categorical levels + numeric keys
        levels: dict[str, set] = {c: set() for c in CATEGORICAL}
        numeric_keys: set = set()
        for r in rows:
            for k, v in r.items():
                if k in CATEGORICAL:
                    levels[k].add(v)
                else:
                    numeric_keys.add(k)
        names = sorted(numeric_keys)
        cat_cols = []
        for c in CATEGORICAL:
            for lv in sorted(levels[c]):
                cat_cols.append((c, lv))
        names = names + [f"{c}={lv}" for c, lv in cat_cols]
        X = []
        for r in rows:
            row = [float(r.get(k, 0.0)) for k in sorted(numeric_keys)]
            row += [1.0 if r.get(c) == lv else 0.0 for c, lv in cat_cols]
            X.append(row)
        return X, names


if __name__ == "__main__":
    demo = ("A robe takes 2 bolts of blue fiber and half that much white "
            "fiber. How many bolts in total does it take?")
    import json
    print(json.dumps(question_features(demo), indent=2))