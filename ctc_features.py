"""
ctc_features.py -- gold-free features of a (question, trace) pair.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
The stored `signals` dict in results.jsonl mixes gold-free and gold-dependent
quantities. E, E_struct, A, A_struct, `misses` and `comp_coverage` all read the
reference solution -- `comp_coverage` divides by `s_hat`, which comes from gold --
so a model trained on the stored dict would look excellent on train and be
unusable at test time, where no gold exists.

This module recomputes ONLY the gold-free half, from the question string and the
model's own trace. Nothing here can see a reference solution, which is what makes
a CTC head trained on it deployable on the test split.

The functions used are exactly the ones components.py uses, minus every one whose
signature takes `gold_solution`.
"""
from __future__ import annotations

import config
from components import (V_score, R_score, G_score, coherence_flag,
                        computational_steps)
from extraction import (extract_equations, extract_final_answer, is_truncated,
                        symbolic_line_count, trace_body, unresolved_relation_count)

FEATURE_NAMES = [
    "V", "G", "R", "coherent", "coherence_unknown",
    "n_equations", "bad_eqs", "bad_eq_ratio",
    "n_steps", "n_comp_steps", "comp_ratio",
    "n_symbolic", "n_unresolved", "truncated",
    "has_answer", "answer_in_question", "trace_chars", "trace_lines",
    "q_numbers", "q_number_coverage", "eq_reaches_answer",
]


def _close(a, b, tol=1e-6):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def trace_features(question: str, trace: str) -> dict:
    """Gold-free. Everything here reads only `question` and `trace`."""
    import re

    body = trace_body(trace)
    eqs = extract_equations(body)
    # V_score returns (score, eqs) and coherence_flag returns Optional[bool];
    # pass the equations we already parsed so nothing is extracted twice.
    v_score, _ = V_score(body, eqs)
    coh = coherence_flag(trace, eqs)
    ans = extract_final_answer(trace)
    lines = [l for l in body.splitlines() if l.strip()]
    n_steps = len(lines)
    n_comp = computational_steps(trace)
    bad = sum(1 for e in eqs if not e.is_true)

    qn = [float(m.group()) for m in
          re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", question.replace(",", ""))
          if float(m.group()) not in (0.0, 1.0)]
    ops = set()
    for e in eqs:
        for x in e.operands:
            if isinstance(x, (int, float)):
                ops.add(float(x))
    used = sum(1 for v in qn if any(_close(v, o) for o in ops))

    return {
        "V": float(v_score),
        "G": float(G_score(question, trace)),
        "R": float(R_score(trace)),
        # coherence is TRI-state: True / False / None ("no equations or no
        # answer, cannot say"). Collapsing None to False would encode "incoherent"
        # for traces that simply have nothing to check, so it gets its own column.
        "coherent": float(coh is True),
        "coherence_unknown": float(coh is None),
        "n_equations": float(len(eqs)),
        "bad_eqs": float(bad),
        "bad_eq_ratio": bad / max(len(eqs), 1),
        "n_steps": float(n_steps),
        "n_comp_steps": float(n_comp),
        "comp_ratio": n_comp / max(n_steps, 1),
        "n_symbolic": float(symbolic_line_count(trace)),
        "n_unresolved": float(unresolved_relation_count(trace)),
        "truncated": float(bool(is_truncated(trace))),
        "has_answer": float(ans is not None),
        # the answer being one of the question's own numbers is a classic
        # give-up pattern -- the model echoes a given instead of computing
        "answer_in_question": float(ans is not None and any(_close(ans, v) for v in qn)),
        "trace_chars": float(len(body)),
        "trace_lines": float(n_steps),
        "q_numbers": float(len(qn)),
        "q_number_coverage": used / max(len(qn), 1),
        "eq_reaches_answer": float(ans is not None and any(_close(e.c, ans) for e in eqs)),
    }


def add_ctc_features(feats: dict, ctc_result=None) -> dict:
    """Fold in the perturbation-consistency signal, when it has been computed.

    Kept separate because it costs ONE EXTRA GENERATION, while everything in
    trace_features() is free. Train a head without these first; only pay for the
    perturbation if the free head is not good enough.
    """
    feats = dict(feats)
    if ctc_result is None or not getattr(ctc_result, "applicable", False):
        feats["ctc_applicable"] = 0.0
        feats["ctc_agrees"] = 0.0
        feats["ctc_n_changed"] = 0.0
    else:
        feats["ctc_applicable"] = 1.0
        feats["ctc_agrees"] = float(bool(ctc_result.agrees))
        feats["ctc_n_changed"] = float(ctc_result.n_changed)
    return feats


CTC_FEATURE_NAMES = FEATURE_NAMES + ["ctc_applicable", "ctc_agrees", "ctc_n_changed"]


def vectorize(rows, names=None):
    names = names or FEATURE_NAMES
    return [[float(r.get(n, 0.0)) for n in names] for r in rows], names
