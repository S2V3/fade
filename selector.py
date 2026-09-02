"""Choose one answer from several candidate traces, without gold.

Two policies, so the experiment can separate the diversity source from the
selection rule:

    majority(...)   vote on the final answer, ties broken by candidate order
    symbolic(...)   prefer the trace whose stated equations actually hold,
                    falling back to the detector, then to answer agreement

Measured on paired train data: the arithmetic tier picks correctly 65.2% of the
time on discordant pairs, the detector 54.8%. The cascade puts the stronger
signal first and only consults the weaker one when arithmetic cannot decide.
"""
from __future__ import annotations

from collections import Counter


def _truth(trace):
    """Fraction of the trace's stated equations that hold. None if no chain."""
    from extraction import extract_equations, trace_body
    eqs = extract_equations(trace_body(trace or ""))
    if not eqs:
        return None
    return sum(e.is_true for e in eqs) / len(eqs)


def _key(ans):
    try:
        return round(float(ans), 6)
    except (TypeError, ValueError):
        return str(ans)


def majority(candidates):
    """candidates: [(trace, answer)] -> (index, 'majority')

    Self-consistency. Ties go to the earliest candidate, which is the model's
    unprompted attempt.
    """
    votes = Counter(_key(a) for _, a in candidates if a not in (None, ""))
    if not votes:
        return 0, "empty"
    best, _ = votes.most_common(1)[0]
    for i, (_, a) in enumerate(candidates):
        if _key(a) == best:
            return i, "majority"
    return 0, "majority"


def symbolic(candidates, question=None, detector=None):
    """candidates: [(trace, answer)] -> (index, reason)

    Arithmetic validity first, then the detector, then a plain answer vote.
    `detector` is any object with .predict(question, trace) -> {'p_wrong': ...}.
    """
    if not candidates:
        return 0, "empty"

    # 1. unanimous answer -- nothing to choose
    keys = [_key(a) for _, a in candidates]
    if len(set(keys)) == 1:
        return 0, "agree"

    # 2. arithmetic validity, on candidates that state any equations at all
    scored = [(i, _truth(t)) for i, (t, _) in enumerate(candidates)]
    have = [(i, s) for i, s in scored if s is not None]
    if have:
        top = max(s for _, s in have)
        tied = [i for i, s in have if s >= top - 1e-9]
        if len(tied) == 1:
            return tied[0], "arith"
    else:
        tied = list(range(len(candidates)))

    # 3. the detector breaks a tie among the arithmetically-best candidates
    if detector is not None and question is not None and len(tied) > 1:
        try:
            ps = [(i, detector.predict(question, candidates[i][0])["p_wrong"])
                  for i in tied]
            return min(ps, key=lambda x: x[1])[0], "detector"
        except Exception:
            pass

    # 4. a vote among the tied candidates, then candidate order
    if len(tied) > 1:
        votes = Counter(keys[i] for i in tied)
        best, n = votes.most_common(1)[0]
        if n > 1:
            for i in tied:
                if keys[i] == best:
                    return i, "vote"
    return tied[0], "order"


def apply(policy, candidates, question=None, detector=None):
    if policy == "majority":
        return majority(candidates)
    if policy == "symbolic":
        return symbolic(candidates, question, detector)
    if policy == "first":
        return 0, "first"
    raise ValueError(f"unknown policy {policy!r}")


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
# Ties go to candidate 0 by convention, and candidate 0 is always the model's
#   unprompted first attempt. A selector that cannot decide therefore falls
#   back to doing nothing, never to a coin flip.
# The arithmetic tier is deliberately first: on discordant pairs it is 10
#   points better than the trained detector, which collapses toward chance on
#   exactly the cases where a decision is needed.
