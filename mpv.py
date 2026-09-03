"""
mpv.py -- Metamorphic Program Verification.  Gold-free, judge-free selection and
detection for a frozen model's reasoning traces, built on ctc.py's perturbation
mechanism and extended in three ways that turn a *detector* into a *selector*.

THE ONE-LINE IDEA
-----------------
A candidate trace is a HYPOTHESIS about which program the model computes for this
question.  The model's answers on metamorphic variants of the question (same words,
different numbers) are HELD-OUT DATA.  Pick the hypothesis that predicts the data.

WHAT IS NEW RELATIVE TO ctc.py
------------------------------
ctc.py:   one candidate, one perturbation, one bit (agree / disagree)  -> detector
mpv.py:   k candidates share the SAME probes, so the cost is per question, not per
          candidate; every candidate's chain is executed symbolically on every probe
          and the candidate that explains the most probe answers wins   -> selector

          plus BACK-SUBSTITUTION: the model's trace on the perturbed question is
          itself a program.  Executing it with the ORIGINAL numbers (sympy, exact)
          yields a fresh candidate answer for the original question.  The model may
          read Q' correctly even when it misread Q (retries diverge 82% of the time),
          and sympy executes the plan without arithmetic slips.  This adds candidates
          at zero extra generations.

          plus PROGRAM AGREEMENT: two chains are compared by their abstract program
          (question-slot / result-slot / operator sequence), not only by their final
          number, so an arithmetic slip on a probe does not mask a shared plan.

GOLD RULE
---------
Everything here reads the question and the model's own traces.  No gold anywhere.
Valid on the test split exactly as on train.

PUBLIC API
----------
    abstract_program(question, trace)              -> str signature (or "")
    execute(trace, subs, final_answer)             -> predicted answer under subs
    make_probes(question, traces, n_probes, factors)
                                                   -> [Probe]  (no GPU)
    back_substitute(probe, perturbed_trace)        -> answer for the ORIGINAL question
    score_candidates(question, candidates, probes) -> [CandidateScore]
    select(question, candidates, probes, detector=None)
                                                   -> (index, reason, scores)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from extraction import extract_equations, trace_body, _normalise_math_text
import ctc as CTC

TOL = 1e-6
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
    "half": 0.5, "twice": 2, "double": 2, "triple": 3, "dozen": 12,
    "quarter": 0.25, "third": 1 / 3,
}


def _close(x, y, tol=TOL):
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= tol * max(1.0, abs(y))


def _key(v):
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return None


# ============================================================================
# 1. ABSTRACT PROGRAM  -- the trace as a function of the question's numbers
# ============================================================================
def question_values(question: str) -> list[float]:
    """Every number in the question, literal or written as a word."""
    vals = [v for v, _ in CTC.question_numbers(question)]
    vals += [float(_WORDS[w]) for w in re.findall(r"[a-z]+", question.lower())
             if w in _WORDS]
    out = []
    for v in vals:
        if not any(_close(v, o) for o in out):
            out.append(v)
    return sorted(out)


def abstract_program(question: str, trace: str) -> str:
    """'q0*k;r0-q1;r1*q2'  -- numbers from the question become q-slots, results of
    earlier steps become r-slots, anything else is a constant k.  Empty string if
    the trace states no equation.  Two traces with the same signature computed the
    same plan, whatever numbers they happened to write."""
    qv = question_values(question)
    results: list[float] = []
    steps = []
    for e in extract_equations(trace_body(trace or "")):
        lhs = e.raw.split("=")[0] if "=" in e.raw else e.raw
        lhs = _normalise_math_text(lhs)

        def rep(m):
            v = float(m.group())
            for j, r in enumerate(results):
                if _close(v, r):
                    return f"r{j}"
            for i, q in enumerate(qv):
                if _close(v, q):
                    return f"q{i}"
            return "k"
        steps.append(re.sub(r"\s+", "", _NUM.sub(rep, lhs)))
        results.append(float(e.c))
    return ";".join(steps)


def question_coverage(question: str, trace: str) -> float:
    """Fraction of the question's numbers the chain actually consumes."""
    qv = question_values(question)
    if not qv:
        return 1.0
    used = set()
    for e in extract_equations(trace_body(trace or "")):
        for x in e.operands:
            for i, q in enumerate(qv):
                if _close(x, q):
                    used.add(i)
    return len(used) / len(qv)


# ============================================================================
# 2. SYMBOLIC EXECUTION under a set of substitutions
# ============================================================================
def execute(trace: str, subs: list[tuple[float, float]], final_answer) -> Optional[float]:
    """Run the trace's own chain with `subs` = [(old, new), ...] applied to the
    question numbers, propagating changed intermediate results downstream exactly
    as ctc.predict_answer does -- but for ANY number of substitutions.

    Returns the predicted final answer, or None when the chain cannot be executed
    or does not reach the final answer (the caller should treat that as ABSTAIN)."""
    if final_answer is None:
        return None
    eqs = extract_equations(trace_body(trace or ""))
    if not eqs or not any(_close(e.c, final_answer) for e in eqs):
        return None
    live = [(float(o), float(n)) for o, n in subs]
    for e in eqs:
        lhs = e.raw.split("=")[0] if "=" in e.raw else e.raw
        val = CTC._eval_lhs(lhs, live)
        if val is None:
            continue
        if not _close(val, e.c):
            live.append((float(e.c), float(val)))
    for old, new in live[len(subs):]:
        if _close(final_answer, old):
            return new
    # answer not downstream of any substitution: the chain says it must not move
    return float(final_answer)


# ============================================================================
# 3. PROBES  -- metamorphic variants shared by every candidate of a question
# ============================================================================
@dataclass
class Probe:
    perturbed_question: str
    subs: list = field(default_factory=list)      # [(old, new)]
    target_value: float = 0.0
    new_value: float = 0.0
    factor: float = 2.0
    # filled after generation
    model_trace: str = ""
    model_answer: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        p = Probe(d["perturbed_question"], [tuple(x) for x in d.get("subs", [])],
                  d.get("target_value", 0.0), d.get("new_value", 0.0),
                  d.get("factor", 2.0))
        p.model_trace = d.get("model_trace", "")
        p.model_answer = d.get("model_answer")
        return p


def _load_bearing_targets(question: str, traces: list[str]):
    """Question numbers consumed by ANY candidate's chain, in question order."""
    ops = set()
    for t in traces:
        for e in extract_equations(trace_body(t or "")):
            for x in e.operands:
                if isinstance(x, (int, float)):
                    ops.add(float(x))
    return [(v, sp) for v, sp in CTC.question_numbers(question)
            if any(_close(v, o) for o in ops)]


def _safe_new_value(v, factor, qvals):
    """A replacement that keeps integrality and sign and collides with NO other
    number of the question (literal or word), so every occurrence of the new
    value in a probe trace can be attributed to the perturbation."""
    others = [x for x in qvals if not _close(x, v)]
    tries = [v * factor, v * factor + 1, v * factor + 3, v + 5, v + 7, v * factor + 4]
    for nv in tries:
        if float(v).is_integer():
            nv = float(int(round(nv)))
        if nv <= 0 or nv in (0.0, 1.0) or _close(nv, v):
            continue
        if any(_close(nv, o) for o in others):
            continue
        return nv
    return None


def make_probes(question: str, traces: list[str], n_probes: int = 2,
                factors=(2.0, 3.0)) -> list[Probe]:
    """Up to n_probes single-number perturbations.  Preference order: the LAST
    load-bearing number (ctc.py's choice, usually the operative quantity), then
    earlier ones, then a second factor on the last one.  A probe is kept only if
    it is well-formed: the number occurs exactly once in the question text, the
    replacement is a positive integer when the original was, and at least one
    candidate's prediction actually moves (otherwise nothing is tested)."""
    targets = _load_bearing_targets(question, traces)
    if not targets:
        return []
    qtxt = CTC._THOUSANDS.sub("", question)
    from extraction import extract_final_answer as _fa
    answers = [_fa(t) for t in traces]

    plan = []
    for f in factors:
        for v, sp in reversed(targets):
            plan.append((v, sp, f))
    probes, seen = [], set()
    for v, sp, f in plan:
        if len(probes) >= n_probes:
            break
        if len(re.findall(rf"(?<![\w.]){re.escape(qtxt[sp[0]:sp[1]])}(?![\w.])", qtxt)) != 1:
            continue                                  # ambiguous span
        nv = _safe_new_value(v, f, question_values(question))
        if nv is None or (v, nv) in seen:
            continue
        moved = False
        for t, a in zip(traces, answers):
            p = execute(t, [(v, nv)], a)
            if p is not None and not _close(p, a):
                moved = True
                break
        if not moved:
            continue
        seen.add((v, nv))
        probes.append(Probe(CTC.perturb(question, sp, nv), [(v, nv)], v, nv, f))
    return probes


# ============================================================================
# 4. BACK-SUBSTITUTION  -- the probe's trace, executed on the original numbers
# ============================================================================
def transfer_trace(probe: Probe, perturbed_trace: str) -> Optional[str]:
    """Rewrite the probe trace's chain with the ORIGINAL numbers, executing each
    step exactly.  The result is a synthetic trace for Q ('lhs = c' lines plus a
    '#### N' marker) that can be scored like any other candidate."""
    from extraction import extract_final_answer as _fa
    a = _fa(perturbed_trace)
    eqs = extract_equations(trace_body(perturbed_trace or ""))
    if a is None or not eqs or not any(_close(e.c, a) for e in eqs):
        return None
    live = [(float(n), float(o)) for o, n in probe.subs]
    lines = []
    final = None
    for e in eqs:
        lhs = e.raw.split("=")[0] if "=" in e.raw else e.raw
        txt = _normalise_math_text(lhs)

        def rep(m, live=live):
            v = float(m.group())
            for old, new in live:
                if _close(v, old):
                    return f"{new:.12g}"
            return m.group()
        new_lhs = _NUM.sub(rep, txt)
        val = CTC._eval_lhs(lhs, live)
        if val is None:
            return None
        if not _close(val, e.c):
            live.append((float(e.c), float(val)))
        lines.append(f"{new_lhs} = {val:.12g}")
        if _close(e.c, a):
            final = val
    if final is None:
        return None
    return "\n".join(lines) + f"\n#### {final:.12g}"


def back_substitute(probe: Probe, perturbed_trace: str) -> Optional[float]:
    """Treat the model's solution of Q' as a program and run it on Q.  Exact
    (sympy), so the plan is transferred without the model's arithmetic."""
    from extraction import extract_final_answer as _fa
    a = _fa(perturbed_trace)
    if a is None:
        return None
    rev = [(new, old) for old, new in probe.subs]
    return execute(perturbed_trace, rev, a)


# ============================================================================
# 5. SCORING and SELECTION
# ============================================================================
@dataclass
class CandidateScore:
    index: int
    answer: Optional[float]
    origin: str                   # 'pass1' | 'retry' | 'backsub:<probe idx>'
    n_applicable: int = 0         # probes on which the chain could be executed
    n_agree: int = 0              # ... and predicted the model's probe answer
    n_program_agree: int = 0      # probe trace computed the same abstract program
    arith_truth: Optional[float] = None
    coverage: float = 0.0
    signature: str = ""
    consistency: Optional[float] = None   # n_agree / n_applicable

    def to_dict(self):
        return asdict(self)


def _truth(trace):
    eqs = extract_equations(trace_body(trace or ""))
    if not eqs:
        return None
    return sum(e.is_true for e in eqs) / len(eqs)


def score_candidates(question: str, candidates: list[tuple[str, Optional[float], str]],
                     probes: list[Probe]) -> list[CandidateScore]:
    """candidates: [(trace, answer, origin)].  Probes must carry model_answer."""
    out = []
    for i, (trace, ans, origin) in enumerate(candidates):
        cs = CandidateScore(i, _key(ans), origin,
                            arith_truth=_truth(trace),
                            coverage=question_coverage(question, trace),
                            signature=abstract_program(question, trace))
        src_probe = int(origin.split(":")[1]) if origin.startswith("backsub:") else -1
        for j, p in enumerate(probes):
            if p.model_answer is None or j == src_probe:
                continue                      # a back-substitution never grades its own probe
            pred = execute(trace, p.subs, ans)
            if pred is None:
                continue
            cs.n_applicable += 1
            if _close(pred, p.model_answer):
                cs.n_agree += 1
            if cs.signature and cs.signature == abstract_program(p.perturbed_question, p.model_trace):
                cs.n_program_agree += 1
        if cs.n_applicable:
            cs.consistency = cs.n_agree / cs.n_applicable
        out.append(cs)
    return out


def backsub_candidates(probes: list[Probe]) -> list[tuple[str, Optional[float], str]]:
    """Fresh candidates for the ORIGINAL question, one per usable probe."""
    out = []
    for j, p in enumerate(probes):
        if not p.model_trace:
            continue
        t = transfer_trace(p, p.model_trace)
        if t is None:
            continue
        from extraction import extract_final_answer as _fa
        out.append((t, _fa(t), f"backsub:{j}"))
    return out


def select(question: str, candidates: list[tuple[str, Optional[float], str]],
           probes: list[Probe], detector=None, use_backsub: bool = True,
           min_probes: int = 1):
    """Rank candidates by how well their program explains the probe answers.

    Tiers (each only breaks ties left by the previous one):
        1. unanimous answer                          -> candidate 0
        2. metamorphic consistency  n_agree - n_disagree  (needs >= min_probes)
        3. program agreement with the probe traces
        4. arithmetic validity of the stated chain
        5. detector p_wrong, if given
        6. candidate order  (0 = the model's unprompted first attempt)

    Returns (index, reason, scores).  Back-substituted candidates are appended
    after the given ones, so their indices start at len(candidates)."""
    cands = list(candidates)
    if use_backsub:
        cands += backsub_candidates(probes)
    if not cands:
        return 0, "empty", []
    scores = score_candidates(question, cands, probes)

    keys = [s.answer for s in scores]
    if len(set(k for k in keys if k is not None)) <= 1:
        return 0, "agree", scores

    tied = [s for s in scores if s.answer is not None] or scores

    def _tier(vals, name):
        nonlocal tied
        top = max(vals[s.index] for s in tied)
        best = [s for s in tied if vals[s.index] >= top - 1e-9]
        if len(best) == 1:
            return best[0].index, name
        tied = best
        return None

    # 2. metamorphic consistency.  A candidate whose program predicted at least one
    #    probe answer beats everything that did not; among those, net agreement
    #    decides.  A candidate that was tested and NEVER agreed is refuted: it drops
    #    below untested candidates rather than winning by default.
    cons = {s.index: (s.n_agree - (s.n_applicable - s.n_agree)) for s in scores}
    supported = [s for s in tied if s.n_applicable >= min_probes and s.n_agree > 0]
    if supported:
        tied = supported
        r = _tier(cons, "metamorphic")
        if r:
            return r[0], r[1], scores
    # A back-substituted program is a guess from ONE probe solve; without positive
    # evidence from another probe it is no better than a fresh sample (~20% right
    # on this model) and must never displace an original candidate.
    orig = [s for s in tied if not s.origin.startswith("backsub:")]
    if orig and len(orig) < len(tied):
        tied = orig

    # 3. program agreement
    prog = {s.index: s.n_program_agree for s in scores}
    if any(prog[s.index] for s in tied):
        r = _tier(prog, "program")
        if r:
            return r[0], r[1], scores

    # 4. arithmetic validity
    arith = {s.index: (s.arith_truth if s.arith_truth is not None else -1.0) for s in scores}
    r = _tier(arith, "arith")
    if r:
        return r[0], r[1], scores

    # 5. detector
    if detector is not None:
        try:
            pw = {s.index: -detector.predict(question, cands[s.index][0])["p_wrong"]
                  for s in tied}
            full = {s.index: pw.get(s.index, -9.0) for s in scores}
            r = _tier(full, "detector")
            if r:
                return r[0], r[1], scores
        except Exception:
            pass

    # 6. order: prefer the original candidates over back-substitutions
    tied.sort(key=lambda s: s.index)
    return tied[0].index, "order", scores


def detection_score(question: str, trace: str, answer, probes: list[Probe]) -> Optional[float]:
    """Gold-free wrongness score for ONE trace from the probes: 1 - consistency.
    None when no probe was applicable (abstain)."""
    s = score_candidates(question, [(trace, answer, "x")], probes)[0]
    if not s.n_applicable:
        return None
    return 1.0 - s.n_agree / s.n_applicable
