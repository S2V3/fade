"""
The six judge-free signals plus the two counts. NO composite Q anymore --
classification is count-based, so only the components are computed. (If
analysis or FST later wants a composite, it builds one from these logged
components; nothing here decides anything.)

  E  - execution fidelity: fraction of gold checkpoint values expressed
       ANYWHERE in the trace (value-based, representation-blind).
  V  - arithmetic self-consistency: fraction of stated 'A op B = C'
       equations that are true. ZERO equations -> V = 0 ('no visible work'
       must not score as 'no errors'). Blind to the gold answer by design.
  R  - redundancy: dup_fraction default; ROSCOE's max / mean_pairwise kept.
  A  - step alignment: checkpoints appearing as RESULTS of stated
       equations. A <= E by construction. High V + low A = confidently
       doing irrelevant work.
  G  - grounding: fraction of the question's given numbers the trace uses.
  coherence - flag: final answer == result of the LAST stated equation.
       None when there are no equations (nothing to compare; the V=0 rule
       already punishes the missing work).

  misses  = gold checkpoints absent from the trace   (classification count)
  bad_eqs = stated equations that are false          (classification count)

All parsing comes from extraction.py; segmentation/comparison from
similarity.py. This module only combines them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from config import R_VARIANT, DUP_SIM_THRESHOLD
from extraction import (extract_values, value_in, extract_gold_checkpoints,
                        Equation, extract_equations, extract_final_answer,
                        is_correct)
from similarity import split_steps, pairwise_similarity


# ----------------------------------------------------------------- E
def checkpoint_hits(trace: str, gold_solution: str) -> tuple[int, int]:
    """(hits, total) over gold checkpoints -- the raw counts behind E and
    behind misses = total - hits."""
    checkpoints = extract_gold_checkpoints(gold_solution)
    trace_vals = extract_values(trace)
    hits = sum(1 for c in checkpoints if value_in(c, trace_vals))
    return hits, len(checkpoints)


def E_score(trace: str, gold_solution: str) -> float:
    hits, total = checkpoint_hits(trace, gold_solution)
    return hits / total if total else 0.0


# ----------------------------------------------------------------- V
def V_score(trace: str,
            eqs: Optional[list[Equation]] = None) -> tuple[float, list[Equation]]:
    if eqs is None:
        eqs = extract_equations(trace)
    if not eqs:
        return 0.0, []
    return sum(e.is_true for e in eqs) / len(eqs), eqs


# ----------------------------------------------------------------- R
def R_score(trace: str, variant: str = R_VARIANT,
            dup_threshold: float = DUP_SIM_THRESHOLD) -> float:
    """A single-step (or empty) trace has no pairs -> R = 1.0."""
    steps = split_steps(trace)
    n = len(steps)
    if n < 2:
        return 1.0
    sim = pairwise_similarity(steps)

    if variant == "max":
        worst = max(sim[i][j] for i in range(1, n) for j in range(i))
        return float(max(0.0, 1.0 - worst))
    if variant == "mean_pairwise":
        vals = [sim[i][j] for i in range(1, n) for j in range(i)]
        return float(max(0.0, 1.0 - sum(vals) / len(vals)))
    if variant == "dup_fraction":
        dups = sum(1 for i in range(1, n)
                   if max(sim[i][j] for j in range(i)) > dup_threshold)
        return float(1.0 - dups / (n - 1))
    raise ValueError(f"unknown R variant: {variant!r}")


# ----------------------------------------------------------------- A
def A_score(trace: str, gold_solution: str,
            eqs: Optional[list[Equation]] = None) -> float:
    """Checkpoint must appear as the RESULT (c) of a stated equation --
    actually computed by a stated operation, not merely mentioned."""
    checkpoints = extract_gold_checkpoints(gold_solution)
    if not checkpoints:
        return 0.0
    if eqs is None:
        eqs = extract_equations(trace)
    results = {e.c for e in eqs}
    hits = sum(1 for c in checkpoints if value_in(c, results))
    return hits / len(checkpoints)


# ----------------------------------------------------------------- G
def G_score(question: str, trace: str) -> float:
    q_vals = extract_values(question)
    if not q_vals:
        return 1.0
    t_vals = extract_values(trace)
    used = sum(1 for q in q_vals if value_in(q, t_vals))
    return used / len(q_vals)


# ------------------------------------------------------- coherence flag
def coherence_flag(trace: str,
                   eqs: Optional[list[Equation]] = None) -> Optional[bool]:
    if eqs is None:
        eqs = extract_equations(trace)
    if not eqs:
        return None
    ans = extract_final_answer(trace)
    if ans is None:
        return None
    return math.isclose(eqs[-1].c, ans, rel_tol=1e-6, abs_tol=1e-6)


# --------------------------------------------------------- all together
@dataclass
class ComponentScores:
    E: float
    V: float
    R: float
    A: float
    G: float
    coherent: Optional[bool]
    correct: bool
    final_answer: Optional[float]
    misses: int           # classification count 1
    bad_eqs: int          # classification count 2
    n_steps: int          # s      (model step count)
    n_checkpoints: int    # s_hat  (expected step count)
    n_equations: int
    r_variant: str
    equations: list[Equation] = field(default_factory=list, repr=False)

    def signals(self) -> dict:
        """JSON-safe dict of everything the store persists per trace."""
        return {
            "E": self.E, "V": self.V, "R": self.R, "A": self.A, "G": self.G,
            "coherent": self.coherent, "correct": self.correct,
            "final_answer": self.final_answer,
            "misses": self.misses, "bad_eqs": self.bad_eqs,
            "n_steps": self.n_steps, "n_checkpoints": self.n_checkpoints,
            "n_equations": self.n_equations, "r_variant": self.r_variant,
        }


def compute_components(question: str, trace: str, gold_solution: str,
                       gold_answer: float,
                       r_variant: str = R_VARIANT) -> ComponentScores:
    """Every signal + both counts for one (question, trace) pair."""
    eqs = extract_equations(trace)
    V, eqs = V_score(trace, eqs)
    hits, total = checkpoint_hits(trace, gold_solution)

    return ComponentScores(
        E=hits / total if total else 0.0,
        V=V,
        R=R_score(trace, variant=r_variant),
        A=A_score(trace, gold_solution, eqs),
        G=G_score(question, trace),
        coherent=coherence_flag(trace, eqs),
        correct=is_correct(trace, gold_answer),
        final_answer=extract_final_answer(trace),
        misses=total - hits,
        bad_eqs=sum(1 for e in eqs if not e.is_true),
        n_steps=len(split_steps(trace)),
        n_checkpoints=total,
        n_equations=len(eqs),
        r_variant=r_variant,
        equations=eqs,
    )