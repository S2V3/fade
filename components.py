"""
The judge-free signals plus the counts. NO composite Q -- classification is
count-based, so only the components are computed.

  E  - execution fidelity: fraction of gold checkpoint values expressed
       ANYWHERE in the trace (value-based, representation-blind).
  E_struct / A_struct - the STRUCTURAL versions: was the checkpoint produced by
       the right OPERATION on the right OPERANDS, not merely landed on.
  E_eff - [AUDIT D6] the blend the cascade actually reads (see below).
  V  - arithmetic self-consistency: fraction of stated equations that are true.
       ZERO equations -> V = 0 ('no visible work' must not score as 'no errors').
  R  - redundancy. [AUDIT D7] DEMOTED to a logged diagnostic: it separates
       correct from wrong traces by +0.010 and no longer gates anything.
  A  - step alignment: checkpoints appearing as RESULTS of stated equations.
       [AUDIT D8] PROMOTED: +0.397 separation, second only to E, and previously
       read by nothing. Now feeds NR and the GOOD rule.
  G  - grounding: fraction of the question's given numbers the trace uses.
  coherence - flag: final answer == result of the LAST stated equation.

  misses  = gold checkpoints absent from the trace   (classification count)
  bad_eqs = stated equations that are false          (classification count)

  n_symbolic - [AUDIT D12] '='-lines carrying a genuine algebraic variable.
  truncated  - [AUDIT D13] the model's own output stopped mid-thought.

All parsing comes from extraction.py; segmentation/comparison from similarity.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from config import (R_VARIANT, DUP_SIM_THRESHOLD, R_REQUIRE_SHARED_VALUE,
                    E_STRUCT_BLEND)
from extraction import (extract_values, value_in, extract_gold_checkpoints,
                        extract_gold_checkpoint_terms,
                        Equation, extract_equations, extract_final_answer,
                        is_correct, symbolic_line_count, is_truncated,
                        unresolved_relation_count)
from similarity import split_steps, pairwise_similarity


# ----------------------------------------------------------------- E
def checkpoint_hits(trace: str, gold_solution: str) -> tuple[int, int]:
    """(hits, total) over gold checkpoints -- the raw counts behind E and
    behind misses = total - hits.

    [AUDIT D5] Matching is now LITERAL on both sides. The old code emitted both
    percent conventions for every '%' token in the TRACE, which doubled the value
    pool a checkpoint could accidentally land in. 13.8% of eval questions contain
    a percent, and 24.4% of gold checkpoints are single-digit values where a
    collision is cheap -- so that inflation was not hypothetical."""
    checkpoints = extract_gold_checkpoints(gold_solution)
    trace_vals = extract_values(trace)
    hits = sum(1 for c in checkpoints if value_in(c, trace_vals))
    return hits, len(checkpoints)


def E_score(trace: str, gold_solution: str) -> float:
    hits, total = checkpoint_hits(trace, gold_solution)
    return hits / total if total else 0.0


def effective_E(E: float, E_struct: float,
                blend: Optional[float] = None) -> float:
    """[AUDIT D6] The execution-fidelity signal the CASCADE reads.

    Value-based E asks only "did the right NUMBER appear?" -- measured mean E was
    0.397 against mean E_struct 0.106, with 627/1500 traces scoring E > 0 and
    E_struct == 0 (right number, wrong route). But raw E_struct is a strict LOWER
    bound: a chained gold checkpoint that the trace computes across separate
    steps cannot operand-match, so driving the cascade off it alone over-fires NR.

    E_eff = max(E_struct, blend*E + (1-blend)*E_struct)
      blend = 1.0 -> pure value-based E (current default; E_eff == E)
      blend = 0.0 -> pure structural
      blend = 0.5 -> credit structure without punishing decomposition

    [AUDIT D24] `blend` is read from config AT CALL TIME, not bound as a default
    argument. Python evaluates default arguments ONCE at import, so the previous
    signature (`blend: float = E_STRUCT_BLEND`) froze the value the module happened
    to see when it was first imported. Setting `config.E_STRUCT_BLEND = 0.5` in a
    notebook then did NOTHING -- which is precisely the knob whose two-step enable
    procedure this file documents. A config flag that silently ignores you is worse
    than no flag.
    """
    if blend is None:
        import config as _cfg
        blend = _cfg.E_STRUCT_BLEND
    return max(E_struct, blend * E + (1.0 - blend) * E_struct)


# ------------------------------------------------- computational steps
def computational_steps(trace: str) -> int:
    """How many trace lines actually COMPUTE something (contain a verifiable
    equation), as opposed to merely being written.

    [AUDIT D27] `s` (= n_steps) counts LINES, and a chat model's lines are mostly
    prose scaffolding -- "Let's break down the information given:", numbered
    headers, restatements. Comparing that against `s_hat` (gold CHECKPOINTS, every
    one of which is a computation) is comparing two different units, and it
    produced a badly misleading picture:

        all lines / s_hat            = 2.74   "the model is VERBOSE"
        computational lines / s_hat  = 0.67   "the model UNDER-COMPUTES"

    Both are true, and only the second is about reasoning. The unit mismatch is
    why ST (step omission) fired 8 times in 1,500 and looked like a dead type: we
    were measuring the model's prose against gold's arithmetic.

    As a quality signal the comparable unit is also far better:

        coverage = comp/s_hat   correct 0.792  wrong 0.640  separation +0.152
        s/s_hat  (today)        correct 2.472  wrong 2.796  separation -0.324

    Note the sign. The existing length ratio is ANTI-correlated with correctness --
    longer traces are more often wrong -- so any rule reading it as "did enough
    work" was reading it backwards.
    """
    return sum(1 for ln in split_steps(trace) if extract_equations(ln))


# ----------------------------------------------------------------- V
def V_score(trace: str,
            eqs: Optional[list[Equation]] = None) -> tuple[float, list[Equation]]:
    if eqs is None:
        eqs = extract_equations(trace)
    if not eqs:
        return 0.0, []
    return sum(e.is_true for e in eqs) / len(eqs), eqs


# ----------------------------------------------------------------- R
def R_score(trace: str, variant: Optional[str] = None,
            dup_threshold: Optional[float] = None) -> float:
    """A single-step (or empty) trace has no pairs -> R = 1.0.

    [AUDIT D7] Two changes, because the old settings flagged legitimately
    PARALLEL math steps as duplicates ("In April, Natalia sold 48 clips." vs
    "In May, she sold half as many, 48/2 = 24") and denied a perfect trace the
    pool:
      1. dup_threshold raised (config.DUP_SIM_THRESHOLD 0.90 -> 0.97);
      2. a flagged pair must ALSO share a numeric value -- parallel PHRASING
         about DIFFERENT numbers is not redundancy (config.R_REQUIRE_SHARED_VALUE).
    R is now a logged diagnostic only; it gates nothing.

    [AUDIT D24] `variant` and `dup_threshold` are read from config AT CALL TIME --
    same default-argument freezing bug as effective_E.
    """
    import config as _cfg
    if variant is None:
        variant = _cfg.R_VARIANT
    if dup_threshold is None:
        dup_threshold = _cfg.DUP_SIM_THRESHOLD
    steps = split_steps(trace)
    n = len(steps)
    if n < 2:
        return 1.0
    try:
        sim = pairwise_similarity(steps)
    except Exception as e:
        # [AUDIT D7] R gates NOTHING now (separation +0.010), so an unavailable
        # embedding backend must not be able to fail SCORING. Degrade to "no
        # duplicates detected" and warn once, rather than taking down the run.
        if not getattr(R_score, "_warned", False):
            print(f"  [warn] R unavailable ({e.__class__.__name__}); "
                  f"reporting R = 1.0. R is a logged diagnostic only.")
            R_score._warned = True
        return 1.0

    if variant == "max":
        worst = max(sim[i][j] for i in range(1, n) for j in range(i))
        return float(max(0.0, 1.0 - worst))
    if variant == "mean_pairwise":
        vals = [sim[i][j] for i in range(1, n) for j in range(i)]
        return float(max(0.0, 1.0 - sum(vals) / len(vals)))
    if variant == "dup_fraction":
        _req_shared = _cfg.R_REQUIRE_SHARED_VALUE
        step_vals = [extract_values(s) for s in steps] if _req_shared else None

        def is_dup(i: int, j: int) -> bool:
            if sim[i][j] <= dup_threshold:
                return False
            if not _req_shared:
                return True
            a, b = step_vals[i], step_vals[j]
            if not a and not b:
                return True                      # two value-free lines: real dup
            return bool(a & b)

        dups = sum(1 for i in range(1, n) if any(is_dup(i, j) for j in range(i)))
        return float(1.0 - dups / (n - 1))
    raise ValueError(f"unknown R variant: {variant!r}")


# ------------------------------------------------- E_struct / A_struct
def structural_hits(gold_solution: str, eqs: list) -> tuple:
    """(e_struct_hits, a_struct_hits, total) over gold checkpoints.
    A checkpoint (result r, operands O, ops P) is an E_struct hit if some TRUE
    trace equation has result==r AND the same operand multiset O; an A_struct hit
    if additionally its operators cover P. This is the dependency check that
    value-based E/A cannot do: landing on r by the wrong route is NOT a hit.

    LIMITATION: a chained checkpoint the trace computes in SEPARATE steps won't
    operand-match, so E_struct is a strict LOWER BOUND -- which is exactly why
    the cascade reads effective_E() rather than E_struct directly."""
    gcs = extract_gold_checkpoint_terms(gold_solution)
    if not gcs:
        return 0, 0, 0
    true_eqs = [e for e in eqs if e.is_true]
    e_hits = a_hits = 0
    for gc in gcs:
        e_ok = a_ok = False
        for e in true_eqs:
            if value_in(gc.result, {e.c}) and e.operands == gc.operands:
                e_ok = True
                if gc.ops <= e.ops:
                    a_ok = True
                    break
        e_hits += e_ok
        a_hits += a_ok
    return e_hits, a_hits, len(gcs)


# ----------------------------------------------------------------- A
def A_score(trace: str, gold_solution: str,
            eqs: Optional[list[Equation]] = None) -> float:
    """Checkpoint must appear as the RESULT (c) of a stated equation -- actually
    computed by a stated operation, not merely mentioned."""
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
    E_struct: float = 0.0   # structural (right operands)
    A_struct: float = 0.0   # structural (right operands + operators)
    E_eff: float = 0.0      # [AUDIT D6] what the cascade reads
    n_comp_steps: int = 0   # [AUDIT D27] lines that actually compute
    comp_coverage: float = 0.0  # [AUDIT D27] n_comp_steps / s_hat -- the ST signal
    n_symbolic: int = 0     # [AUDIT D12] '='-lines carrying an algebraic variable (logged)
    n_unresolved: int = 0   # [AUDIT D12] '='-lines that resolved to NO arithmetic (AL trigger)
    truncated: bool = False  # [AUDIT D13] model output stopped mid-thought
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
            "E_struct": self.E_struct, "A_struct": self.A_struct,
            "E_eff": self.E_eff,
            "n_comp_steps": self.n_comp_steps, "comp_coverage": self.comp_coverage,
            "n_symbolic": self.n_symbolic, "n_unresolved": self.n_unresolved,
            "truncated": self.truncated,
        }


def compute_components(question: str, trace: str, gold_solution: str,
                       gold_answer: float,
                       r_variant: Optional[str] = None) -> ComponentScores:
    """Every signal + both counts for one (question, trace) pair.

    [AUDIT D24] r_variant defaults to None and resolves from config at call time.
    """
    import config as _cfg
    if r_variant is None:
        r_variant = _cfg.R_VARIANT
    eqs = extract_equations(trace)
    V, eqs = V_score(trace, eqs)
    hits, total = checkpoint_hits(trace, gold_solution)
    e_sh, a_sh, tot_s = structural_hits(gold_solution, eqs)

    _ncomp = computational_steps(trace)
    E = hits / total if total else 0.0
    E_struct = e_sh / tot_s if tot_s else 0.0

    return ComponentScores(
        E=E,
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
        E_struct=E_struct,
        A_struct=(a_sh / tot_s if tot_s else 0.0),
        E_eff=effective_E(E, E_struct),
        n_comp_steps=_ncomp,
        comp_coverage=(_ncomp / total if total else 1.0),
        n_symbolic=symbolic_line_count(trace),
        n_unresolved=unresolved_relation_count(trace),
        truncated=is_truncated(trace),
        equations=eqs,
    )