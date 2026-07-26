"""
One diagnostic cascade for EVERYTHING non-GOOD -- all four cells, correct
and wrong alike. For failures it reads "why did this fail"; for weak
successes, "what is deficient in this reasoning." One taxonomy, one
vocabulary, one set of typed cures.

Inputs are all already-computed signals:
    E        - execution fidelity          (scoring.E_score)
    V        - arithmetic self-consistency (scoring.V_score)
    s        - model step count            (steps.step_count)
    s_hat    - expected step count         (checkpoints.expected_step_count)
    G        - grounding                   (scoring.G_score)        [v5]
    coherent - answer/work coherence flag  (scoring.coherence_flag) [v5]

Ordered cascade (v5):
    1. NR -- nonsensical reasoning (checked FIRST): E < 0.30 AND the model
       stated NO verifiable arithmetic at all (n_equations == 0). "Never
       engaged" means no work, not "worked and slipped": a trace with valid
       equations that merely fails coherence is NOT nonsensical (that was the
       old bug -- a trace with four true equations got mislabelled NR because
       its final answer didn't match its last line). Checked first because
       genuine nonsense is short and low-E -- ST and SM would swallow it and
       corrupt their statistics (and every FST cell built on them). An
       NR-diagnosed CORRECT trace is, by definition, a guess.
    2. ST -- step omission: s < 0.6*s_hat AND s_hat >= 3. Real-but-compressed
       work (NR-first guarantees "real").
    3. ABSTAIN margin: |E - 0.30| < 0.05 -> UNCLASSIFIED. One boundary to
       guard now (the 0.70 bound is gone) -> higher coverage than v3.
    4. SM -- semantic misunderstanding: E < 0.30, V > 0.40, s >= 0.7*s_hat,
       AND grounded (G >= 0.5) AND coherent. SM now means what it says: the
       model COHERENTLY solved a DIFFERENT problem. Same E/V profile without
       grounding/coherence -> that trace already went to NR.
    5. CE -- calculation error: E >= 0.30, V > 0.40, s >= 0.7*s_hat.
       NO upper E bound (v5): a wrong answer that hit 75% -- or 100% -- of
       checkpoints with valid arithmetic is the PUREST CE, a slip near the
       end of the gold path. (The old <0.70 ceiling sent exactly these to
       UNCLASSIFIED; a Phase-0 hand label caught it.)
    6. UNCLASSIFIED -> generic treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import (NR_E_MAX, NR_V_MAX, G_MIN, ST_LEN_RATIO, ST_MIN_SHAT,
                    SM_E_MAX, CE_E_MIN, V_MIN, V_MAX, FULL_LEN_RATIO, ABSTAIN_MARGIN)


class FailureType(str, Enum):
    NR = "NR"                       # never engaged      -> simplest templates
    ST = "ST"                       # step omission      -> decomposition
    SM = "SM"                       # wrong problem      -> comprehension
    CE = "CE"                       # execution slip     -> hygiene
    UNCLASSIFIED = "UNCLASSIFIED"   # generic treatment


TYPED_INSTRUCTION = {   # the one-line hint-free typed instructions (Layer 2)
    FailureType.NR: "Restate what the question asks and list its given numbers before solving.",
    FailureType.SM: "Read carefully what the question asks.",
    FailureType.CE: "Verify each computation.",
    FailureType.ST: "Solve step by step, do not skip steps.",
    FailureType.UNCLASSIFIED: "",
}

TYPED_CURE = {          # what the typed positive exemplars look like (section 5)
    FailureType.NR: "simplest complete demonstrations (<=2 ops, target-naming) -- inverts Fu et al. on purpose",
    FailureType.SM: "comprehension exemplars (>=3 quantities, <=2 computations, 'we need to find' language)",
    FailureType.CE: "hygiene exemplars (V >= 0.85, verification language)",
    FailureType.ST: "fully-decomposed exemplars (steps >= expected)",
    FailureType.UNCLASSIFIED: "generic retrieval",
}


@dataclass
class Diagnosis:
    ftype: FailureType
    reason: str          # which rule fired / why it abstained


def diagnose(E: float, V: float, s: int, s_hat: int,
             G: float, coherent: Optional[bool],
             n_equations: int = 0,
             margin: float = ABSTAIN_MARGIN) -> Diagnosis:
    """Diagnose ONE non-GOOD trace (correct or wrong). Rules fire in order."""
    coh = bool(coherent)          # None -> not coherent

    # 1 -- NR first: nonsense would masquerade as ST or SM. NR now means the
    #      model showed NO verifiable work (no stated equations) AND low
    #      execution fidelity -- not merely "engaged but incoherent/wrong".
    if E < NR_E_MAX and n_equations == 0:
        return Diagnosis(FailureType.NR,
                         f"E={E:.3f} < {NR_E_MAX} and no stated equations "
                         f"(n_equations=0): model showed no verifiable work")

    # 2 -- ST: real-but-compressed work
    if s_hat >= ST_MIN_SHAT and s < ST_LEN_RATIO * s_hat:
        return Diagnosis(FailureType.ST,
                         f"s={s} < {ST_LEN_RATIO}*s_hat={ST_LEN_RATIO * s_hat:.1f} "
                         f"and s_hat={s_hat} >= {ST_MIN_SHAT}")

    # 3 -- abstain margin on the single remaining E boundary (strict '<')
    if margin > 0 and abs(E - SM_E_MAX) < margin:
        return Diagnosis(FailureType.UNCLASSIFIED,
                         f"abstain: |E={E:.3f} - {SM_E_MAX}| < {margin}")

    full_length = s >= FULL_LEN_RATIO * s_hat

    # 4 -- SM: coherently solved a DIFFERENT problem
    if E < SM_E_MAX and V > V_MIN and full_length and G >= G_MIN and coh:
        return Diagnosis(FailureType.SM,
                         f"E={E:.3f} < {SM_E_MAX}, V={V:.3f} > {V_MIN}, "
                         f"s={s} >= {FULL_LEN_RATIO}*s_hat={FULL_LEN_RATIO * s_hat:.1f}, "
                         f"G={G:.3f} >= {G_MIN}, coherent")

    # 5 -- CE: on the path, execution slipped (no upper E bound in v5)
    if E >= CE_E_MIN and V < V_MAX and full_length:
        return Diagnosis(FailureType.CE,
                         f"E={E:.3f} >= {CE_E_MIN}, V={V:.3f} < {V_MAX}, "
                         f"s={s} >= {FULL_LEN_RATIO}*s_hat={FULL_LEN_RATIO * s_hat:.1f}")

    # 6 -- everything else
    return Diagnosis(FailureType.UNCLASSIFIED, "no rule fired")