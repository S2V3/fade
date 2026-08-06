"""
One diagnostic cascade for EVERYTHING non-GOOD -- all four cells, correct
and wrong alike. For failures it reads "why did this fail"; for weak
successes, "what is deficient in this reasoning." One taxonomy, one
vocabulary, one set of typed cures.

Inputs are all already-computed signals:
    E        - execution fidelity      (scoring.E_score)
    V        - arithmetic self-consistency (scoring.V_score)
    s        - model step count        (steps.step_count)
    s_hat    - expected step count     (checkpoints.expected_step_count)
    G        - grounding               (scoring.G_score)        [v5]
    coherent - answer/work coherence flag (scoring.coherence_flag) [v5]

Ordered cascade (v5):
  1. NR -- nonsensical reasoning (checked FIRST): E < 0.30 AND the model
     stated NO verifiable arithmetic at all (n_equations == 0).
  2. ST -- step omission: s < 0.6*s_hat AND s_hat >= ST_MIN_SHAT AND E is
     ALSO low (E < ST_E_MAX). The E gate stops terse-but-faithful traces
     from being mislabelled ST and starving CE.
  3. ABSTAIN margin: |E - 0.30| < 0.05 -> UNCLASSIFIED.
  4. SM -- semantic misunderstanding: E < 0.30, V > 0.40, s >= 0.7*s_hat,
     grounded (G >= 0.5) AND coherent -- coherently solved a DIFFERENT problem.
  5. CE -- calculation error: E >= 0.30, V > 0.40, s >= 0.7*s_hat (no upper E).
  6. UNCLASSIFIED -> generic treatment.

CONFIDENCE + ABSTAIN (new):
  UNCLASSIFIED is NOT a fifth failure type -- it is the cascade's *abstain*.
  Its cure is deliberately generic retrieval, because when you cannot name
  the failure the worst move is a targeted (possibly backwards) cure.

  Each fired rule now reports a heuristic `confidence` in [0,1]: how far
  INSIDE its thresholds the deciding signals sit (the minimum normalised
  slack across the rule's conditions). A trace sitting right on a boundary
  scores low. If a TYPED diagnosis fires with confidence < DIAG_MIN_CONFIDENCE
  it is downgraded to UNCLASSIFIED (abstain -> generic) instead of committing
  a confidently-wrong typed cure. With DIAG_MIN_CONFIDENCE = 0.0 (default)
  nothing is downgraded and behaviour is identical to before; the confidence
  is still emitted so FST can drop low-confidence labels from its training
  data (they are noise a predictor would otherwise reproduce).

  The confidence is a heuristic ranking signal, not a calibrated probability;
  it exists to be compared against ONE dev-tuned threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import (NR_E_MAX, NR_V_MAX, G_MIN, ST_LEN_RATIO, ST_MIN_SHAT,
                    ST_E_MAX, SM_E_MAX, CE_E_MIN, V_MIN, V_MAX,
                    FULL_LEN_RATIO, ABSTAIN_MARGIN, DIAG_MIN_CONFIDENCE)


class FailureType(str, Enum):
    NR = "NR"                      # never engaged -> simplest templates
    ST = "ST"                      # step omission -> decomposition
    SM = "SM"                      # wrong problem -> comprehension
    CE = "CE"                      # execution slip -> hygiene
    UNCLASSIFIED = "UNCLASSIFIED"  # ABSTAIN -> generic treatment (not a type)


TYPED_INSTRUCTION = {  # the one-line hint-free typed instructions (Layer 2)
    FailureType.NR: "Restate what the question asks and list its given numbers before solving.",
    FailureType.SM: "Read carefully what the question asks.",
    FailureType.CE: "Verify each computation.",
    FailureType.ST: "Solve step by step, do not skip steps.",
    FailureType.UNCLASSIFIED: "",
}

TYPED_CURE = {  # what the typed positive exemplars look like (section 5)
    FailureType.NR: "simplest complete demonstrations (<=2 ops, target-naming) -- inverts Fu et al. on purpose",
    FailureType.SM: "comprehension exemplars (>=3 quantities, <=2 computations, 'we need to find' language)",
    FailureType.CE: "hygiene exemplars (V >= 0.85, verification language)",
    FailureType.ST: "fully-decomposed exemplars (steps >= expected)",
    FailureType.UNCLASSIFIED: "generic retrieval (abstain -- no targeted cure)",
}


@dataclass
class Diagnosis:
    ftype: FailureType
    reason: str            # which rule fired / why it abstained
    confidence: float = 1.0  # heuristic [0,1]; low = near a boundary


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _slack_above(value: float, thresh: float) -> float:
    """How far `value` sits ABOVE a lower bound `thresh`, normalised so the
    headroom (1 - thresh) maps to 1.0. Used for 'value > thresh' conditions."""
    denom = max(1.0 - thresh, 1e-9)
    return _clip01((value - thresh) / denom)


def _slack_below(value: float, thresh: float) -> float:
    """How far `value` sits BELOW an upper bound `thresh`, normalised so the
    room (thresh) maps to 1.0. Used for 'value < thresh' conditions."""
    denom = max(thresh, 1e-9)
    return _clip01((thresh - value) / denom)


def _len_slack(s: int, target: float, *, below: bool) -> float:
    """Normalised distance of step count s from the length target."""
    denom = max(target, 1e-9)
    return _clip01(((target - s) if below else (s - target)) / denom)


def diagnose(E: float, V: float, s: int, s_hat: int,
             G: float, coherent: Optional[bool],
             n_equations: int = 0,
             margin: float = ABSTAIN_MARGIN,
             min_confidence: float = DIAG_MIN_CONFIDENCE) -> Diagnosis:
    """Diagnose ONE non-GOOD trace (correct or wrong). Rules fire in order.
    A typed diagnosis whose confidence < min_confidence is downgraded to the
    UNCLASSIFIED abstain (generic cure)."""
    coh = bool(coherent)  # None -> not coherent

    def finalize(d: Diagnosis) -> Diagnosis:
        # Abstain if a TYPED rule fired but too weakly. UNCLASSIFIED never
        # downgrades (it is already the abstain).
        if (d.ftype is not FailureType.UNCLASSIFIED
                and min_confidence > 0.0 and d.confidence < min_confidence):
            return Diagnosis(
                FailureType.UNCLASSIFIED,
                f"abstain: {d.ftype.value} fired but confidence "
                f"{d.confidence:.2f} < {min_confidence:.2f} ({d.reason})",
                confidence=d.confidence)
        return d

    # 1 -- NR first: no verifiable work at all
    if E < NR_E_MAX and n_equations == 0:
        conf = _slack_below(E, NR_E_MAX)  # n_equations==0 is exact
        return finalize(Diagnosis(
            FailureType.NR,
            f"E={E:.3f} < {NR_E_MAX} and no stated equations "
            f"(n_equations=0): model showed no verifiable work",
            confidence=conf))

    # 2 -- ST: real-but-compressed work, gated on ALSO-low execution fidelity
    if s_hat >= ST_MIN_SHAT and s < ST_LEN_RATIO * s_hat and E < ST_E_MAX:
        conf = min(_len_slack(s, ST_LEN_RATIO * s_hat, below=True),
                   _slack_below(E, ST_E_MAX))
        return finalize(Diagnosis(
            FailureType.ST,
            f"s={s} < {ST_LEN_RATIO}*s_hat={ST_LEN_RATIO * s_hat:.1f}, "
            f"s_hat={s_hat} >= {ST_MIN_SHAT}, and E={E:.3f} < {ST_E_MAX}",
            confidence=conf))

    # 3 -- abstain margin on the single remaining E boundary (strict '<')
    if margin > 0 and abs(E - SM_E_MAX) < margin:
        return Diagnosis(
            FailureType.UNCLASSIFIED,
            f"abstain: |E={E:.3f} - {SM_E_MAX}| < {margin}",
            confidence=0.0)

    full_length = s >= FULL_LEN_RATIO * s_hat
    len_target = FULL_LEN_RATIO * s_hat

    # 4 -- SM: coherently solved a DIFFERENT problem
    if E < SM_E_MAX and V > V_MIN and full_length and G >= G_MIN and coh:
        conf = min(_slack_below(E, SM_E_MAX), _slack_above(V, V_MIN),
                   _len_slack(s, len_target, below=False), _slack_above(G, G_MIN))
        return finalize(Diagnosis(
            FailureType.SM,
            f"E={E:.3f} < {SM_E_MAX}, V={V:.3f} > {V_MIN}, "
            f"s={s} >= {FULL_LEN_RATIO}*s_hat={len_target:.1f}, "
            f"G={G:.3f} >= {G_MIN}, coherent",
            confidence=conf))

    # 5 -- CE: on the path, execution slipped (no upper E bound in v5)
    if E >= CE_E_MIN and V < V_MAX and full_length:
        conf = min(_slack_above(E, CE_E_MIN), _slack_below(V, V_MAX),
                   _len_slack(s, len_target, below=False))
        return finalize(Diagnosis(
            FailureType.CE,
            f"E={E:.3f} >= {CE_E_MIN}, V={V:.3f} < {V_MAX}, "
            f"s={s} >= {FULL_LEN_RATIO}*s_hat={len_target:.1f}",
            confidence=conf))

    # 6 -- everything else -> abstain
    return Diagnosis(FailureType.UNCLASSIFIED, "no rule fired", confidence=0.0)