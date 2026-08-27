"""
One diagnostic cascade for EVERYTHING non-GOOD -- all cells, correct and wrong
alike. For failures it reads "why did this fail"; for weak successes, "what is
deficient in this reasoning." One taxonomy, one vocabulary, one set of typed cures.

=============================================================================
AUDIT REVISION -- why this file changed (full evidence in FADE_AUDIT.md)
=============================================================================
On the 1,500-problem train run the old cascade abstained on 52.2% of traces
(713/1367). That is not a taxonomy, it is a coin toss with extra steps. The
decomposition of those 713:

    216 (30.3%)  the abstain band |E - 0.30| < 0.05
    243 (34.1%)  CE blocked by V >= 0.85  -- valid arithmetic, wrong answer
    137 (19.2%)  SM blocked by V <= 0.40
     85 (11.9%)  SM blocked by incoherent
     15 ( 2.1%)  SM blocked by G < 0.5
     17 ( 2.4%)  blocked by not full_length

Five changes, each independently measured:

 [D10] ABSTAIN_MARGIN 0.05 -> 0.02. E = hits/s_hat is a ratio of SMALL INTEGERS.
       All 216 banded traces sat at exactly three values: 1/3 (142), 1/4 (71),
       2/7 (3). A continuous +/-0.05 guard on a signal with ~6 distinct values
       below 0.5 swallows whole denominators. 0.02 leaves 3/216.
       Effect alone: UNCLASSIFIED 52.2% -> 43.2%.

 [D11] WP (wrong plan) -- NEW. The 243 V>=0.85 traces are NOT calculation errors:
       CE fires at mean V = 0.193 while this group sits at mean V = 0.999. Their
       arithmetic is SOUND and their PLAN is wrong. Do NOT fix this by dropping
       CE's upper V bound -- that hands "verify each computation" to traces whose
       computations are already correct, the exact confidently-wrong cure the
       abstain design exists to prevent.
       Effect with D10: UNCLASSIFIED -> 21.7%.

 [D12] AL (algebraic non-execution) -- NEW. 400/1500 traces (26.7%) set up
       symbolic variables and never resolve them ("x + 3x/2 = 4x/2" -> "#### 4"
       when gold was 5250). Accuracy 12.0% when algebraic vs 21.3% when not: a
       9.3-point cost with no type describing it. Detected from the TRACE with no
       gold, which also makes it FST's most promising target class.

 [D13] TR (truncated) -- NEW, and it fires FIRST. 154/1500 traces stop
       mid-thought. These are GENERATION failures wearing a reasoning-failure
       costume. Diagnosing them as NR/SM/CE poisons FST's training labels with
       noise that has nothing to do with the question. A TR trace should be
       RE-GENERATED, not cured.

 [D14] ST redefined. ST fired 7 times in 1,500 because it was a LENGTH test
       against a model that writes 2.74x MORE lines than gold has steps (mean
       s = 7.12 vs s_hat = 3.10); only 12 traces satisfy s < 0.6*s_hat at all.
       The concept is real, the operationalisation was wrong. ST is now
       semantic: "you wrote plenty but never actually COMPUTED most of the
       required intermediates" -- A_struct < 0.34 with the length floor as a
       guard rather than the trigger. Restore the old rule with
       config.ST_USE_LEGACY = True.

 [D6]  Every E test now reads E_eff (components.effective_E), not raw E.

Cascade order (order matters: nonsense must not masquerade as something subtler):
    TR -> NR -> AL -> ST -> abstain-band -> SM -> CE -> WP -> UNCLASSIFIED

CONFIDENCE + ABSTAIN
    UNCLASSIFIED is NOT a failure type -- it is the cascade's *abstain*, and its
    cure is deliberately generic retrieval, because when you cannot name the
    failure the worst move is a targeted (possibly backwards) cure.
    Each fired rule reports a heuristic confidence in [0,1]: how far INSIDE its
    thresholds the deciding signals sit. A TYPED diagnosis firing below
    DIAG_MIN_CONFIDENCE is downgraded to the abstain. Default 0.0 = off.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from config import (NR_E_MAX, NR_V_MAX, NR_A_MAX, G_MIN,
                    ST_LEN_RATIO, ST_MIN_SHAT, ST_E_MAX, ST_USE_LEGACY,
                    ST_ASTRUCT_MAX, ST_MODE, ST_COVERAGE_MAX,
                    AL_MIN_VAR_LINES, TR_ENABLED, WP_ENABLED,
                    NR_USE_A,
                    SM_E_MAX, CE_E_MIN, V_MIN, V_MAX,
                    FULL_LEN_RATIO, ABSTAIN_MARGIN, DIAG_MIN_CONFIDENCE)


class FailureType(str, Enum):
    TR = "TR"                      # truncated generation -> re-generate, don't cure
    NR = "NR"                      # never engaged -> simplest templates
    AL = "AL"                      # algebraic non-execution -> numeric-only examples
    ST = "ST"                      # step omission -> decomposition
    SM = "SM"                      # wrong problem -> comprehension
    CE = "CE"                      # execution slip -> arithmetic hygiene
    WP = "WP"                      # wrong plan -> planning
    UNCLASSIFIED = "UNCLASSIFIED"  # ABSTAIN -> generic treatment (not a type)


# The one-line hint-free typed instructions. NEVER reveal the answer.
TYPED_INSTRUCTION = {
    FailureType.TR: "",   # re-generate; nothing to instruct
    FailureType.NR: "Restate what the question asks and list its given numbers before solving.",
    FailureType.AL: "Compute a number on every line. Do not write relations between named quantities.",
    FailureType.ST: "Compute and write out every intermediate value the problem needs, one per line.",
    FailureType.SM: "Read carefully what the question asks.",
    FailureType.CE: "Verify each computation.",
    FailureType.WP: "Decide what quantity the question asks for before you compute anything.",
    FailureType.UNCLASSIFIED: "",
}

# What the typed positive exemplars look like.
TYPED_CURE = {
    FailureType.TR: "re-generate (generation failure, not a reasoning failure)",
    FailureType.NR: "simplest complete demonstrations (<=2 ops, target-naming)",
    FailureType.AL: "numeric-only worked examples (no symbols; every line ends in a number)",
    FailureType.ST: "fully-decomposed exemplars (one computed value per line)",
    FailureType.SM: "comprehension exemplars (>=3 quantities, <=2 computations, 'we need to find')",
    FailureType.CE: "arithmetic-hygiene exemplars (V >= 0.85, verification language)",
    FailureType.WP: "planning exemplars (state the target quantity, then order the operations)",
    FailureType.UNCLASSIFIED: "generic retrieval (abstain -- no targeted cure)",
}

# Types that are real failure MODES and therefore valid FST targets.
# TR is excluded: it is a generation artefact, not a property of the question.
FST_TARGET_TYPES = (FailureType.NR, FailureType.AL, FailureType.ST,
                    FailureType.SM, FailureType.CE, FailureType.WP)


@dataclass
class Diagnosis:
    ftype: FailureType
    reason: str            # which rule fired / why it abstained
    confidence: float = 1.0  # heuristic [0,1]; low = near a boundary


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _slack_above(value: float, thresh: float) -> float:
    """How far `value` sits ABOVE a lower bound, normalised so the headroom
    (1 - thresh) maps to 1.0."""
    denom = max(1.0 - thresh, 1e-9)
    return _clip01((value - thresh) / denom)


def _slack_below(value: float, thresh: float) -> float:
    """How far `value` sits BELOW an upper bound, normalised so the room
    (thresh) maps to 1.0."""
    denom = max(thresh, 1e-9)
    return _clip01((thresh - value) / denom)


def _len_slack(s: int, target: float, *, below: bool) -> float:
    denom = max(target, 1e-9)
    return _clip01(((target - s) if below else (s - target)) / denom)


def diagnose(E: float, V: float, s: int, s_hat: int,
             G: float, coherent: Optional[bool],
             n_equations: int = 0,
             margin: float = ABSTAIN_MARGIN,
             min_confidence: float = DIAG_MIN_CONFIDENCE,
             *,
             A: float = 0.0,
             A_struct: float = 0.0,
             n_symbolic: int = 0,
             n_unresolved: int = 0,
             comp_coverage: float = 1.0,
             truncated: bool = False,
             correct: bool = False) -> Diagnosis:
    """Diagnose ONE non-GOOD trace (correct or wrong). Rules fire in order.

    `E` should be E_eff (components.effective_E). The new keyword-only arguments
    all default to their old no-op values, so any legacy caller passing only the
    original positional signature gets the pre-audit behaviour for those rules.
    """
    coh = bool(coherent)  # None -> not coherent

    def finalize(d: Diagnosis) -> Diagnosis:
        if (d.ftype not in (FailureType.UNCLASSIFIED, FailureType.TR)
                and min_confidence > 0.0 and d.confidence < min_confidence):
            return Diagnosis(
                FailureType.UNCLASSIFIED,
                f"abstain: {d.ftype.value} fired but confidence "
                f"{d.confidence:.2f} < {min_confidence:.2f} ({d.reason})",
                confidence=d.confidence)
        return d

    # 0 -- TR FIRST [D13]: a truncated generation is not a reasoning failure.
    # Diagnosing it would attribute a decoding artefact to the question and
    # poison FST's labels.
    # [AUDIT SELF-REVIEW] `and not correct`: an early version fired TR on 11
    # traces that were scored CORRECT. Whatever the surface looks like, a trace
    # that reached the right answer did not fail by truncation.
    if TR_ENABLED and truncated and not correct:
        return Diagnosis(
            FailureType.TR,
            "model output stopped mid-thought (dangling function word, trailing "
            "operator, or partial '####'): generation failure -- re-generate "
            "rather than cure",
            confidence=1.0)

    # 1 -- NR: no verifiable work at all.
    # [D8 CORRECTED] Deliberately still the equation-count test, NOT an A test.
    # Defining NR as "A <= 0.15" was tried and measured: NR went 183 -> 688
    # (50.6% of all diagnoses), because 616 traces have A <= 0.15 while stating
    # real equations -- they did work, on the wrong quantities. That is SM/WP,
    # not "never engaged". A is used in the GOOD rule and (as A_struct) in ST.
    _nr_fires = (E < NR_E_MAX and n_equations == 0) if not NR_USE_A else \
                (E < NR_E_MAX and A <= NR_A_MAX)
    if _nr_fires:
        conf = _slack_below(E, NR_E_MAX)
        return finalize(Diagnosis(
            FailureType.NR,
            f"E_eff={E:.3f} < {NR_E_MAX} and no stated equations "
            f"(n_equations={n_equations}): model showed no verifiable work",
            confidence=conf))

    # 2 -- AL [D12]: symbolic set-up that never resolves to a number
    # [AUDIT SELF-REVIEW] fires on UNRESOLVED RELATIONS, not on the variable
    # regex -- see extraction.unresolved_relation_count for why.
    if n_unresolved >= AL_MIN_VAR_LINES and not correct:
        conf = _clip01((n_unresolved - AL_MIN_VAR_LINES + 1) / 3.0)
        return finalize(Diagnosis(
            FailureType.AL,
            f"{n_unresolved} '=' lines resolved to no verifiable arithmetic "
            f"(>= {AL_MIN_VAR_LINES}): the model stated relations between named "
            f"quantities/variables instead of computing numbers",
            confidence=conf))

    # 3 -- ST: step omission
    # [AUDIT D27] The trigger is COMPUTATION COVERAGE, not line length.
    # `s` counts LINES and a chat model's lines are mostly prose, so s/s_hat = 2.74
    # ("verbose") while computational_lines/s_hat = 0.67 ("under-computes"). Only
    # the second is about reasoning. Under the old length rule ST fired 8 times in
    # 1,500 and looked dead; on the comparable unit it is a real, well-populated
    # class. Threshold 0.5 was swept on train (see config.ST_MODE).
    if ST_MODE == "coverage":
        if (s_hat >= ST_MIN_SHAT and n_equations >= 1
                and comp_coverage < ST_COVERAGE_MAX):
            conf = _slack_below(comp_coverage, ST_COVERAGE_MAX)
            return finalize(Diagnosis(
                FailureType.ST,
                f"computation coverage {comp_coverage:.2f} < {ST_COVERAGE_MAX} "
                f"(the trace computes fewer than half the {s_hat} values the "
                f"problem requires, despite stating {n_equations} equation(s))",
                confidence=conf))
    elif ST_MODE == "astruct":
        if (s_hat >= ST_MIN_SHAT and A_struct < ST_ASTRUCT_MAX
                and E >= CE_E_MIN and s >= FULL_LEN_RATIO * s_hat):
            conf = min(_slack_below(A_struct, ST_ASTRUCT_MAX),
                       _slack_above(E, CE_E_MIN))
            return finalize(Diagnosis(
                FailureType.ST,
                f"A_struct={A_struct:.3f} < {ST_ASTRUCT_MAX} with E_eff={E:.3f}",
                confidence=conf))
    else:  # "length" -- the original rule, kept for ablation
        if s_hat >= ST_MIN_SHAT and s < ST_LEN_RATIO * s_hat and E < ST_E_MAX:
            conf = min(_len_slack(s, ST_LEN_RATIO * s_hat, below=True),
                       _slack_below(E, ST_E_MAX))
            return finalize(Diagnosis(
                FailureType.ST,
                f"s={s} < {ST_LEN_RATIO}*s_hat={ST_LEN_RATIO * s_hat:.1f}",
                confidence=conf))

    # 4 -- abstain margin on the single remaining E boundary (strict '<') [D10]
    if margin > 0 and abs(E - SM_E_MAX) < margin:
        return Diagnosis(
            FailureType.UNCLASSIFIED,
            f"abstain: |E_eff={E:.3f} - {SM_E_MAX}| < {margin}",
            confidence=0.0)

    full_length = s >= FULL_LEN_RATIO * s_hat
    len_target = FULL_LEN_RATIO * s_hat

    # 5 -- SM: coherently solved a DIFFERENT problem
    if E < SM_E_MAX and V > V_MIN and full_length and G >= G_MIN and coh:
        conf = min(_slack_below(E, SM_E_MAX), _slack_above(V, V_MIN),
                   _len_slack(s, len_target, below=False), _slack_above(G, G_MIN))
        return finalize(Diagnosis(
            FailureType.SM,
            f"E_eff={E:.3f} < {SM_E_MAX}, V={V:.3f} > {V_MIN}, "
            f"s={s} >= {FULL_LEN_RATIO}*s_hat={len_target:.1f}, "
            f"G={G:.3f} >= {G_MIN}, coherent",
            confidence=conf))

    # 6 -- CE: on the path, arithmetic slipped
    if E >= CE_E_MIN and V < V_MAX and full_length:
        conf = min(_slack_above(E, CE_E_MIN), _slack_below(V, V_MAX),
                   _len_slack(s, len_target, below=False))
        return finalize(Diagnosis(
            FailureType.CE,
            f"E_eff={E:.3f} >= {CE_E_MIN}, V={V:.3f} < {V_MAX}, "
            f"s={s} >= {FULL_LEN_RATIO}*s_hat={len_target:.1f}",
            confidence=conf))

    # 7 -- WP [D11]: arithmetic is SOUND, the plan is wrong
    if WP_ENABLED and E >= CE_E_MIN and V >= V_MAX and full_length and not correct:
        conf = min(_slack_above(E, CE_E_MIN), _slack_above(V, V_MAX),
                   _len_slack(s, len_target, below=False))
        return finalize(Diagnosis(
            FailureType.WP,
            f"E_eff={E:.3f} >= {CE_E_MIN}, V={V:.3f} >= {V_MAX} (arithmetic is "
            f"sound), full length, answer still wrong: the PLAN is wrong, not "
            f"the computation",
            confidence=conf))

    # 8 -- everything else -> abstain
    return Diagnosis(FailureType.UNCLASSIFIED, "no rule fired", confidence=0.0)


def diagnose_components(c, margin: float = ABSTAIN_MARGIN,
                        min_confidence: float = DIAG_MIN_CONFIDENCE) -> Diagnosis:
    """Convenience wrapper: diagnose straight from a ComponentScores object so
    callers cannot forget to pass the new signals (or to pass E_eff instead of
    raw E). This is the preferred entry point."""
    return diagnose(
        c.E_eff, c.V, c.n_steps, c.n_checkpoints, c.G, c.coherent,
        n_equations=c.n_equations, margin=margin, min_confidence=min_confidence,
        A=c.A, A_struct=c.A_struct, n_symbolic=c.n_symbolic,
        n_unresolved=c.n_unresolved, comp_coverage=c.comp_coverage,
        truncated=c.truncated, correct=c.correct)