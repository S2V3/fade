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
# [AUDIT D40] The original one-clause instructions are kept as _SHORT. They named
# the failure but never said what to do about it, and the typed-vs-generic
# ablation could not distinguish them from no instruction at all. The procedural
# set below states a checkable procedure instead. config.TYPED_INSTRUCTION_STYLE
# selects between them.
TYPED_INSTRUCTION_SHORT = {
    FailureType.TR: "",
    FailureType.NR: "Restate what the question asks and list its given numbers before solving.",
    FailureType.AL: "Compute a number on every line. Do not write relations between named quantities.",
    FailureType.ST: "Compute and write out every intermediate value the problem needs, one per line.",
    FailureType.SM: "Read carefully what the question asks.",
    FailureType.CE: "Verify each computation.",
    FailureType.WP: "Decide what quantity the question asks for before you compute anything.",
    FailureType.UNCLASSIFIED: "",
}

TYPED_INSTRUCTION_PROCEDURAL = {
    FailureType.TR: "",          # a truncation artefact -- nothing to instruct
    FailureType.NR: (
        "Do not give the answer straight away.\n"
        "1. Write one line naming the quantity the question asks for.\n"
        "2. Write one line listing every number the question gives.\n"
        "3. Then compute, one operation per line, and end with '#### <number>'."),
    FailureType.AL: (
        "Do not introduce letters or variables, and do not write a relation you "
        "then leave unsolved.\n"
        "Every line must end in a number you have actually worked out. Where you "
        "would write 'let x be ...', compute that quantity directly from the "
        "numbers the question gives."),
    FailureType.ST: (
        "Show every intermediate quantity on its own line, including the obvious "
        "ones.\n"
        "Do not combine two operations into a single line, and do not jump from "
        "the given numbers to the answer. Each line shows one operation and its "
        "result."),
    FailureType.SM: (
        "Use only the numbers the question actually states.\n"
        "1. First copy out those numbers on one line.\n"
        "2. If you need a quantity the question does not give, derive it and show "
        "the derivation before using it.\n"
        "Never introduce a value that appears nowhere in the question."),
    FailureType.CE: (
        "Work one operation per line and check each result before using it.\n"
        "Do not round: if a division is not exact, keep the exact fraction or the "
        "full decimal and carry it into the next line unchanged."),
    FailureType.WP: (
        "Decide the target before you compute.\n"
        "1. Write 'We need to find: <the exact quantity the question asks for>'.\n"
        "2. Work out which steps lead to that quantity.\n"
        "3. Compute them in order, and make sure your last line reports that same "
        "quantity -- not an intermediate one."),
    FailureType.UNCLASSIFIED: "",
}


def _instruction_table():
    """Selected at call time so config can be changed without reimporting."""
    import config as _cfg
    style = getattr(_cfg, "TYPED_INSTRUCTION_STYLE", "short")
    return (TYPED_INSTRUCTION_PROCEDURAL if style == "procedural"
            else TYPED_INSTRUCTION_SHORT)


class _InstructionView(dict):
    """dict-like, but resolves through config on every read, so existing callers
    that do TYPED_INSTRUCTION.get(ftype, "") pick up the configured style."""
    def get(self, key, default=""):
        return _instruction_table().get(key, default)

    def __getitem__(self, key):
        return _instruction_table()[key]

    def __iter__(self):
        return iter(_instruction_table())

    def items(self):
        return _instruction_table().items()

    def keys(self):
        return _instruction_table().keys()

    def values(self):
        return _instruction_table().values()

    def __len__(self):
        return len(_instruction_table())


TYPED_INSTRUCTION = _InstructionView()

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


# ============================================================================
# [AUDIT D41] APPROACH DEMONSTRATIONS
# ============================================================================
# A 7B chat model follows FORMAT far more reliably than it follows INSTRUCTIONS.
# The typed instruction sits in the system message, outside the block the model's
# next-token objective actually imitates, so "Verify each computation." can be
# read and ignored at no cost. A DEMONSTRATION of the same procedure sits inside
# the exemplar block, where imitation pressure is highest.
#
# Each demo below is a deliberately trivial problem solved in exactly the shape
# the cure asks for. They are hand-written, not drawn from GSM8K, so no gold is
# involved and nothing is leaked.
#
# BUDGET PARITY: when enabled, typed shows (budget - 1) retrieved exemplars plus
# ONE demo, so both arms still show exactly `budget` exemplars. Without that the
# comparison would be 9-vs-8 and the extra slot alone could explain a win.
TYPED_APPROACH = {
    FailureType.NR: {
        "question": "A box holds 6 pens. Ana buys 4 boxes. How many pens does she have?",
        "trace": ("We need to find: the total number of pens.\n"
                  "Given: 6 pens in a box, 4 boxes.\n"
                  "Total pens: 6 * 4 = 24\n"
                  "#### 24"),
    },
    FailureType.AL: {
        "question": "Sam has 3 times as many marbles as Ken. Ken has 7 marbles. How many does Sam have?",
        "trace": ("Ken has 7 marbles.\n"
                  "Sam has 3 times that many: 3 * 7 = 21\n"
                  "Sam has 21 marbles.\n"
                  "#### 21"),
    },
    FailureType.ST: {
        "question": "A shirt costs 15 dollars. Mia buys 3 shirts and pays with 50 dollars. How much change does she get?",
        "trace": ("One shirt costs 15 dollars.\n"
                  "Three shirts cost: 15 * 3 = 45\n"
                  "She pays 50 dollars.\n"
                  "Change: 50 - 45 = 5\n"
                  "#### 5"),
    },
    FailureType.SM: {
        "question": "A recipe needs 2 cups of flour and 3 cups of sugar. Leo triples the recipe. How many cups of flour does he need?",
        "trace": ("The question gives: 2 cups of flour, 3 cups of sugar, recipe tripled.\n"
                  "It asks only about the flour, so the sugar is not used.\n"
                  "Flour needed: 2 * 3 = 6\n"
                  "#### 6"),
    },
    FailureType.CE: {
        "question": "A ribbon 7 metres long is cut into 4 equal pieces. How long is each piece?",
        "trace": ("Each piece: 7 / 4 = 1.75\n"
                  "Check that this is right: 1.75 * 4 = 7\n"
                  "Each piece is 1.75 metres.\n"
                  "#### 1.75"),
    },
    FailureType.WP: {
        "question": "Tom reads 20 pages a day for 5 days. The book has 150 pages. How many pages are left?",
        "trace": ("We need to find: the pages LEFT, not the pages read.\n"
                  "Pages read: 20 * 5 = 100\n"
                  "Pages left: 150 - 100 = 50\n"
                  "The last line reports pages left, which is what was asked.\n"
                  "#### 50"),
    },
}


def approach_demo(ftype):
    """An exemplar-shaped record demonstrating the cure, or None."""
    d = TYPED_APPROACH.get(ftype)
    if not d:
        return None
    return {"question": d["question"], "trace": d["trace"],
            "id": f"demo_{getattr(ftype, 'value', ftype)}",
            "is_approach_demo": True}


# ============================================================================
# [AUDIT D43] RESPONSE PREFILL -- the strongest typed-only lever
# ============================================================================
# An instruction can be ignored. A demonstration can be ignored. A PREFILL cannot:
# the prompt ends mid-sentence and the model has no option but to continue it.
#
# "Decide what quantity the question asks for" is advice. Ending the prompt with
# "Solution: We need to find:" FORCES the model to name the target quantity as its
# very first act, before any arithmetic exists to anchor on. For a 7B that is the
# difference between a suggestion and a constraint.
#
# This is typed-exclusive by construction -- generic ends at "Solution:" with
# nothing after it -- and it costs no exemplar slot, no extra tokens and no extra
# generation. The prefill is prepended back onto the returned text before scoring,
# so the stored trace is exactly what the model would have produced unaided.
TYPED_PREFILL = {
    FailureType.NR: "The question asks for",
    FailureType.AL: "Working with numbers only, not letters.\n",
    FailureType.ST: "Step 1:",
    FailureType.SM: "The numbers the question gives are",
    # [AUDIT D45] CE's prefill used to be
    #   "I will do one operation per line and check each result.\nStep 1:"
    # which imposed a whole new structure on a trace whose PLAN was already
    # correct -- CE means right plan, wrong arithmetic. The re-run showed CE
    # traces losing a third of their equations (2.7 -> 1.8) and recovery
    # collapsing. A CE cure must change how carefully the model computes, not how
    # it lays the solution out, so there is no prefill for CE any more.
    FailureType.CE: "",
    FailureType.WP: "We need to find:",
}


def question_numbers_phrase(question: str, limit: int = 8) -> str:
    """'48, 24 and 3' -- the numbers the question actually states."""
    import re
    seen, out = set(), []
    # The trailing guard is (?!\d), not (?![\w.]). The stricter version dropped
    # every number at the end of a sentence -- "She has 50." lost the 50 -- which
    # is exactly the kind of given an SM failure then invents a replacement for.
    for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?!\d)",
                         (question or "").replace(",", "")):
        v = m.group()
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= limit:
            break
    if not out:
        return ""
    return out[0] if len(out) == 1 else ", ".join(out[:-1]) + " and " + out[-1]


def prefill_for(ftype, question=None):
    """Text appended to the prompt after 'Solution:', or '' when there is none.

    [AUDIT D51] SM's prefill is COMPLETED for the model, not left open.
    SM means the trace used a value the question never gave, and 70% of SM traces
    do exactly that (177 of 253, measured). Asking the model to list the givens
    leaves it free to hallucinate one at the very first step. Extracting them with
    a regex and writing them into the prompt removes that opportunity entirely --
    the model starts from a correct, complete list it did not have to produce.
    """
    import config as _cfg
    if not getattr(_cfg, "TYPED_PREFILL_ENABLED", False):
        return ""
    base = TYPED_PREFILL.get(ftype, "")
    if (ftype is FailureType.SM and question
            and getattr(_cfg, "SM_INJECT_QUESTION_NUMBERS", False)):
        nums = question_numbers_phrase(question)
        if nums:
            return f"{base} {nums}. Use only these values and what follows from them.\n"
    return base


# ============================================================================
# [AUDIT D44] TYPE-CONDITIONED ACCEPTANCE TESTS  -- the diagnosis defines both
#                                                   the cure AND the criterion
# ============================================================================
# Every cure so far is fire-and-forget: build a prompt, take whatever comes back,
# score it against gold. The diagnosis picks the treatment and is then discarded.
#
# But each failure type has a STRUCTURAL SIGNATURE the cure is supposed to remove,
# and every one of those signatures is checkable GOLD-FREE with machinery FADE
# already has. So the diagnosis can also say whether the treatment TOOK:
#
#     CE  the new trace states no arithmetically false equation
#     AL  no relation is left unresolved
#     NR  at least two lines actually compute something
#     ST  more computational steps than the trace that failed
#     SM  every number used comes from the question or from an earlier line
#     WP  the final answer equals the last equation's result (it reports what it computed)
#     TR  the trace is no longer truncated
#
# If the test fails, the cure demonstrably did not work and it is worth one
# resample. Generic has no diagnosis, so it has no test -- it must accept whatever
# it gets. That asymmetry is not a trick: it IS what having a diagnosis buys, and
# no published training-free self-correction method uses a symbolic, gold-free,
# per-error-type acceptance criterion this way.
#
# FAIRNESS: this gives typed more generations than generic on the problems that
# fail the test. config.GENERIC_MATCH_RESAMPLE makes generic resample at the same
# RATE (chosen at random, keeping the second sample) so compute is matched and the
# only difference is WHICH candidate gets kept. Leave it True for the paper.
def acceptance_test(ftype, new_trace, prev_trace=None):
    """Did the cure take? Gold-free. Returns (passed: bool, reason: str).

    [AUDIT D45] EVERY test carries an anti-degeneracy guard.

    The first version was gameable. "States no arithmetically false equation" is
    satisfied most cheaply by stating NO equations, and "no unresolved relation"
    by writing nothing at all. So when a greedy CE retry failed its test, the
    temperature-0.7 resample that computed LESS passed it and was kept. Measured
    on the partial re-run: CE recovery fell 26.6% -> 9.6% and equations per CE
    trace fell 2.7 -> 1.8. The test was selecting traces that AVOID arithmetic
    rather than traces that get it right.

    Fix: a candidate must clear the type's criterion AND not compute less than the
    trace it is replacing. A test that can be passed by doing less is not a test.
    """
    from extraction import (extract_equations, extract_final_answer, is_truncated,
                            trace_body, unresolved_relation_count)
    from components import computational_steps

    body = trace_body(new_trace)
    eqs = extract_equations(body)
    ans = extract_final_answer(new_trace)

    if ftype is FailureType.TR:
        return (not is_truncated(new_trace)), "still truncated"

    # [AUDIT D45] anti-degeneracy floor, shared by the tests that could be
    # satisfied by computing less.
    n_new = computational_steps(new_trace)
    n_prev = computational_steps(prev_trace) if prev_trace else 0
    degenerate = (n_prev > 0 and n_new < n_prev) or len(eqs) == 0

    if ftype is FailureType.CE:
        bad = [e for e in eqs if not e.is_true]
        if degenerate:
            return False, (f"degenerate: {len(eqs)} equations, {n_new} computed "
                           f"lines vs {n_prev} before")
        return (not bad), (f"still states a false equation: {bad[0].raw}" if bad else "")

    if ftype is FailureType.AL:
        n = unresolved_relation_count(new_trace)
        if degenerate:
            return False, f"degenerate: {n_new} computed lines vs {n_prev} before"
        return (n == 0), (f"{n} relations still unresolved" if n else "")

    if ftype is FailureType.NR:
        n = computational_steps(new_trace)
        return (n >= 2), (f"only {n} computational lines" if n < 2 else "")

    if ftype is FailureType.ST:
        n = computational_steps(new_trace)
        prev = computational_steps(prev_trace) if prev_trace else 0
        return (n > prev), (f"{n} computational lines vs {prev} before" if n <= prev else "")

    if ftype is FailureType.SM:
        # every operand must come from the question or from an earlier line
        import re
        qn = {float(m.group()) for m in
              re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", body)}  # placeholder
        known = set()
        for e in eqs:
            for x in e.operands:
                if isinstance(x, (int, float)) and not any(
                        abs(float(x) - k) < 1e-9 for k in known | qn):
                    pass          # cannot decide without the question; see below
            known.add(float(e.c))
        return True, ""           # handled by acceptance_test_sm(), which has the question

    if ftype is FailureType.WP:
        if degenerate:
            return False, f"degenerate: {n_new} computed lines vs {n_prev} before"
        if ans is None or not eqs:
            return False, "no answer or no computation to check against"
        ok = any(abs(float(e.c) - float(ans)) <= 1e-6 * max(1.0, abs(float(ans)))
                 for e in eqs)
        return ok, ("" if ok else "the reported answer is not any computed value")

    return True, ""


def acceptance_test_sm(question, new_trace):
    """SM needs the question: every number used must be given or derived."""
    import re
    from extraction import extract_equations, trace_body
    qn = {float(m.group()) for m in
          re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", question.replace(",", ""))}
    known = set(qn) | {0.0, 1.0, 2.0, 100.0}      # identities and percent base
    for e in extract_equations(trace_body(new_trace)):
        for x in e.operands:
            if isinstance(x, (int, float)):
                if not any(abs(float(x) - k) <= 1e-9 for k in known):
                    return False, f"uses {x:g}, which is not in the question"
        known.add(float(e.c))
    return True, ""


def cure_took(ftype, question, new_trace, prev_trace=None):
    """Single entry point. -> (passed, reason). A type with no usable test always
    passes, so nothing is resampled on noise."""
    import config as _cfg
    if not getattr(_cfg, "TYPED_ACCEPT_TEST", False):
        return True, ""
    allowed = getattr(_cfg, "ACCEPT_TEST_TYPES", ())
    if getattr(ftype, "value", ftype) not in allowed:
        return True, ""
    if ftype is FailureType.SM:
        return acceptance_test_sm(question, new_trace)
    return acceptance_test(ftype, new_trace, prev_trace)


# ============================================================================
# [AUDIT D52] ABSTENTION IS RIGHT FOR THE TAXONOMY, WRONG FOR THE CURE
# ============================================================================
# UNCLASSIFIED is a deliberate design feature: a taxonomy that always fires
# cannot be wrong, and the abstain class is what makes the per-type recovery
# rates interpretable. That reasoning applies to REPORTING.
#
# It does not apply to choosing a cure. On the train run, 88 failures were
# UNCLASSIFIED and every one of them received the generic cure with no
# instruction -- the worst-performing group in the study, at ~8-9% recovery. But
# 84% of them carry an obvious gold-free signature anyway:
#
#     looks CE (states a false equation)      66   75%
#     looks AL (leaves a relation unresolved)  8    9%
#     no signature                            14   16%
#
# So the cascade abstains, and the cure bank guesses. The RECORDED diagnosis
# stays UNCLASSIFIED -- the taxonomy's honesty is preserved and the paper still
# reports an abstain rate -- but the treatment is the matched one.
def cure_type_for(ftype, trace):
    """The type whose CURE should be applied. Equals `ftype` except for
    UNCLASSIFIED, which is routed on trace evidence alone. Returns (ftype, note)."""
    import config as _cfg
    if ftype is not FailureType.UNCLASSIFIED:
        return ftype, ""
    if not getattr(_cfg, "UNCLASSIFIED_FALLBACK_CURE", False):
        return ftype, ""
    from extraction import extract_equations, trace_body, is_truncated, \
        unresolved_relation_count
    from components import computational_steps
    eqs = extract_equations(trace_body(trace or ""))
    # Order matters. is_truncated is the LOOSEST of these tests -- it fires on a
    # trace that merely ends on a dangling word -- so checking it first swallowed
    # cases with far stronger evidence. A stated-and-false equation, or an
    # unresolved relation, is concrete; a dangling word is a heuristic. TR is
    # therefore checked LAST, and only when nothing definite was found.
    if any(not e.is_true for e in eqs):
        return FailureType.CE, "abstained; false equation -> CE cure"
    if unresolved_relation_count(trace or "") > 0:
        return FailureType.AL, "abstained; unresolved relation -> AL cure"
    if is_truncated(trace or ""):
        return FailureType.TR, "abstained; truncated -> TR cure"
    if computational_steps(trace or "") < 2:
        return FailureType.NR, "abstained; no computation -> NR cure"
    return FailureType.UNCLASSIFIED, "abstained; no signature -> generic cure"
