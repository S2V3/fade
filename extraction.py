"""
ALL parsing lives here, and only here. Four sections:

  1. NUMERIC VALUES  -- representation-blind value extraction:
       plain numbers (commas stripped, negatives ok), fractions a/b by their
       evaluated value (3/4 -> 0.75), percents.
       [AUDIT D5] Percents no longer emit BOTH conventions everywhere -- see
       `dual_percent` below.
  2. GOLD CHECKPOINTS -- <<expr=result>> annotations from gold solutions.
       Feed E, A, the misses count, and s_hat for the cascade.
  3. EQUATIONS        -- "<arithmetic> = C" statements from traces, verified by
       sympy. [AUDIT D4] REWRITTEN: anchored on the '=' and scanned BACKWARD,
       not forward from the first digit in the sentence. Feed V, A, bad_eqs,
       coherence, and the structural E/A operand check.
  4. FINAL ANSWER     -- the extraction ladder:
       '#### N' -> '\\boxed{N}' -> 'the answer is N' -> last number.
       [AUDIT D9] This is now the ONLY answer ladder in the codebase;
       generation.py imports it instead of carrying a weaker duplicate.

Downstream: components.py computes E/V/R/A/G/coherence from these;
classification.py uses is_correct; diagnosis reads the derived signals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

import sympy

from config import VALUE_MATCH_TOL, PERCENT_DUAL_EVERYWHERE

# ============================================================================
# 1. NUMERIC VALUES
# ============================================================================

# One number token: with thousands commas, or plain int/decimal; optional minus.
NUM = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?"

RE_NUMBER = re.compile(NUM)
RE_PERCENT = re.compile(rf"({NUM})\s*%")
# Fraction: digits/digits, not embedded in a longer number.
RE_FRACTION = re.compile(r"(?<![\d.,])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])")


def to_float(token: str) -> Optional[float]:
    """'1,234.5' -> 1234.5 ; strip currency symbols before calling."""
    try:
        return float(token.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def rational(v: float | str) -> sympy.Rational:
    """Exact rational -- symbolic equivalence before float coercion."""
    return sympy.Rational(str(v))


def extract_values(text: str, dual_percent: Optional[bool] = None) -> set[float]:
    """All numeric VALUES expressed in the text, representation-blind.

    [AUDIT D5] `dual_percent` controls whether "20%" contributes BOTH 20 and 0.20
    to the pool. GSM8K's own <<...>> checkpoints are inconsistent about the
    convention, so dual emission is correct when MATCHING GOLD CHECKPOINTS -- but
    when scoring a TRACE it silently doubles the surface area for a coincidental
    checkpoint hit (inflating E) and a free grounding hit (inflating G). 13.8% of
    eval questions contain a percent token.

    Default follows config.PERCENT_DUAL_EVERYWHERE (False = literal value only).
    Callers that genuinely need both conventions pass dual_percent=True.
    """
    if dual_percent is None:
        dual_percent = PERCENT_DUAL_EVERYWHERE
    values: set[float] = set()
    for m in RE_PERCENT.finditer(text):
        v = to_float(m.group(1))
        if v is not None:
            values.add(v)                       # 75% -> 75
            if dual_percent:
                values.add(v / 100.0)           # 75% -> 0.75 as well
    for m in RE_FRACTION.finditer(text):
        a, b = to_float(m.group(1)), to_float(m.group(2))
        if a is not None and b not in (None, 0.0):
            values.add(a / b)       # 3/4 -> 0.75 (a, b caught below anyway)
    for m in RE_NUMBER.finditer(text):
        v = to_float(m.group(0))
        if v is not None:
            values.add(v)
    return values


def value_in(value: float, pool: set[float],
             tol: float = VALUE_MATCH_TOL) -> bool:
    return any(math.isclose(v, value, rel_tol=tol, abs_tol=tol) for v in pool)


# ============================================================================
# 2. GOLD CHECKPOINTS
# ============================================================================

RE_CHECKPOINT = re.compile(r"<<([^<>=]+)=([^<>]+)>>")


def extract_gold_checkpoints(gold_solution: str) -> list[float]:
    """One entry per <<...=result>>, in order. Duplicates KEPT: each
    checkpoint is a distinct step, even if two steps share a value."""
    out: list[float] = []
    for m in RE_CHECKPOINT.finditer(gold_solution):
        v = to_float(m.group(2))
        if v is not None:
            out.append(v)
    return out


def expected_step_count(gold_solution: str) -> int:
    """s_hat -- expected step count for the cascade (and FST bands later)."""
    return len(extract_gold_checkpoints(gold_solution))


# ---- structural view: operands + operators of an expression --------------
_OP_NORMALISE = {"x": "*", "X": "*"}
_OP_CHARS = set("+-*/xX")


def _expr_terms(expr: str) -> tuple:
    """(sorted operand multiset, operator set) from an expression string.
    Both gold checkpoint and trace equation are parsed the SAME way, so they
    stay comparable."""
    operands = tuple(sorted(
        v for v in (to_float(t) for t in RE_NUMBER.findall(expr)) if v is not None))
    ops = frozenset(_OP_NORMALISE.get(ch, ch) for ch in expr if ch in _OP_CHARS)
    return operands, ops


@dataclass
class GoldCheckpoint:
    result: float
    operands: tuple    # sorted multiset of input numbers in the <<expr>>
    ops: frozenset     # operator symbols in the <<expr>>
    raw: str


def extract_gold_checkpoint_terms(gold_solution: str) -> list:
    """Structural companion to extract_gold_checkpoints: keeps each
    checkpoint's RESULT plus the OPERANDS + OPERATOR(S) gold used to produce it."""
    out = []
    for m in RE_CHECKPOINT.finditer(gold_solution):
        r = to_float(m.group(2))
        if r is None:
            continue
        operands, ops = _expr_terms(m.group(1))
        out.append(GoldCheckpoint(result=r, operands=operands, ops=ops,
                                  raw=m.group(0)))
    return out


# ============================================================================
# 3. EQUATIONS   [AUDIT D4 -- REWRITTEN]
# ============================================================================
#
# THE BUG THIS REPLACES
# ---------------------
# The old RE_EQ_GENERAL required the LHS to *start* with [0-9(] and allowed
# a-zA-Z inside it, and re.finditer takes the EARLIEST viable start. So the match
# began at the first digit anywhere in the sentence and swallowed prose up to the
# '='. _clean_expr then deleted the prose words and left two juxtaposed numbers:
#
#   "He sold 20 kg to the market, so he has 60 - 20 = 40 kg left."
#      old cleaned LHS -> "20               60 - 20"  -> sympify fails -> DISCARDED
#      the true equation  60 - 20 = 40  was right there.
#
# Measured on the 1,500-problem train run: 334/1500 traces (22.3%) parsed ZERO
# equations -- and 312 of those 334 CONTAIN an '=' sign. Zero equations forces
# V = 0 and coherent = None, which makes the trace automatically BAD.
#
# THE FIX
# -------
# Anchor on the '=', not on the first digit. For each '=' boundary in a line,
# take the LONGEST SUFFIX of the left-hand side that tokenises to a well-formed
# arithmetic expression, and REJECT any candidate containing two adjacent numeric
# tokens -- that single rule is what kills the prose-swallowing bug. Chained
# equalities (a = b = c) are handled by iterating every boundary.
#
# Measured effect: equations 2282 -> 2939 (1.29x), zero-equation traces
# 22.3% -> 17.6%, mean V 0.542 -> 0.600, GOOD pool 133 -> 150 (+12.8%).

_CURRENCY = "$£€₹¥"
_PER_UNIT = re.compile(r"/\s*[a-zA-Z]+")        # "$12/hour" -> "12" (a unit, not a division)
# Tolerate the space chat models insert after a thousands comma ("70, 000"):
# without it the comma survives as a junk token, the tokeniser splits the number,
# and a suffix can start mid-number -- producing artefacts like "000/10 = 10".
_RE_THOUSANDS = re.compile(r"(\d),\s*(?=\d{3}(?!\d))")


def _normalise_math_text(s: str) -> str:
    """Canonicalise a fragment for arithmetic parsing. Representation only --
    never changes a value."""
    s = (s.replace("×", "*").replace("÷", "/")
           .replace("−", "-").replace("–", "-").replace("—", "-"))
    for c in _CURRENCY:
        s = s.replace(c, "")
    prev = None
    while prev != s:                             # "1,234,567" -> "1234567"
        prev = s
        s = _RE_THOUSANDS.sub(r"\1", s)
    s = _PER_UNIT.sub(" ", s)                    # drop per-unit denominators
    s = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", s)   # "12 x 5" -> "12*5"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\b", r"(\1/100)*", s, flags=re.I)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    return s


_TOKEN = re.compile(rf"({NUM})|([()+\-*/])|([A-Za-z]+)|(\S)")


def _tokenise(s: str) -> list[tuple[str, str]]:
    """(kind, text). kind: 'n' number, 'o' operator/paren, 'w' word, 'x' junk.

    [AUDIT D23] BINARY MINUS vs NEGATIVE SIGN. `NUM` allows a leading '-', and the
    tokeniser scans left to right, so a subtraction written WITHOUT a space after
    the minus was read as a negative number:

        "60-20"  ->  [('n','60'), ('n','-20')]     two adjacent numbers
                 ->  _is_wellformed rejects it (that is the prose-swallowing guard)
                 ->  the equation was SILENTLY DROPPED

    Only '-' was affected; '+', '*' and '/' have no sign role, so "48+24=72" and
    "48/2=24" always worked. That asymmetry is what made it easy to miss.
    Measured on the 1,500-problem run: 14 '=' lines across 13 traces (0.9%) --
    small here, but it is silent data loss and the rate depends on how the model
    happens to space its arithmetic, which changes run to run.

    Fix: after tokenising, if a numeric token carries a leading '-' AND the token
    before it is a number or a closing paren, that '-' is a BINARY OPERATOR, not a
    sign. Split it. A '-' at the start of an expression, or after another operator,
    stays a sign ("-20", "3*-2").
    """
    raw: list[tuple[str, str]] = []
    for m in _TOKEN.finditer(s):
        if m.group(1):
            raw.append(("n", m.group(1)))
        elif m.group(2):
            raw.append(("o", m.group(2)))
        elif m.group(3):
            raw.append(("w", m.group(3)))
        else:
            raw.append(("x", m.group(4)))

    out: list[tuple[str, str]] = []
    for kind, text in raw:
        if (kind == "n" and text.startswith("-") and len(text) > 1
                and out and (out[-1][0] == "n" or out[-1][1] == ")")):
            out.append(("o", "-"))
            out.append(("n", text[1:]))
        else:
            out.append((kind, text))
    return out


def _is_wellformed(toks: list[tuple[str, str]]) -> bool:
    """A token run is a usable arithmetic expression iff it has no words/junk,
    at least one real operator, balanced parens, no two adjacent numbers (the
    prose-swallowing signature), and sympy can evaluate it."""
    if not toks:
        return False
    if any(k in ("w", "x") for k, _ in toks):
        return False
    if not any(k == "o" and v in "+-*/" for k, v in toks):
        return False
    for a, b in zip(toks, toks[1:]):
        if a[0] == "n" and b[0] == "n":
            return False                         # <- the bug, caught
    s = "".join(v for _, v in toks)
    if s.count("(") != s.count(")"):
        return False
    if toks[0][0] == "o" and toks[0][1] not in "(-":
        return False
    try:
        float(sympy.sympify(s))
    except Exception:
        return False
    return True


def _longest_arith_suffix(text: str) -> Optional[str]:
    """The longest suffix of `text` that is a well-formed arithmetic expression.

    Unit WORDS ('minutes', 'pages') are dropped, because chat models write
    "50 minutes / 60 = 0.83". But a word sitting BETWEEN two numbers is kept as a
    barrier: deleting it would juxtapose them and manufacture a bogus expression,
    which is precisely how the old parser mangled mid-sentence arithmetic.
    """
    toks = _tokenise(_normalise_math_text(text))
    cleaned: list[tuple[str, str]] = []
    for i, (k, v) in enumerate(toks):
        if k == "w":
            prev = cleaned[-1][0] if cleaned else None
            nxt = toks[i + 1][0] if i + 1 < len(toks) else None
            if prev == "n" and nxt == "n":
                cleaned.append(("x", "|"))       # barrier, never valid
            continue
        cleaned.append((k, v))
    for i in range(len(cleaned)):                # longest first
        # Never START on a numeric token with a redundant leading zero ("000",
        # "00"): that is the signature of a thousands separator normalisation
        # missed, so the suffix would begin in the MIDDLE of a real number.
        # (A prose comma is fine to start after -- do not filter on commas, they
        # are far more often ordinary punctuation than a broken separator.)
        tk, tv = cleaned[i]
        if tk == "n" and len(tv) > 1 and tv[0] == "0" and not tv.startswith("0."):
            continue
        if _is_wellformed(cleaned[i:]):
            return "".join(v for _, v in cleaned[i:])
    return None


@dataclass
class Equation:
    a: float         # evaluated LHS value
    op: str          # 'expr' (the parser is general; kept for record shape)
    b: float
    c: float         # stated RHS
    raw: str         # matched text, for inspection
    is_true: bool = False
    operands: tuple = ()          # structural E/A: input numbers in the LHS
    ops: frozenset = frozenset()  # structural E/A: operator symbols in the LHS


def check_equation(a: float, op: str, b: float, c: float) -> bool:
    """Symbolic first; float closeness only as last resort.
    Retained for backward compatibility with any caller doing binary checks."""
    try:
        A, B, C = rational(a), rational(b), rational(c)
        if op == "+":
            lhs = A + B
        elif op == "-":
            lhs = A - B
        elif op == "*":
            lhs = A * B
        elif op == "/":
            if B == 0:
                return False
            lhs = A / B
        else:
            return False
        return sympy.simplify(lhs - C) == 0
    except Exception:
        try:
            lhs = {"+": a + b, "-": a - b, "*": a * b,
                   "/": (a / b) if b else float("nan")}[op]
            return math.isclose(lhs, c, rel_tol=1e-4, abs_tol=1e-4)
        except Exception:
            return False


_RE_RHS_NUM = re.compile(rf"^\s*({NUM})")


def _written_decimals(token: str) -> int:
    """How many decimal places the model actually WROTE. '0.83' -> 2, '25' -> 0."""
    t = token.replace(",", "").strip()
    return len(t.split(".")[1]) if "." in t else 0


def _equation_holds(lhs_val: float, c: float, rhs_token: str) -> bool:
    """Is a stated equation true, allowing for HONEST ROUNDING?

    [AUDIT D25] The old test was a flat `math.isclose(rel_tol=1e-4)`, which scores
    `50/60 = 0.83` and `2/3 = 0.67` as FALSE. Those are not arithmetic errors --
    they are the model rounding a repeating decimal, which is ordinary behaviour
    (the prompt header even names it: "Keep exact fractions, e.g. 50/60 = 5/6, not
    0.83"). Marking them false inflates `bad_eqs`, depresses V, and pushes the
    trace toward a CE diagnosis whose cure is "verify each computation" -- when the
    computation was fine and only the presentation was rounded. That is exactly the
    confidently-wrong cure the taxonomy is supposed to avoid.

    The rule: an equation also holds if the LHS rounds to the RHS **at the
    precision the model itself wrote**. `50/60 = 0.83` -> round(0.8333, 2) == 0.83.

    Deliberately restricted to `dp >= 1`. Allowing it at dp == 0 would make
    `7/2 = 3` true, which is a real error, not a rounding. Integers must still
    match tightly.

    Measured: 25 of 746 false equations (3.4%) are roundings of this kind.
    """
    if math.isclose(lhs_val, c, rel_tol=1e-4, abs_tol=1e-4):
        return True
    dp = _written_decimals(rhs_token)
    if dp >= 1:
        try:
            return round(lhs_val, dp) == round(c, dp)
        except (ValueError, OverflowError):
            return False
    return False


def extract_equations(text: str) -> list[Equation]:
    """Every stated arithmetic equation, verified, in order.

    Handles binary ('48 + 24 = 72'), chained/parenthesised ('(100-(15+30)) = 55'),
    percent ('20% of 50 = 10'), unit-bearing ('50 minutes / 60 = 0.83') and
    CHAINED EQUALITIES ('20/8 = 2.5 minutes per page = 2.5'), and -- unlike the
    previous implementation -- arithmetic embedded mid-sentence.
    """
    eqs: list[Equation] = []
    for line in text.splitlines():
        if "=" not in line:
            continue
        parts = line.split("=")
        for i in range(len(parts) - 1):
            lhs = _longest_arith_suffix(parts[i])
            if not lhs:
                continue
            m = _RE_RHS_NUM.match(_normalise_math_text(parts[i + 1]))
            if not m:
                continue
            c = to_float(m.group(1))
            if c is None:
                continue
            try:
                lhs_val = float(sympy.sympify(lhs))
            except Exception:
                continue
            is_true = _equation_holds(lhs_val, c, m.group(1))
            operands, ops = _expr_terms(lhs)
            eqs.append(Equation(a=lhs_val, op="expr", b=0.0, c=c,
                                raw=f"{lhs} = {m.group(1)}", is_true=is_true,
                                operands=operands, ops=ops))
    return eqs


# ============================================================================
# 3b. SYMBOLIC / ALGEBRAIC SET-UP DETECTION   [AUDIT D12 -- new]
# ============================================================================
# 400/1500 traces (26.7%) introduce symbolic variables and never resolve them to
# a number. Accuracy when algebraic is 12.0% vs 21.3% when not -- a 9.3-point
# cost, and no existing failure type describes it. Canonical case:
#     "x + 3x/2 = 4x/2"  ->  "#### 4"   (gold was 5250)
#
# Gold-free by construction: this reads the TRACE only, which is also what makes
# it FST's most promising target class.

# 'x' between digits is a TIMES sign ("48 x 2"), not a variable. Kill those first.
_RE_TIMES = re.compile(r"\d\s*[xX]\s*[\d(]")
# a real variable: optional numeric coefficient then a bare single letter
_RE_VAR = re.compile(r"(?:(?<=^)|(?<=[\s(=+\-*/]))\d*\.?\d*([xXyYzZ])(?![A-Za-z])")


def symbolic_line_count(text: str) -> int:
    """How many '='-bearing lines carry a genuine algebraic variable.

    [AUDIT SELF-REVIEW] KEPT AS A LOGGED FEATURE, but no longer the AL trigger.
    It has real false positives: it fires on 'x' used as a TIMES sign after a
    WORD ("3 pages/week x 52 weeks/year", "Coconut trees x 5/2"), because
    _RE_TIMES only strips 'x' between DIGITS. See unresolved_relation_count for
    what AL actually fires on now.
    """
    n = 0
    for line in text.splitlines():
        if "=" not in line:
            continue
        if _RE_VAR.search(_RE_TIMES.sub(" ", line)):
            n += 1
    return n


def unresolved_relation_count(text: str) -> int:
    """'='-bearing lines that yield NO verifiable arithmetic -- the model stated a
    RELATION instead of COMPUTING a number.

    [AUDIT SELF-REVIEW] This is the AL trigger, replacing the variable regex.
    Three reasons it is the better definition:

      1. PRECISE. It has no notion of 'x' to get wrong; it simply asks whether the
         line resolved to a checkable computation. The variable regex fired on
         'Coconut trees x 5/2' (a times sign) and missed named-quantity algebra
         entirely.
      2. UNIFIED. It covers BOTH failure shapes in one class -- symbolic variables
         ("x + 3x/2 = 4x/2") and named-quantity relations ("Total revenue =
         Number of DVDs sold/day x Price of each DVD"). Measured, they behave the
         same way: both are the model deferring execution.
      3. NOT NR. Of the 585 traces this flags at >= 2, **384 also state at least
         one real, verifiable equation** -- so this is not 'did no work', it is
         'wrote relations it never resolved'. NR would not catch them.

    Measured on the 1,500-problem run at the >= 2 threshold: 585 traces (39.0%),
    accuracy 14.2% vs 21.7% for the rest -- a 7.6-point penalty. (The old variable
    regex isolated a 9.3-point penalty on 400 traces, but by conflating two
    unrelated surface patterns, which makes the CURE incoherent. A cure has to
    name something real: 'compute a number on every line' does, 'stop using the
    letter x' does not.)
    """
    n = 0
    for line in text.splitlines():
        if "=" not in line:
            continue
        if not extract_equations(line):
            n += 1
    return n


# ============================================================================
# 3c. TERMINATION / TRUNCATION DETECTION   [AUDIT D13 -- new]
# ============================================================================
# 154/1500 traces end mid-thought. These are GENERATION failures wearing a
# reasoning-failure costume; diagnosing them as NR/SM/CE poisons FST's training
# labels with noise that has nothing to do with the question.

_RE_APPENDED_HASH = re.compile(r"\n#{3,}\s*-?[\d,]+(?:\.\d+)?\s*$")


def trace_body(trace: str) -> str:
    """The trace with a synthesised trailing '#### N' marker removed, so
    truncation is judged on what the MODEL actually wrote."""
    m = _RE_APPENDED_HASH.search(trace)
    return trace[:m.start()] if m else trace


# A trailing FUNCTION word is the real signature of a cut-off generation
# ("...Molecular weight of", "...Pieces left"). A trailing CONTENT word usually is
# not ("...$4.60 per person", "...bought 38 stamps") -- that is just a complete
# sentence missing its full stop.
_DANGLING_WORDS = {
    "of", "the", "a", "an", "and", "or", "to", "by", "for", "from", "with", "in",
    "on", "at", "as", "is", "are", "was", "were", "be", "been", "that", "which",
    "so", "then", "we", "he", "she", "it", "they", "will", "would", "can",
    "could", "should", "each", "per", "this", "these", "those", "his", "her",
    "their", "its", "if", "since", "because", "but", "when", "while", "after",
    "before", "than", "into", "over", "under", "about", "plus", "minus", "times",
    "divided", "equals", "total", "number", "amount", "cost", "price", "left",
    "remaining", "many", "much", "how", "what", "let", "lets", "now", "next",
    "first", "second", "third", "step", "steps", "therefore", "thus", "hence",
}
_RE_LAST_WORD = re.compile(r"([A-Za-z]+)\W*$")


def is_truncated(trace: str) -> bool:
    """True when the model's own output stops mid-thought.

    [AUDIT SELF-REVIEW] The first version of this test was simply
    `b[-1] not in ".!?)0123456789"`, i.e. "does it end in punctuation or a digit".
    That was far too loose: of the 147 traces it flagged, 46 ended in 's' -- lines
    like "...each friend will pay: $4.60 per person" and "...bought 38 stamps",
    which are COMPLETE sentences merely missing a full stop. 11 of them were even
    scored CORRECT, and a correct answer is by definition not a truncation.

    The precise signatures of a real cut are: a partial '#' marker, an expression
    that stops on an operator or '=', a trailing ':' or ',', or a dangling
    FUNCTION word. Note the cascade additionally requires `not correct` before
    firing TR (diagnosis.py), so a correct trace can never be labelled truncated.
    """
    b = trace_body(trace).rstrip()
    if not b:
        return True
    if re.search(r"#{1,3}\s*$", b):          # started '####' and got cut
        return True
    if b[-1] in "+-*/=(,:":                  # stops mid-expression or mid-clause
        return True
    if b[-1] in ".!?)0123456789":            # a clean ending
        return False
    m = _RE_LAST_WORD.search(b)
    return bool(m) and m.group(1).lower() in _DANGLING_WORDS


# ============================================================================
# 4. FINAL ANSWER + CORRECTNESS
# ============================================================================

RE_HASH = re.compile(rf"#{{3,}}\s*(\$?\s*{NUM})")   # models sometimes write "### N"
RE_BOXED = re.compile(rf"\\boxed\{{\s*({NUM})\s*\}}")
RE_ANSWER_IS = re.compile(
    rf"(?:the\s+)?answer\s*(?:is|must be|=|:)\s*:?\s*(\$?\s*{NUM})", re.IGNORECASE)


def _last_value_in_line(line: str) -> Optional[float]:
    """The last VALUE stated on a line, treating `a/b` as one fraction.

    [AUDIT D26] The old fallback was `RE_NUMBER.findall(line)[-1]`, which reads a
    bare fractional answer as its DENOMINATOR:

        "So he gets 2/3 of a pizza"   ->  3.0     (should be 0.667)
        "#### 2/3"                    ->  2.0     (should be 0.667)

    It must NOT over-correct, though: in "96 / 2 = 48" the concluding value is 48,
    and the `96 / 2` earlier in the line is working, not the answer. So the rule is
    positional -- take whichever match STARTS last, and prefer the fraction only
    when the last plain number sits INSIDE it.

    Prevalence on GSM8K is ~0 (its answers are integers or decimals, never bare
    fractions), so this changes no number in the current run. It is fixed because
    it is a real defect that would bite on any dataset with fractional answers, and
    because the same ladder decides `is_correct` -- the accuracy figure itself.
    """
    nums = list(RE_NUMBER.finditer(line))
    if not nums:
        return None
    last_num = nums[-1]
    fracs = list(RE_FRACTION.finditer(line))
    if fracs:
        f = fracs[-1]
        if f.start() <= last_num.start() and last_num.end() <= f.end():
            a, b = to_float(f.group(1)), to_float(f.group(2))
            if a is not None and b not in (None, 0.0):
                return a / b
    return to_float(last_num.group(0))


def extract_final_answer(trace: str) -> Optional[float]:
    """THE answer ladder. [AUDIT D9] generation.py used to carry a second, weaker
    copy of this, and because _ensure_hash_line stamped ITS result as '#### N'
    (on 87.9% of traces) the weaker ladder silently overrode this one. There is
    now exactly one ladder and generation.py imports it."""
    m = RE_HASH.search(trace)
    if m:
        return to_float(m.group(1).replace("$", ""))
    m = RE_BOXED.search(trace)
    if m:
        return to_float(m.group(1))
    last = None
    for m in RE_ANSWER_IS.finditer(trace):
        last = m                     # take the LAST occurrence
    if last:
        return to_float(last.group(1).replace("$", ""))
    # No explicit marker. Prefer the last number on the FINAL non-empty line
    # (the concluding sentence): measured on real traces, preferring the last
    # *equation* result here LOSES answers, because the last equation is usually
    # mid-reasoning while the conclusion ("So Jack gets 400 grams") is last.
    for line in reversed([l for l in trace.splitlines() if l.strip()]):
        v = _last_value_in_line(line)
        if v is not None:
            return v
    nums = RE_NUMBER.findall(trace)
    return to_float(nums[-1]) if nums else None


def is_correct(trace: str, gold_answer: float) -> bool:
    pred = extract_final_answer(trace)
    if pred is None:
        return False
    try:
        return sympy.simplify(rational(pred) - rational(gold_answer)) == 0
    except Exception:
        return math.isclose(pred, gold_answer, rel_tol=1e-6, abs_tol=1e-6)