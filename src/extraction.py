"""
ALL parsing lives here, and only here. Four sections:

  1. NUMERIC VALUES  -- representation-blind value extraction (the v3 fix):
       plain numbers (commas stripped, negatives ok), fractions a/b by their
       evaluated value (3/4 -> 0.75), percents in BOTH conventions
       (75% -> 75 and 0.75; GSM8K's own checkpoints are inconsistent).
  2. GOLD CHECKPOINTS -- <<expr=result>> annotations from gold solutions.
       Feed E, A, the misses count, and s_hat for the cascade.
  3. EQUATIONS        -- "A op B = C" statements from traces, verified by
       sympy SYMBOLIC equivalence (symbolic first, float last, so
       3/4 = 0.75 verifies exactly). Feed V, A, bad_eqs, coherence.
  4. FINAL ANSWER     -- the extraction ladder:
       '#### N' -> '\\boxed{N}' -> 'the answer is N' -> last number,
       and correctness vs gold (symbolic first). Feeds classification.

Downstream: components.py computes E/V/R/A/G/coherence from these;
classification.py uses is_correct; diagnosis reads the derived signals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

import sympy

from config import VALUE_MATCH_TOL

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


def extract_values(text: str) -> set[float]:
    """All numeric VALUES expressed in the text, representation-blind."""
    values: set[float] = set()
    for m in RE_PERCENT.finditer(text):
        v = to_float(m.group(1))
        if v is not None:
            values.add(v)           # 75% -> 75
            values.add(v / 100.0)   # 75% -> 0.75 (both conventions)
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


# ============================================================================
# 3. EQUATIONS
# ============================================================================

# A op B = C, op in + - * x (times) / (div); '$' tolerated on C.
RE_EQUATION = re.compile(
    rf"({NUM})\s*([+\-*/x×÷])\s*({NUM})\s*=\s*(\$?\s*{NUM})"
)
_OPS = {"+": "+", "-": "-", "*": "*", "x": "*", "×": "*", "/": "/", "÷": "/"}


@dataclass
class Equation:
    a: float
    op: str          # normalised to + - * /
    b: float
    c: float
    raw: str         # matched text, for inspection
    is_true: bool = False


def check_equation(a: float, op: str, b: float, c: float) -> bool:
    """Symbolic first; float closeness only as last resort."""
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


def extract_equations(text: str) -> list[Equation]:
    """All BINARY equation statements, verified, in order. Chained forms
    ('4*3+10=22') are out of scope by design -- phase 0 measures the miss."""
    eqs: list[Equation] = []
    for m in RE_EQUATION.finditer(text):
        a = to_float(m.group(1))
        op = _OPS.get(m.group(2))
        b = to_float(m.group(3))
        c = to_float(m.group(4).replace("$", ""))
        if None in (a, b, c) or op is None:
            continue
        eq = Equation(a=a, op=op, b=b, c=c, raw=m.group(0))
        eq.is_true = check_equation(a, op, b, c)
        eqs.append(eq)
    return eqs


# ============================================================================
# 4. FINAL ANSWER + CORRECTNESS
# ============================================================================

RE_HASH = re.compile(rf"####\s*(\$?\s*{NUM})")
RE_BOXED = re.compile(rf"\\boxed\{{\s*({NUM})\s*\}}")
RE_ANSWER_IS = re.compile(rf"the answer (?:is|must be)\s*:?\s*(\$?\s*{NUM})",
                          re.IGNORECASE)


def extract_final_answer(trace: str) -> Optional[float]:
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