"""
ctc.py -- Counterfactual Trace Consistency. A gold-free, model-free, per-instance
correctness signal for a frozen model's reasoning trace.

THE QUESTION IT ANSWERS
-----------------------
FST asked "can we predict failure from the QUESTION, before attempting?" Measured
answer: no. The type predictor loses to a constant, and failure ranks at AUROC 0.664
while recoverability ranks at 0.433.

CTC asks a different question: "can we detect failure from the model's own
BEHAVIOUR, with no gold and no judge?" That is the question that matters at test
time, because it is what decides whether a problem enters the deferred-retry queue
when there is no reference answer to tell you it failed.

THE MECHANISM
-------------
A trace that encodes a correct procedure is a FUNCTION of the question's numbers.
So:

  1. extract the trace's equation chain                        (extraction.py)
  2. find a question number the chain actually consumes         find_targets()
  3. rewrite the question with that number changed              perturb()
  4. predict, SYMBOLICALLY, from the trace's own chain, what    predict_answer()
     the answer must become under that substitution             <- costs nothing
  5. re-ask the model the perturbed question                    <- 1 generation
  6. compare the model's new answer to its own chain's prediction

  AGREE    -> the model executed a stable, inspectable procedure
  DISAGREE -> the first trace was not a procedure; it landed on a number

Step 4 is the hinge. The prediction comes from the model's OWN trace, resolved by
sympy -- not from a second model, not from token uncertainty, not from a trained
verifier. That is what makes it free and deterministic, and it is only possible
because FADE already built a symbolic equation extractor.

WHY IT SHOULD CATCH WHAT THE ARITHMETIC CHECK CANNOT
----------------------------------------------------
Measured on the 1,500-problem train run:

    failures WITH a false equation   351  (32.9%)   arithmetic check sees these
    failures with NO false equation  717  (67.1%)   arithmetic check is BLIND

The 717 are WP (254), SM (172), ST (82), NR (73), AL (56) -- every step valid, the
plan wrong. A wrong plan is a DIFFERENT FUNCTION of the inputs than the right one,
so it moves differently under perturbation. The signal is orthogonal to arithmetic
validity by construction, which is exactly what is needed.

COVERAGE -- measured, not assumed
---------------------------------
    connected chain (question number -> ... -> final answer)  1052/1500 = 70.1%
    no extractable equation                                    144       = 9.6%
    no question number used as an operand                      147       = 9.8%
    chain does not reach the final answer                      157       = 10.5%

CTC abstains on the other 30%. Connectivity is itself weakly informative --
78.5% of correct traces are connected versus 66.8% of wrong ones -- so the abstain
rate is reported, never silently folded into a verdict.

GOLD RULE
---------
Everything here reads the QUESTION and the model's own TRACE. No gold is touched at
any point, so CTC is valid on the test split. `perturbed_gold()` exists only to
SCORE the pilot on train and must never be called at inference; it is kept in a
separate function for exactly that reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import sympy

from extraction import extract_equations, trace_body, _longest_arith_suffix, _normalise_math_text

_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
# thousands separators only -- '1,200' -> '1200'. A blanket .replace(",", "")
# also deleted ordinary punctuation ("in April, and then" -> "in April and
# then"), which changed the question the model was re-asked. The perturbed
# question must differ from the original in EXACTLY one number.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
TOL = 1e-6


def _close(x, y, tol=TOL):
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= tol * max(1.0, abs(y))


# ------------------------------------------------------------------ targets
def question_numbers(question: str):
    """(value, span) for every literal number in the question, excluding 0 and 1.

    0 and 1 are excluded because perturbing them usually changes the PROBLEM
    ('half' -> 'a third'), not just a quantity, and because they appear as
    identity elements inside expressions where a substitution is ambiguous.
    """
    q = _THOUSANDS.sub("", question)
    out = []
    for m in _NUM.finditer(q):
        v = float(m.group())
        if v not in (0.0, 1.0):
            out.append((v, m.span()))
    return out


def chain_operands(eqs):
    ops = set()
    for e in eqs:
        for x in e.operands:
            if isinstance(x, (int, float)):
                ops.add(float(x))
    return ops


def find_targets(question: str, eqs):
    """Question numbers the trace's chain actually consumes -- the load-bearing
    ones. Perturbing a number the chain never reads proves nothing: the answer
    SHOULD stay the same, and so both a good and a bad trace look consistent."""
    ops = chain_operands(eqs)
    return [(v, sp) for v, sp in question_numbers(question)
            if any(_close(v, o) for o in ops)]


def perturb(question: str, span, new_value):
    """Rewrite the question with one number replaced. Integers stay integers so
    the wording stays natural ('48 clips', not '48.0 clips')."""
    s = f"{int(new_value)}" if float(new_value).is_integer() else f"{new_value:g}"
    q = _THOUSANDS.sub("", question)
    return q[:span[0]] + s + q[span[1]:]


def choose_perturbation(value, factor=2.0, rng=None):
    """A new value that is clearly different, still plausible, and not degenerate.

    Doubling is the default because it keeps integrality, keeps sign, and cannot
    collide with the original. Values that would collide with a common constant
    are nudged instead.
    """
    nv = value * factor
    if float(value).is_integer():
        nv = float(int(round(nv)))
    if _close(nv, value) or nv in (0.0, 1.0):
        nv = value + 3.0
    return nv


# --------------------------------------------------------- symbolic predict
def _eval_lhs(expr_text, subs):
    """Evaluate an LHS expression with numeric substitutions applied."""
    txt = _normalise_math_text(expr_text)
    def rep(m):
        v = float(m.group())
        for old, new in subs:
            if _close(v, old):
                return f"{new:.12g}"
        return m.group()
    txt = _NUM.sub(rep, txt)
    try:
        return float(sympy.sympify(txt))
    except Exception:
        return None


def predict_answer(trace: str, old_value, new_value, original_answer):
    """What the answer MUST become if the trace's own chain is a real procedure.

    Walks the equations in order. Each equation whose value changes contributes a
    further substitution (old result -> new result), so the change propagates
    down the chain exactly as the model's own arithmetic dictates.

    Returns (predicted_answer or None, n_equations_changed).
    """
    eqs = extract_equations(trace_body(trace))
    if not eqs:
        return None, 0
    subs = [(float(old_value), float(new_value))]
    changed = 0
    for e in eqs:
        lhs = e.raw.split("=")[0] if "=" in e.raw else e.raw
        val = _eval_lhs(lhs, subs)
        if val is None:
            continue
        if not _close(val, e.c):
            subs.append((float(e.c), float(val)))
            changed += 1
    if original_answer is None:
        return None, changed
    for old, new in subs[1:]:
        if _close(original_answer, old):
            return new, changed
    # the answer is not downstream of the perturbation -> the chain says it
    # should NOT move
    return float(original_answer), changed


# ------------------------------------------------------------------ verdict
@dataclass
class CTCResult:
    applicable: bool = False
    reason: str = ""
    target_value: float | None = None
    new_value: float | None = None
    perturbed_question: str = ""
    predicted: float | None = None
    n_changed: int = 0
    model_answer: float | None = None
    agrees: bool | None = None

    def to_dict(self):
        return asdict(self)


def prepare(question: str, trace: str, final_answer, factor=2.0):
    """Everything up to (but not including) the extra generation. No GPU, no gold."""
    r = CTCResult()
    eqs = extract_equations(trace_body(trace))
    if not eqs:
        r.reason = "no extractable equation"; return r
    targets = find_targets(question, eqs)
    if not targets:
        r.reason = "no question number is used by the chain"; return r
    if final_answer is None:
        r.reason = "no final answer to track"; return r
    if not any(_close(e.c, final_answer) for e in eqs):
        r.reason = "chain does not reach the final answer"; return r

    # perturb the LAST load-bearing number: earliest ones are often scene-setting
    # ("3 friends"), and the last is more often the operand that actually drives
    # the computation.
    value, span = targets[-1]
    new_value = choose_perturbation(value, factor)
    pred, changed = predict_answer(trace, value, new_value, final_answer)
    if pred is None:
        r.reason = "chain could not be re-evaluated"; return r
    if _close(pred, final_answer):
        r.reason = "prediction is unchanged -- perturbation is not load-bearing"
        return r

    r.applicable = True
    r.target_value, r.new_value = value, new_value
    r.perturbed_question = perturb(question, span, new_value)
    r.predicted, r.n_changed = pred, changed
    return r


def judge(r: CTCResult, model_answer, tol=TOL):
    """Fill in the model's answer to the perturbed question and decide."""
    r.model_answer = model_answer
    r.agrees = (model_answer is not None and _close(model_answer, r.predicted, tol))
    return r


# ------------------------------------------------- SCORING ONLY -- train only
def perturbed_gold(gold_solution: str, old_value, new_value):
    """The TRUE answer to the perturbed question, from GSM8K's <<...>> annotations.

    FOR PILOT SCORING ON TRAIN ONLY. Never call this at inference -- it reads gold.
    Kept in its own function, below a loud banner, so a misuse is obvious in review.
    """
    calc = re.findall(r"<<([^>]+)>>", gold_solution or "")
    if not calc:
        return None
    subs = [(float(old_value), float(new_value))]
    last = None
    for c in calc:
        if "=" not in c:
            continue
        lhs, rhs = c.split("=", 1)
        val = _eval_lhs(lhs, subs)
        try:
            stated = float(rhs)
        except ValueError:
            continue
        if val is None:
            continue
        if not _close(val, stated):
            subs.append((stated, val))
        last = val
    return last
