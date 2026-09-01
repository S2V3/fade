"""
symbolic_repair.py -- [AUDIT D46] a NON-GENERATIVE cure.

THE IDEA
--------
Every cure in FADE so far is a better prompt. But prompting cannot fix arithmetic:
the re-run showed WP +6.0, ST +8.2 and TR +6.1 over generic -- all STRUCTURAL
failures a prompt can control -- while CE went -16.0, because "compute more
carefully" is not something a frozen 7B can act on.

So for the one failure type where re-prompting is hopeless, do not re-prompt. Fix
the arithmetic directly.

WHEN IT FIRES, AND WHY IT NEEDS NO GOLD
---------------------------------------
A trace is SELF-INCONSISTENT when the answer it reports is the stated right-hand
side of an equation that is arithmetically false:

    "48 + 24 = 72"        <- false, 48 + 24 = 72 is fine; a real one:
    "11 + 20 + 7 = 48"    <- false, the true value is 38
    "#### 48"             <- the reported answer IS that false value

The model has contradicted itself. No reference solution is needed to know the
answer is wrong, and no reference solution is needed to know what the model's own
arithmetic actually implies. Substituting the symbolically correct value is a
repair the trace itself licenses.

MEASURED ON THE PARTIAL RE-RUN (738 problems, zero GPU)
-------------------------------------------------------
    type   n     repaired   rate     new (the generative retry missed these)
    CE     94    20         21.3%    20
    SM    168     4          2.4%     4
    AL     65     3          4.6%     3
    UNCL   53     3          5.7%     3
    TR     33     1          3.0%     1
    WP/ST/NR       0          0.0%     0     <- arithmetic was never the problem

31 problems the generative cure missed, +4.2 points, at no GPU cost. And it fires
exactly where prompting fails: nothing on WP, ST or NR, everything on CE.

CE with repair: 9.6% -> 30.9%, against generic's 25.5%.

IT ALSO SAVES GPU
-----------------
Repair runs BEFORE the generative retry. When it succeeds, no generation happens
at all -- roughly 20 of every 94 CE retries become free.

IT IS NOT UNCONDITIONALLY SAFE -- MEASURED
------------------------------------------
An earlier draft of this file claimed the rule was "safe by construction", on the
reasoning that a trace contradicting its own arithmetic must be wrong. The
measurement says otherwise: on the 432 CORRECT pass-1 traces the rule fires 7
times and breaks all 7 (-1.6 points). The parser occasionally reads a false
equation out of a correct trace, and that is enough.

That cost is ZERO in the deferred-retry arm, because repair only ever runs on
traces that already failed. It becomes real at test time, where CTC flags some
correct traces too. Budget for it: apply repair only to CTC-flagged traces, and
subtract ~1.6 points times the false-positive rate.

TWO OTHER RULES WERE TESTED AND REJECTED
----------------------------------------
    rule                       fires  correct  precision   breaks correct traces
    answer == false eq RHS       180       42     23.3%      7  (-1.6 pts)
    transcription slip           143        7      4.9%     29  (-6.7 pts)
    answer matches nothing       141        5      3.5%     26  (-6.0 pts)

The last two overwrite an answer using a heuristic about which equation "should"
have been final. They gain +0.8 points on failures and lose six times that on
correct traces. Not shipped.

THE OBJECTION, STATED HONESTLY
------------------------------
A reviewer will say generic could apply this too. Two answers, and the paper
should give both:
  1. It is cure SELECTION -- FADE applies it because the diagnosis says the
     failure is computational. That is the same logic that sends WP to a planning
     cure. An untyped arm has nothing to select on.
  2. Report the ablation anyway: applied blindly to every failure it fixes 44 of
     1,068 (4.1%); routed by diagnosis it fixes 31 of the 738 the generative cure
     missed. Owning that comparison is stronger than hiding it.
"""
from __future__ import annotations

import re

from extraction import extract_equations, trace_body

TOL = 1e-6


def _close(a, b, tol=TOL):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def _fmt(v):
    v = float(v)
    return f"{int(v)}" if v.is_integer() else f"{v:g}"


def find_repair(trace: str, final_answer):
    """-> (wrong_value, correct_value, equation_text) or None. Gold-free."""
    if final_answer is None:
        return None
    for e in extract_equations(trace_body(trace)):
        if not e.is_true and _close(e.c, final_answer):
            return float(e.c), float(e.a), e.raw
    return None


def repair(trace: str, final_answer):
    """Rewrite the trace with the corrected value. -> (new_trace, new_answer, note)
    or None when nothing is repairable."""
    found = find_repair(trace, final_answer)
    if not found:
        return None
    wrong, right, eq = found
    w, r = _fmt(wrong), _fmt(right)

    body = trace_body(trace)
    # Replace the wrong value only where it stands as a whole number token, so
    # "48" inside "480" or inside an unrelated quantity is left alone.
    pat = re.compile(rf"(?<![\w.]){re.escape(w)}(?![\w.])")
    fixed_body, n = pat.subn(r, body)

    note = (f"\n[symbolic repair] '{eq}' is false; "
            f"{w} corrected to {r} ({n} occurrence(s)).")
    new_trace = fixed_body.rstrip() + note + f"\n#### {r}"
    return new_trace, right, note.strip()


def main():
    import argparse, json
    from collections import Counter
    ap = argparse.ArgumentParser(description="audit symbolic repair on a store")
    ap.add_argument("--results", required=True)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    p1 = [r for r in rows if r.get("phase") == "pass1"]
    wrong = [r for r in p1 if not (r.get("signals") or {}).get("correct")]

    fired = hit = 0
    by = Counter(); byhit = Counter()
    shown = 0
    for r in wrong:
        out = repair(r.get("trace", ""), (r.get("signals") or {}).get("final_answer"))
        if not out:
            continue
        fired += 1
        by[r.get("diagnosis") or "UNCL"] += 1
        ok = _close(out[1], r.get("gold_answer"))
        hit += ok
        byhit[r.get("diagnosis") or "UNCL"] += ok
        if ok and shown < a.show:
            shown += 1
            print(f"\n  Q: {r['question'][:78]}...")
            print(f"     gold {r['gold_answer']} | model said "
                  f"{(r.get('signals') or {}).get('final_answer')} -> repaired to {out[1]}")
            print(f"     {out[2]}")

    print("\n" + "=" * 66)
    print(f"  SYMBOLIC REPAIR on {len(wrong)} wrong traces")
    print("=" * 66)
    print(f"  fired (trace self-inconsistent) : {fired}  ({fired/max(len(wrong),1):.1%})")
    print(f"  corrected value == gold         : {hit}   ({hit/max(fired,1):.1%} of those)")
    print(f"  accuracy effect                 : +{hit/max(len(p1),1)*100:.1f} pts, 0 generations")
    print(f"\n  {'type':<14}{'fired':>7}{'correct':>9}{'precision':>11}")
    for t in sorted(by, key=lambda x: -by[x]):
        print(f"  {t:<14}{by[t]:>7}{byhit[t]:>9}{byhit[t]/max(by[t],1):>11.1%}")
    print("\n  Precision below 100% means the trace was self-inconsistent AND the")
    print("  corrected value still is not gold -- the plan was wrong too. Those")
    print("  cost nothing: the answer was already wrong before the repair.")


if __name__ == "__main__":
    main()
