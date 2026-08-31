"""
negatives.py -- the counter-example bank: what going wrong on this KIND of
question actually looks like.

THE IDEA
--------
Every arm so far shows the model successes only. The pool is 432 correct traces;
the 1,068 wrong ones are used to diagnose and then discarded as prompt material.
But the diagnosis already tells us WHERE questions like this break, and a warning
carrying that is different information from another correct demonstration.

So the cure becomes two-sided:

    positives   k correct traces of the matching structural type   (what to do)
    negatives   a described failure mode, drawn from real wrong     (what breaks)
                traces of the PREDICTED type, on similar questions

THE RISK, AND THE DESIGN THAT CONTAINS IT
-----------------------------------------
Showing a model wrong reasoning can make it copy the wrong reasoning. Contrastive
prompting results in the literature are mixed for exactly this reason, and the
failure is silent -- you get a fluent trace with the demonstrated error in it.

Three rules keep that from happening here, and they are the whole design:

 1. NEGATIVES ARE NEVER DEMONSTRATIONS. Positives go in the exemplar block, where
    the model's next-token objective pushes it to imitate. Negatives go in the
    SYSTEM message, as prose about a mistake, in the same channel the typed
    instruction already uses. Nothing in the imitation path is ever wrong.
 2. NO WRONG ANSWER IS EVER RENDERED. `_render` cuts the trace at the '####' line
    and drops the final value. A wrong number in context is the one thing most
    likely to be copied verbatim.
 3. AT MOST 2, AT MOST ~40 WORDS EACH. A long wrong trace is a demonstration no
    matter which message it sits in.

LOCALISATION -- gold-free
-------------------------
For 351 of the 1,068 wrong traces (32.9%) extraction.extract_equations finds a
stated equation that is arithmetically FALSE, so the warning can quote the exact
step. Those concentrate in CE (119), SM (81), AL (38) and ST (32) -- the
computational types. For WP and NR, where nothing local is wrong (the plan was
wrong, or there was no reasoning at all), the warning falls back to the type's
signature. That asymmetry is real and worth reporting: the types we can localise
are not the types that dominate the error mass.

GOLD RULE: records are passed through classification.sanitize_exemplar on the way
in, so gold_solution/gold_answer cannot reach a prompt. Selecting WHICH traces are
wrong used gold on the TRAIN split, which is allowed -- the same as the positives.

USAGE
    from negatives import NegativeBank
    bank = NegativeBank.from_store("stores/store_typed_train_audit2")
    warning = bank.warning_for(question, "CE", k=2)   # -> str, possibly ""
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from classification import sanitize_exemplar

MAX_NEGATIVES = 2
MAX_WORDS = 40

# What the failure mode IS, for types with no localisable arithmetic error.
_SIGNATURE = {
    "WP": "set up a plan for a different quantity than the one asked for, and "
          "computed it correctly",
    "NR": "jumped straight to a number with no intermediate reasoning",
    "TR": "stopped mid-sentence before reaching an answer",
    "SM": "worked with a value that never appeared in the question",
    "ST": "skipped the intermediate quantities and answered from the given "
          "numbers directly",
    "AL": "left a relation stated but never resolved it to a number",
    "CE": "made an arithmetic slip inside an otherwise correct plan",
}


def _strip_answer(trace: str) -> str:
    """Cut at the answer marker and drop the final value. Rule 2."""
    body = re.split(r"####", trace)[0]
    body = re.sub(r"(?i)\b(the answer is|answer:)\s*[-+]?[\d,]*\.?\d+.*$", "",
                  body, flags=re.M)
    return body.strip()


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"[-+]?\d*\.?\d+", str(text).replace(",", "")))


def _same_number(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except Exception:
        return str(a) == str(b)


def _first_false_equation(trace: str, final_answer=None) -> str | None:
    """The first stated equation that is arithmetically false. Gold-free.

    Rule 2 is enforced HERE, not only in _strip_answer. A false equation very
    often ENDS in the wrong final answer -- "11+20+7 = 48" when the trace went on
    to answer 48 -- so cutting the '####' line is not enough. An audit of the
    first 60 traces per type found 9 such warnings quoting the wrong answer back
    at the model, which is precisely the number most likely to be copied. Any
    equation carrying the final answer is skipped in favour of an earlier-failing
    one, and if none is clean we fall back to the type signature.
    """
    try:
        from extraction import extract_equations
        for e in extract_equations(_strip_answer(trace)):
            if e.is_true:
                continue
            txt = getattr(e, "text", None) or getattr(e, "raw", None)
            if not txt:
                continue
            txt = " ".join(str(txt).split())[:80]
            if final_answer is not None and any(
                    _same_number(n, final_answer) for n in _numbers(txt)):
                continue                      # would show the wrong answer
            return txt
    except Exception:
        pass
    return None


class NegativeBank:
    """Wrong traces of the train split, grouped by diagnosed type."""

    def __init__(self, records):
        self.by_type = defaultdict(list)
        for r in records:
            self.by_type[r.get("diagnosis") or "UNCLASSIFIED"].append(r)
        self._sim = None

    # ------------------------------------------------------------- loading
    @classmethod
    def from_records(cls, records):
        clean = []
        for r in records:
            s = sanitize_exemplar(r)
            assert "gold_solution" not in s and "gold_answer" not in s, \
                "GOLD LEAK in the negative bank"
            if s.get("trace"):
                clean.append(s)
        return cls(clean)

    @classmethod
    def from_store(cls, store_dir):
        """Read the MEDIUM+BAD queues, or fall back to results.jsonl."""
        root = Path(store_dir)
        recs = []
        for name in ("medium_queue.jsonl", "bad_queue.jsonl"):
            p = root / name
            if p.exists():
                recs += [json.loads(l) for l in open(p) if l.strip()]
        if not recs:
            p = root / "results.jsonl"
            if p.exists():
                for l in open(p):
                    if not l.strip():
                        continue
                    r = json.loads(l)
                    if (r.get("phase") == "pass1"
                            and not (r.get("signals") or {}).get("correct")):
                        recs.append(r)
        # a MEDIUM_CORRECT record sits in the medium queue but is not a failure
        recs = [r for r in recs
                if not (r.get("signals") or {}).get("correct")]
        return cls.from_records(recs)

    # ----------------------------------------------------------- retrieval
    def _similar(self, question, cands, k):
        if not cands:
            return []
        try:
            from similarity import rank_by_similarity
            return rank_by_similarity(question, cands, k)
        except Exception:
            pass
        qw = set(re.findall(r"[a-z]+", question.lower()))
        def ov(r):
            w = set(re.findall(r"[a-z]+", (r.get("question") or "").lower()))
            return len(qw & w) / max(len(qw | w), 1)
        return sorted(cands, key=ov, reverse=True)[:k]

    # ------------------------------------------------------------ renderit
    def _render(self, rec):
        t = rec.get("diagnosis") or "UNCLASSIFIED"
        fa = (rec.get("signals") or {}).get("final_answer")
        eq = _first_false_equation(rec.get("trace", ""), fa)
        if eq:
            return f'wrote "{eq}", which is false, and carried the result forward'
        # No free-text fallback. Quoting raw trace body was worse than useless --
        # it surfaced the model's preamble ("Great, let's solve this problem!")
        # as if it were the mistake, and on 5 UNCLASSIFIED traces it dragged the
        # wrong final answer along with it. If there is neither a false equation
        # nor a type signature, there is nothing worth saying, and an abstain is
        # the right output. UNCLASSIFIED deliberately has no signature.
        return _SIGNATURE.get(t)

    def warning_for(self, question, ptype, k=MAX_NEGATIVES):
        """A system-message warning block, or '' when there is nothing to say."""
        if not ptype or k <= 0:
            return ""
        ptype = getattr(ptype, "value", ptype)
        cands = self.by_type.get(ptype) or []
        if not cands:
            return ""
        # Over-fetch: most types fall back to a single fixed signature string,
        # so two picks would render the identical sentence twice. Dedupe on the
        # rendered text and stop at k distinct ones.
        picked = self._similar(question, cands, min(k, MAX_NEGATIVES) * 4)
        lines, seen = [], set()
        for r in picked:
            d = self._render(r)
            if not d:
                continue
            d = " ".join(str(d).split()[:MAX_WORDS])
            # Last-mile guard. _first_false_equation already skips equations
            # carrying the wrong answer, but the free-text fallback quotes trace
            # body and can still surface it. Nothing containing this record's
            # final answer is ever emitted.
            fa = (r.get("signals") or {}).get("final_answer")
            if fa is not None and any(_same_number(n, fa) for n in _numbers(d)):
                continue
            if d in seen:
                continue
            seen.add(d)
            lines.append(f"- On a question like this, a previous attempt {d}.")
            if len(lines) >= min(k, MAX_NEGATIVES):
                break
        if not lines:
            return ""
        return ("Common failure mode on questions of this shape:\n"
                + "\n".join(lines)
                + "\nDo not reproduce that. Show each intermediate value and "
                  "check it before using it.")

    # ---------------------------------------------------------------- info
    def report(self):
        out = {}
        for t, rs in sorted(self.by_type.items(), key=lambda x: -len(x[1])):
            loc = sum(1 for r in rs if _first_false_equation(
                r.get("trace", ""), (r.get("signals") or {}).get("final_answer")))
            out[t] = {"n": len(rs), "localisable": loc,
                      "pct_localisable": round(loc / max(len(rs), 1), 3)}
        return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="inspect the negative bank")
    ap.add_argument("--store", required=True)
    ap.add_argument("--show", type=int, default=3)
    a = ap.parse_args()

    bank = NegativeBank.from_store(a.store)
    total = sum(len(v) for v in bank.by_type.values())
    print("=" * 70)
    print(f"  NEGATIVE BANK · {total} wrong traces from {a.store}")
    print("=" * 70)
    print(f"  {'type':<16}{'n':>6}{'localisable':>14}{'%':>8}")
    for t, d in bank.report().items():
        print(f"  {t:<16}{d['n']:>6}{d['localisable']:>14}{d['pct_localisable']:>8.1%}")
    print("\n  'localisable' = a stated equation was found to be arithmetically")
    print("  false, so the warning can quote the exact step. Otherwise it falls")
    print("  back to the type signature.\n")

    for t in ("CE", "WP", "SM", "AL"):
        if t not in bank.by_type:
            continue
        q = bank.by_type[t][0].get("question", "")
        w = bank.warning_for(q, t, k=2)
        print("-" * 70)
        print(f"  {t}  |  {q[:64]}...")
        print("  " + (w.replace("\n", "\n  ") if w else "(nothing to say)"))

    print("\n" + "=" * 70)
    print("  SAFETY CHECK -- rule 2: no rendered line may carry the wrong final")
    print("  answer OF THE TRACE IT CAME FROM")
    print("=" * 70)
    # The invariant is per-SOURCE-trace. An earlier version compared a warning
    # against the QUERYING record's answer and flagged 17 "leaks" -- all of them
    # another trace's quoted equation coincidentally containing the same number.
    # There is no causal path from a test question's answer into its own prompt,
    # so that check was measuring noise. This checks the real thing.
    bad = checked = 0
    offenders = []
    for t, rs in bank.by_type.items():
        for r in rs:
            d = bank._render(r)
            if not d:
                continue
            checked += 1
            fa = (r.get("signals") or {}).get("final_answer")
            if fa is not None and any(_same_number(n, fa) for n in _numbers(d)):
                bad += 1
                if len(offenders) < 5:
                    offenders.append((t, fa, str(d)[:70]))
    print(f"  source traces rendered: {checked}")
    print(f"  carrying their own wrong answer: {bad}   "
          f"{'OK' if bad == 0 else '<-- LEAK'}")
    for t, fa, d in offenders:
        print(f"    {t} answer={fa}: {d}")

    print("\n  Rule 1 (negatives never enter the exemplar block) is enforced by")
    print("  cure_bank.cure_for: the warning is returned in the INSTRUCTION slot,")
    print("  which build_retry_prompt routes to extra_system -- never into the")
    print("  demonstration block the model imitates.")


if __name__ == "__main__":
    main()
