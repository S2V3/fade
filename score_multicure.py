"""Score the 2x2: diversity source against selection policy.

    rows        cure-enumeration  vs  temperature resampling
    columns     majority vote     vs  symbolic selection

Reports coverage (does ANY candidate get it right) separately from capture
(does the policy find it), because they answer different questions and only
coverage is bounded by the model.

USAGE
    python score_multicure.py --candidates <dir>/candidates.jsonl --ctc <joblib>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import kaggle_run as K
import selector as SEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ctc", default=None)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    a = ap.parse_args()

    K.SPLIT_TAG, K.MODEL_ID = "test", a.model
    gold = {p["question"]: p for p in K._load_split("test", None)}
    recs = [json.loads(l) for l in open(a.candidates) if l.strip()]
    print(f"  {len(recs)} problems, {len(recs[0]['cures'])} cures + "
          f"{len(recs[0]['temps'])} resamples each")

    det = None
    if a.ctc:
        import importlib, joblib
        for m in ("ctc_train", "ctc_train2"):
            try:
                importlib.import_module(m)
            except Exception:
                pass
        det = joblib.load(a.ctc)

    def ok(q, t):
        g = gold[q]
        return bool(K.score_trace(q, t, g["answer"], g["gold_answer"])[0].correct)

    scored = [r for r in recs if r["question"] in gold]
    n = len(scored)
    base = sum(ok(r["question"], r["pass1_trace"]) for r in scored)
    print(f"\n  pass-1 on these {n}: {base}/{n} = {base/n:.1%}"
          "   (all were predicted wrong, so this is the detector's error rate)")

    res = {}
    for src in ("cures", "temps"):
        # candidate 0 is always pass-1, so a selector that abstains does nothing
        sets = [[(r["pass1_trace"], r["pass1_answer"])]
                + [(c["trace"], c["answer"]) for c in r[src]] for r in scored]
        truth = [[ok(r["question"], t) for t, _ in s] for r, s in zip(scored, sets)]
        cov = sum(1 for t in truth if any(t))
        res[(src, "coverage")] = cov
        for pol in ("majority", "symbolic"):
            hit = 0
            why = Counter()
            for r, s, t in zip(scored, sets, truth):
                i, reason = SEL.apply(pol, s, r["question"], det)
                hit += t[i]
                why[(reason, "right" if t[i] else "wrong")] += 1
            res[(src, pol)] = hit
            res[(src, pol, "why")] = why

    print("\n" + "=" * 66)
    print("  COVERAGE -- does ANY candidate get it right?")
    print("=" * 66)
    for src in ("cures", "temps"):
        c = res[(src, "coverage")]
        print(f"  {src:<8}{c:>5}/{n} = {c/n:>6.1%}")
    print(f"  difference {(res[('cures','coverage')]-res[('temps','coverage')])/n:+.1%}"
          "   <- is cure-diversity better than temperature?")

    print("\n" + "=" * 66)
    print("  THE 2x2 -- accuracy after selection")
    print("=" * 66)
    print(f"  {'':<10}{'majority':>12}{'symbolic':>12}{'coverage':>12}")
    for src in ("cures", "temps"):
        m, s, c = res[(src, "majority")], res[(src, "symbolic")], res[(src, "coverage")]
        print(f"  {src:<10}{m/n:>11.1%}{s/n:>12.1%}{c/n:>12.1%}")
    print(f"\n  capture rate (selection / coverage)")
    for src in ("cures", "temps"):
        c = max(res[(src, 'coverage')], 1)
        print(f"    {src:<8}majority {res[(src,'majority')]/c:>6.1%}"
              f"   symbolic {res[(src,'symbolic')]/c:>6.1%}")

    print("\n  which tier decided, and was it right (cures + symbolic)")
    for (reason, verdict), k in sorted(res[("cures", "symbolic", "why")].items()):
        print(f"    {reason:<10}{verdict:<6}{k:>5}")

    try:
        from stats import mcnemar
        ids = [r["id"] for r in scored]
        A = {i: bool(ok(r["question"], r["pass1_trace"]))
             for i, r in zip(ids, scored)}
        for src in ("cures", "temps"):
            for pol in ("majority", "symbolic"):
                B = {}
                for i, r in zip(ids, scored):
                    s = [(r["pass1_trace"], r["pass1_answer"])] + \
                        [(c["trace"], c["answer"]) for c in r[src]]
                    j, _ = SEL.apply(pol, s, r["question"], det)
                    B[i] = ok(r["question"], s[j][0])
                mc = mcnemar(A, B)
                print(f"  {src:<7}{pol:<10} McNemar p = {mc['p']:.2e}"
                      + ("  SIGNIFICANT" if mc["p"] < 0.05 else ""))
    except Exception as e:
        print("  McNemar:", e)


if __name__ == "__main__":
    main()
