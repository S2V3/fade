"""
recompute_stage2.py -- re-read a finished stage-2 store. NO GPU, seconds.

WHY THIS EXISTS
---------------
run_ctc_validate.py reports final accuracy as `correct OR retry_correct` -- the
UNION of both attempts. That is how the train harness has always scored, and on
train it is defensible: the retry is an extra attempt and the union is the
system's best output.

It is NOT defensible at test time, and stage 3 is a test-time claim. Picking the
better of two answers requires knowing which one is right, and that is gold. A
deployed FADE has flagged the first answer as wrong and REPLACED it; it cannot
quietly fall back to the answer it just rejected.

So this script reports both, side by side:

    UNION    correct OR retry_correct          -- oracle selection, train-style
    REPLACE  flagged -> retry, else pass-1     -- what a deployed system outputs

REPLACE is the number that survives review. Report it as the headline and keep
UNION as the upper bound, clearly labelled.

It also re-derives the two scorecards and the budget sweep from the raw rows, so
a finished run can be re-analysed at any threshold without spending GPU again.

USAGE
    python recompute_stage2.py --store stores/store_ctcval_train_audit2_a2
    python recompute_stage2.py --store <dir> --sweep-gate
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(store):
    p = Path(store) / "results.jsonl"
    if not p.exists():
        p = Path(store) / "merged.jsonl"
    rows = []
    for ln in open(p):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return [r for r in rows if "correct" in r]


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def score(rows, thr=None, gate=None):
    """-> (pass1, union, replace) counts at a threshold, re-deciding the flag."""
    p1 = union = repl = 0
    for r in rows:
        c1 = bool(r["correct"])
        flagged = (bool(r.get("flagged")) if thr is None
                   else r["p_wrong"] >= thr)
        has_retry = "retry_correct" in r
        c2 = bool(r.get("retry_correct"))
        p1 += c1
        if flagged and has_retry:
            union += (c1 or c2)
            repl += c2                      # the rejected answer is GONE
        else:
            union += c1
            repl += c1
    return p1, union, repl


def main():
    ap = argparse.ArgumentParser(description="re-read a stage-2 store, no GPU")
    ap.add_argument("--store", required=True)
    ap.add_argument("--sweep-gate", action="store_true",
                    help="also show accuracy vs the type-confidence gate")
    a = ap.parse_args()

    rows = load(a.store)
    n = len(rows)
    if not n:
        raise SystemExit("no scored rows found")
    thr_used = rows[0].get("threshold")
    print("=" * 74)
    print(f"  STAGE 2 RE-READ · {n} scored questions · store {Path(a.store).name}")
    print("=" * 74)

    # ---------------------------------------------------------------- 1
    print("\n  1 · HOW THE FINAL ANSWER IS SCORED -- this is the whole ballgame")
    print("  " + "-" * 70)
    p1, un, rp = score(rows)
    print(f"    pass 1                                {p1:>4}/{n} = {p1/n:>6.1%}")
    print(f"    UNION    correct OR retry_correct     {un:>4}/{n} = {un/n:>6.1%}"
          f"   ({(un-p1)/n:+.1%})")
    print(f"    REPLACE  flagged -> retry answer      {rp:>4}/{n} = {rp/n:>6.1%}"
          f"   ({(rp-p1)/n:+.1%})")
    print(f"\n    The gap is {(un-rp)/n:.1%} = {un-rp} questions where the retry was WRONG")
    print("    and the union quietly kept the pass-1 answer instead. Doing that")
    print("    needs gold to know which to keep, so REPLACE is the test-time claim.")

    try:
        from stats import mcnemar
        for lbl, key in (("UNION", "u"), ("REPLACE", "r")):
            after = {}
            for r in rows:
                fl = bool(r.get("flagged")) and "retry_correct" in r
                c2 = bool(r.get("retry_correct"))
                after[r["id"]] = ((bool(r["correct"]) or c2) if key == "u"
                                  else (c2 if fl else bool(r["correct"])))
            mc = mcnemar({r["id"]: bool(r["correct"]) for r in rows}, after)
            print(f"    {lbl:<8} McNemar p = {mc['p']:.2e}  "
                  f"gained {mc['n01']}, lost {mc['n10']}"
                  + ("   SIGNIFICANT" if mc["p"] < 0.05 else ""))
    except Exception as e:
        print(f"    (McNemar unavailable: {e})")

    # ---------------------------------------------------------------- 2
    print("\n  2 · WAS THE JUDGE RIGHT?")
    print("  " + "-" * 70)
    y = [not r["correct"] for r in rows]
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y, [r["p_wrong"] for r in rows])
    except Exception:
        auroc = float("nan")
    print(f"    AUROC {auroc:.3f}   base wrong rate {sum(y)/n:.1%}")
    # A sweep can only TIGHTEN. Loosening flags traces that were never retried,
    # and there is no retry answer to score them with -- an earlier version of
    # this table silently fell back to the pass-1 answer for those rows and
    # printed an identical accuracy at every budget, which looked like a
    # flat curve and was really a missing measurement.
    n_ret = sum(1 for r in rows if "retry_correct" in r)
    print(f"\n    {'budget':>7}{'thr':>7}{'flagged':>9}{'prec':>8}{'recall':>8}"
          f"{'REPLACE acc':>13}{'vs pass1':>10}")
    srt = sorted(rows, key=lambda r: (-r["p_wrong"], r["id"]))
    shown = 0
    for bud in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0):
        kb = max(1, min(n, int(round(bud * n))))
        sel = srt[:kb]
        if sum(1 for r in sel if "retry_correct" not in r):
            print(f"    {bud:>7.0%}{'--':>7}{kb:>9}{'':>8}{'':>8}"
                  "    not retried -- needs a new run")
            continue
        t_ = sel[-1]["p_wrong"]
        tp = sum(1 for r in sel if not r["correct"])
        _, _, rp2 = score(rows, thr=t_)
        print(f"    {bud:>7.0%}{t_:>7.3f}{kb:>9}{tp/max(kb,1):>8.1%}"
              f"{tp/max(sum(y),1):>8.1%}{rp2/n:>13.1%}{(rp2-p1)/n:>+10.1%}")
        shown += 1
    print(f"\n    Only budgets at or inside the {n_ret} questions this run actually")
    print("    retried can be scored. A looser budget needs GPU, not arithmetic.")
    print("    Recall is a CHOICE set by the budget, not a ceiling. The")
    print("    threshold-free number is AUROC; precision is the operational one.")

    # ---------------------------------------------------------------- 3
    print("\n  3 · DID NAMING THE TYPE PAY OFF?")
    print("  " + "-" * 70)
    rr = [r for r in rows if "retry_correct" in r and r.get("true_type")
          and not r["correct"]]
    ung = [r for r in rr if not r.get("type_gated")]
    gat = [r for r in rr if r.get("type_gated")]
    if ung:
        ok = [r for r in ung if r["predicted_type"] == r["true_type"]]
        no = [r for r in ung if r["predicted_type"] != r["true_type"]]
        for lbl, s_ in (("cure matched the true type", ok),
                        ("cure was mismatched", no)):
            if s_:
                print(f"    {lbl:<28}{sum(bool(r['retry_correct']) for r in s_)}/{len(s_)}"
                      f" = {sum(bool(r['retry_correct']) for r in s_)/len(s_):>6.1%}")
    if gat:
        print(f"    {'gated -> GENERIC cure':<28}"
              f"{sum(bool(r['retry_correct']) for r in gat)}/{len(gat)}"
              f" = {sum(bool(r['retry_correct']) for r in gat)/len(gat):>6.1%}")
    print("\n    If matched does not beat mismatched, the type head is not earning")
    print("    its place and the honest claim is 'gold-free detection + retry',")
    print("    not 'matched cure'. That is still a claim -- just a different one.")

    # per-type recovery, where the cure actually was typed
    if ung:
        by = defaultdict(lambda: [0, 0])
        for r in ung:
            by[r["true_type"]][1] += 1
            by[r["true_type"]][0] += bool(r["retry_correct"])
        print(f"\n    {'true type':<16}{'n':>5}{'recovered':>11}")
        for t, (c, m) in sorted(by.items(), key=lambda x: -x[1][1]):
            print(f"    {t:<16}{m:>5}{c/max(m,1):>11.1%}")

    if a.sweep_gate:
        print("\n  4 · TYPE-CONFIDENCE GATE (re-decided offline)")
        print("  " + "-" * 70)
        print(f"    {'gate':>6}{'typed':>8}{'generic':>9}{'REPLACE acc':>13}")
        for g in (0.0, 0.4, 0.5, 0.6, 0.7, 1.01):
            nt = sum(1 for r in rows if r.get("retry_correct") is not None
                     and r.get("type_confidence", 0) >= g)
            ng = sum(1 for r in rows if "retry_correct" in r) - nt
            print(f"    {g:>6.2f}{nt:>8}{ng:>9}"
                  "            (needs a re-run to change the cure)")
        print("    NOTE: the gate cannot be swept offline -- it changed which cure")
        print("    RAN. This only shows how many rows each setting would route.")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
