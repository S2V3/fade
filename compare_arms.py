"""
compare_arms.py -- the EDER ablation: typed vs generic vs immediate, PAIRED.

WHY THIS IS NOW THE MOST IMPORTANT SCRIPT IN THE PROJECT
--------------------------------------------------------
The clean 1,500-problem run showed EDER working:

    pass-1  28.8%  ->  final  40.3%      (+11.5 points, 172 recovered)
    MEDIUM_WRONG recovers 21.8% vs BAD_WRONG 14.6%   Fisher p = 0.013

But "+11.5 points" is not yet a claim about FADE. It could be the gain from
*retrying at all*. Three arms, on the SAME problems, separate the three factors:

    immediate   re-sample the same prompt          -> does retrying help at all?
    generic     8 untyped exemplars, no instruction -> do MORE EXEMPLARS help?
    typed       8 type-matched exemplars + cure     -> does the DIAGNOSIS help?

The claim the paper needs is `typed > generic > immediate`. Every arm uses the
same exemplar budget and the same retry count, so exactly one factor changes at
each step.

WHY PAIRED (McNEMAR) AND NOT TWO PROPORTIONS
--------------------------------------------
The arms share problems. A two-proportion z-test throws that pairing away and is
anticonservative here. McNemar conditions on the discordant pairs -- the problems
where the two arms disagree -- which is the actual evidence.

USAGE
    python compare_arms.py \
        --typed     stores/store_typed_train_audit2 \
        --generic   stores/store_generic_train_audit2 \
        --immediate stores/store_immediate_train_audit2
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

from stats import mcnemar, wilson_ci, fisher_exact_2x2


def load_arm(store: str):
    """-> (pass1_correct{id:bool}, final_correct{id:bool}, diag{id:type}, meta)"""
    path = os.path.join(store, "results.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no results.jsonl in {store}")
    p1c, finc, diag, prev_label = {}, {}, {}, {}
    mode = None
    for ln in open(path):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("phase") == "pass1":
            ok = bool((r.get("signals") or {}).get("correct"))
            p1c[r["id"]] = ok
            finc[r["id"]] = ok
            diag[r["id"]] = r.get("diagnosis")
            prev_label[r["id"]] = r.get("label")
        else:
            mode = mode or r.get("retry_mode")
            if r.get("newly_correct"):
                finc[r["id"]] = True
    snap = os.path.join(store, "config_snapshot.json")
    ver = None
    if os.path.exists(snap):
        try:
            ver = json.load(open(snap)).get("FADE_CODE_VERSION")
        except Exception:
            pass
    return p1c, finc, diag, {"retry_mode": mode, "version": ver, "n": len(p1c)}


def main():
    ap = argparse.ArgumentParser(description="paired EDER arm comparison")
    ap.add_argument("--typed", required=True)
    ap.add_argument("--generic")
    ap.add_argument("--immediate")
    a = ap.parse_args()

    arms = {}
    for name, path in (("typed", a.typed), ("generic", a.generic),
                       ("immediate", a.immediate)):
        if path:
            arms[name] = load_arm(path)

    print("=" * 72)
    print("  ARMS LOADED")
    print("=" * 72)
    vers = set()
    for name, (p1c, finc, diag, meta) in arms.items():
        vers.add(meta["version"])
        print(f"  {name:<11} n={meta['n']:<6} retry_mode={str(meta['retry_mode']):<10} "
              f"code={meta['version']}")
    if len(vers) > 1:
        print(f"\n  !! ARMS WERE PRODUCED BY DIFFERENT CODE VERSIONS {vers}.")
        print("     They are not comparable. Re-run the odd one out.")

    # ---- the paired problem set -----------------------------------------
    common = set.intersection(*[set(v[0]) for v in arms.values()])
    print(f"\n  problems present in EVERY arm: {len(common)}")
    if not common:
        raise SystemExit("no shared problems -- the arms did not run the same slice")
    for name, (p1c, _, _, _) in arms.items():
        agree = sum(1 for i in common if p1c[i] == arms["typed"][0][i])
        if name != "typed" and agree != len(common):
            print(f"  !! {name}: pass-1 outcome differs from typed on "
                  f"{len(common)-agree} shared problems.")
            print("     Pass-1 is decoded greedily and should be IDENTICAL across arms;")
            print("     a difference means the arms did not see the same prompts.")

    # ---- headline table ---------------------------------------------------
    print("\n" + "=" * 72)
    print("  ACCURACY ON THE SHARED PROBLEMS")
    print("=" * 72)
    print(f"  {'arm':<12}{'pass-1':>10}{'final':>10}{'recovered':>12}{'gain':>9}")
    base_p1 = None
    for name in ("immediate", "generic", "typed"):
        if name not in arms:
            continue
        p1c, finc, _, _ = arms[name]
        p1 = sum(p1c[i] for i in common) / len(common)
        fin = sum(finc[i] for i in common) / len(common)
        rec = sum(1 for i in common if finc[i] and not p1c[i])
        base_p1 = base_p1 if base_p1 is not None else p1
        print(f"  {name:<12}{p1:>10.1%}{fin:>10.1%}{rec:>12}{fin-p1:>+9.1%}")

    # ---- the paired tests -------------------------------------------------
    print("\n" + "=" * 72)
    print("  PAIRED COMPARISONS (McNemar, exact)")
    print("=" * 72)
    order = [n for n in ("immediate", "generic", "typed") if n in arms]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            lo, hi = order[i], order[j]
            A = {k: arms[lo][1][k] for k in common}
            B = {k: arms[hi][1][k] for k in common}
            m = mcnemar(A, B)
            better = hi if m["n01"] > m["n10"] else lo
            print(f"\n  {hi} vs {lo}")
            print(f"    {hi} right / {lo} wrong : {m['n01']}")
            print(f"    {lo} right / {hi} wrong : {m['n10']}")
            print(f"    accuracy {lo} {m['acc_a']:.1%}  ->  {hi} {m['acc_b']:.1%}"
                  f"   ({m['acc_b']-m['acc_a']:+.1%})")
            print(f"    exact p = {m['p']:.4f}"
                  f"   {'SIGNIFICANT' if m['p'] < 0.05 else 'not significant'}"
                  f"   (favours {better})")

    # ---- per-type recovery, typed only -----------------------------------
    if "typed" in arms:
        print("\n" + "=" * 72)
        print("  TYPED ARM — recovery by diagnosed type (shared problems)")
        print("=" * 72)
        p1c, finc, diag, _ = arms["typed"]
        by = defaultdict(lambda: [0, 0])
        for i in common:
            if p1c[i]:
                continue
            t = diag.get(i) or "None"
            by[t][1] += 1
            by[t][0] += bool(finc[i])
        tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
        for t, (c, n) in sorted(by.items(), key=lambda x: -x[1][1]):
            lo_, hi_ = wilson_ci(c, n)
            print(f"    {t:<14}{c:>4}/{n:<5} = {c/max(n,1):>6.1%}  [{lo_:.3f},{hi_:.3f}]")
        print(f"    {'OVERALL':<14}{tot[0]:>4}/{tot[1]:<5} = {tot[0]/max(tot[1],1):>6.1%}")

    print("\n" + "=" * 72)
    print("  HOW TO READ THIS")
    print("=" * 72)
    print("  typed > generic  ->  the DIAGNOSIS is doing work, not just the exemplars")
    print("  generic > immediate -> extra exemplars help beyond a plain re-sample")
    print("  typed ~ generic  ->  your gain is 'more exemplars', not 'matched cure'.")
    print("                       That is still a result, but it is a DIFFERENT claim")
    print("                       and the paper must say so.")


if __name__ == "__main__":
    main()
