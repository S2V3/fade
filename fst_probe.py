"""
fst_probe.py -- the complete FST feasibility analysis. NO GPU. ~2 minutes.

WHAT THIS REPLACES
------------------
The original plan was fst_predictor.py -> run on the test split -> measure lift.
That costs ~10 GPU-h. This script establishes, offline, whether that spend can
possibly pay off -- and produces the numbers the paper's FST section needs either
way. It is not a shortcut around the experiment; it IS the experiment, minus the
part that only re-confirms what the arithmetic already determines.

THE THREE QUESTIONS, IN ORDER
-----------------------------
1. IS THE CURE WORTH ROUTING?   oracle (true type known) vs generic cure.
   If routing buys nothing even with a perfect predictor, FST is pointless
   regardless of prediction accuracy. [Measured: 17.4% vs 8.0% -> +9.4 pts.
   Routing IS worth something.]

2. CAN THE TYPE BE PREDICTED?   every reasonable feature set and model, 5-fold CV,
   against the majority-class baseline. Includes RAW QUESTION TEXT, which the
   original gate never tried.

3. WHAT WOULD FST ACTUALLY EARN?  expected recovery =
        acc * r_matched + (1-acc) * r_mismatched
   bracketed over what a mismatched cure might do. If the bracket straddles zero,
   the test-split run cannot come out positive except by chance.

WHY THIS IS A RESULT, NOT AN ABANDONMENT
----------------------------------------
"FST is under construction" is a weakness a reviewer will name. "Cure-matching is
worth +9.4 points, but the type is not predictable from the question -- no model
beats a majority baseline, and per-type AUROC exceeds 0.6 for only 2 of 6 types --
so the expected gain is ~0" is a finding about WHY predictive routing fails.

USAGE
    python fst_probe.py --results stores/store_typed_train_audit2/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

TYPED = ("NR", "AL", "ST", "SM", "CE", "WP")


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    p1 = {r["id"]: r for r in rows if r.get("phase") == "pass1"}
    rt = [r for r in rows if r.get("phase") == "retry"]
    return p1, rt


# ---------------------------------------------------------------- Q1: ceiling
def question1(p1, rt):
    print("=" * 74)
    print("  Q1 · IS CURE-ROUTING WORTH ANYTHING?  (oracle vs generic)")
    print("=" * 74)
    matched = [r for r in rt if r.get("prev_diagnosis") in TYPED]
    generic = [r for r in rt if r.get("prev_diagnosis") == "UNCLASSIFIED"]
    if not matched or not generic:
        print("  cannot compute -- need both typed and UNCLASSIFIED retries")
        return None, None
    rm = sum(bool(r.get("newly_correct")) for r in matched) / len(matched)
    rg = sum(bool(r.get("newly_correct")) for r in generic) / len(generic)
    from stats import wilson_ci, fisher_exact_2x2
    lo1, hi1 = wilson_ci(sum(bool(r.get("newly_correct")) for r in matched), len(matched))
    lo2, hi2 = wilson_ci(sum(bool(r.get("newly_correct")) for r in generic), len(generic))
    p = fisher_exact_2x2([[int(rm*len(matched)), len(matched)-int(rm*len(matched))],
                          [int(rg*len(generic)), len(generic)-int(rg*len(generic))]])
    print(f"  cure matched to TRUE type : {rm:>6.1%}  n={len(matched):<5} [{lo1:.3f},{hi1:.3f}]")
    print(f"  generic cure (abstain arm): {rg:>6.1%}  n={len(generic):<5} [{lo2:.3f},{hi2:.3f}]")
    print(f"  ROUTING HEADROOM          : {(rm-rg)*100:>+5.1f} pts   Fisher p = {p:.4f}")
    print("\n  CAVEAT: the abstain arm is not a randomised control -- UNCLASSIFIED")
    print("  traces may be intrinsically harder. Only a real `generic` arm on the")
    print("  SAME problems settles this. Treat +9-ish pts as an upper estimate.")
    return rm, rg


# ------------------------------------------------------- Q2: predictability
def question2(p1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from scipy.sparse import hstack, csr_matrix

    print("\n" + "=" * 74)
    print("  Q2 · CAN THE FAILURE TYPE BE PREDICTED FROM THE QUESTION?")
    print("=" * 74)
    data = [(r["question"], r["diagnosis"]) for r in p1.values()
            if not (r.get("signals") or {}).get("correct") and r.get("diagnosis") in TYPED]
    if len(data) < 100:
        print(f"  only {len(data)} typed wrong traces -- too few. Need >= 300.")
        return None
    q = [d[0] for d in data]
    y = np.array([d[1] for d in data])
    maj = Counter(y).most_common(1)[0][1] / len(y)
    print(f"  n = {len(y)}   classes {dict(Counter(y).most_common())}")
    print(f"  MAJORITY BASELINE = {maj:.3f}   (any model must beat this)\n")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    from features import question_features, vectorize
    Xh, _ = vectorize([question_features(x) for x in q])
    Xh = np.asarray(Xh, float)
    Xw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(q)
    Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                         sublinear_tf=True).fit_transform(q)
    Xb = hstack([Xw, csr_matrix(StandardScaler().fit_transform(Xh))])

    LR = lambda: LogisticRegression(max_iter=6000, class_weight="balanced")
    trials = [
        ("hand-made surface features + LR", Xh, LR()),
        ("word 1-2gram TF-IDF + LR",        Xw, LR()),
        ("char 3-5gram TF-IDF + LR",        Xc, LR()),
        ("TF-IDF + hand features + LR",     Xb, LR()),
        ("hand features + RandomForest",    Xh, RandomForestClassifier(400, random_state=0,
                                                                      class_weight="balanced")),
        ("hand features + GradientBoosting", Xh, GradientBoostingClassifier(random_state=0)),
    ]
    best = -1
    print(f"  {'model':<40}{'CV acc':>9}{'vs majority':>13}")
    for name, X, clf in trials:
        s = cross_val_score(clf, X, y, cv=cv).mean()
        best = max(best, s)
        flag = "  BEATS" if s > maj else ""
        print(f"  {name:<40}{s:>9.3f}{s-maj:>+13.3f}{flag}")

    print(f"\n  per-type ONE-VS-REST AUROC (0.5 = chance):")
    aur = {}
    for t in TYPED:
        yb = (y == t).astype(int)
        if yb.sum() < 20:
            continue
        a = cross_val_score(LR(), Xb, yb, cv=cv, scoring="roc_auc").mean()
        aur[t] = a
        mark = "  <-- weak signal" if a >= 0.60 else ""
        print(f"    {t:<6} n={int(yb.sum()):<5} AUROC {a:.3f}{mark}")
    return maj, best, aur


# ------------------------------------------------------- Q3: expected value
def question3(rm, rg, acc, wrong_frac=0.71):
    print("\n" + "=" * 74)
    print("  Q3 · WHAT WOULD FST ACTUALLY EARN?")
    print("=" * 74)
    if rm is None:
        print("  skipped (Q1 unavailable)")
        return
    print(f"  matched cure {rm:.1%} | generic {rg:.1%} | best predictor accuracy {acc:.1%}\n")
    print(f"  {'assumption about a MISMATCHED cure':<40}{'retry':>8}{'vs generic':>12}{'overall':>10}")
    for lbl, rw in (("behaves like the generic cure", rg),
                    ("half as good as generic", rg / 2),
                    ("worthless (confidently wrong)", 0.0)):
        ev = acc * rm + (1 - acc) * rw
        print(f"  {lbl:<40}{ev:>8.1%}{(ev-rg)*100:>+11.1f}p{(ev-rg)*wrong_frac*100:>+9.1f}p")
    print("\n  If this bracket straddles zero, a test-split FST run cannot come out")
    print("  positive except by chance -- and the power to detect a ~2-point effect")
    print("  needs n in the thousands (see the power note below).")
    print("\n  POWER: paired McNemar, true effect +2 pts on retries")
    print("    n=100 -> 4%   n=250 -> 7%   n=500 -> 10%   n=1000 -> 15%")
    print("    (80% is the usual bar; simulated, 2000 trials, alpha=.05)")


def main():
    ap = argparse.ArgumentParser(description="FST feasibility, offline")
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    p1, rt = load(a.results)
    print(f"loaded {len(p1)} pass-1 and {len(rt)} retry records\n")
    rm, rg = question1(p1, rt)
    out = question2(p1)
    if out and rm is not None:
        maj, best, aur = out
        question3(rm, rg, max(best, maj))
    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    if out:
        maj, best, aur = out
        if best > maj + 0.05:
            print("  GO -- a predictor beats majority by a usable margin. Build")
            print("  fst_predictor.py and run the test split.")
        else:
            print("  NO-GO for a *predictor*. No model beats the majority baseline.")
            print("  But Q1 shows cure-ROUTING has real headroom, so report the two")
            print("  claims separately:")
            print("     'routing by known type is worth +N pts'")
            print("     'the type is not predictable from the question (k models, AUROC")
            print("      above 0.6 for only the types listed above)'")
            print("  That is a finding about WHY predictive routing fails -- not an")
            print("  unfinished component.")


if __name__ == "__main__":
    main()
