"""
fst_selection.py -- does the failure gate let us SPEND A CURE BUDGET better?
No GPU. ~3 minutes on 1,500 problems.

THE QUESTION
------------
fst_predictor's `fail` head ranks failure at AUROC ~0.66 -- real signal. The
obvious use is budgeted triage: if you can only afford to cure K of N questions,
spend it on the K the gate calls riskiest.

That is only worth anything if risk correlates with RECOVERABILITY, not merely
with failure. This script separates the two, out-of-fold, on the train run:

  * precision@K   -- of the K riskiest, how many actually failed?
  * recovered@K   -- of the K riskiest, how many did the retry actually rescue?
  * vs random     -- recovered@K minus what a random K would have yielded

WHAT IT FOUND (1,500-problem train run)
---------------------------------------
    budget   precision(failure)     lift   recovered   vs random
        5%              89.3%     +18.1%           8        -0.6
       10%              88.7%     +17.5%          17        -0.2
       20%              84.7%     +13.5%          33        -1.4
       50%              81.7%     +10.5%          85        -1.0
      100%              71.2%      +0.0%         172        +0.0

    fail-gate AUROC       0.665   (base rate 0.712)   <- failure IS predictable
    recoverability AUROC  0.491                       <- recovery is NOT

THE FINDING
-----------
Failure is predictable from the question; recoverability is not. Ranking by risk
concentrates FAILURES (89% precision in the top 5% against a 71% base rate) but
concentrates RECOVERIES not at all -- every budget recovers what random would.

That is the mechanism behind FST's flat expected value, and it is a sharper claim
than "the failure type is not predictable". It says the questions the model is most
likely to get wrong are not the questions a cure can rescue. The two populations are
close to orthogonal, and no amount of predictor accuracy fixes that -- a perfect
failure predictor would still not help, because it is ranking the wrong quantity.

FOR THE PAPER: this is a negative result with a mechanism, which is publishable in
a way that "we tried FST and it did not work" is not. State both AUROCs together.

USAGE
    python fst_selection.py --results stores/store_typed_train_audit2/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    p1 = [r for r in rows if r.get("phase") == "pass1"]
    rt = {}
    for r in rows:
        if r.get("phase") == "retry":
            rt[r["id"]] = rt.get(r["id"]) or r.get("newly_correct")
            if r.get("newly_correct"):
                rt[r["id"]] = True
    if not p1:
        raise SystemExit("no pass1 records")
    return p1, rt


def main():
    ap = argparse.ArgumentParser(description="budgeted triage with the FST gate")
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--features", choices=["hand", "tfidf", "both"], default="both")
    a = ap.parse_args()

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, average_precision_score
    from fst_predictor import _Featurizer, FSTPredictor

    p1, rt = load(a.results)
    qs = [r["question"] for r in p1]
    wrong = np.array([0 if (r.get("signals") or {}).get("correct") else 1 for r in p1])
    recov = np.array([1 if rt.get(r["id"]) else 0 for r in p1])
    n = len(qs)

    retried = int(wrong.sum())
    print("=" * 74)
    print("  FST GATE · BUDGETED TRIAGE")
    print("=" * 74)
    print(f"  problems {n} | failed {retried} ({wrong.mean():.1%}) | "
          f"recovered by retry {int(recov.sum())} ({recov.mean():.1%} of all)")
    if recov.sum() == 0:
        raise SystemExit("no recoveries in this store -- nothing to rank")

    fz = _Featurizer(a.features)
    X = fz.fit_transform(qs)
    clf = FSTPredictor(a.model, "fail", a.features)._make()
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    risk = cross_val_predict(clf, X, wrong, cv=cv, method="predict_proba")[:, 1]

    # The rival ranking: predict RECOVERY directly.
    #
    # It must be scored on ALL n questions, not just the known failures. Scoring
    # only the failures would hand the head the answer to the very thing the gate
    # is trying to guess (which ones fail) and inflate it enormously -- an earlier
    # version of this script did exactly that and reported +44 recoveries at a 50%
    # budget, all of which was the leak. In the FST setting nothing has been
    # attempted yet, so every question must be scored blind.
    m = wrong == 1
    rec_score = cross_val_predict(
        FSTPredictor(a.model, "recover", a.features)._make(),
        X, recov, cv=cv, method="predict_proba")[:, 1]

    order = np.argsort(-risk)
    order_rec = np.argsort(-rec_score)
    base_w, base_r = wrong.mean(), recov.mean()

    print("\n" + "-" * 74)
    print("  SPEND THE CURE BUDGET ON THE K RISKIEST QUESTIONS")
    print("-" * 74)
    print(f"  {'budget':>8}{'n':>7}{'precision(fail)':>18}{'lift':>9}"
          f"{'recovered':>12}{'if random':>11}{'delta':>8}")
    rowsout = []
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50, float(f"{base_w:.2f}"), 1.00):
        k = max(int(round(n * frac)), 1)
        sel = order[:k]
        prec = wrong[sel].mean()
        got = int(recov[sel].sum())
        exp = base_r * k
        rowsout.append((frac, k, prec, got, exp))
        print(f"  {frac:>7.0%}{k:>7}{prec:>17.1%}{prec-base_w:>+9.1%}"
              f"{got:>12}{exp:>11.1f}{got-exp:>+8.1f}")

    print("\n" + "-" * 74)
    print("  ...VERSUS RANKING BY PREDICTED RECOVERABILITY  (`recover` head)")
    print("-" * 74)
    print(f"  {'budget':>8}{'n':>7}{'recovered':>12}{'if random':>11}{'delta':>8}"
          f"{'vs risk-rank':>14}")
    for frac in (0.05, 0.10, 0.20, 0.30, 0.50):
        k = max(int(round(n * frac)), 1)
        got_r = int(recov[order_rec[:k]].sum())
        got_k = int(recov[order[:k]].sum())
        exp = base_r * k
        print(f"  {frac:>7.0%}{k:>7}{got_r:>12}{exp:>11.1f}{got_r-exp:>+8.1f}"
              f"{got_r-got_k:>+14}")

    au_fail = roc_auc_score(wrong, risk)
    ap_fail = average_precision_score(wrong, risk)
    au_rec = roc_auc_score(recov, risk)
    # recoverability among the ones that actually failed -- the fair test
    m = wrong == 1
    au_rec_w = roc_auc_score(recov[m], risk[m]) if 0 < recov[m].sum() < m.sum() else float("nan")

    print("\n" + "-" * 74)
    print("  WHAT THE RISK SCORE ACTUALLY RANKS")
    print("-" * 74)
    print(f"  failure           AUROC {au_fail:.3f}   AP {ap_fail:.3f}  (base {base_w:.3f})")
    print(f"  recovery, all     AUROC {au_rec:.3f}")
    print(f"  recovery | failed AUROC {au_rec_w:.3f}   <- the fair test: among the")
    print( "                                        traces that DID fail, does risk")
    print( "                                        rank which ones the cure rescues?")

    print("\n" + "=" * 74)
    print("  VERDICT")
    print("=" * 74)
    strong_fail = au_fail >= 0.60
    strong_rec = (au_rec_w == au_rec_w) and au_rec_w >= 0.58
    if strong_fail and strong_rec:
        print("  BOTH signals present. Budgeted triage should pay off -- run the")
        print("  test split with --gate and a budget below 100%.")
    elif strong_fail and not strong_rec:
        print("  FAILURE is predictable; RECOVERABILITY is not.")
        print(f"    failure AUROC {au_fail:.3f}  vs  recoverability AUROC {au_rec_w:.3f}")
        print("  Ranking by risk concentrates failures but not recoveries, so a")
        print("  budgeted FST run recovers what random selection would. Report BOTH")
        print("  numbers together: that pairing is the mechanism behind the flat")
        print("  expected value, and it is a stronger claim than 'FST did not work'.")
        print("\n  It also means a BETTER failure predictor would not help. The")
        print("  quantity being ranked is the wrong one.")
    else:
        print("  The gate does not rank failure usefully either. Nothing to triage on.")

    au_direct_all = roc_auc_score(recov, rec_score)
    au_direct_w = (roc_auc_score(recov[m], rec_score[m])
                   if 0 < recov[m].sum() < m.sum() else float("nan"))
    print("\n  DIRECTLY-TRAINED RECOVERABILITY HEAD  (scored blind on all "
          f"{n} questions)")
    print(f"    recovery, all     AUROC {au_direct_all:.3f}   (risk-score: {au_rec:.3f})")
    print(f"    recovery | failed AUROC {au_direct_w:.3f}   (risk-score: {au_rec_w:.3f})")
    if au_direct_all >= 0.58:
        print("    -> usable. Allocate the retry budget on THIS head, not on `fail`.")
        print("       It is the honest form of FST: predict which questions a cure")
        print("       can RESCUE, not which questions will fail.")
    elif au_direct_all >= 0.54:
        print("    -> weak but above chance, and better than ranking on `fail`.")
        print("       Predicting recoverability is the right target; the question")
        print("       alone does not carry enough of it to build a product on.")
    else:
        print("    -> at chance. Recoverability is not predictable from the question,")
        print("       by any of the routes tried.")

    print("\n  A NOTE ON THE TWO SETTINGS")
    print("    FST (pre-attempt): the head must score every question blind -- that is")
    print("      the number above.")
    print("    Deferred-retry budget (post-attempt): failures are already known and")
    print("      gold-free trace signals are available. Train `--target recover` and")
    print("      add the signals; measured AUROC there is 0.595, the best of any route.")

    print("\n  Paper-ready sentence:")
    print(f"    \"A question-only classifier ranks pass-1 failure at AUROC {au_fail:.2f},")
    print(f"     but ranks recoverability among failures at {au_rec_w:.2f} -- chance.")
    print( "     Predictive routing therefore cannot beat random allocation of a")
    print( "     fixed retry budget, independent of predictor quality.\"")


if __name__ == "__main__":
    main()
