"""
ctc_train2.py -- CTC v2. Stronger heads, same gold-free inputs. NO GPU, ~3 min.

WHAT CHANGED FROM ctc_train.py
------------------------------
v1 used 21 hand-made features and one RandomForest per head. Measured, 5-fold CV,
on the same 2,568 traces from the final typed run:

    head     v1 (hand + RF)     v2 (hand + trace text + question, ensembled)
    detect   AUROC 0.687        AUROC 0.752
    type     acc   0.621        acc   0.657   (majority 0.216, lift +0.441)

Three changes, each measured before being kept:

1. THE WORDS IN THE TRACE. A failure leaves lexical fingerprints -- "let x be",
   "Step 1:", "The answer is" -- that no hand feature captures. Trace TF-IDF alone
   is weak (type 0.302) but combined with the hand features it is the single
   largest gain: type 0.599 -> 0.644.
2. THE QUESTION TEXT, added on top: type 0.644 -> 0.653.
3. ENSEMBLING a linear model on the sparse text with a tree model on the dense
   hand features. They fail differently, so averaging beats either:
   detect 0.745/0.726 -> 0.752, type 0.653/0.623 -> 0.657.

Character n-grams were tried and dropped (0.643 vs 0.653 for type, no detect gain).

THE EVALUATION WAS WRONG, AND THE NUMBERS WERE INFLATED
-------------------------------------------------------
Every trace is one example, but a question contributes TWO -- its pass-1 attempt
and its retry. StratifiedKFold splits on traces, so one question's pass-1 could
land in the training fold while its retry sat in the test fold. The model could
then look the question up rather than read the trace.

    head     StratifiedKFold        GroupKFold by question (honest)
    detect   AUROC 0.752            AUROC 0.753     (no leakage)
    type     acc   0.657            acc   0.613     (-0.044, leakage)

Detection was unaffected -- it reads the trace. The TYPE head was inflated by 4.4
points, and the culprit was the question TF-IDF: it let the model memorise which
type each question tends to produce. Under grouped CV those features stop helping
and start hurting:

    type, GroupKFold        hand only  hand+trace  hand+trace+question
      linear                  0.603      0.616          0.608
      0.7 linear + 0.3 RF     0.604      0.608          0.601

So the type head now uses hand + TRACE text only, linear only. The table above is
from the config search (linear head alone); the SHIPPED model reports detect AUROC
0.756 and type accuracy 0.627 (majority 0.216, lift +0.411). Trust what fit()
prints over this docstring.

Everything reported from here uses GroupKFold. It is the only split that answers
the question CTC will actually face: an unseen problem.

ONE NUMBER HIDES TWO DISTRIBUTIONS
----------------------------------
A question contributes both a pass-1 trace and a retry trace, and they are not
equally legible. Out of fold, sliced by phase:

    slice    n      wrong%   detect AUROC   type acc
    pass-1   1500   71.2%    0.776          0.598
    retry    1068   81.5%    0.683          0.679
    both     2568   75.5%    0.751          0.635

DEPLOYMENT ONLY EVER SEES PASS-1. run_ctc_validate.py reads a first attempt and
nothing else, so it should land near 0.776 for detection and 0.598 for type --
above the headline on one, below it on the other. Neither is a regression, and
comparing stage 2 against the headline instead of against the pass-1 row will
make a working detector look broken and a broken type head look fine.

Training on pass-1 rows alone was tried, since that IS the deployment
distribution: detect +0.006 (bootstrap 95% CI [-0.004, +0.016], P(gain>0)=0.88),
type -0.008. Not significant, and it halves the data, so the retries stay in.

WHAT DID NOT WORK, AND WHY IT IS NOT HERE
-----------------------------------------
Pooling traces from EVERY run gives 5,677 examples instead of 2,568. It is not
used: earlier runs were produced by different configurations, so their traces are
off-distribution relative to what the final system emits. Doubling the data by
mixing in traces the deployed pipeline would never produce trains the detector on
the wrong thing. --extra-files re-enables it for an ablation.

PER-TYPE RECALL IS VERY UNEVEN, AND THAT MATTERS MORE THAN THE MEAN
-------------------------------------------------------------------
    TR 98.9%   NR 96.4%   AL 88.1%   WP 74.5%
    ST 62.8%   UNCL 55.7%   SM 42.0%   CE 41.7%

Structural failures carry crisp signatures; semantic ones do not. SM and CE are two
of the three largest classes, so roughly a coin-flip on the cure they get. Report
the spread, never the mean alone.

GOLD RULE: features read the question and the model's own trace. Only the LABEL
comes from gold, and only on train.

USAGE
    python ctc_data.py --files <results.jsonl> --out ctc_dataset.jsonl
    python ctc_train2.py --data ctc_dataset.jsonl --out-dir stores/ctc_v2
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# [AUDIT D64] Weights and features re-chosen under GroupKFold. See below.
W_DETECT_LINEAR = 0.60      # 0.6*LR + 0.4*GB -- best under grouped CV (0.753)
W_TYPE_LINEAR = 1.00        # linear only; the tree blend HURT under grouped CV
USE_QUESTION_TFIDF = False  # [D64] leakage, not signal -- see below


class CTC2:
    """detect + type over gold-free (question, trace) features."""

    def __init__(self):
        self.hand_names = None
        self.tf_trace = self.tf_question = self.scaler = None
        self.det_lin = self.det_tree = None
        self.typ_lin = self.typ_tree = None
        self.classes_ = None
        self.report = {}

    # ------------------------------------------------------------ features
    def _dense(self, rows, fit=False):
        from ctc_features import trace_features, vectorize
        from sklearn.preprocessing import StandardScaler
        feats = [trace_features(q, t) for q, t in rows]
        X, names = vectorize(feats, self.hand_names)
        X = np.asarray(X, float)
        if fit:
            self.hand_names = names
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def _sparse(self, rows, fit=False, with_question=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.sparse import hstack, csr_matrix
        if with_question is None:
            with_question = USE_QUESTION_TFIDF
        qs = [q for q, _ in rows]
        ts = [t for _, t in rows]
        if fit:
            self.tf_trace = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                            sublinear_tf=True, max_features=8000).fit(ts)
            self.tf_question = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                                               sublinear_tf=True).fit(qs)
        parts = [csr_matrix(self._dense(rows)), self.tf_trace.transform(ts)]
        if with_question:
            parts.append(self.tf_question.transform(qs))
        return hstack(parts).tocsr()

    # ----------------------------------------------------------------- fit
    def fit(self, rows, correct, types, verbose=True, phases=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score, accuracy_score

        Xd = self._dense(rows, fit=True)
        Xs = self._sparse(rows, fit=True)
        y = np.array([0 if c else 1 for c in correct])
        # [AUDIT D64] group by QUESTION. A question contributes both its pass-1 and
        # its retry trace; splitting on traces puts them in different folds and the
        # model looks the question up instead of reading the trace.
        groups = np.array([hash(q) % 10**9 for q, _ in rows])
        cv = GroupKFold(5)

        LIN = lambda: LogisticRegression(max_iter=4000, class_weight="balanced")
        pl = cross_val_predict(LIN(), Xs, y, cv=cv, groups=groups,
                               method="predict_proba")[:, 1]
        pg = cross_val_predict(GradientBoostingClassifier(random_state=0), Xd, y,
                               cv=cv, groups=groups, method="predict_proba")[:, 1]
        blend = W_DETECT_LINEAR * pl + (1 - W_DETECT_LINEAR) * pg
        det = {"auroc": float(roc_auc_score(y, blend)),
               "auroc_linear": float(roc_auc_score(y, pl)),
               "auroc_tree": float(roc_auc_score(y, pg)),
               "base_wrong_rate": float(y.mean())}

        # [AUDIT D77] CALIBRATION FOR ONLINE USE. Stage 2 and stage 3 decide
        # per question, the moment the trace exists -- there is no distribution
        # to take a quantile of, because the other 399 questions have not been
        # attempted yet. So the mapping from "retry the worst 40%" to an actual
        # p_wrong cut has to be measured HERE, out of fold, and shipped with the
        # model. Restricted to pass-1 rows when the phase is known, since that is
        # the only kind of trace a deployed detector ever reads.
        _m = (np.array([p == "pass1" for p in phases]) if phases is not None
              else np.ones(len(y), bool))
        if _m.sum() < 50:
            _m = np.ones(len(y), bool)
        bcal, ycal = blend[_m], y[_m]
        det["calibration_slice"] = ("pass1" if phases is not None and _m.sum() < len(y)
                                    else "all")
        det["calibration_n"] = int(_m.sum())
        det["budget_thresholds"] = {}
        det["budget_precision"] = {}
        for bud in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            kb = max(1, int(round(bud * len(bcal))))
            t_ = float(np.sort(bcal)[::-1][kb - 1])
            sel = bcal >= t_
            det["budget_thresholds"][f"{bud:.1f}"] = t_
            det["budget_precision"][f"{bud:.1f}"] = float(ycal[sel].mean())
        self.det_lin = LIN().fit(Xs, y)
        self.det_tree = GradientBoostingClassifier(random_state=0).fit(Xd, y)

        m = y == 1
        yw = np.array(types)[m]
        maj = Counter(yw).most_common(1)[0][1] / len(yw)
        gw = groups[m]
        Pl = cross_val_predict(LIN(), Xs[m], yw, cv=cv, groups=gw,
                               method="predict_proba")
        Pr = cross_val_predict(RandomForestClassifier(600, random_state=0,
                                                      class_weight="balanced"),
                               Xd[m], yw, cv=cv, groups=gw, method="predict_proba")
        cls = np.array(sorted(set(yw)))
        P = W_TYPE_LINEAR * Pl + (1 - W_TYPE_LINEAR) * Pr
        pred = cls[P.argmax(1)]
        typ = {"accuracy": float(accuracy_score(yw, pred)), "majority": float(maj),
               "lift": float(accuracy_score(yw, pred) - maj),
               "per_type_recall": {c: float((pred[yw == c] == c).mean()) for c in cls},
               "classes": {str(k): int(v) for k, v in Counter(yw).items()}}
        self.typ_lin = LIN().fit(Xs[m], yw)
        self.typ_tree = RandomForestClassifier(600, random_state=0,
                                               class_weight="balanced").fit(Xd[m], yw)
        self.classes_ = self.typ_lin.classes_
        self.report = {"n": int(len(y)), "n_wrong": int(m.sum()),
                       "detect": det, "type": typ}

        if verbose:
            print(f"\n  DETECT  n={len(y)}  base wrong rate {y.mean():.1%}"
                  f"   [GroupKFold by question]")
            print(f"    linear on text+hand   AUROC {det['auroc_linear']:.3f}")
            print(f"    trees on hand         AUROC {det['auroc_tree']:.3f}")
            print(f"    ENSEMBLE              AUROC {det['auroc']:.3f}")
            if det["auroc"] < 0.70:
                print("    !! below 0.70 -- usable for RANKING, not as a hard gate.")
            print(f"\n    calibration for ONLINE use "
                  f"({det['calibration_slice']} traces, n={det['calibration_n']}):")
            print(f"    {'budget':>8}{'p_wrong cut':>13}{'precision':>11}")
            for b in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.7", "1.0"):
                print(f"    {float(b):>8.0%}{det['budget_thresholds'][b]:>13.3f}"
                      f"{det['budget_precision'][b]:>11.1%}")
            print("    run_ctc_validate.py reads these to turn a budget into a cut")
            print("    it can apply to ONE question at a time.")
            print(f"\n  TYPE    n={int(m.sum())}  majority {maj:.3f}")
            print(f"    ENSEMBLE accuracy {typ['accuracy']:.3f}  lift {typ['lift']:+.3f}")
            print(f"\n    {'type':<16}{'n':>5}{'recall':>9}")
            for c, r in sorted(typ["per_type_recall"].items(), key=lambda x: -x[1]):
                print(f"    {c:<16}{typ['classes'].get(c,0):>5}{r:>9.1%}")
            print("\n    Structural types are nearly solved; SM and CE are near a")
            print("    coin flip and are two of the three largest classes.")
        return self

    # ----------------------------------------------------------- inference
    def predict(self, question, trace):
        rows = [(question, trace)]
        Xd, Xs = self._dense(rows), self._sparse(rows)
        p = (W_DETECT_LINEAR * self.det_lin.predict_proba(Xs)[0][1]
             + (1 - W_DETECT_LINEAR) * self.det_tree.predict_proba(Xd)[0][1])
        P = (W_TYPE_LINEAR * self.typ_lin.predict_proba(Xs)[0]
             + (1 - W_TYPE_LINEAR) * self.typ_tree.predict_proba(Xd)[0])
        i = int(P.argmax())
        return {"p_wrong": float(p), "predicted_type": str(self.classes_[i]),
                "type_confidence": float(P[i])}

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        return joblib.load(path)


def main():
    ap = argparse.ArgumentParser(description="train CTC v2")
    ap.add_argument("--data", required=True, help="ctc_dataset.jsonl from ctc_data.py")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    ex = [json.loads(l) for l in open(a.data) if l.strip()]
    print("=" * 66)
    print(f"  CTC v2 · {len(ex)} traces")
    print("=" * 66)
    print(f"  correct {sum(1 for e in ex if e['correct'])} | "
          f"wrong {sum(1 for e in ex if not e['correct'])}")
    if len(ex) < 1000:
        print("  !! under 1,000 traces -- the type head will be unstable.")

    rows = [(e["question"], e["trace"]) for e in ex]
    m = CTC2().fit(rows, [e["correct"] for e in ex], [e["type"] for e in ex],
                phases=[e.get("phase") for e in ex])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    m.save(out / "ctc2.joblib")
    (out / "ctc2_report.json").write_text(json.dumps(m.report, indent=2))
    print(f"\n  saved -> {out/'ctc2.joblib'}")

    print("\n  smoke test (gold-free path):")
    for e in ex[:3]:
        p = m.predict(e["question"], e["trace"])
        print(f"    p_wrong {p['p_wrong']:.2f} -> {p['predicted_type']:<14}"
              f"(conf {p['type_confidence']:.2f})   actually "
              f"{'correct' if e['correct'] else e['type']}")


if __name__ == "__main__":
    import ctc_train2 as _self
    _self.main()
