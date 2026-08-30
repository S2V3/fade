"""
fst_predictor.py -- question -> predicted failure type. Trains on the TRAIN run's
diagnoses; used at TEST time before the model has attempted anything.

GOLD RULE, enforced by construction
-----------------------------------
Features come from features.question_features(question_string). That function
takes a string and raises on anything else, so no gold-derived value can enter.
The TARGET is gold-dependent (the diagnosis needed the reference solution), which
is fine: gold may TRAIN, never PREDICT.

WHAT THE TRAINING DATA SAYS -- read before trusting a prediction
----------------------------------------------------------------
On the 1,500-problem train run (924 wrong+typed traces, majority class 27.5%):

    hand-made features + LR      0.208     word TF-IDF + LR    0.219
    char TF-IDF + LR             0.233     TF-IDF + hand + LR  0.239
    hand + GradientBoosting      0.262     hand + RandomForest 0.264

NOT ONE BEATS MAJORITY. Per-type one-vs-rest AUROC clears 0.60 for only NR (0.615)
and AL (0.618); ST, SM and CE sit at chance. `fit()` reports this comparison every
time and refuses to claim success it has not earned.

CONFIDENCE GATING
-----------------
Accuracy DOES rise with the model's own confidence -- top-20% 0.332, top-5% 0.391
against 0.261 overall -- so `predict()` can abstain below a threshold and fall back
to the generic cure. Measured trade-off: gating raises precision on what it routes
but cuts coverage faster, so expected recovery FALLS (route-all +2.5 pts vs
route-top-20% +0.6). Default MIN_CONFIDENCE = 0.0 (route everything); raise it only
if you have measured that a mismatched cure is actively harmful.

USAGE
    python fst_predictor.py --train store_typed_train/results.jsonl --out fst.joblib
    python fst_predictor.py --train ... --out fst.joblib --model rf --min-confidence 0.35
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

TYPED = ("NR", "AL", "ST", "SM", "CE", "WP")


class FSTPredictor:
    def __init__(self, model="rf", min_confidence=0.0):
        self.model_name = model
        self.min_confidence = float(min_confidence)
        self.clf = None
        self.names = None
        self.classes_ = None
        self.train_report = {}

    # ------------------------------------------------------------------ fit
    def fit(self, questions, types, verbose=True):
        from features import question_features, vectorize
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold

        X, self.names = vectorize([question_features(q) for q in questions])
        X = np.asarray(X, float)
        y = np.asarray(types)
        maj = Counter(y).most_common(1)[0][1] / len(y)

        self.clf = (RandomForestClassifier(500, random_state=0, class_weight="balanced")
                    if self.model_name == "rf"
                    else LogisticRegression(max_iter=6000, class_weight="balanced"))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = cross_val_score(self.clf, X, y, cv=cv).mean()
        self.clf.fit(X, y)
        self.classes_ = self.clf.classes_
        self.train_report = {"n": int(len(y)), "majority": float(maj),
                             "cv_accuracy": float(acc), "lift": float(acc - maj),
                             "classes": dict(Counter(y))}
        if verbose:
            print(f"  trained on {len(y)} traces | classes {dict(Counter(y).most_common())}")
            print(f"  majority baseline {maj:.3f} | 5-fold CV accuracy {acc:.3f}"
                  f" | lift {acc-maj:+.3f}")
            if acc <= maj:
                print("  !! THE PREDICTOR DOES NOT BEAT THE MAJORITY BASELINE.")
                print("     Routing on it applies a MISMATCHED cure most of the time.")
                print("     Report this; do not present the predictor as working.")
        return self

    # -------------------------------------------------------------- predict
    def predict(self, question):
        """-> (type_or_None, confidence). None means abstain -> generic cure."""
        from features import question_features, vectorize
        X, _ = vectorize([question_features(question)])
        X = np.asarray(X, float)
        # vectorize() one-hots per call, so column sets can differ from training
        X = self._align(X, _)
        proba = self.clf.predict_proba(X)[0]
        i = int(proba.argmax())
        conf = float(proba[i])
        if conf < self.min_confidence:
            return None, conf
        return str(self.classes_[i]), conf

    def _align(self, X, names):
        """Pad/reorder a single-row feature matrix to the training column set."""
        if names == self.names:
            return X
        out = np.zeros((X.shape[0], len(self.names)))
        idx = {n: j for j, n in enumerate(names)}
        for j, n in enumerate(self.names):
            if n in idx:
                out[:, j] = X[:, idx[n]]
        return out

    # ------------------------------------------------------------- persist
    def save(self, path):
        import joblib
        joblib.dump({"clf": self.clf, "names": self.names, "classes": self.classes_,
                     "min_confidence": self.min_confidence,
                     "model_name": self.model_name, "report": self.train_report}, path)

    @staticmethod
    def load(path):
        import joblib
        b = joblib.load(path)
        p = FSTPredictor(b["model_name"], b["min_confidence"])
        p.clf, p.names, p.classes_ = b["clf"], b["names"], b["classes"]
        p.train_report = b.get("report", {})
        return p


def load_training(results_path):
    qs, ts = [], []
    for ln in open(results_path):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("phase") != "pass1":
            continue
        if (r.get("signals") or {}).get("correct"):
            continue
        if r.get("diagnosis") in TYPED:
            qs.append(r["question"])
            ts.append(r["diagnosis"])
    return qs, ts


def main():
    ap = argparse.ArgumentParser(description="train the FST failure-type predictor")
    ap.add_argument("--train", required=True, help="a TRAIN-split results.jsonl")
    ap.add_argument("--out", default="fst.joblib")
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()

    qs, ts = load_training(a.train)
    print(f"FST predictor  ({a.model}, min_confidence={a.min_confidence})")
    if len(qs) < 300:
        print(f"  !! only {len(qs)} training traces. Below ~300 this is unstable.")
    p = FSTPredictor(a.model, a.min_confidence).fit(qs, ts)
    p.save(a.out)
    print(f"  saved -> {a.out}")
    print("\n  sanity check on 3 training questions:")
    for q in qs[:3]:
        t, c = p.predict(q)
        print(f"    conf {c:.2f}  ->  {t}   | {q[:58]}...")


if __name__ == "__main__":
    main()
