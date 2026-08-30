"""
fst_predictor.py -- question -> failure risk / severity / type, trained on the
MEDIUM+BAD pool. No GPU: this whole file runs on a CPU Kaggle session.

WHAT CHANGED AND WHY (read this, it reframes FST)
-------------------------------------------------
The first version trained one 6-way type head on the 924 wrong+TYPED traces. That
threw away three things the train run actually gives us:

  * the 144 wrong traces diagnosed TR or UNCLASSIFIED  -> pool is 1068, not 924
  * the MEDIUM_WRONG / BAD_WRONG severity label        -> recovery 21.8% vs 14.6%
  * the 432 CORRECT traces                             -> the "will it fail?" signal

and it created a train/serve mismatch: at test time we predict a type for EVERY
question, including ones the model would have answered correctly, but the head had
never seen a "no failure" class.

THREE TARGETS, all trainable from one train-split results.jsonl:

  --target fail       will pass-1 be wrong?      n=1500  base rate 71.2%
  --target severity   MEDIUM_WRONG vs BAD_WRONG  n=1068  base rate 79.4%
  --target type       8-way, full medium+bad     n=1068  majority   23.8%
  --target recover    will the retry RESCUE it?  n=1068  base rate 16.1%

MEASURED ON THE 1,500-PROBLEM TRAIN RUN
---------------------------------------
                                accuracy   vs majority    AUROC
  fail   (tfidf+hand, RF)         0.709      -0.003        0.665   <- real signal
  fail   (tfidf+hand, LR)         0.658      -0.054        0.668
  sever. (tfidf+hand, RF)         0.795      +0.001        0.599
  type   8-way full pool, RF      0.254      +0.016          --    <- first config
  type   6-way typed-only, RF     0.264      -0.011          --       to beat majority
  recov. (tfidf+hand, RF)         --           --           0.561   <- the one that
  recov. + gold-free signals      --           --           0.595      should be ranked

Accuracy is the WRONG metric for `fail` and `severity` -- a 71/29 and an 80/20 split
make the majority baseline nearly unbeatable on accuracy while AUROC 0.67 and 0.60
show the question does carry information. Read AUROC for those two; read accuracy
for `type`.

Note the direction flip on `type`: training on the FULL medium+bad pool (+0.012)
beats training on typed-only (-0.011). Including TR and UNCLASSIFIED as real classes
helps -- an abstain that the model can PREDICT is better than an abstain it has to
infer from low confidence.

THE FINDING THAT MATTERS MOST -- see fst_selection.py
-----------------------------------------------------
The `fail` gate ranks FAILURE well (AUROC 0.664; precision 90.7% in the top 5%
against a 71.2% base rate) but ranks RECOVERABILITY among failures at 0.433 --
below chance. Spending a fixed cure budget on the riskiest questions recovers
slightly LESS than spending it at random, because the riskiest questions are the
hardest and the hardest are the least rescuable.

So `fail` is the wrong quantity to rank. Ranking `recover` DIRECTLY works better
(AUROC 0.561 from the question alone, 0.595 once gold-free trace signals are
available at retry time). Weak, but it is the head to spend a budget on -- and it
is the honest form of the FST idea: not "predict the failure" but "predict which
failures a cure can actually rescue".

GOLD RULE: features read the question STRING only. Targets are gold-derived, which
is fine -- gold may TRAIN, never PREDICT.

USAGE
    python fst_predictor.py --train store/results.jsonl --target fail     --out fst_gate.joblib
    python fst_predictor.py --train store/results.jsonl --target recover  --out fst_recover.joblib
    python fst_predictor.py --train store/results.jsonl --target type     --out fst_type.joblib
    python fst_predictor.py --train store/results.jsonl --target severity --out fst_sev.joblib
    python fst_predictor.py --train store/results.jsonl --target all      --out-dir stores/
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

TYPED = ("NR", "AL", "ST", "SM", "CE", "WP")
ALL_TYPES = TYPED + ("TR", "UNCLASSIFIED")
TARGETS = ("fail", "severity", "type", "recover")


# ------------------------------------------------------------------ features
class _Featurizer:
    """Question string -> matrix. Fitted vectorizers are PERSISTED, so predict()
    produces the same columns training saw. The old version re-derived one-hot
    columns per call and papered over the drift with _align()."""

    def __init__(self, kind="both"):
        assert kind in ("hand", "tfidf", "both")
        self.kind = kind
        self.tfidf = None
        self.scaler = None
        self.names = None

    def _hand(self, questions):
        from features import question_features, vectorize
        X, names = vectorize([question_features(q) for q in questions])
        return np.asarray(X, float), names

    def fit_transform(self, questions):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from scipy.sparse import hstack, csr_matrix

        if self.kind in ("hand", "both"):
            Xh, self.names = self._hand(questions)
        if self.kind == "hand":
            return Xh
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xw = self.tfidf.fit_transform(questions)
        if self.kind == "tfidf":
            return Xw
        self.scaler = StandardScaler().fit(Xh)
        return hstack([Xw, csr_matrix(self.scaler.transform(Xh))]).tocsr()

    def transform(self, questions):
        from scipy.sparse import hstack, csr_matrix

        if self.kind in ("hand", "both"):
            Xh, names = self._hand(questions)
            Xh = self._align(Xh, names)
        if self.kind == "hand":
            return Xh
        Xw = self.tfidf.transform(questions)
        if self.kind == "tfidf":
            return Xw
        return hstack([Xw, csr_matrix(self.scaler.transform(Xh))]).tocsr()

    def _align(self, X, names):
        if names == self.names:
            return X
        out = np.zeros((X.shape[0], len(self.names)))
        idx = {n: j for j, n in enumerate(names)}
        for j, n in enumerate(self.names):
            if n in idx:
                out[:, j] = X[:, idx[n]]
        return out


# ------------------------------------------------------------------ predictor
class FSTPredictor:
    def __init__(self, model="rf", target="type", features="both", min_confidence=0.0):
        self.model_name = model
        self.target = target
        self.features = features
        self.min_confidence = float(min_confidence)
        self.fz = None
        self.clf = None
        self.classes_ = None
        self.positive_ = None      # the "bad news" class, for risk()
        self.train_report = {}

    def _make(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        if self.model_name == "rf":
            return RandomForestClassifier(500, random_state=0, class_weight="balanced")
        return LogisticRegression(max_iter=6000, class_weight="balanced")

    def fit(self, questions, labels, verbose=True):
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score

        self.fz = _Featurizer(self.features)
        X = self.fz.fit_transform(questions)
        y = np.asarray(labels)
        counts = Counter(y)
        maj = counts.most_common(1)[0][1] / len(y)
        binary = len(counts) == 2

        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = float(cross_val_score(self._make(), X, y, cv=cv).mean())
        auroc = None
        if binary:
            self.positive_ = counts.most_common()[-1][0]      # minority = bad news
            if self.target == "fail":
                self.positive_ = "WRONG"
            elif self.target == "recover":
                self.positive_ = "RESCUED"
            p = cross_val_predict(self._make(), X, y, cv=cv, method="predict_proba")
            pos_col = list(sorted(counts)).index(self.positive_)
            auroc = float(roc_auc_score((y == self.positive_).astype(int), p[:, pos_col]))

        self.clf = self._make().fit(X, y)
        self.classes_ = self.clf.classes_
        self.train_report = {
            "target": self.target, "features": self.features, "model": self.model_name,
            "n": int(len(y)), "majority": float(maj), "cv_accuracy": acc,
            "lift": float(acc - maj), "auroc": auroc,
            "classes": {str(k): int(v) for k, v in counts.items()},
        }

        if verbose:
            print(f"  target={self.target}  features={self.features}  model={self.model_name}")
            print(f"  n={len(y)}  classes={dict(counts.most_common())}")
            print(f"  majority {maj:.3f} | 5-fold CV accuracy {acc:.3f} | lift {acc-maj:+.3f}"
                  + (f" | AUROC {auroc:.3f}" if auroc is not None else ""))
            self._verdict(acc, maj, auroc, binary)
        return self

    def _verdict(self, acc, maj, auroc, binary):
        if binary:
            if auroc is None or auroc < 0.55:
                print("  !! AUROC at or near chance -- this head carries no usable signal.")
            elif auroc < 0.70:
                print(f"  ~  AUROC {auroc:.3f}: real but weak ranking signal. Usable for")
                print("     RANKING (who to spend budget on), not for a hard decision.")
                print("     Note accuracy loses to majority; on a skewed target that is")
                print("     expected and is not the metric to report.")
            else:
                print(f"  OK AUROC {auroc:.3f} -- usable.")
        else:
            if acc <= maj:
                print("  !! DOES NOT BEAT THE MAJORITY BASELINE. Routing on this head")
                print("     applies a mismatched cure most of the time. Report it; do")
                print("     not present the predictor as working.")
            else:
                print(f"  OK beats majority by {acc-maj:+.3f} (thin -- confirm on held-out data).")

    # ------------------------------------------------------------- inference
    def predict(self, question):
        """-> (label_or_None, confidence). None = abstain -> generic cure."""
        proba = self.clf.predict_proba(self.fz.transform([question]))[0]
        i = int(proba.argmax())
        conf = float(proba[i])
        if conf < self.min_confidence:
            return None, conf
        return str(self.classes_[i]), conf

    def risk(self, question):
        """-> P(bad news). Binary heads only. This is the number to RANK on."""
        if self.positive_ is None:
            raise ValueError(f"risk() is for binary targets; this head is '{self.target}'")
        proba = self.clf.predict_proba(self.fz.transform([question]))[0]
        return float(proba[list(self.classes_).index(self.positive_)])

    def risk_batch(self, questions):
        if self.positive_ is None:
            raise ValueError(f"risk() is for binary targets; this head is '{self.target}'")
        P = self.clf.predict_proba(self.fz.transform(list(questions)))
        return P[:, list(self.classes_).index(self.positive_)]

    # --------------------------------------------------------------- persist
    def save(self, path):
        import joblib
        joblib.dump({"fz": self.fz, "clf": self.clf, "classes": self.classes_,
                     "positive": self.positive_, "target": self.target,
                     "features": self.features, "model_name": self.model_name,
                     "min_confidence": self.min_confidence,
                     "report": self.train_report}, path)

    @staticmethod
    def load(path):
        import joblib
        b = joblib.load(path)
        p = FSTPredictor(b["model_name"], b.get("target", "type"),
                         b.get("features", "both"), b["min_confidence"])
        p.fz, p.clf, p.classes_ = b["fz"], b["clf"], b["classes"]
        p.positive_ = b.get("positive")
        p.train_report = b.get("report", {})
        return p


# --------------------------------------------------------------------- data
def load_training(results_path, target):
    """The medium+bad pool (or all pass-1, for the fail gate)."""
    p1 = []
    for ln in open(results_path):
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("phase") == "pass1":
            p1.append(r)
    if not p1:
        raise SystemExit(f"no pass1 records in {results_path}")

    if target == "fail":
        qs = [r["question"] for r in p1]
        ys = ["RIGHT" if (r.get("signals") or {}).get("correct") else "WRONG" for r in p1]
        return qs, ys

    if target == "recover":
        rescued = set()
        for ln in open(results_path):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("phase") == "retry" and r.get("newly_correct"):
                rescued.add(r["id"])
        pool = [r for r in p1 if not (r.get("signals") or {}).get("correct")]
        if not rescued:
            raise SystemExit("no recoveries in this store -- cannot train `recover`")
        return ([r["question"] for r in pool],
                ["RESCUED" if r["id"] in rescued else "STUCK" for r in pool])

    pool = [r for r in p1 if not (r.get("signals") or {}).get("correct")]
    bad = [r for r in pool if r.get("label") not in ("MEDIUM_WRONG", "BAD_WRONG")]
    if bad:
        print(f"  note: {len(bad)} wrong traces carry an unexpected label "
              f"{Counter(r.get('label') for r in bad)} -- kept for `type`, dropped for `severity`")
    if target == "severity":
        pool = [r for r in pool if r.get("label") in ("MEDIUM_WRONG", "BAD_WRONG")]
        return [r["question"] for r in pool], [r["label"] for r in pool]
    return ([r["question"] for r in pool],
            [(r.get("diagnosis") or "UNCLASSIFIED") for r in pool])


def train_one(results, target, out, model, features, min_conf):
    qs, ys = load_training(results, target)
    print(f"\n{'-'*70}\nTRAIN · {target}")
    if len(qs) < 200:
        print(f"  !! only {len(qs)} rows -- unstable below ~300.")
    p = FSTPredictor(model, target, features, min_conf).fit(qs, ys)
    p.save(out)
    print(f"  saved -> {out}")
    return p


def main():
    ap = argparse.ArgumentParser(description="train FST heads on the medium+bad pool")
    ap.add_argument("--train", required=True, help="a TRAIN-split results.jsonl")
    ap.add_argument("--target", choices=TARGETS + ("all",), default="all")
    ap.add_argument("--out", help="output .joblib (single target)")
    ap.add_argument("--out-dir", default=".", help="output dir when --target all")
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--features", choices=["hand", "tfidf", "both"], default="both")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    a = ap.parse_args()

    print("=" * 70)
    print("  FST HEADS -- trained on the MEDIUM+BAD pool.  No GPU needed.")
    print("=" * 70)

    if a.target == "all":
        import os
        os.makedirs(a.out_dir, exist_ok=True)
        made = {}
        for t in TARGETS:
            made[t] = train_one(a.train, t, os.path.join(a.out_dir, f"fst_{t}.joblib"),
                                a.model, a.features, a.min_confidence)
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  {'head':<10}{'n':>6}{'majority':>10}{'CV acc':>9}{'lift':>8}{'AUROC':>8}")
        for t, p in made.items():
            r = p.train_report
            au = f"{r['auroc']:.3f}" if r.get("auroc") is not None else "  --"
            print(f"  {t:<10}{r['n']:>6}{r['majority']:>10.3f}{r['cv_accuracy']:>9.3f}"
                  f"{r['lift']:>+8.3f}{au:>8}")
        print("\n  Read AUROC for `fail`, `severity` and `recover` (skewed targets make")
        print("  accuracy uninformative there); read accuracy vs majority for `type`.")
        print("  `recover` is the head worth spending a budget on -- see fst_selection.py.")
        print("  Next: python fst_selection.py --results <results.jsonl>")
    else:
        train_one(a.train, a.target, a.out or f"fst_{a.target}.joblib",
                  a.model, a.features, a.min_confidence)


if __name__ == "__main__":
    # Re-import under the real module name before doing anything. Run as a script,
    # this file is `__main__`, so joblib would pickle _Featurizer as
    # `__main__._Featurizer` and every later `FSTPredictor.load()` from an importing
    # process would die with "Can't get attribute '_Featurizer' on __main__".
    import fst_predictor as _self
    _self.main()
