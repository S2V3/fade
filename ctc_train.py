"""
ctc_train.py -- train CTC from the finished train run. NO GPU. ~2 minutes.

WHAT CTC IS
-----------
Two heads over GOLD-FREE features of a (question, trace) pair:

    detect  : is this trace WRONG?           -> should it go back for a retry?
    type    : WHICH failure is it?           -> which cure to load

This is what FST was trying to be, except FST predicted from the QUESTION, before
the model had written anything. Measured on the same 1,500 problems, the difference
is not subtle:

                                    FST (question only)      CTC (question + trace)
    is it wrong?      AUROC              0.664                     0.717
    which type?       CV acc        0.254 vs 0.238 maj        0.614 vs 0.215 maj
                      lift               +0.016                    +0.399

Same data, same labels, same model class. The trace carries what the question does
not. That contrast is worth reporting in its own right: a frozen model's failure is
not predictable from the problem, but it is highly legible in its own output.

THE HONEST FRAMING OF THE TYPE HEAD
-----------------------------------
The `type` labels come from diagnosis.py's cascade, which itself reads gold (E, A,
misses). So the type head is a GOLD-FREE SURROGATE for a gold-dependent diagnostic,
distilled at ~69% fidelity -- not an independent oracle. Its ceiling is the
cascade's own quality, and it should be described that way in the paper. That is
still exactly what makes a test-time pipeline possible: on test there is no gold,
so a distilled diagnosis is the only diagnosis available.

Per-type recall of the distillation (5-fold CV, with retries):
    TR 100%   NR 89.2%   AL 87.8%   WP 67.8%   UNCL 59.1%   ST 52.2%
    SM 39.4%   CE 36.8%
TR, NR and AL are nearly solved -- they have crisp structural signatures. CE and SM
are the weak pair and they are also two of the three biggest classes, so the routing
will be right on the structural types and roughly a coin-flip on the semantic ones.
Say that plainly rather than quoting only the mean.

A CORRECTION, RECORDED SO IT IS NOT REPEATED
--------------------------------------------
The first probe of this idea reported AUROC 0.791 and type accuracy 0.691. Those
numbers were WRONG: the feature list included `n_checkpoints`, which comes from
extract_gold_checkpoints(gold_solution) and is therefore gold-derived. Removing it
gives the honest figures above. The conclusion is unchanged -- CTC beats FST by a
wide margin because the trace carries what the question does not -- but the margin
is +0.399 on the type head, not +0.453, and the detector is 0.717, not 0.791.

If you ever add a feature here, check its provenance in components.py first. Any
function whose signature takes `gold_solution` is disqualified, and so is anything
derived from one (comp_coverage divides by s_hat; n_checkpoints counts gold
checkpoints).

PASS-1 ONLY vs INCLUDING RETRIES  (measured both ways)
------------------------------------------------------
    pass-1 only  (1500 rows)   detect AUROC 0.723   type 0.575  lift +0.337
    with retries (2568 rows)   detect AUROC 0.717   type 0.614  lift +0.399

Retries are included by default: the type head gains 6 points of lift, the detector
is unchanged within noise, and retry traces are the only examples of what a trace
looks like AFTER a cure -- which is the distribution CTC meets on a second attempt.
`--pass1-only` reproduces the other row.

NOTE ON R
---------
R_score needs sentence-transformers. Without it R fails soft to 1.0 and the numbers
above are a slight UNDER-estimate. On Kaggle, with the package installed, R is real.

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
E, E_struct, A, A_struct, misses and comp_coverage all read the reference solution
(comp_coverage divides by s_hat, which is gold-derived). Training on the stored
`signals` dict wholesale would score beautifully on train and be unusable on test.
ctc_features.trace_features() recomputes only the gold-free half from the raw
question and trace, and was verified to reproduce the stored values exactly for
V, G, n_equations, bad_eqs, n_comp_steps, n_unresolved and n_symbolic.

USAGE
    python ctc_train.py --train stores/store_typed_train_audit2/results.jsonl \
                        --out-dir stores/ctc_audit2
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ctc_features import FEATURE_NAMES, trace_features, vectorize


class CTCModel:
    """detect + type, over gold-free (question, trace) features."""

    def __init__(self, model="rf"):
        self.model_name = model
        self.detect = None
        self.type = None
        self.names = list(FEATURE_NAMES)
        self.report = {}

    def _make(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        if self.model_name == "rf":
            return RandomForestClassifier(500, random_state=0, class_weight="balanced")
        return LogisticRegression(max_iter=6000, class_weight="balanced")

    def fit(self, feats, correct, types, verbose=True):
        from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
        from sklearn.metrics import roc_auc_score

        X, self.names = vectorize(feats, self.names)
        X = np.asarray(X, float)
        y = np.asarray([0 if c else 1 for c in correct])        # 1 = WRONG
        cv = StratifiedKFold(5, shuffle=True, random_state=0)

        # ---- head 1: is it wrong -----------------------------------------
        p = cross_val_predict(self._make(), X, y, cv=cv, method="predict_proba")[:, 1]
        auroc = float(roc_auc_score(y, p))
        acc = float(cross_val_score(self._make(), X, y, cv=cv).mean())
        maj = float(max(y.mean(), 1 - y.mean()))
        self.detect = self._make().fit(X, y)

        # ---- head 2: which type, among the wrong -------------------------
        m = y == 1
        Xw, yw = X[m], np.asarray(types)[m]
        tmaj = Counter(yw).most_common(1)[0][1] / len(yw)
        tacc = float(cross_val_score(self._make(), Xw, yw, cv=cv).mean())
        pw = cross_val_predict(self._make(), Xw, yw, cv=cv)
        recall = {c: float((pw[yw == c] == c).mean()) for c in sorted(set(yw))}
        self.type = self._make().fit(Xw, yw)

        self.report = {
            "n": int(len(y)), "n_wrong": int(m.sum()),
            "detect": {"auroc": auroc, "cv_accuracy": acc, "majority": maj,
                       "base_wrong_rate": float(y.mean())},
            "type": {"cv_accuracy": tacc, "majority": float(tmaj),
                     "lift": float(tacc - tmaj), "per_type_recall": recall,
                     "classes": {str(k): int(v) for k, v in Counter(yw).items()}},
            "features": self.names,
        }

        if verbose:
            print(f"\n  HEAD 1 · is this trace WRONG?")
            print(f"    n={len(y)}  base wrong rate {y.mean():.1%}")
            print(f"    AUROC {auroc:.3f} | CV acc {acc:.3f} vs majority {maj:.3f}")
            for q, lab in ((20, "flagged MOST likely wrong (bottom 20%)"),
                           (10, "bottom 10%")):
                th = np.percentile(p, 100 - q)
                sel = p >= th
                print(f"      {lab:<42} precision {y[sel].mean():>6.1%}  n={int(sel.sum())}")
            if auroc < 0.70:
                print("    !! AUROC below 0.70 -- weak for a gate. Report, do not deploy.")
            print(f"\n  HEAD 2 · WHICH failure, among the wrong?")
            print(f"    n={int(m.sum())}  majority {tmaj:.3f} | CV acc {tacc:.3f} "
                  f"| lift {tacc-tmaj:+.3f}")
            for c, r in sorted(recall.items(), key=lambda x: -x[1]):
                print(f"      {c:<14} recall {r:>6.1%}")
            if tacc <= tmaj + 0.05:
                print("    !! barely beats majority -- the cure routing will be noise.")
        return self

    # ------------------------------------------------------------ inference
    def predict(self, question, trace):
        """-> dict. GOLD-FREE: reads only the question and the model's own trace."""
        f = trace_features(question, trace)
        X = np.asarray(vectorize([f], self.names)[0], float)
        p_wrong = float(self.detect.predict_proba(X)[0][list(self.detect.classes_).index(1)])
        tp = self.type.predict_proba(X)[0]
        i = int(tp.argmax())
        return {"p_wrong": p_wrong,
                "predicted_type": str(self.type.classes_[i]),
                "type_confidence": float(tp[i]),
                "features": f}

    def save(self, path):
        import joblib
        joblib.dump({"detect": self.detect, "type": self.type, "names": self.names,
                     "model_name": self.model_name, "report": self.report}, path)

    @staticmethod
    def load(path):
        import joblib
        b = joblib.load(path)
        m = CTCModel(b["model_name"])
        m.detect, m.type, m.names = b["detect"], b["type"], b["names"]
        m.report = b.get("report", {})
        return m


def load_training(results_path, include_retries=True):
    """Every attempt in the run -- pass-1 AND retries.

    Retries are included on purpose. They are additional (question, trace,
    outcome) triples produced by the same model, they roughly double the training
    set, and they are the only examples of what a trace looks like AFTER a cure has
    been applied -- which is exactly the distribution CTC will meet at test time,
    where it runs on a second attempt too.
    """
    feats, correct, types, origin = [], [], [], []
    for ln in open(results_path):
        if not ln.strip():
            continue
        r = json.loads(ln)
        ph = r.get("phase")
        if ph == "pass1":
            q, tr = r.get("question"), r.get("trace")
            ok = bool((r.get("signals") or {}).get("correct"))
            ty = r.get("diagnosis") or "UNCLASSIFIED"
        elif ph == "retry" and include_retries:
            q, tr = r.get("question"), r.get("trace")
            ok = bool((r.get("signals") or {}).get("correct")
                      or r.get("newly_correct"))
            ty = r.get("diagnosis") or r.get("prev_diagnosis") or "UNCLASSIFIED"
        else:
            continue
        if not q or not tr:
            continue
        feats.append(trace_features(q, tr))
        correct.append(ok)
        types.append(ty)
        origin.append(ph)
    return feats, correct, types, origin


def main():
    ap = argparse.ArgumentParser(description="train CTC from a finished train run")
    ap.add_argument("--train", required=True, help="results.jsonl of the train run")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--pass1-only", action="store_true",
                    help="exclude retry traces from training")
    a = ap.parse_args()

    print("=" * 70)
    print("  CTC · training a gold-free failure detector from the train run")
    print("=" * 70)
    feats, correct, types, origin = load_training(a.train, not a.pass1_only)
    print(f"  training rows {len(feats)}   {dict(Counter(origin))}")
    print(f"  wrong {sum(1 for c in correct if not c)} | correct {sum(correct)}")
    print(f"  types {dict(Counter(t for t, c in zip(types, correct) if not c).most_common())}")
    if len(feats) < 400:
        print("  !! under 400 rows -- unstable.")

    m = CTCModel(a.model).fit(feats, correct, types)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    m.save(out / "ctc.joblib")
    (out / "ctc_report.json").write_text(json.dumps(m.report, indent=2))
    print(f"\n  saved -> {out/'ctc.joblib'}")

    print("\n  smoke test on 3 training questions (gold-free path):")
    import itertools
    for ln in itertools.islice((l for l in open(a.train) if l.strip()), 3):
        r = json.loads(ln)
        if r.get("phase") != "pass1":
            continue
        p = m.predict(r["question"], r["trace"])
        truth = "correct" if (r.get("signals") or {}).get("correct") else "WRONG"
        print(f"    p_wrong {p['p_wrong']:.2f} -> {p['predicted_type']:<14}"
              f"(conf {p['type_confidence']:.2f})   actually {truth}"
              f"  [cascade said {r.get('diagnosis')}]")


if __name__ == "__main__":
    import ctc_train as _self
    _self.main()
