"""
FST GO/NO-GO GATE  --  run this BEFORE building any predictor.

Answers the one make-or-break question of Phase 3: are the frozen model's
FAILURE TYPES predictable from GOLD-FREE features of the question? It also
produces the descriptive table you actually want first -- "how many errors of
what kind, on what kind of question" -- because that table IS the gate's
evidence.

Reads a pass-1 `results.jsonl` (produced by a `--split train` run of
kaggle_run.py). No GPU, no model. Pure analysis.

What it prints:
  1. Pass-1 accuracy + label mix (GOOD / MEDIUM / BAD) + diagnosis histogram,
     including the UNCLASSIFIED (abstain) rate.
  2. The (question category x failure type) contingency table.
  3. The statistical test: chi-square + Cramer's V on that table
     -> is failure TYPE associated with question category?
  4. Baselines to beat: overall majority-type, and category->majority-type
     lookup. Optionally a logistic-regression probe on the full gold-free
     feature set (sklearn, stratified CV) vs majority.
  5. A binary side-check: is FAILURE ITSELF (correct vs wrong) associated with
     category?
  6. A GO / WEAK / NO-GO verdict, with caveats.

Targets (per the agreed framing):
  * "succeeds" = correct answer, "failed" = wrong answer  (binary, side-check).
  * failure TYPE target = {NR, ST, SM, CE} on WRONG traces only
    (UNCLASSIFIED is ABSTAIN, not a type -> excluded from the target).

CAVEATS this script prints for you, because a reviewer will ask:
  - PRELIMINARY while E/A are value-based (structural E/A not yet in): the
    diagnosis labels may shift, so re-run the gate after that change.
  - Sparse cells (expected < 5) make chi-square unreliable -> merge rare
    categories or gather more data; the script flags this.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

TYPED = ("NR", "ST", "SM", "CE")           # real failure types (targets)
ABSTAIN = "UNCLASSIFIED"                    # not a type

# ----------------------------------------------------------------- loading
def load_pass1(path: str, min_confidence: float = 0.0) -> list[dict]:
    """Return one row per pass-1 attempt with the fields the gate needs.
    Rows whose typed diagnosis has confidence < min_confidence are treated as
    abstain (their diagnosis is blanked to UNCLASSIFIED)."""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("phase") != "pass1":
                continue
            sig = r.get("signals") or {}
            cat = (r.get("category") or {}).get("category_type", "unknown")
            diag = r.get("diagnosis")
            conf = r.get("diagnosis_confidence")
            if (min_confidence > 0.0 and diag in TYPED
                    and conf is not None and conf < min_confidence):
                diag = ABSTAIN  # low-confidence typed label -> abstain
            rows.append({
                "id": r.get("id"),
                "question": r.get("question", ""),
                "category": cat,
                "correct": bool(sig.get("correct", False)),
                "label": r.get("label"),
                "diagnosis": diag,
                "confidence": conf,
            })
    return rows


# ----------------------------------------------------------------- stats
def chi_square(table: dict[str, Counter], cols: list[str]):
    """Chi-square test of independence on a rows x cols contingency table.
    Returns (chi2, dof, p, cramers_v, min_expected). Uses scipy if present,
    else a pure-python chi2 with a survival-function approximation."""
    rows = list(table.keys())
    obs = [[table[r].get(c, 0) for c in cols] for r in rows]
    row_tot = [sum(rr) for rr in obs]
    col_tot = [sum(obs[i][j] for i in range(len(rows))) for j in range(len(cols))]
    n = sum(row_tot)
    if n == 0 or len(rows) < 2 or len(cols) < 2:
        return 0.0, 0, 1.0, 0.0, 0.0
    exp = [[row_tot[i] * col_tot[j] / n for j in range(len(cols))]
           for i in range(len(rows))]
    chi2 = 0.0
    min_exp = float("inf")
    for i in range(len(rows)):
        for j in range(len(cols)):
            e = exp[i][j]
            min_exp = min(min_exp, e)
            if e > 0:
                chi2 += (obs[i][j] - e) ** 2 / e
    dof = (len(rows) - 1) * (len(cols) - 1)
    k = min(len(rows), len(cols))
    cramers_v = math.sqrt(chi2 / (n * (k - 1))) if n and k > 1 else 0.0
    try:
        from scipy.stats import chi2 as _c
        p = float(_c.sf(chi2, dof))
    except Exception:
        p = _chi2_sf(chi2, dof)
    return chi2, dof, p, cramers_v, min_exp


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square (regularized upper incomplete gamma),
    pure-python fallback when scipy is absent."""
    if x <= 0 or k <= 0:
        return 1.0
    a = k / 2.0
    xx = x / 2.0
    # series for lower incomplete gamma P(a,xx); Q = 1 - P
    term = 1.0 / a
    summ = term
    for n_ in range(1, 500):
        term *= xx / (a + n_)
        summ += term
        if term < 1e-12 * summ:
            break
    logP = -xx + a * math.log(xx) - math.lgamma(a) + math.log(summ)
    P = math.exp(logP)
    return max(0.0, min(1.0, 1.0 - P))


# ----------------------------------------------------------------- printing
def _print_table(title, table, cols, row_order=None):
    rows = row_order or list(table.keys())
    w = max([len(title)] + [len(r) for r in rows]) + 2
    header = " " * w + "".join(f"{c:>8}" for c in cols) + f"{'TOT':>8}"
    print(title)
    print(header)
    col_tot = Counter()
    for r in rows:
        line = f"{r:<{w}}"
        rt = 0
        for c in cols:
            v = table[r].get(c, 0)
            rt += v
            col_tot[c] += v
            line += f"{v:>8}"
        line += f"{rt:>8}"
        print(line)
    foot = f"{'TOT':<{w}}" + "".join(f"{col_tot[c]:>8}" for c in cols)
    foot += f"{sum(col_tot.values()):>8}"
    print(foot)


# ----------------------------------------------------------------- gate
def run_gate(rows: list[dict], use_features: bool = True):
    n = len(rows)
    if n == 0:
        print("No pass-1 rows found. Did you point --results at a train run?")
        return
    n_correct = sum(r["correct"] for r in rows)
    print("=" * 68)
    print("FST GATE  --  pass-1 summary")
    print("=" * 68)
    print(f"problems (pass-1 rows): {n}")
    print(f"pass-1 accuracy       : {n_correct}/{n} = {n_correct/n:.3f}")

    labels = Counter(r["label"] for r in rows)
    print("\nlabel mix:")
    for lab, c in labels.most_common():
        print(f"  {lab:<16} {c:>5}  ({c/n:.1%})")

    diag_all = Counter(r["diagnosis"] for r in rows if r["diagnosis"] is not None)
    n_diag = sum(diag_all.values())
    print("\ndiagnosis histogram (all non-GOOD):")
    for d, c in diag_all.most_common():
        print(f"  {d:<16} {c:>5}  ({c/max(1,n_diag):.1%})")
    unc = diag_all.get(ABSTAIN, 0)
    print(f"\nUNCLASSIFIED (abstain) rate: {unc}/{n_diag} = "
          f"{unc/max(1,n_diag):.1%}  <-- large (>~20%) means the taxonomy is "
          f"leaking signal; re-check after structural E/A")

    # ---- failure TYPE target: wrong traces with a typed diagnosis ----
    fail_rows = [r for r in rows if not r["correct"] and r["diagnosis"] in TYPED]
    print("\n" + "=" * 68)
    print(f"FAILURE-TYPE TARGET  (wrong traces, typed) : {len(fail_rows)} rows")
    print("=" * 68)
    if len(fail_rows) < 30:
        print("  !! very few typed failures -- results below are unstable; "
              "gather more train data before trusting the verdict.")

    cats = sorted({r["category"] for r in fail_rows})
    present_types = [t for t in TYPED
                     if any(r["diagnosis"] == t for r in fail_rows)]
    table = {c: Counter() for c in cats}
    for r in fail_rows:
        table[r["category"]][r["diagnosis"]] += 1
    print()
    _print_table("(category x failure-type)", table, present_types, cats)

    chi2, dof, p, v, min_exp = chi_square(table, present_types)
    print(f"\nchi-square = {chi2:.2f}   dof = {dof}   p = {p:.4g}")
    print(f"Cramer's V = {v:.3f}   (0=none, .1 small, .3 medium, .5 large)")
    if min_exp < 5:
        print(f"  !! smallest expected cell = {min_exp:.1f} (<5): chi-square "
              f"unreliable -> merge rare categories or gather more data.")

    # ---- baselines to beat ----
    type_counts = Counter(r["diagnosis"] for r in fail_rows)
    maj_type, maj_n = (type_counts.most_common(1)[0] if type_counts else ("-", 0))
    maj_acc = maj_n / max(1, len(fail_rows))
    # category -> majority type lookup, accuracy on the same rows
    cat_major = {c: table[c].most_common(1)[0][0] for c in cats if table[c]}
    look_correct = sum(1 for r in fail_rows
                       if cat_major.get(r["category"]) == r["diagnosis"])
    look_acc = look_correct / max(1, len(fail_rows))
    print(f"\nbaseline  majority-type ('{maj_type}')      : {maj_acc:.3f}")
    print(f"baseline  category->majority-type lookup : {look_acc:.3f}"
          f"   (lift {look_acc-maj_acc:+.3f})")

    lr_acc = None
    if use_features:
        lr_acc = _feature_probe(fail_rows)

    # ---- binary side-check: is failing-at-all tied to category? ----
    print("\n" + "-" * 68)
    print("SIDE-CHECK: correct-vs-wrong by category")
    btable = {c: Counter() for c in sorted({r["category"] for r in rows})}
    for r in rows:
        btable[r["category"]]["correct" if r["correct"] else "wrong"] += 1
    _print_table("(category x outcome)", btable, ["correct", "wrong"])
    bchi2, bdof, bp, bv, _ = chi_square(btable, ["correct", "wrong"])
    print(f"chi-square = {bchi2:.2f}  p = {bp:.4g}  Cramer's V = {bv:.3f}")

    # ---- verdict ----
    _verdict(p, v, maj_acc, look_acc, lr_acc, len(fail_rows))


def _feature_probe(fail_rows):
    """Optional: can a simple logistic regression on the FULL gold-free
    feature set beat majority (stratified CV)? Returns mean acc or None."""
    try:
        from features import question_features, vectorize
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        import numpy as np
    except Exception as e:
        print(f"\n(feature probe skipped: {e})")
        return None
    y = [r["diagnosis"] for r in fail_rows]
    if len(set(y)) < 2:
        print("\n(feature probe skipped: only one failure type present)")
        return None
    feats = [question_features(r["question"]) for r in fail_rows]
    X, names = vectorize(feats)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    # class counts must allow k folds
    k = min(5, min(Counter(y).values()))
    if k < 2:
        print("\n(feature probe skipped: a class has <2 members)")
        return None
    accs = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    m, s = float(np.mean(accs)), float(np.std(accs))
    print(f"\nfeature probe  logistic-regression ({k}-fold CV): "
          f"{m:.3f} +/- {s:.3f}   ({len(names)} features)")
    return m


def _verdict(p, v, maj_acc, look_acc, lr_acc, n_fail):
    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    beat = max(look_acc, lr_acc or 0.0)
    lift = beat - maj_acc
    significant = p < 0.05
    effect = v >= 0.20
    beats_majority = lift >= 0.05
    if n_fail < 30:
        print("INCONCLUSIVE -- too few typed failures. Gather more train data.")
    elif significant and effect and beats_majority:
        print("GO. Failure type is associated with the question "
              f"(p={p:.3g}, V={v:.2f}) AND a simple model beats majority "
              f"(+{lift:.3f}). Proceed to fst_predictor.")
    elif significant and (effect or beats_majority):
        print("WEAK / PROMISING. Some signal (p="
              f"{p:.3g}, V={v:.2f}, lift={lift:+.3f}) but not decisive. "
              "Try structural E/A, confidence filtering, or more data before "
              "committing.")
    else:
        print("NO-GO (so far). No usable association "
              f"(p={p:.3g}, V={v:.2f}, lift={lift:+.3f}). A frozen model's "
              "failure types may not be predictable from the question -- "
              "which is itself a publishable null. Re-check after structural "
              "E/A; if it holds, write the null.")
    print("\nCAVEATS: (1) PRELIMINARY while E/A are value-based -- re-run after "
          "structural E/A. (2) Small expected cells make chi-square shaky. "
          "(3) This tests predictability, not yet recovery lift -- the real "
          "payoff is measured later in run_predictive.py.")


def main():
    ap = argparse.ArgumentParser(description="FST go/no-go gate")
    ap.add_argument("--results", required=True,
                    help="path to a pass-1 results.jsonl (train split)")
    ap.add_argument("--min-confidence", type=float, default=0.0,
                    help="treat typed diagnoses below this confidence as abstain")
    ap.add_argument("--no-features", action="store_true",
                    help="skip the sklearn logistic-regression feature probe")
    a = ap.parse_args()
    rows = load_pass1(a.results, min_confidence=a.min_confidence)
    run_gate(rows, use_features=not a.no_features)


if __name__ == "__main__":
    main()