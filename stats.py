"""
stats.py -- the statistics the paper needs, with no silent assumptions.

Nothing here touches the model or the gold rule; it consumes already-scored
records. Four families:

  1. PROPORTIONS      wilson_ci        -- CIs that behave at small n and near 0/1
                      diff_ci          -- CI on a difference of independent rates
  2. PAIRED TESTS     mcnemar          -- the RIGHT test for "same problems,
                                          two arms" (typed vs generic vs immediate)
  3. INDEPENDENT      fisher_exact_2x2 -- MEDIUM_WRONG vs BAD_WRONG recovery
  4. AGREEMENT        fleiss_kappa     -- is the failure TYPE stable across
                      cohen_kappa         re-samples of the SAME question?

Why (4) matters: FST's gate asks "is failure type predictable from the question?"
If the type is not even reproducible when you ask the SAME question twice, then no
question-only feature can predict it, and the gate's NO-GO has a measured
mechanism rather than being an unexplained negative. kappa is that ceiling.

Pure python + math; scipy is used when present but never required.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional, Sequence


# ============================================================================
# 1. PROPORTIONS
# ============================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k/n. Preferred over normal-approximation
    ('Wald') because Wald gives nonsense at small n or near 0/1 -- and FADE's
    recovery rates live near 0.05."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def diff_ci(k1: int, n1: int, k2: int, n2: int,
            z: float = 1.96) -> tuple[float, float, float]:
    """(difference, lo, hi) for two INDEPENDENT proportions, Newcombe's
    method (built from two Wilson intervals -- correct near the boundaries
    where the naive normal interval crosses 0/1)."""
    p1, p2 = (k1 / n1 if n1 else 0.0), (k2 / n2 if n2 else 0.0)
    l1, u1 = wilson_ci(k1, n1, z)
    l2, u2 = wilson_ci(k2, n2, z)
    d = p1 - p2
    lo = d - z * math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2) / z
    hi = d + z * math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2) / z
    return d, lo, hi


# ============================================================================
# 2. PAIRED TEST -- the one to use for arm comparisons
# ============================================================================
def mcnemar(a_correct: dict, b_correct: dict,
            exact: bool = True) -> dict:
    """Paired comparison of two arms run on the SAME problems.

    a_correct / b_correct: {problem_id: bool}. Only ids present in BOTH are used.

    Returns n01 (a wrong, b right), n10 (a right, b wrong), statistic, p, and the
    two arms' accuracies. Use this -- NOT a two-proportion z-test -- whenever the
    arms share problems, which every FADE ablation does. The unpaired test throws
    away the pairing and is anticonservative here.

    `exact` uses the binomial test (correct at any n); otherwise the
    continuity-corrected chi-square.
    """
    ids = sorted(set(a_correct) & set(b_correct))
    n01 = sum(1 for i in ids if not a_correct[i] and b_correct[i])
    n10 = sum(1 for i in ids if a_correct[i] and not b_correct[i])
    n = n01 + n10
    if n == 0:
        p, stat = 1.0, 0.0
    elif exact:
        # two-sided exact binomial with p=0.5
        tail = sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1))
        p = min(1.0, 2.0 * tail / (2 ** n))
        stat = float(min(n01, n10))
    else:
        stat = (abs(n01 - n10) - 1) ** 2 / n
        p = _chi2_sf(stat, 1)
    return {
        "n_paired": len(ids), "n01": n01, "n10": n10,
        "statistic": stat, "p": p,
        "acc_a": sum(a_correct[i] for i in ids) / max(1, len(ids)),
        "acc_b": sum(b_correct[i] for i in ids) / max(1, len(ids)),
    }


# ============================================================================
# 3. INDEPENDENT 2x2
# ============================================================================
def fisher_exact_2x2(table: Sequence[Sequence[int]]) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]. Exact at any cell count,
    which matters because FADE's typed cells get small fast."""
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact(table)[1])
    except Exception:
        pass
    (a, b), (c, d) = table
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def hyp(x):
        return (math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1))

    lo, hi = max(0, c1 - r2), min(r1, c1)
    obs = hyp(a)
    return min(1.0, sum(hyp(x) for x in range(lo, hi + 1)
                        if hyp(x) <= obs * (1 + 1e-9)))


def _chi2_sf(x: float, k: int) -> float:
    """Chi-square survival function; scipy when available, series otherwise."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, k))
    except Exception:
        pass
    if x <= 0 or k <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    term = 1.0 / a
    summ = term
    for i in range(1, 500):
        term *= xx / (a + i)
        summ += term
        if term < 1e-12 * summ:
            break
    logP = -xx + a * math.log(xx) - math.lgamma(a) + math.log(summ)
    return max(0.0, min(1.0, 1.0 - math.exp(logP)))


# ============================================================================
# 4. AGREEMENT -- the reliability ceiling for FST
# ============================================================================
def fleiss_kappa(ratings: Sequence[Sequence[str]],
                 categories: Optional[Sequence[str]] = None) -> dict:
    """Fleiss' kappa over `ratings`, one inner list per ITEM holding that item's
    category labels from k interchangeable raters.

    Here an "item" is a QUESTION and a "rater" is one re-sample of the frozen
    model. That is exactly Fleiss' setting: raters are not identified across
    items, only counted -- so independent samples qualify.

    Handles a VARYING number of raters per item (the generalised form), which is
    needed when you restrict to, say, only the wrong samples of each question.
    Items with fewer than 2 ratings carry no agreement information and are
    dropped.

    Interpretation (Landis & Koch): <0 none, 0-.20 slight, .21-.40 fair,
    .41-.60 moderate, .61-.80 substantial, >.80 almost perfect.

    A kappa near 0 means the failure type is NOT a reproducible property of the
    question -- which upper-bounds any question-only predictor and explains an
    FST gate NO-GO rather than leaving it unexplained.
    """
    items = [list(r) for r in ratings if len(r) >= 2]
    if not items:
        return {"kappa": float("nan"), "n_items": 0, "note": "no item had >=2 ratings"}
    cats = list(categories) if categories else sorted({c for it in items for c in it})
    if len(cats) < 2:
        return {"kappa": float("nan"), "n_items": len(items),
                "note": f"only one category present ({cats}) -- kappa undefined"}

    # P_i: observed agreement within item i;  p_j: marginal share of category j
    P_bar = 0.0
    total_ratings = 0
    cat_tot = Counter()
    for it in items:
        n_i = len(it)
        counts = Counter(it)
        cat_tot.update(counts)
        total_ratings += n_i
        P_bar += (sum(c * c for c in counts.values()) - n_i) / (n_i * (n_i - 1))
    P_bar /= len(items)
    p = {j: cat_tot[j] / total_ratings for j in cats}
    P_e = sum(v * v for v in p.values())
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-12 else float("nan")
    return {
        "kappa": kappa, "P_observed": P_bar, "P_expected": P_e,
        "n_items": len(items), "n_ratings": total_ratings,
        "categories": cats, "marginals": {j: round(p[j], 4) for j in cats},
        "interpretation": _kappa_label(kappa),
    }


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> dict:
    """Two paired raters (e.g. sample 1 vs sample 2 of the same question)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"kappa": float("nan"), "n": 0}
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(ca) | set(cb))
    k = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else float("nan")
    return {"kappa": k, "p_observed": po, "p_expected": pe, "n": n,
            "interpretation": _kappa_label(k)}


def _kappa_label(k: float) -> str:
    if k != k:
        return "undefined"
    if k < 0:
        return "worse than chance"
    for hi, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                     (0.80, "substantial"), (1.01, "almost perfect")):
        if k <= hi:
            return name
    return "?"


# ============================================================================
# convenience: the recovery table the paper reports
# ============================================================================
def recovery_table(rows: Sequence[dict], group_key: str,
                   recovered_key: str = "recovered") -> list[dict]:
    """rows -> per-group recovery rate with Wilson CIs, sorted by n desc."""
    g = {}
    for r in rows:
        k = r.get(group_key)
        g.setdefault(k, [0, 0])
        g[k][1] += 1
        g[k][0] += bool(r.get(recovered_key))
    out = []
    for k, (c, n) in g.items():
        lo, hi = wilson_ci(c, n)
        out.append({"group": k, "recovered": c, "n": n, "rate": c / n if n else 0.0,
                    "ci_lo": lo, "ci_hi": hi})
    return sorted(out, key=lambda r: -r["n"])


if __name__ == "__main__":
    # sanity: the numbers from the 1,500-problem train run
    print("MEDIUM_WRONG vs BAD_WRONG recovery")
    print("  fisher p =", round(fisher_exact_2x2([[13, 129], [60, 1016]]), 4))
    print("  MW 95% CI", tuple(round(x, 3) for x in wilson_ci(13, 142)))
    print("  BW 95% CI", tuple(round(x, 3) for x in wilson_ci(60, 1076)))
    print("\nfleiss demo (3 questions x 5 samples, unstable types)")
    print(" ", fleiss_kappa([["CE", "SM", "CE", "NR", "UNCLASSIFIED"],
                            ["SM", "SM", "CE", "CE", "NR"],
                            ["NR", "CE", "UNCLASSIFIED", "SM", "CE"]]))
