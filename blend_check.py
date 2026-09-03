"""
blend_check.py -- the detection numbers only, on CPU, in about a minute.

WHY THIS EXISTS SEPARATELY FROM score_mpv.py
--------------------------------------------
score_mpv.py re-scores every arm: it calls score_trace() on ~2,600 traces to
rebuild the failure taxonomy, then computes several O(n^2) AUROCs. That is the
right thing to run once, at the end. But the *blend* number needed one fix --
the weight had been chosen by maximising on the test set -- and re-deriving it
should not cost ten minutes and a full re-score.

This script computes ONLY the detector comparison:

    MPV alone            1 - support/applicable, from the probes
    p_wrong alone        the shipped CTC score, already in results.jsonl
    blend                w*MPV + (1-w)*p_wrong, over a fixed grid

WHAT "BLEND" IS, AND WHY IT IS WORTH REPORTING
----------------------------------------------
Two gold-free wrongness scores exist for the same pass-1 trace:

    p_wrong   learned, reads STATIC FEATURES of that one trace
    MPV       symbolic, reads how the trace's PROGRAM behaves across the
              question's number-family

They are orthogonal by construction. p_wrong is blind on WP (it flags wrong
plans at 12% while flagging correct traces at 18.3% -- below chance); MPV is
strongest exactly there. So a weighted sum should beat either alone, and that
single number is the cleanest evidence that MPV carries signal the trained
detector does not.

THE STATISTICAL TRAP THIS SCRIPT FIXES
--------------------------------------
The first run reported blend = 0.778 at w = 0.25, having picked w by maximising
AUROC ON THE TEST SET. Three candidate weights makes the optimism small, but it
is still fitting a parameter to the evaluation and reporting the fit. A reviewer
will catch it.

Two defensible options, both implemented here:

  1. PRE-REGISTER w = 0.5 (the neutral choice), report the whole grid as a
     sensitivity analysis, and claim "the blend beats p_wrong at every weight"
     rather than "at the best weight". This is the default headline.

  2. FIT w ON TRAIN with --fit-on <train store>, then report the test number at
     that fixed weight. Stronger, but only available if you have probes on a
     train store (you do not, unless you run run_mpv.py on the stage-2 store).

A paired bootstrap over questions gives a CI on (blend - p_wrong), so the claim
is an interval and not a point.

CORRECTNESS LABELS
------------------
Only the final answer is needed here, not the failure taxonomy, so this uses
extraction.is_correct() -- the same answer ladder score_trace() uses -- instead
of the full scoring path. That is the whole speed difference.

USAGE
    python blend_check.py --store stores/store_ctctest_audit2_a1 --split test
    python blend_check.py --store <dir> --gold-jsonl gold.jsonl      # offline
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import mpv
from extraction import extract_final_answer, is_correct

GRID = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
HEADLINE_W = 0.5


def auroc(scores, labels):
    """labels: 1 = wrong. Rank-based with tie correction, O(n log n)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    # average ranks over ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_rows(store, probes_path=None):
    root = Path(store)
    rows = [json.loads(l) for l in open(root / "results.jsonl") if l.strip()]
    probes = defaultdict(dict)
    p = Path(probes_path) if probes_path else root / "mpv_probes.jsonl"
    n_raw = 0
    if p.exists():
        for ln in open(p):
            if not ln.strip():
                continue
            d = json.loads(ln); n_raw += 1
            if d.get("model_answer") is None:
                continue
            key = (round(float(d["target_value"]), 6), round(float(d["new_value"]), 6))
            probes[d["id"]][key] = mpv.Probe.from_dict(d)
    return rows, probes, n_raw


def build(store, gold, probes_path=None):
    """-> [(mpv_score, p_wrong, wrong?)] for every testable pass-1 trace."""
    rows, probes, n_raw = load_rows(store, probes_path)
    out, n_no_probe, n_no_pw, n_abstain = [], 0, 0, 0
    for r in rows:
        g = gold.get(r["question"])
        if g is None:
            continue
        pw = r.get("p_wrong")
        if pw is None:
            n_no_pw += 1
            continue
        pr = [probes[r["id"]][k] for k in sorted(probes.get(r["id"], {}))]
        if not pr:
            n_no_probe += 1
            continue
        ans = extract_final_answer(r["trace"])
        ms = mpv.detection_score(r["question"], r["trace"], ans, pr)
        if ms is None:
            n_abstain += 1
            continue
        y = 0 if is_correct(r["trace"], g["gold_answer"]) else 1
        out.append((ms, float(pw), y))
    print(f"  {len(rows)} rows | {n_raw} probes written | {len(out)} traces scorable")
    print(f"  dropped: no probe {n_no_probe}, MPV abstained {n_abstain}, no p_wrong {n_no_pw}")
    return out


def report(data, label, fixed_w=None):
    ys = [y for _, _, y in data]
    a_m = auroc([m for m, _, _ in data], ys)
    a_p = auroc([p for _, p, _ in data], ys)
    grid = {w: auroc([w * m + (1 - w) * p for m, p, _ in data], ys) for w in GRID}
    w_head = HEADLINE_W if fixed_w is None else fixed_w
    a_b = grid[w_head] if w_head in grid else auroc(
        [w_head * m + (1 - w_head) * p for m, p, _ in data], ys)

    rng = random.Random(0)
    diffs = []
    for _ in range(2000):
        samp = [data[rng.randrange(len(data))] for _ in range(len(data))]
        y2 = [y for _, _, y in samp]
        if len(set(y2)) < 2:
            continue
        b = auroc([w_head * m + (1 - w_head) * p for m, p, _ in samp], y2)
        q = auroc([p for _, p, _ in samp], y2)
        if b is not None and q is not None:
            diffs.append(b - q)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]

    print(f"\n  === {label}  (n = {len(data)}, {sum(ys)} wrong / {len(ys)-sum(ys)} correct) ===")
    print(f"    MPV alone       {a_m:.3f}")
    print(f"    p_wrong alone   {a_p:.3f}")
    print(f"    BLEND w={w_head:<4}    {a_b:.3f}   gain {a_b - a_p:+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  (paired bootstrap, 2000 resamples)")
    print("    sensitivity -- no weight was tuned on this split:")
    for w in GRID:
        star = "  <- headline" if w == w_head else ""
        print(f"      w={w:<5} {grid[w]:.3f}   gain {grid[w] - a_p:+.3f}{star}")
    beats = sum(1 for w in GRID if 0 < w < 1 and grid[w] > a_p)
    n_int = sum(1 for w in GRID if 0 < w < 1)
    print(f"    blend beats p_wrong at {beats}/{n_int} interior weights"
          + ("  -- the claim is robust to the weight." if beats == n_int else
             "  -- NOT robust; report the grid, not a single number."))
    return {"n": len(data), "auroc_mpv": a_m, "auroc_pwrong": a_p,
            "blend_w": w_head, "auroc_blend": a_b, "gain_ci": [lo, hi],
            "grid": {str(w): grid[w] for w in GRID}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gold-jsonl", default=None,
                    help="offline gold rows {question, answer, gold_answer}; "
                         "default loads the split from HuggingFace")
    ap.add_argument("--fit-on", default=None,
                    help="a TRAIN store with probes: choose w there, report it here")
    ap.add_argument("--fit-gold-jsonl", default=None)
    ap.add_argument("--out", default=None, help="default <store>/blend_report.json")
    a = ap.parse_args()

    def load_gold(path, split):
        if path:
            g = {}
            for ln in open(path):
                if ln.strip():
                    d = json.loads(ln); g[d["question"]] = d
            return g
        import kaggle_run as K
        K.SPLIT_TAG = split
        return {p["question"]: p for p in K._load_split(split, None)}

    print("=" * 74)
    print("  BLEND CHECK -- detector comparison only, CPU")
    print("=" * 74)

    fixed_w = None
    if a.fit_on:
        print(f"\n  fitting the weight on {a.fit_on} (train)")
        fit_data = build(a.fit_on, load_gold(a.fit_gold_jsonl, "train"))
        ys = [y for _, _, y in fit_data]
        best = max(GRID, key=lambda w: auroc([w * m + (1 - w) * p for m, p, _ in fit_data], ys))
        fixed_w = best
        print(f"  -> w = {best} chosen on TRAIN; it is now frozen for the test report.")

    data = build(a.store, load_gold(a.gold_jsonl, a.split), a.probes)
    if not data:
        raise SystemExit("  nothing scorable -- are the probes in this store?")
    rep = report(data, f"{a.split} split", fixed_w)

    out = Path(a.out) if a.out else Path(a.store) / "blend_report.json"
    json.dump(rep, open(out, "w"), indent=2)
    print(f"\n  -> {out}")
    print("\n  Put the w=0.5 row in the paper as the headline and the grid in the")
    print("  appendix. Do NOT quote the best cell of the grid as the result.")


if __name__ == "__main__":
    main()
