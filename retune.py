"""
retune.py -- re-derive the tuned thresholds on NEW data. CPU only, ~1 minute.

WHY THIS EXISTS
---------------
Four thresholds in config.py were swept on the ORIGINAL 1,500-problem traces, which
were produced under a decoding bug (no_repeat_ngram blocked the '####' terminator,
so 88% of answers were synthesised) and with corrupted retry prompts. Once the model
generates cleanly its traces change shape, and a threshold fitted to the old shape is
no longer the right one.

None of these will BREAK anything if left alone -- they will just be slightly
mis-tuned, which quietly costs you taxonomy quality and therefore FST label quality.

    ABSTAIN_MARGIN       the |E - 0.30| guard band
    AL_MIN_VAR_LINES     unresolved-relation lines that trigger AL
    ST_COVERAGE_MAX      computation coverage below which ST fires
    GOOD_A_MIN           the A gate on the exemplar pool

THE OBJECTIVE
-------------
A taxonomy is well-conditioned when (a) the abstain rate is low, (b) no single class
dominates, and (c) the majority-class baseline an FST predictor must beat is low --
because a 60%-majority target is nearly unbeatable and tells you nothing.

We score each candidate by the MAJORITY BASELINE over the typed target set (lower is
better), with the abstain rate reported alongside. That is the number that actually
governs whether the gate can ever say GO.

USAGE
    python retune.py --results store_typed_train/results.jsonl
    python retune.py --results store_canary/results.jsonl --quick
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import config
import diagnosis as D
from components import compute_components
from classification import classify, Label
from preprocessing import normalize_math


def score_all(path: str, phase: str = "pass1"):
    """Score every trace ONCE; the sweeps then only re-run the cheap cascade."""
    out = []
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("phase") != phase:
                continue
            c = compute_components(
                r["question"], r.get("trace_scored") or r["trace"],
                normalize_math(r["gold_solution"]), r["gold_answer"])
            out.append(c)
    return out


def taxonomy(comps, **overrides):
    """Re-run the cascade with temporary config overrides."""
    saved = {k: getattr(config, k) for k in overrides}
    saved_d = {k: getattr(D, k, None) for k in overrides if hasattr(D, k)}
    try:
        for k, v in overrides.items():
            setattr(config, k, v)
            if hasattr(D, k):
                setattr(D, k, v)
        counts = Counter()
        for c in comps:
            if classify(c) is Label.GOOD:
                continue
            if c.correct and not config.DIAGNOSE_CORRECT:
                continue
            counts[D.diagnose_components(c).ftype.value] += 1
        return counts
    finally:
        for k, v in saved.items():
            setattr(config, k, v)
        for k, v in saved_d.items():
            if v is not None:
                setattr(D, k, v)


def quality(counts):
    """(abstain rate, majority baseline over the typed target set, n_typed)."""
    n = sum(counts.values()) or 1
    tgt = {k: v for k, v in counts.items() if k not in ("UNCLASSIFIED", "TR")}
    tt = sum(tgt.values()) or 1
    return counts.get("UNCLASSIFIED", 0) / n, max(tgt.values()) / tt, tt


# A candidate must beat the CURRENT value by at least this much on the score to be
# recommended. Without it the sweep "wins" on exact ties and tells you to change
# knobs that provably do nothing -- which is worse than no advice, because you then
# believe you tuned something.
MIN_GAIN = 0.005          # 0.5 percentage points


def sweep(name, values, comps, quick=False):
    print(f"\n{'='*68}\n  {name}\n{'='*68}")
    print(f"  {'value':<10}{'abstain':>10}{'maj-base':>11}{'typed n':>10}   taxonomy")
    cur = getattr(config, name)
    if cur not in values:
        values = sorted(set(list(values) + [cur]), key=float)
    scores = {}
    for v in values:
        counts = taxonomy(comps, **{name: v})
        ab, mb, tt = quality(counts)
        scores[v] = (mb + 0.25 * ab, ab, mb, tt, counts)
    cur_score = scores[cur][0]
    # keep the CURRENT value unless a candidate is materially better
    best = cur
    for v, (sc, *_rest) in scores.items():
        if sc < scores[best][0] - MIN_GAIN:
            best = v
    for v in values:
        sc, ab, mb, tt, counts = scores[v]
        mix = " ".join(f"{k}{counts.get(k,0)}" for k in
                       ("TR", "NR", "AL", "ST", "SM", "CE", "WP"))
        tag = "  <-- BEST" if v == best else ("  (current)" if v == cur else "")
        print(f"  {str(v):<10}{ab:>9.1%}{mb:>11.1%}{tt:>10}   {mix}{tag}")
    if best == cur:
        print(f"\n  KEEP {name} = {cur}  (no candidate beat it by >{MIN_GAIN:.1%})")
    else:
        print(f"\n  CHANGE {name}: {cur} -> {best}  "
              f"(score {cur_score:.3f} -> {scores[best][0]:.3f})")
    return best


def main():
    ap = argparse.ArgumentParser(description="re-derive tuned thresholds on new data")
    ap.add_argument("--results", required=True)
    ap.add_argument("--phase", default="pass1")
    ap.add_argument("--quick", action="store_true", help="fewer candidate values")
    a = ap.parse_args()

    print(f"scoring {a.results} ...")
    comps = score_all(a.results, a.phase)
    print(f"  {len(comps)} traces scored")
    if len(comps) < 100:
        print("  !! fewer than 100 traces -- these sweeps will be noisy. Use them as a\n"
              "     sanity check only; re-run on the full train set before committing.")

    base = taxonomy(comps)
    ab, mb, tt = quality(base)
    print(f"\nCURRENT CONFIG: abstain {ab:.1%} | majority baseline {mb:.1%} | typed {tt}")
    print(f"  {dict(base.most_common())}")

    best = {}
    best["ABSTAIN_MARGIN"] = sweep("ABSTAIN_MARGIN",
                                   [0.0, 0.01, 0.02, 0.05] if not a.quick else [0.0, 0.02],
                                   comps)
    best["AL_MIN_VAR_LINES"] = sweep("AL_MIN_VAR_LINES",
                                     [2, 3, 4, 5] if not a.quick else [2, 3, 4], comps)
    best["ST_COVERAGE_MAX"] = sweep("ST_COVERAGE_MAX",
                                    [0.34, 0.5, 0.67] if not a.quick else [0.34, 0.5],
                                    comps)
    best["GOOD_A_MIN"] = sweep("GOOD_A_MIN",
                               [0.0, 0.25, 0.5, 0.75] if not a.quick else [0.0, 0.5],
                               comps)

    combined = taxonomy(comps, **best)
    ab2, mb2, tt2 = quality(combined)
    print(f"\n{'='*68}\n  ALL BEST VALUES TOGETHER\n{'='*68}")
    print(f"  abstain {ab:.1%} -> {ab2:.1%}   majority {mb:.1%} -> {mb2:.1%}   "
          f"typed {tt} -> {tt2}")
    print(f"  {dict(combined.most_common())}")

    changed = {k: v for k, v in best.items() if v != getattr(config, k)}
    print(f"\n{'='*68}")
    if not changed:
        print("  NO CHANGES NEEDED -- the current config is already best on this data.")
    else:
        print("  PASTE INTO config.py (and note in CHANGES.md that they were re-swept):")
        for k, v in changed.items():
            print(f"    {k} = {v}      # was {getattr(config, k)}, re-swept on {a.results}")
        print("\n  Then re-run fst_gate.py -- every label has moved.")
    print("=" * 68)
    print("\nCAVEAT: sweeping on the data you will report is mild overfitting. If you"
          "\nhave the budget, sweep on a held-out slice (e.g. the first 300 problems)"
          "\nand freeze the values before scoring the rest.")


if __name__ == "__main__":
    main()