"""
Deep-dive on ONE exemplar: every intermediate the labels are built from, so
when run_phase0.py disagrees with your hand label you can see exactly WHY --
which checkpoint missed (-> misses), which equation is false (-> bad_eqs),
which steps duplicated (-> R), which cascade rule fired.

Usage:
    python inspect_trace.py q06_medium_wrong_last_step_slip
    python inspect_trace.py e_nr2_ungrounded_valid_arithmetic
    python inspect_trace.py my_id --data my_exemplars.json --r-variant max
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from extraction import (extract_values, value_in, extract_gold_checkpoints,
                        extract_equations, extract_final_answer)
from similarity import split_steps, pairwise_similarity, backend_name
from components import compute_components
from classification import classify, CONSEQUENCE, needs_diagnosis
from diagnosis import diagnose, TYPED_INSTRUCTION, TYPED_CURE


def find_item(data: dict, item_id: str) -> dict:
    for section in ("quality_set", "error_set"):
        for it in data.get(section, []):
            if it["id"] == item_id:
                return it
    raise SystemExit(f"id {item_id!r} not found. Available: " + ", ".join(
        it["id"] for s in ("quality_set", "error_set") for it in data.get(s, [])))


def main(data_path: Path, item_id: str, r_variant: str, margin: float):
    it = find_item(json.loads(data_path.read_text()), item_id)
    q, tr, gold = it["question"], it["trace"], it["gold_solution"]

    print("=" * 78)
    print(f"ID: {it['id']}")
    print("=" * 78)
    print(f"\nQUESTION:\n  {q}")
    print(f"\nGOLD SOLUTION:\n  {gold}")
    print(f"\nTRACE:\n" + "\n".join("  " + ln for ln in tr.splitlines()))
    if it.get("note"):
        print(f"\nNOTE: {it['note']}")

    # --------------------------------------------- E / misses detail
    cps = extract_gold_checkpoints(gold)
    tvals = extract_values(tr)
    print("\n--- E and misses: checkpoint matching (value-based) ---")
    print(f"trace values extracted: {sorted(tvals)}")
    for c in cps:
        print(f"  checkpoint {c:>10}: {'HIT' if value_in(c, tvals) else 'MISS'}")

    # --------------------------------------------- V / bad_eqs detail
    eqs = extract_equations(tr)
    print("\n--- V and bad_eqs: extracted equations (sympy symbolic) ---")
    if not eqs:
        print("  (none extracted -> V = 0, bad_eqs = 0, coherent = None)")
    for e in eqs:
        print(f"  {e.raw!r:40} -> {'TRUE' if e.is_true else 'FALSE'}")

    # --------------------------------------------- R detail
    stps = split_steps(tr)
    print(f"\n--- R: steps and near-duplicates (backend: {backend_name()}, "
          f"variant: {r_variant}) ---")
    for i, s in enumerate(stps):
        print(f"  [{i}] {s}")
    if len(stps) >= 2:
        sim = pairwise_similarity(stps)
        dups = [(i, j, sim[i][j]) for i in range(1, len(stps))
                for j in range(i) if sim[i][j] > config.DUP_SIM_THRESHOLD]
        for i, j, s_ in dups:
            print(f"  near-duplicate: step {i} ~ step {j} (cos={s_:.3f})")
        if not dups:
            print(f"  no near-duplicate pairs above {config.DUP_SIM_THRESHOLD}")

    # --------------------------------------------- answer
    print("\n--- answer extraction ladder ---")
    print(f"  extracted final answer: {extract_final_answer(tr)}  "
          f"(gold: {it['gold_answer']})")

    # --------------------------------------------- components + label
    c = compute_components(q, tr, gold, it["gold_answer"], r_variant=r_variant)
    label = classify(c)
    print("\n--- components (no composite Q -- counts decide) ---")
    print(f"  E={c.E:.3f}  V={c.V:.3f}  R={c.R:.3f}  A={c.A:.3f}  "
          f"G={c.G:.3f}  coherent={c.coherent}")
    print(f"  counts: misses={c.misses}  bad_eqs={c.bad_eqs}   "
          f"s={c.n_steps}  s_hat={c.n_checkpoints}   correct={c.correct}")

    print(f"\n--- count-based label ---")
    print(f"  LABEL: {label.value}  ->  {CONSEQUENCE[label]}")
    print(f"  rule inputs: correct={c.correct}, misses={c.misses}, "
          f"bad_eqs={c.bad_eqs}, coherent={bool(c.coherent)}, "
          f"R={c.R:.2f} (GOOD needs >= {config.GOOD_R_MIN}), "
          f"G={c.G:.2f} (MEDIUM-wrong needs >= {config.G_MIN})")
    if it.get("human_expected_label"):
        print(f"  your expected label: {it['human_expected_label']}  "
              f"({'MATCH' if it['human_expected_label'] == label.value else 'MISMATCH'})")

    # --------------------------------------------- cascade (every non-GOOD)
    if needs_diagnosis(label):
        d = diagnose(c.E, c.V, c.n_steps, c.n_checkpoints,
                     c.G, c.coherent, margin=margin)
        mode = "failure" if not c.correct else "weak-success (polish)"
        print(f"\n--- diagnostic cascade ({mode}) ---")
        print(f"  type: {d.ftype.value}")
        print(f"  rule: {d.reason}")
        print(f"  typed instruction: {TYPED_INSTRUCTION[d.ftype] or '(none -- generic)'}")
        print(f"  typed cure:        {TYPED_CURE[d.ftype]}")
        if it.get("human_expected_diagnosis"):
            print(f"  your expected diagnosis: {it['human_expected_diagnosis']}  "
                  f"({'MATCH' if it['human_expected_diagnosis'] == d.ftype.value else 'MISMATCH'})")
    else:
        print("\n--- diagnostic cascade ---\n"
              "  (GOOD trace: not diagnosed; enters the pool)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FADE phase 0 single-trace inspector")
    p.add_argument("item_id")
    p.add_argument("--data", type=Path,
                   default=Path(__file__).parent / "sample_exemplars.json")
    p.add_argument("--r-variant", default=config.R_VARIANT,
                   choices=["max", "mean_pairwise", "dup_fraction"])
    p.add_argument("--margin", type=float, default=config.ABSTAIN_MARGIN)
    a = p.parse_args()
    main(a.data, a.item_id, a.r_variant, a.margin)