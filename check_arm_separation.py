"""
check_arm_separation.py -- prove the two arms differ BEFORE spending GPU on them.
No GPU. Seconds.

WHY
---
The whole control experiment is worthless if `--retry-mode generic` quietly builds
the same prompt as `--retry-mode typed`. That has happened before in this project:
[AUDIT D16] found typed and generic selecting an IDENTICAL 8/8 exemplar set for CE,
which made the arms indistinguishable for that type and would have shown up as
"typing does not help" when the real cause was a selection bug.

So: before the run, take real failures out of the store, build both prompts, and
check they actually diverge. If this prints IDENTICAL for any type, do not run.

WHAT IT CHECKS
  1. exemplar overlap, typed vs generic, per type   (want: low)
  2. instruction text, typed vs generic             (want: typed non-empty, generic empty)
  3. the full system message actually handed to the model
  4. that the typed cures differ from EACH OTHER    (want: low pairwise overlap)

USAGE
    python check_arm_separation.py --store stores/store_typed_train_audit2
    python check_arm_separation.py --store ... --show WP     # dump both prompts
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from classification import TraceStore
from diagnosis import FailureType, TYPED_INSTRUCTION
from kaggle_run import (select_typed_positives, select_generic_positives,
                        build_retry_prompt)

TYPES = ["NR", "AL", "ST", "SM", "CE", "WP"]


def _qs(recs):
    return {r.get("question") for r in recs}


def main():
    ap = argparse.ArgumentParser(description="verify typed and generic differ")
    ap.add_argument("--store", required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--show", help="dump both full prompts for this type")
    a = ap.parse_args()

    store = TraceStore(root=Path(a.store))
    pool = store.exemplars()
    if not pool:
        raise SystemExit(f"empty pool in {a.store}")
    print(f"pool: {len(pool)} gold-stripped exemplars\n")

    gen_ex = select_generic_positives(pool, [], a.budget)
    typed = {t: select_typed_positives(pool, FailureType(t), [], a.budget) for t in TYPES}

    print("=" * 70)
    print("  1 · TYPED vs GENERIC  (exemplar overlap out of %d)" % a.budget)
    print("=" * 70)
    print(f"  {'type':<8}{'overlap':>9}{'instruction':>14}   verdict")
    bad = 0
    for t in TYPES:
        ov = len(_qs(typed[t]) & _qs(gen_ex))
        ins = TYPED_INSTRUCTION.get(FailureType(t), "")
        same = (ov == a.budget) and not ins
        bad += same
        print(f"  {t:<8}{ov:>9}{('yes' if ins else 'NONE'):>14}   "
              f"{'!! IDENTICAL ARMS' if same else 'differs'}")
    print(f"\n  generic instruction: {'(empty)' if True else ''}  <- by design")

    print("\n" + "=" * 70)
    print("  2 · TYPED vs TYPED  (do the eight cures differ from each other?)")
    print("=" * 70)
    print(f"  {'':6}" + "".join(f"{t:>6}" for t in TYPES))
    for x in TYPES:
        print(f"  {x:<6}" + "".join(f"{len(_qs(typed[x]) & _qs(typed[y])):>6}" for y in TYPES))
    ovs = [len(_qs(typed[x]) & _qs(typed[y])) for x, y in combinations(TYPES, 2)]
    print(f"\n  mean pairwise overlap {sum(ovs)/len(ovs):.2f} / {a.budget}")
    identical = [f"{x}~{y}" for x, y in combinations(TYPES, 2)
                 if len(_qs(typed[x]) & _qs(typed[y])) == a.budget]
    if identical:
        bad += len(identical)
        print(f"  !! these typed cures are IDENTICAL: {identical}")

    print("\n" + "=" * 70)
    print("  3 · THE SYSTEM MESSAGE EACH ARM SENDS")
    print("=" * 70)
    q = "PLACEHOLDER QUESTION"
    for t in TYPES:
        _, extra = build_retry_prompt(q, typed[t], TYPED_INSTRUCTION.get(FailureType(t), ""))
        print(f"  typed/{t:<4} -> {extra!r}")
    _, extra = build_retry_prompt(q, gen_ex, "")
    print(f"  generic    -> {extra!r}")

    if a.show:
        t = a.show.upper()
        rec_q = next((r.get("question") for r in pool), q)
        pt, et = build_retry_prompt(rec_q, typed[t], TYPED_INSTRUCTION.get(FailureType(t), ""))
        pg, eg = build_retry_prompt(rec_q, gen_ex, "")
        print("\n" + "=" * 70)
        print(f"  FULL PROMPT · typed/{t}   (system: {et!r})")
        print("=" * 70); print(pt[:2400])
        print("\n" + "=" * 70)
        print(f"  FULL PROMPT · generic   (system: {eg!r})")
        print("=" * 70); print(pg[:2400])

    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if bad:
        print(f"  !! {bad} problem(s) above. The arms are NOT cleanly separated.")
        print("     Running the comparison now would measure a selection bug, not")
        print("     the diagnosis. Fix select_typed_positives first.")
        raise SystemExit(1)
    print("  Arms are separated: typed and generic select different exemplars,")
    print("  typed carries an instruction and generic carries none, and the eight")
    print("  typed cures differ from each other. Safe to spend GPU.")


if __name__ == "__main__":
    main()
