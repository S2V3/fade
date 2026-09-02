"""
ctc_data.py -- assemble every (question, trace, outcome) triple FADE has produced.

WHY
---
ctc_train.py read a single store: 1,500 pass-1 rows plus that run's retries, 2,568
in all. But pass-1 is FORKED and shared across every arm, so each additional run
contributes ~1,000 genuinely new retry traces that were simply being ignored.

    typed (d63)     1,068 retries
    typed (d61)     1,068
    typed (v1)      1,068
    generic (new)   1,074
    generic (old)   1,068
    partial           636
    pass-1          1,500  (shared, counted once)

Deduplicated on (question, trace): 5,286 unique traces, 4,243 of them wrong.

MORE DATA IS THE CHEAPEST IMPROVEMENT AVAILABLE. It costs no GPU -- these traces
already exist -- and it doubles the training set.

WHAT COUNTS AS ONE EXAMPLE
--------------------------
A (question, trace) pair with its outcome, wherever it came from. A retry trace is
as legitimate a training example as a pass-1 trace: same model, same task, and it
is the distribution CTC actually meets on a second attempt.

DEDUPLICATION matters. Greedy decoding means the same prompt yields the same trace,
and arms share exemplars, so near-identical rows recur across runs. Training on
duplicates inflates cross-validation scores because copies of one trace land in
both the train and test folds.

GOLD RULE: only the LABEL comes from gold. Features are computed by
ctc_features.trace_features() from the question and the trace alone.

USAGE
    python ctc_data.py --stores dirA dirB --out ctc_dataset.jsonl
    python ctc_data.py --files a.jsonl b.jsonl --out ctc_dataset.jsonl --report
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def _rows(path):
    with open(path) as f:
        for ln in f:
            if ln.strip():
                try:
                    yield json.loads(ln)
                except Exception:
                    continue


def collect(paths, include_retries=True):
    """-> (examples, provenance). One example per unique (question, trace)."""
    uniq, prov = {}, Counter()
    for p in paths:
        src = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        for r in _rows(p):
            ph = r.get("phase")
            if ph == "pass1":
                ok = bool((r.get("signals") or {}).get("correct"))
                ty = r.get("diagnosis")
            elif ph == "retry" and include_retries:
                ok = bool((r.get("signals") or {}).get("correct")
                          or r.get("newly_correct"))
                # the diagnosis OF THIS TRACE, not of the attempt it replaced
                ty = r.get("diagnosis")
            else:
                continue
            q, tr = r.get("question"), r.get("trace")
            if not q or not tr:
                continue
            key = (q.strip()[:200], tr.strip()[:400])
            if key in uniq:
                prov["duplicate"] += 1
                continue
            uniq[key] = {"question": q, "trace": tr, "correct": ok,
                         "type": (ty or "UNCLASSIFIED"), "phase": ph, "source": src}
            prov[f"{src}:{ph}"] += 1
    return list(uniq.values()), prov


def main():
    ap = argparse.ArgumentParser(description="assemble the CTC training set")
    ap.add_argument("--files", nargs="*", default=[], help="results.jsonl paths")
    ap.add_argument("--stores", nargs="*", default=[], help="store dirs")
    ap.add_argument("--out", default="ctc_dataset.jsonl")
    ap.add_argument("--pass1-only", action="store_true")
    a = ap.parse_args()

    paths = list(a.files) + [os.path.join(s, "results.jsonl") for s in a.stores]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no readable results.jsonl given")

    ex, prov = collect(paths, include_retries=not a.pass1_only)
    print("=" * 66)
    print(f"  CTC DATASET from {len(paths)} file(s)")
    print("=" * 66)
    for k, v in sorted(prov.items(), key=lambda x: -x[1]):
        print(f"  {k:<34}{v:>6}")
    lab = Counter("correct" if e["correct"] else "wrong" for e in ex)
    print(f"\n  unique (question, trace) : {len(ex)}")
    print(f"  labels                   : {dict(lab)}")
    wrong = [e for e in ex if not e["correct"]]
    print(f"  types among wrong        : "
          f"{dict(Counter(e['type'] for e in wrong).most_common())}")

    with open(a.out, "w") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    print(f"\n  wrote -> {a.out}")


if __name__ == "__main__":
    main()
