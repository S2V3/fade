"""
nm_pick_budget.py -- choose the new model's flag budget with the paper's own rule.

Llama-2's 40% budget was the argmax of the expected net
    net(b) = N_flag(b) * ( pi(b) * R  -  (1 - pi(b)) * D )
over the out-of-fold budget table, with R measured on its train retries and D a
prior. A model that is right 85% of the time has a completely different pi(b), so
reusing 40% would not be "the same pipeline" -- reusing the RULE is. This script
applies that rule to the new model's own detector and train run.

    R  = recovery rate of the typed retries in the new model's train store (measured)
    D  = prior (default 0.746, Llama-2's measured test damage). Stated in the paper;
         the held-out run then measures the real D.

If no budget has a positive expected net, the smallest budget (10%) is used so R
and D still get measured, and the script says so -- that outcome is itself the
break-even result for this model.

    python nm_pick_budget.py --ctc-report <dir>/ctc2_report.json --train-store <store> \\
        --out <dir>/budget.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc-report", required=True)
    ap.add_argument("--train-store", required=True)
    ap.add_argument("--d-prior", type=float, default=0.746)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rep = json.load(open(a.ctc_report))
    det = rep.get("detect", {})
    prec = det.get("budget_precision") or {}
    thr = det.get("budget_thresholds") or {}
    n_cal = det.get("calibration_n") or rep.get("n")
    if not prec:
        raise SystemExit("ctc2_report has no budget table -- retrain with the current ctc_train2.py")

    rows = [json.loads(l) for l in open(Path(a.train_store) / "results.jsonl") if l.strip()]
    retries = [r for r in rows if r.get("phase") == "retry" and int(r.get("iter", 1) or 1) == 1]
    rec = sum(bool(r.get("newly_correct")) for r in retries)
    R = rec / max(len(retries), 1)
    D = a.d_prior
    print(f"  train retries {len(retries)}, recovered {rec} -> R = {R:.1%} | D prior = {D:.1%}"
          f" | break-even precision pi* = {D / (R + D):.1%}")
    print(f"  {'budget':>7}{'cut':>8}{'precision':>11}{'expected net':>14}")
    table, best = [], None
    for b in sorted(prec, key=float):
        pi = prec[b]
        nflag = float(b) * n_cal
        net = nflag * (pi * R - (1 - pi) * D)
        table.append({"budget": float(b), "threshold": thr.get(b), "precision": pi, "net": net})
        print(f"  {float(b):>7.0%}{thr.get(b, float('nan')):>8.3f}{pi:>11.1%}{net:>+14.1f}")
        if best is None or net > best["net"]:
            best = table[-1]
    pays = best["net"] > 0
    choice = best if pays else min(table, key=lambda t: t["budget"])
    note = ("argmax of expected net" if pays else
            "NO budget has positive expected net at this R and D -- using the smallest "
            "budget so R and D are still measured. This is the break-even result.")
    print(f"\n  chosen budget {choice['budget']:.0%} (p_wrong >= {choice['threshold']:.3f}): {note}")
    Path(a.out).write_text(json.dumps({"budget": choice["budget"], "threshold": choice["threshold"],
                                       "R_train": R, "n_train_retries": len(retries),
                                       "D_prior": D, "pi_star": D / (R + D), "pays": pays,
                                       "note": note, "table": table}, indent=2))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
