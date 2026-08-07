"""
why_unclassified.py -- break down WHY traces land in the abstain bucket.

UNCLASSIFIED is 52.2% of all diagnoses on the 1,500-problem train run (713/1367).
That is the single biggest weakness in the taxonomy, and "the cascade abstained"
is not a finding you can put in a paper. This script names the actual cause for
every abstained trace, so the limitations section is specific and the fixes are
ranked by how many traces they would actually recover.

It replays the cascade's conditions on already-scored records -- no model, no GPU.

THE TWO CAUSES IT SEPARATES
---------------------------
1. THE ABSTAIN BAND is discretisation noise, not uncertainty. E = hits/s_hat is a
   ratio of small integers, so near 0.30 it can only take the values 1/4=0.250,
   2/7=0.286, 1/3=0.333 -- all of which fall inside |E-0.30| < 0.05 and are forced
   to abstain. On the train run that is 216 traces sitting at exactly three values,
   213 of which resolve if ABSTAIN_MARGIN drops to 0.02. A continuous guard band on
   a signal with ~6 distinct values below 0.5 has no principled justification.

2. A GENUINE TAXONOMY GAP. CE requires V < V_MAX (0.85) and SM requires E < 0.30,
   so a trace with sound arithmetic, full length, and a wrong answer matches
   neither. On the train run that is 243 traces. They are NOT calculation errors:
   CE fires at mean V=0.224 while this group sits at V>=0.85. The arithmetic is
   right and the PLAN is wrong. Do not fix this by dropping CE's upper bound --
   that would hand "verify each computation" to traces whose computations are
   already correct, which is exactly the confidently-wrong cure the abstain design
   exists to prevent. Add a fifth type (WP, wrong plan) with a planning cure.

USAGE
    python why_unclassified.py --results results_clean.jsonl
    python why_unclassified.py --results results_clean.jsonl --simulate
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from config import (SM_E_MAX, CE_E_MIN, V_MIN, V_MAX, FULL_LEN_RATIO, G_MIN,
                    ABSTAIN_MARGIN)


def load(path, phase="pass1"):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if phase and r.get("phase") != phase:
                continue
            if r.get("diagnosis") is None:      # GOOD -- never diagnosed
                continue
            rows.append(r)
    return rows


def classify_cause(sig: dict) -> str:
    """Name the single rule that stopped this trace from getting a typed label."""
    E, V, G = sig["E"], sig["V"], sig["G"]
    s, shat = sig["n_steps"], sig["n_checkpoints"]
    coh = bool(sig["coherent"])
    full = s >= FULL_LEN_RATIO * shat

    if E >= CE_E_MIN and V >= V_MAX and full:
        return "CE blocked by V>=V_MAX  (valid arithmetic, wrong answer -> 5th type 'WP')"
    if E >= CE_E_MIN and not full:
        return "CE blocked by not full_length  (terse trace; split_steps undercount)"
    if E < SM_E_MAX and not full:
        return "SM blocked by not full_length  (terse trace; split_steps undercount)"
    if E < SM_E_MAX and V <= V_MIN:
        return "SM blocked by V<=V_MIN  (low E AND broken arithmetic)"
    if E < SM_E_MAX and not coh:
        return "SM blocked by incoherent  (answer != last equation)"
    if E < SM_E_MAX and G < G_MIN:
        return "SM blocked by G<G_MIN  (ungrounded)"
    return "other / no rule matched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--phase", default="pass1",
                    help="'pass1', 'retry', or '' for every attempt")
    ap.add_argument("--simulate", action="store_true",
                    help="also simulate margin changes and the WP 5th type")
    a = ap.parse_args()

    rows = load(a.results, a.phase or None)
    unc = [r for r in rows if r["diagnosis"] == "UNCLASSIFIED"]
    print("=" * 74)
    print(f"UNCLASSIFIED BREAKDOWN   ({len(unc)}/{len(rows)} = "
          f"{len(unc)/max(1,len(rows)):.1%} of diagnosed traces)")
    print("=" * 74)

    band = [r for r in unc if "abstain" in (r.get("diagnosis_reason") or "")]
    fell = [r for r in unc if r not in band]

    # ---- cause 1: the abstain band ---------------------------------------
    print(f"\n[1] ABSTAIN BAND  |E - {SM_E_MAX}| < {ABSTAIN_MARGIN}   "
          f"{len(band)} traces ({len(band)/max(1,len(unc)):.1%} of UNCLASSIFIED)")
    evals = Counter(round(r["signals"]["E"], 3) for r in band)
    for e, c in evals.most_common():
        frac = _as_fraction(e)
        print(f"      E = {e:<6} {c:>5} traces{frac}")
    print("      ^ a ratio of small integers, not a continuum. These are")
    print("        discretisation artefacts, not genuine boundary uncertainty.")
    for m in (0.02, 0.01, 0.0):
        rem = sum(1 for r in band if abs(r["signals"]["E"] - SM_E_MAX) < m)
        print(f"      ABSTAIN_MARGIN={m:<5} would still abstain {rem}/{len(band)}")

    # ---- cause 2: real rule failures --------------------------------------
    print(f"\n[2] NO RULE FIRED   {len(fell)} traces "
          f"({len(fell)/max(1,len(unc)):.1%} of UNCLASSIFIED)")
    causes = Counter(classify_cause(r["signals"]) for r in fell)
    for cause, c in causes.most_common():
        print(f"      {c:>5}  {cause}")

    # ---- the 5th-type case, characterised ---------------------------------
    wp = [r for r in fell if classify_cause(r["signals"]).startswith("CE blocked by V")]
    ce = [r for r in rows if r["diagnosis"] == "CE"]
    if wp and ce:
        print(f"\n[3] IS THE V>=V_MAX GROUP REALLY 'CE'?   (n={len(wp)} vs CE n={len(ce)})")
        for name, grp in (("CE (fires)", ce), ("V>=V_MAX blocked", wp)):
            print(f"      {name:<20} mean E={_m(grp,'E'):.3f}  mean V={_m(grp,'V'):.3f}  "
                  f"mean misses={_m(grp,'misses'):.2f}  mean G={_m(grp,'G'):.3f}")
        print("      ^ different populations. CE's arithmetic is BROKEN; this group's")
        print("        arithmetic is SOUND. Same cure for both would be a wrong cure.")

    # ---- simulation --------------------------------------------------------
    if a.simulate:
        _simulate(rows)


def _simulate(rows):
    import diagnosis as D
    print("\n" + "=" * 74)
    print("SIMULATED RE-DIAGNOSIS")
    print("=" * 74)

    def run(margin, wp=False):
        c = Counter()
        for r in rows:
            s = r["signals"]
            d = D.diagnose(s["E"], s["V"], s["n_steps"], s["n_checkpoints"],
                           s["G"], s["coherent"], s["n_equations"], margin=margin)
            t = d.ftype.value
            if (wp and t == "UNCLASSIFIED" and s["E"] >= CE_E_MIN
                    and s["V"] >= V_MAX
                    and s["n_steps"] >= FULL_LEN_RATIO * s["n_checkpoints"]):
                t = "WP"
            c[t] += 1
        return c

    n = len(rows)
    for name, args in (("current", (ABSTAIN_MARGIN, False)),
                       ("margin=0.02", (0.02, False)),
                       ("margin=0.00", (0.0, False)),
                       ("margin=0.02 + WP 5th type", (0.02, True))):
        c = run(*args)
        mix = "  ".join(f"{k}={v}" for k, v in c.most_common())
        print(f"  {name:<28} UNCL {c['UNCLASSIFIED']/n:>6.1%}   {mix}")
    print("\n  NOTE: every row here shifts the FST training labels, so re-run")
    print("  fst_gate.py after adopting any of these changes.")


def _as_fraction(e: float) -> str:
    for num in range(1, 8):
        for den in range(1, 12):
            if abs(num / den - e) < 5e-4:
                return f"   (= {num}/{den}: {den} checkpoints, {num} hit)"
    return ""


def _m(rows, key):
    vals = [r["signals"][key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    main()
