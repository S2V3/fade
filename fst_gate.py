"""
fst_gate.py -- decide whether FST is justified, from a completed run.

FST's premise: failure TYPE depends on problem CATEGORY. That is testable, not
assumed. This runs a chi-square test of independence on the (category, diagnosis)
table from store/results.jsonl.

  p < 0.05  -> types cluster by category -> FST is justified, build it.
  p >= 0.05 -> independent -> FST premise false for this model; report and skip.

Usage:  python fst_gate.py            # reads store/results.jsonl
        python fst_gate.py --file path/to/results.jsonl
"""
import argparse, json
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="store/results.jsonl")
    ap.add_argument("--min-cell", type=int, default=5,
                    help="collapse rare (cat,type) rows/cols below this to keep "
                         "the chi-square valid")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.file)]
    p1 = [r for r in rows if r.get("phase") == "pass1"
          and r.get("diagnosis")]           # non-GOOD, diagnosed
    if len(p1) < 30:
        print(f"Only {len(p1)} diagnosed failures -- too few for a reliable "
              f"chi-square. Run more problems before deciding on FST.")
        return

    # contingency table: category x diagnosis
    table = defaultdict(Counter)
    for r in p1:
        cat = r["category"]["category_type"] if isinstance(r["category"], dict) else r["category"]
        table[cat][r["diagnosis"]] += 1

    cats = sorted(table)
    types = sorted({t for c in table for t in table[c]})
    print(f"Diagnosed failures: {len(p1)}  |  categories: {len(cats)}  "
          f"types: {len(types)}\n")

    # print the table
    hdr = "category".ljust(12) + "".join(t[:6].rjust(8) for t in types) + "   total"
    print(hdr); print("-" * len(hdr))
    for c in cats:
        row = table[c]
        line = c.ljust(12) + "".join(str(row.get(t, 0)).rjust(8) for t in types)
        print(line + str(sum(row.values())).rjust(8))
    print()

    try:
        from scipy.stats import chi2_contingency
        import numpy as np
        mat = np.array([[table[c].get(t, 0) for t in types] for c in cats])
        # drop all-zero rows/cols
        mat = mat[mat.sum(1) > 0][:, mat.sum(0) > 0]
        chi2, p, dof, exp = chi2_contingency(mat)
        low = (exp < 5).mean()
        print(f"chi2 = {chi2:.2f}   dof = {dof}   p = {p:.4f}")
        if low > 0.2:
            print(f"  ! {low:.0%} of expected cells < 5 -- p is approximate; "
                  f"consider collapsing rare categories or more data.")
        print()
        if p < 0.05:
            print("DECISION: p < 0.05 -> failure type DEPENDS on category. "
                  "FST is justified. Build it.")
        else:
            print("DECISION: p >= 0.05 -> type is INDEPENDENT of category for "
                  "this model. FST's premise does not hold here -- report this "
                  "as a finding and do NOT build FST.")
    except ImportError:
        print("scipy not installed. pip install scipy, then re-run. "
              "(The table above is still the input to the test.)")


if __name__ == "__main__":
    main()