"""Score the select-instead-of-replace policy on a finished --no-score store.

Run on Kaggle, where the GSM8K test split is reachable. Reads the store's
results.jsonl plus stage3_selector.jsonl (the gold-free picks) and reports
replace-always against select, on the same traces.

    python score_selector.py --store <dir> --picks stage3_selector.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--picks", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    a = ap.parse_args()

    import kaggle_run as K
    K.SPLIT_TAG, K.MODEL_ID = "test", a.model
    gold = {p["question"]: p for p in K._load_split("test", None)}

    rows = [json.loads(l) for l in open(f"{a.store}/results.jsonl") if l.strip()]
    pick = {p["id"]: p for p in (json.loads(l) for l in open(a.picks) if l.strip())}

    def ok(q, t):
        g = gold[q]
        return bool(K.score_trace(q, t, g["answer"], g["gold_answer"])[0].correct)

    p1 = repl = sel = 0
    # only the discordant pairs can move the number; count them by which way
    gain_r = lose_r = gain_s = lose_s = 0
    by_why = Counter()

    for r in rows:
        q = r["question"]
        if q not in gold:
            continue
        c1 = ok(q, r["trace"])
        p1 += c1
        if not r.get("retry_trace"):
            repl += c1
            sel += c1
            continue
        c2 = ok(q, r["retry_trace"])
        repl += c2                                   # replace-always
        p = pick.get(r["id"], {})
        keep_pass1 = p.get("pick") == "pass1"
        cs = c1 if keep_pass1 else c2
        sel += cs
        if c1 != c2:                                 # discordant
            gain_r += (not c1) and c2
            lose_r += c1 and (not c2)
            gain_s += (not c1) and cs
            lose_s += c1 and (not cs)
            by_why[(p.get("why"), "right" if cs == max(c1, c2) else "wrong")] += 1

    n = len(rows)
    print("=" * 62)
    print(f"  {n} problems, {sum(1 for r in rows if r.get('retry_trace'))} retried")
    print("=" * 62)
    print(f"  pass 1                {p1:>5}/{n} = {p1/n:>6.1%}")
    print(f"  REPLACE always        {repl:>5}/{n} = {repl/n:>6.1%}  ({(repl-p1)/n:+.1%})")
    print(f"  SELECT (gold-free)    {sel:>5}/{n} = {sel/n:>6.1%}  ({(sel-p1)/n:+.1%})")
    print(f"\n  on discordant pairs")
    print(f"    replace   gained {gain_r:>3}  lost {lose_r:>3}  net {gain_r-lose_r:+d}")
    print(f"    select    gained {gain_s:>3}  lost {lose_s:>3}  net {gain_s-lose_s:+d}")

    print(f"\n  which signal decided, and was it right")
    for why in ("agree", "arith", "detector"):
        r_ = by_why[(why, "right")]
        w_ = by_why[(why, "wrong")]
        if r_ + w_:
            print(f"    {why:<10}{r_:>4} right {w_:>4} wrong   {r_/(r_+w_):>6.1%}")

    try:
        from stats import mcnemar
        for lbl, after in (("REPLACE", "r"), ("SELECT", "s")):
            A, B = {}, {}
            for r in rows:
                q = r["question"]
                if q not in gold:
                    continue
                c1 = ok(q, r["trace"])
                A[r["id"]] = c1
                if not r.get("retry_trace"):
                    B[r["id"]] = c1
                else:
                    c2 = ok(q, r["retry_trace"])
                    B[r["id"]] = c2 if after == "r" else (
                        c1 if pick.get(r["id"], {}).get("pick") == "pass1" else c2)
            mc = mcnemar(A, B)
            print(f"  {lbl:<9} McNemar p = {mc['p']:.2e}"
                  + ("  SIGNIFICANT" if mc["p"] < 0.05 else ""))
    except Exception as e:
        print("  McNemar:", e)


if __name__ == "__main__":
    main()
