"""
score_mpv.py -- the ONLY place gold is opened for MPV, once, at the end.

Reads a finished store (pass-1 + retry traces) and its mpv_probes.jsonl, then
reports, on the same questions, paired:

    pass 1                    the model's first answer
    REPLACE                   flagged -> retry            (what stage 3 shipped)
    SELECT symbolic           selector.symbolic over {pass-1, retry}   (fade_selector)
    SELECT MPV                metamorphic selection over {pass-1, retry}
    SELECT MPV + backsub      ... plus the back-substituted probe programs

and, for detection:

    MPV wrongness of pass-1   1 - consistency, AUROC against gold, by failure type
    p_wrong (CTC)             the shipped detector on the same rows
    blend                     and whether adding MPV moves it

Nothing here influences generation.  Run it after run_mpv.py.

USAGE
    python score_mpv.py --store stores/store_ctctest_audit2_a1 --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import mpv
import selector as SEL
from stats import mcnemar, wilson_ci
from extraction import extract_final_answer


def _auroc(scores, labels):
    """labels 1 = wrong. Rank-based, ties at 0.5. None if one class."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    return tot / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--probes", default=None, help="default <store>/mpv_probes.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="default <store>/mpv_report.json")
    ap.add_argument("--min-probes", type=int, default=1)
    ap.add_argument("--gold-jsonl", default=None,
                    help="offline gold: rows {question, answer(annotated), gold_answer}; "
                         "default loads the GSM8K split from HF")
    a = ap.parse_args()

    import kaggle_run as K
    K.SPLIT_TAG = a.split
    root = Path(a.store)
    rows = [json.loads(l) for l in open(root / "results.jsonl") if l.strip()]
    leak = [k for r in rows for k in ("gold_answer", "correct", "true_type") if k in r]
    assert not leak, f"GOLD IN THE RUN OUTPUT: {set(leak)}"

    probes = defaultdict(dict)
    ppath = Path(a.probes) if a.probes else root / "mpv_probes.jsonl"
    if ppath.exists():
        for ln in open(ppath):
            if ln.strip():
                d = json.loads(ln)
                probes[d["id"]][d["probe_index"]] = mpv.Probe.from_dict(d)
    print(f"  {len(rows)} rows | probes for {len(probes)} questions "
          f"({sum(len(v) for v in probes.values())} probes)")

    if a.gold_jsonl:
        gold = {}
        for ln in open(a.gold_jsonl):
            if ln.strip():
                d = json.loads(ln); gold[d["question"]] = d
    else:
        gold = {p["question"]: p for p in K._load_split(a.split, None)}

    def correct_trace(q, trace, g):
        return bool(K.score_trace(q, trace, g["answer"], g["gold_answer"])[0].correct)

    def true_type(q, trace, g):
        comps, label, diag, _ = K.score_trace(q, trace, g["answer"], g["gold_answer"])
        if comps.correct:
            return "CORRECT"
        ft = getattr(diag, "ftype", None)
        return str(getattr(ft, "value", ft) or "UNCLASSIFIED")

    arms = {k: {} for k in ("pass1", "replace", "select_symbolic", "select_mpv",
                            "select_mpv_backsub")}
    reasons = Counter(); backsub_wins = Counter(); n_backsub = 0
    det = []          # (id, mpv_score or None, p_wrong, wrong?, true type)
    per_q = []

    for r in rows:
        q = r["question"]; g = gold.get(q)
        if g is None:
            continue
        qid = r["id"]
        c1 = correct_trace(q, r["trace"], g)
        a1 = extract_final_answer(r["trace"])
        cands = [(r["trace"], a1, "pass1")]
        cr = None
        if r.get("retry_trace"):
            cr = correct_trace(q, r["retry_trace"], g)
            cands.append((r["retry_trace"], extract_final_answer(r["retry_trace"]), "retry"))
        pr = [probes[qid][j] for j in sorted(probes.get(qid, {}))]

        arms["pass1"][qid] = c1
        arms["replace"][qid] = cr if cr is not None else c1

        # symbolic selector (the fade_selector baseline), same candidates
        i_sym, _ = SEL.symbolic([(t, a_) for t, a_, _ in cands], question=q)
        arms["select_symbolic"][qid] = [c1, cr][i_sym] if cr is not None else c1

        # MPV without back-substitution
        i_m, why_m, _ = mpv.select(q, cands, pr, use_backsub=False, min_probes=a.min_probes)
        arms["select_mpv"][qid] = [c1, cr][i_m] if cr is not None else c1

        # MPV with back-substitution
        i_b, why_b, scores = mpv.select(q, cands, pr, use_backsub=True, min_probes=a.min_probes)
        if i_b < len(cands):
            cb = [c1, cr][i_b] if cr is not None else c1
        else:
            cb = mpv._close(scores[i_b].answer, g["gold_answer"]) if scores[i_b].answer is not None else False
            n_backsub += 1; backsub_wins[cb] += 1
        arms["select_mpv_backsub"][qid] = cb
        reasons[why_b] += 1

        # detection on the pass-1 trace
        ms = mpv.detection_score(q, r["trace"], a1, pr) if pr else None
        det.append((qid, ms, r.get("p_wrong"), 0 if c1 else 1, str(true_type(q, r["trace"], g))))
        per_q.append({"id": qid, "pass1": c1, "retry": cr, "select": cb,
                      "reason": why_b, "n_probes": len(pr),
                      "chosen": scores[i_b].origin if scores else "pass1"})

    n = len(arms["pass1"])
    print(f"\n  paired on {n} questions\n")
    base = arms["pass1"]
    rep = {}
    for k, v in arms.items():
        acc = sum(v.values()) / n
        mc = mcnemar(base, v)
        mr = mcnemar(arms["replace"], v)
        lo, hi = wilson_ci(sum(v.values()), n)
        print(f"  {k:<20} {sum(v.values()):>5}/{n} = {acc:6.1%}  [{lo:.1%},{hi:.1%}]"
              f"   vs pass1: +{mc['n01']} -{mc['n10']} p={mc['p']:.4f}"
              f"   vs REPLACE: +{mr['n01']} -{mr['n10']} p={mr['p']:.4f}")
        rep[k] = {"correct": sum(v.values()), "n": n, "acc": acc,
                  "vs_pass1": mc, "vs_replace": mr}
    print(f"\n  selection reasons: {dict(reasons)}")
    print(f"  back-substituted candidate chosen {n_backsub} times: "
          f"right {backsub_wins[True]}  wrong {backsub_wins[False]}")

    # ---- detection ---------------------------------------------------------
    have = [(s, pw, y, t) for _, s, pw, y, t in det if s is not None and pw is not None]
    print(f"\n  DETECTION on pass-1 traces  (MPV applicable on {len(have)}/{len(det)} = "
          f"{len(have)/max(len(det),1):.1%}; the rest abstain)")
    if have:
        au_m = _auroc([s for s, _, _, _ in have], [y for _, _, y, _ in have])
        au_p = _auroc([pw for _, pw, _, _ in have], [y for _, _, y, _ in have])
        best = None
        for w in (0.25, 0.5, 0.75):
            au_b = _auroc([w * s + (1 - w) * pw for s, pw, _, _ in have], [y for _, _, y, _ in have])
            if best is None or au_b > best[1]:
                best = (w, au_b)
        print(f"    AUROC  MPV alone {au_m:.3f}   p_wrong alone {au_p:.3f}   "
              f"blend(w={best[0]}) {best[1]:.3f}")
        rep["detection"] = {"n": len(have), "auroc_mpv": au_m, "auroc_pwrong": au_p,
                            "auroc_blend": best[1], "blend_w": best[0]}
        # by true failure type: fraction of wrong traces with MPV score = 1 (fully refuted)
        print("    refuted (consistency 0) by TRUE failure type:")
        by = defaultdict(lambda: [0, 0])
        for s, _, y, t in have:
            by[t][1] += 1; by[t][0] += (s >= 1.0 - 1e-9)
        rep["refuted_by_type"] = {}
        for t, (k, m) in sorted(by.items(), key=lambda x: -x[1][1]):
            print(f"      {t:<12} {k:>4}/{m:<4} = {k/m:6.1%}")
            rep["refuted_by_type"][t] = {"refuted": k, "n": m}
        print("    (CORRECT row = false-refutation rate; lower is better. Compare WP/SM "
              "against the 12%/28% the shipped detector flags.)")

    out = Path(a.out) if a.out else root / "mpv_report.json"
    rep["reasons"] = dict(reasons)
    rep["backsub"] = {"chosen": n_backsub, "right": backsub_wins[True], "wrong": backsub_wins[False]}
    json.dump(rep, open(out, "w"), indent=2)
    with open(root / "mpv_per_question.jsonl", "w") as f:
        for d in per_q:
            f.write(json.dumps(d) + "\n")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
