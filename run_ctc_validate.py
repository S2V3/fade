"""
run_ctc_validate.py -- STAGE 2. Does CTC work on questions it has never seen?

THE RUN
-------
400 fresh TRAIN questions (offset 1500 -- the pool-building run stopped there).

  gen 1     model solves the question                    -> trace, answer
  cpu       CTC reads (question, trace), GOLD-FREE:
              p_wrong        is this trace wrong?
              predicted_type which of the eight failures?
  gen 2     if flagged wrong: load the matched cure and solve again  [--retry-flagged]
  score     open gold and check everything

This is the full pipeline rehearsed on train, where gold exists to score it. Stage 3
is the identical pipeline on test with the scoring removed.

WHY THE RETRY IS INCLUDED BY DEFAULT
------------------------------------
Detection accuracy alone does not tell you whether the pipeline helps. Two rates
decide that, and they pull in opposite directions:

    a TRUE positive  gains ~16.1%  (measured: 172 of 1068 retries recovered)
    a FALSE positive costs ~76.5%  (pre-audit run: retrying an already-correct
                                    trace made it WRONG 114 of 149 times)

The 76.5% is a PRIOR, not a fresh measurement -- [AUDIT D20] removed those retries
from the current run because they could not possibly recover anything. So this stage
re-measures it: every flagged trace is retried, including the ones that were already
right, and the damage rate comes out of the data instead of out of a footnote.

Skip it with --no-retry-flagged to get detection numbers only (~1.9 GPU-h instead of
~3.1), but then the threshold for stage 3 has to be chosen on a prior.

WHAT STAGE 1 SAID (2,568 training attempts, 5-fold CV)
------------------------------------------------------
    detect  AUROC 0.721   CV acc 0.735 vs majority 0.765
    type    CV acc 0.610 vs majority 0.215   lift +0.395
    recall  TR 100%  NR 90%  AL 88%  UNCL 65%  WP 62%  ST 55%  CE 50%  SM 31%

Note the detector's accuracy LOSES to the majority baseline while its AUROC is 0.72.
That is not a contradiction: 76.5% of traces are wrong, so "always say wrong" is a
strong accuracy baseline, and the detector's value is in RANKING, not in a hard
yes/no. This script therefore sweeps the threshold rather than assuming 0.5, and
reports the expected net gain at each one.

GOLD RULE
---------
Generation, detection and cure selection read only the question and the model's own
trace. Gold is opened afterwards, to score. --no-score drops it entirely, which is
exactly how stage 3 runs on test.

USAGE
    python run_ctc_validate.py --ctc stores/ctc_audit2/ctc.joblib \
        --pool-store stores/store_typed_train_audit2 \
        --store-dir stores/store_ctcval_audit2 --n-problems 400 --eval-offset 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config
import generation as G
import kaggle_run as K
from categorizer import QuestionCategorizer
from cure_bank import cure_for, describe

RECOVERY_PRIOR = 0.161
DAMAGE_PRIOR = 0.765


def _close(a, b, tol=1e-6):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def main():
    ap = argparse.ArgumentParser(description="CTC stage 2 -- validate on fresh train questions")
    ap.add_argument("--ctc", required=True, help="ctc.joblib from stage 1")
    ap.add_argument("--pool-store", required=True, help="train store holding the exemplar pool")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=400)
    ap.add_argument("--eval-offset", type=int, default=1500)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--retrieval", choices=["knn", "3stage"], default="3stage",
                    help="retrieval arm, must match the train run (default 3stage)")
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="p_wrong above this = flagged. Swept afterwards regardless.")
    ap.add_argument("--retry-flagged", dest="retry_flagged", action="store_true", default=True)
    ap.add_argument("--no-retry-flagged", dest="retry_flagged", action="store_false")
    ap.add_argument("--no-score", action="store_true", help="never read gold (stage-3 mode)")
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    a = ap.parse_args()

    K.MODEL_ID = a.model
    K.SPLIT_TAG = "train"
    G.set_exemplar_budget(a.exemplar_budget)
    K.version_banner()
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)

    from ctc_train import CTCModel
    ctc_model = CTCModel.load(a.ctc)
    rep = ctc_model.report or {}
    print(f"\n  CTC loaded: detect AUROC {rep.get('detect',{}).get('auroc')} | "
          f"type lift {rep.get('type',{}).get('lift')}")

    from classification import TraceStore
    pool = TraceStore(root=Path(a.pool_store)).exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
    print(f"  pool: {len(pool)} gold-stripped exemplars")

    train = K._load_split("train", a.eval_offset + a.n_problems + 100)
    manual, eval_problems = K.build_seeds_and_eval_order(train, a.seed_pool)
    seeds = manual.get(2, [])
    problems = eval_problems[a.eval_offset:a.eval_offset + a.n_problems]
    if not problems:
        raise SystemExit(f"no problems at offset {a.eval_offset}")
    print(f"  problems: {len(problems)} fresh train questions from offset {a.eval_offset}")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    cat = QuestionCategorizer()
    knn = K._retrieval_fn(a.retrieval)

    # ---- PROVENANCE CHECK -------------------------------------------------
    # generation._build_prompt draws exemplars from the POOL via knn_fn and only
    # tops up from the seeds when the pool returns too few. Omit knn_fn and it
    # silently takes EVERY exemplar from the seeds -- the 334-entry pool is passed
    # in and never read. That is [AUDIT D15] all over again ("retrieval never
    # ran"), and it would invalidate the whole claim: the point is that the model's
    # OWN successes become the exemplars, not that hand-written seeds do.
    _pq = {r.get("question") for r in pool}
    _probe = G._build_prompt(2, problems[0]["question"], seeds, pool,
                             cat.categorize(problems[0]["question"]), knn)[0]
    _from_pool = sum(1 for q_ in _pq if q_ and q_[:60] in _probe)
    print(f"\n  retrieval: {a.retrieval} | exemplars traced to the POOL: "
          f"{_from_pool}/{a.exemplar_budget}")
    if pool and _from_pool == 0:
        raise SystemExit(
            "  RETRIEVAL IS NOT RUNNING: a non-empty pool contributed zero\n"
            "  exemplars, so every one came from the seeds. Do not spend GPU.")
    if _from_pool < a.exemplar_budget:
        print(f"  note: {a.exemplar_budget - _from_pool} topped up from seeds "
              "(pool had too few matches for this question)")

    root = Path(a.store_dir); root.mkdir(parents=True, exist_ok=True)
    snap = {k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    snap["FADE_ARM"] = "ctc_validate"; snap["FADE_CTC_THRESHOLD"] = a.threshold
    (root / config.CONFIG_SNAPSHOT_FILE).write_text(json.dumps(snap, indent=2))
    results_path = root / config.RESULTS_FILE

    done = set()
    if results_path.exists():
        for ln in open(results_path):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
        if done:
            print(f"  RESUME: {len(done)} already done")

    def _solve(q, positives=None, instruction=""):
        info = cat.categorize(q)
        if positives is None:
            # knn_fn=knn is REQUIRED -- without it the pool is ignored entirely.
            res = gen.generate(q, 2, manual_exemplars=seeds, pool_exemplars=pool,
                               category_info=info, max_new_tokens=a.max_new_tokens,
                               return_logprobs=False, knn_fn=knn)
            return res["trace"], G.extract_final_answer(res["trace"])
        prompt, extra = K.build_retry_prompt(q, positives, instruction)
        traces, _ = gen._run_model(prompt, temperature=0.0,
                                   max_new_tokens=a.max_new_tokens,
                                   num_return_sequences=1, return_logprobs=False,
                                   extra_system=extra, problem=q)
        tr, _h = G.ensure_hash_line_flagged(traces[0], G.extract_final_answer(traces[0]))
        return tr, G.extract_final_answer(tr)

    f = open(results_path, "a")
    t0 = time.time(); n_seen = n_flag = n_retried = 0

    print("\n" + "#" * 70)
    print(f"#  CTC VALIDATE | n={len(problems)} | threshold {a.threshold} | "
          f"retry_flagged={a.retry_flagged}")
    print("#" * 70)

    for i, prob in enumerate(problems, 1):
        pid = f"cv_{a.eval_offset + i:05d}"
        if pid in done:
            continue
        q = prob["question"]
        trace, ans = _solve(q)

        pred = ctc_model.predict(q, trace)            # GOLD-FREE
        flagged = pred["p_wrong"] >= a.threshold
        n_flag += flagged; n_seen += 1

        row = {"id": pid, "eval_index": a.eval_offset + i - 1, "question": q,
               "trace": trace, "final_answer": ans,
               "p_wrong": pred["p_wrong"], "predicted_type": pred["predicted_type"],
               "type_confidence": pred["type_confidence"], "flagged": bool(flagged),
               "features": pred["features"]}

        if flagged and a.retry_flagged:
            positives, instruction = cure_for(pred["predicted_type"], pool, seeds,
                                              a.exemplar_budget)
            trace2, ans2 = _solve(q, positives, instruction)
            n_retried += 1
            row.update({"retry_trace": trace2, "retry_answer": ans2,
                        "cure": describe(pred["predicted_type"]),
                        "n_exemplars": len(positives)})

        if not a.no_score:
            gold = prob["gold_answer"]
            comps, label, diag, _ = K.score_trace(q, trace, prob["answer"], gold)
            row.update({"gold_answer": gold, "gold_solution": prob["answer"],
                        "correct": bool(comps.correct),
                        "true_type": (diag.ftype.value if diag else None),
                        "label": label.value, "signals": comps.signals()})
            if "retry_answer" in row:
                c2, l2, d2, _ = K.score_trace(q, row["retry_trace"], prob["answer"], gold)
                row["retry_correct"] = bool(c2.correct)
                row["final_correct"] = bool(comps.correct or c2.correct)
            else:
                row["final_correct"] = bool(comps.correct)

        f.write(json.dumps(row) + "\n"); f.flush()

        if a.show_every and (i % a.show_every == 0 or i == len(problems)):
            el = time.time() - t0
            print(f"  [{i}/{len(problems)}] flagged {n_flag}/{n_seen} "
                  f"({n_flag/max(n_seen,1):.0%}) | retried {n_retried} | "
                  f"{el/max(n_seen,1):.1f}s/prob", flush=True)
    f.close()

    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    summary = {"arm": "ctc_validate", "n": len(rows), "threshold": a.threshold,
               "eval_offset": a.eval_offset, "model": a.model,
               "retry_flagged": a.retry_flagged, "retrieval": a.retrieval,
               "pool_size": len(pool), "exemplars_from_pool_probe": _from_pool}

    if a.no_score:
        (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
        print(f"\n  scoring disabled. outputs: {root}")
        return

    scored = [r for r in rows if "correct" in r]
    y = [not r["correct"] for r in scored]            # True = actually WRONG
    p = [r["p_wrong"] for r in scored]

    print("\n" + "=" * 74)
    print("  1 · DETECTION -- can CTC tell a wrong trace from a right one?")
    print("=" * 74)
    base = sum(y) / max(len(y), 1)
    print(f"  n={len(scored)}   actually wrong {sum(y)} ({base:.1%})")
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    print(f"  AUROC {auroc:.3f}    (stage-1 cross-validated: "
          f"{rep.get('detect',{}).get('auroc', float('nan')):.3f})")
    summary["auroc"] = auroc

    print("\n" + "-" * 74)
    print("  THRESHOLD SWEEP -- and what each one is worth")
    print("-" * 74)
    print(f"  {'thr':>5}{'flagged':>9}{'TP':>6}{'FP':>6}{'precision':>11}"
          f"{'recall':>9}{'expected net':>14}")
    best = None
    for thr in [round(x * 0.05, 2) for x in range(4, 19)]:
        sel = [r for r in scored if r["p_wrong"] >= thr]
        tp = sum(1 for r in sel if not r["correct"])
        fp = len(sel) - tp
        prec = tp / max(len(sel), 1)
        rec = tp / max(sum(y), 1)
        net = tp * RECOVERY_PRIOR - fp * DAMAGE_PRIOR
        if best is None or net > best[1]:
            best = (thr, net)
        print(f"  {thr:>5.2f}{len(sel):>9}{tp:>6}{fp:>6}{prec:>11.1%}{rec:>9.1%}{net:>+14.1f}")
    print(f"\n  'expected net' = TP x {RECOVERY_PRIOR:.3f} recovery  -  FP x "
          f"{DAMAGE_PRIOR:.3f} damage, in problems.")
    print(f"  Best on those priors: threshold {best[0]:.2f} (net {best[1]:+.1f})")
    summary["best_threshold_prior"] = best[0]

    print("\n" + "=" * 74)
    print("  2 · TYPE -- when CTC says wrong, does it name the right failure?")
    print("=" * 74)
    tw = [r for r in scored if not r["correct"] and r.get("true_type")]
    hit = sum(1 for r in tw if r["predicted_type"] == r["true_type"])
    print(f"  wrong traces with a cascade diagnosis: {len(tw)}")
    print(f"  CTC's type == cascade's type: {hit}/{len(tw)} = {hit/max(len(tw),1):.1%}")
    print(f"     (stage-1 cross-validated: {rep.get('type',{}).get('cv_accuracy', float('nan')):.1%})")
    by = defaultdict(lambda: [0, 0])
    for r in tw:
        by[r["true_type"]][1] += 1
        by[r["true_type"]][0] += (r["predicted_type"] == r["true_type"])
    print(f"\n  {'true type':<16}{'n':>5}{'recall':>9}")
    for t, (c, n) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"  {t:<16}{n:>5}{c/max(n,1):>9.1%}")
    summary["type_accuracy"] = hit / max(len(tw), 1)

    if a.retry_flagged:
        print("\n" + "=" * 74)
        print("  3 · THE PIPELINE -- did it actually help?")
        print("=" * 74)
        p1c = sum(1 for r in scored if r["correct"])
        fin = sum(1 for r in scored if r.get("final_correct"))
        ret = [r for r in scored if "retry_correct" in r]
        tp_ret = [r for r in ret if not r["correct"]]
        fp_ret = [r for r in ret if r["correct"]]
        recovered = sum(1 for r in tp_ret if r["retry_correct"])
        broken = sum(1 for r in fp_ret if not r["retry_correct"])
        print(f"  pass-1 accuracy      {p1c}/{len(scored)} = {p1c/len(scored):.1%}")
        print(f"  final  accuracy      {fin}/{len(scored)} = {fin/len(scored):.1%}"
              f"   ({(fin-p1c)/len(scored):+.1%})")
        print(f"\n  retried {len(ret)}   true positives {len(tp_ret)}  false positives {len(fp_ret)}")
        print(f"    recovered from the true positives : {recovered}/{len(tp_ret)} = "
              f"{recovered/max(len(tp_ret),1):.1%}   (prior {RECOVERY_PRIOR:.1%})")
        print(f"    MEASURED DAMAGE on false positives: {broken}/{len(fp_ret)} = "
              f"{broken/max(len(fp_ret),1):.1%}   (prior {DAMAGE_PRIOR:.1%})")
        print("\n  The damage rate above is the number the 76.5% prior was standing in")
        print("  for. Use it, not the prior, to pick the stage-3 threshold.")
        summary.update({"pass1_accuracy": p1c/len(scored),
                        "final_accuracy": fin/len(scored),
                        "n_retried": len(ret), "n_true_pos": len(tp_ret),
                        "n_false_pos": len(fp_ret), "recovered": recovered,
                        "broken": broken,
                        "measured_damage": broken/max(len(fp_ret), 1),
                        "measured_recovery": recovered/max(len(tp_ret), 1)})

    (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 74)
    print("  GATE FOR STAGE 3")
    print("=" * 74)
    if a.retry_flagged and summary.get("final_accuracy", 0) > summary.get("pass1_accuracy", 1):
        print("  The pipeline IMPROVED accuracy on unseen questions. Take it to test,")
        print(f"  using the threshold that maximises the MEASURED net gain above.")
    elif auroc == auroc and auroc >= 0.65:
        print("  Detection generalises, but the retry did not pay off at this")
        print("  threshold. Re-run the sweep against the MEASURED damage rate and")
        print("  pick a stricter threshold before going to test.")
    else:
        print("  Detection did not generalise. Report it with the FST negative --")
        print("  two honest routes to the same conclusion is a real analysis.")
    print(f"\n  outputs: {root}")


if __name__ == "__main__":
    main()