"""
run_ctc.py -- the CTC pilot. Fresh TRAIN questions, two generations each, no EDER.

WHAT IT MEASURES
----------------
For each question:

  gen 1   solve it normally                              -> trace, answer
  (cpu)   ctc.prepare(): pick a load-bearing number,
          perturb it, and predict SYMBOLICALLY from the
          trace's own chain what the answer must become  -> predicted
  gen 2   solve the PERTURBED question                   -> model_answer
  (cpu)   agrees = (model_answer == predicted)

Then, using gold ONLY to score the pilot: does `agrees` predict whether gen 1 was
right? That is the number that decides whether CTC is worth building on.

WHY FRESH QUESTIONS
-------------------
The 1,500 already run built the exemplar pool, so their pass-1 traces were produced
against a pool that was still growing. Regenerating both sides here, on unseen
train questions, under identical conditions, removes that confound. --eval-offset
defaults to 1500 so it continues where the pool-building run stopped.

WHAT THE OFFLINE PRE-CHECK ALREADY SHOWED  (1,500 traces, no GPU)
-----------------------------------------------------------------
    CTC applicable                                       1007/1500 = 67.1%
    chain prediction == TRUE perturbed answer
        on CORRECT traces                                247/330  = 74.8%
        on WRONG traces                                   50/667  =  7.5%

A correct trace's chain is a faithful function of the inputs; a wrong trace's is
not. That is the mechanism CTC rides on -- but it is measured against gold, so it
is a train-only diagnostic, not the deployable signal. This run measures the
deployable version: whether the MODEL, re-asked, agrees with its OWN chain.

GOLD RULE
---------
Generation and the CTC decision touch only the question and the model's own trace.
Gold is read after the fact, to score. `--no-score` drops it entirely, which is how
this would run on test.

USAGE
    python run_ctc.py --store-dir stores/store_ctc_pilot --n-problems 400 \
        --eval-offset 1500 --model meta-llama/Llama-2-7b-chat-hf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config
import ctc
import generation as G
import kaggle_run as K
from categorizer import QuestionCategorizer


def main():
    ap = argparse.ArgumentParser(description="CTC pilot on fresh train questions")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=400)
    ap.add_argument("--eval-offset", type=int, default=1500)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--pool-store", help="optional: exemplar pool from the train run")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--factor", type=float, default=2.0,
                    help="perturbation multiplier for the chosen number")
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    ap.add_argument("--no-score", action="store_true",
                    help="do not read gold at all (how this runs on test)")
    a = ap.parse_args()

    K.MODEL_ID = a.model
    K.SPLIT_TAG = "train"
    G.set_exemplar_budget(a.exemplar_budget)
    K.version_banner()
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)

    train = K._load_split("train", a.eval_offset + a.n_problems + 100)
    manual, eval_problems = K.build_seeds_and_eval_order(train, a.seed_pool)
    seeds = manual.get(2, [])
    problems = eval_problems[a.eval_offset:a.eval_offset + a.n_problems]
    if not problems:
        raise SystemExit(f"no problems at offset {a.eval_offset}")

    pool = []
    if a.pool_store:
        from classification import TraceStore
        pool = TraceStore(root=Path(a.pool_store)).exemplars()
        for r in pool:
            assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
        print(f"  pool: {len(pool)} gold-stripped exemplars from {a.pool_store}")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    cat = QuestionCategorizer()

    root = Path(a.store_dir); root.mkdir(parents=True, exist_ok=True)
    snap = {k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    snap["FADE_ARM"] = "ctc_pilot"; snap["FADE_CTC_FACTOR"] = a.factor
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

    f = open(results_path, "a")
    t0 = time.time()
    n_seen = n_app = n_agree = 0
    reasons = Counter()

    print("\n" + "#" * 70)
    print(f"#  CTC PILOT | n={len(problems)} | offset={a.eval_offset} | 2 generations each")
    print("#" * 70)

    def _solve(q):
        info = cat.categorize(q)
        res = gen.generate(q, 2, manual_exemplars=seeds, pool_exemplars=pool,
                           category_info=info, max_new_tokens=a.max_new_tokens,
                           return_logprobs=False)
        return res["trace"], G.extract_final_answer(res["trace"])

    for i, prob in enumerate(problems, 1):
        pid = f"ctc_{a.eval_offset + i:05d}"
        if pid in done:
            continue
        q = prob["question"]
        trace, ans = _solve(q)

        r = ctc.prepare(q, trace, ans, factor=a.factor)
        model_ans = None
        if r.applicable:
            _, model_ans = _solve(r.perturbed_question)   # the ONLY extra generation
            ctc.judge(r, model_ans)
            n_app += 1
            n_agree += bool(r.agrees)
        else:
            reasons[r.reason] += 1
        n_seen += 1

        row = {"id": pid, "eval_index": a.eval_offset + i - 1, "question": q,
               "trace": trace, "final_answer": ans, "ctc": r.to_dict()}
        if not a.no_score:
            row["gold_answer"] = prob["gold_answer"]
            row["gold_solution"] = prob["answer"]
            row["correct"] = bool(ans is not None
                                  and ctc._close(ans, prob["gold_answer"]))
            if r.applicable:
                pg = ctc.perturbed_gold(prob["answer"], r.target_value, r.new_value)
                row["perturbed_gold"] = pg
                row["chain_matches_true"] = (pg is not None
                                             and ctc._close(r.predicted, pg))
                row["model_correct_on_perturbed"] = (pg is not None and model_ans is not None
                                                     and ctc._close(model_ans, pg))
        f.write(json.dumps(row) + "\n"); f.flush()

        if a.show_every and (i % a.show_every == 0 or i == len(problems)):
            el = time.time() - t0
            print(f"  [{i}/{len(problems)}] applicable {n_app}/{n_seen} "
                  f"({n_app/max(n_seen,1):.0%}) | agree {n_agree}/{max(n_app,1)} "
                  f"({n_agree/max(n_app,1):.0%}) | {el/max(n_seen,1):.1f}s/prob", flush=True)
    f.close()

    # ---------------------------------------------------------------- report
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    app = [r for r in rows if r["ctc"]["applicable"]]
    print("\n" + "=" * 70)
    print("  CTC PILOT")
    print("=" * 70)
    print(f"  problems     {len(rows)}")
    print(f"  applicable   {len(app)}  ({len(app)/max(len(rows),1):.1%})")
    for k, v in reasons.most_common():
        print(f"    skip: {k:<50}{v:>5}")

    summary = {"arm": "ctc_pilot", "n": len(rows), "applicable": len(app),
               "eval_offset": a.eval_offset, "factor": a.factor, "model": a.model}

    if not a.no_score and app:
        agree_ok = [r for r in app if r["ctc"]["agrees"]]
        dis = [r for r in app if not r["ctc"]["agrees"]]
        pc = lambda rs: sum(bool(r.get("correct")) for r in rs) / max(len(rs), 1)
        base = pc(rows)
        print(f"\n  THE NUMBER THAT DECIDES IT")
        print(f"    overall accuracy                      {base:.1%}")
        print(f"    accuracy when CTC AGREES              {pc(agree_ok):.1%}   n={len(agree_ok)}")
        print(f"    accuracy when CTC DISAGREES           {pc(dis):.1%}   n={len(dis)}")
        try:
            from sklearn.metrics import roc_auc_score
            y = [bool(r.get("correct")) for r in app]
            x = [bool(r["ctc"]["agrees"]) for r in app]
            au = roc_auc_score(y, x) if len(set(y)) > 1 else float("nan")
            print(f"    AUROC of the CTC flag                 {au:.3f}")
            summary["auroc"] = au
        except Exception:
            au = float("nan")
        summary.update({"acc_overall": base, "acc_agree": pc(agree_ok),
                        "acc_disagree": pc(dis), "n_agree": len(agree_ok),
                        "n_disagree": len(dis)})
        cm = [r for r in app if r.get("chain_matches_true") is not None]
        if cm:
            c = [r for r in cm if r.get("correct")]; w = [r for r in cm if not r.get("correct")]
            print(f"\n  chain prediction == TRUE perturbed answer (train-only diagnostic)")
            print(f"    on correct traces  {sum(bool(r['chain_matches_true']) for r in c)}/{len(c)}")
            print(f"    on wrong traces    {sum(bool(r['chain_matches_true']) for r in w)}/{len(w)}")

        print(f"\n  VERDICT")
        gap = pc(agree_ok) - pc(dis)
        if au == au and au >= 0.65:
            print(f"    GO. AUROC {au:.3f}, accuracy gap {gap:+.1%}. CTC is a usable")
            print( "    gold-free failure detector -- take it to the test split.")
        elif au == au and au >= 0.58:
            print(f"    MARGINAL. AUROC {au:.3f}, gap {gap:+.1%}. Real but weak. Try a")
            print( "    different --factor, or combine with the false-equation check")
            print( "    before committing more GPU.")
        else:
            print(f"    NO-GO. AUROC {au:.3f}. Report it beside the FST negative --")
            print( "    two independent routes to the same conclusion is a stronger")
            print( "    analysis than one.")

    (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    print(f"\n  outputs: {root}")


if __name__ == "__main__":
    main()
