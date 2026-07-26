"""
compare_baselines.py -- run several methods on the SAME eval problems and
tabulate accuracy, so every comparison is apples-to-apples.

THE POINT: you cannot compare FADE-on-500 to a published number computed on a
different set. Every baseline here runs on the *identical* 500 problems FADE
runs on (same seed selection, same eval order, cached in artifacts/). Published
benchmark numbers become sanity checks only ("my 8-shot ~ 14% matches
literature"), never the comparison itself.

Methods included (each is one row of the final table):
  zero_shot          strategy 0
  zero_shot_cot      strategy 1
  few_shot_8         strategy 2
  cot_few_shot       strategy 13
  self_consistency   strategy 16 (maj@5)
  immediate_retry    strategy 2, but each problem sampled a 2nd time (no
                     diagnosis, no typed exemplars) -- the NULL HYPOTHESIS for
                     FADE's retry: if typed retry doesn't beat this, retry adds
                     nothing.
  fade               strategy 2 + the full typed deferred-retry loop
                     (run separately via kaggle_run.py; its final_accuracy from
                     run_summary.json goes in the same table)

Usage:
    python compare_baselines.py --n 500 --model meta-llama/Llama-2-7b-chat-hf \\
        --methods zero_shot few_shot_8 cot_few_shot self_consistency immediate_retry \\
        --max-new-tokens 320

Writes store/baselines.json and store/baselines.csv, and prints the table.
Resumable per method: re-running skips (method, problem) pairs already scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import kaggle_run as K
import generation as G
from generation import ExemplarGenerator, STRATEGY_NAMES
from categorizer import QuestionCategorizer
from exemplar_selector import ExemplarSelector
from similarity import backend_name

REPO = Path(__file__).resolve().parent
OUT = REPO / "store"
OUT.mkdir(exist_ok=True)

# method name -> (strategy_id, is_immediate_retry)
METHODS = {
    "zero_shot":        (0,  False),
    "zero_shot_cot":    (1,  False),
    "few_shot_8":       (2,  False),
    "cot_few_shot":     (13, False),
    "self_consistency": (16, False),
    "immediate_retry":  (2,  True),
}


def _solve_once(gen, seeds, knn, strategy_id, q, cat_info, max_new_tokens):
    """First attempt: the strategy's own decoding (greedy for strategy 2)."""
    res = gen.generate(q, strategy_id, manual_exemplars=seeds, pool_exemplars=[],
                       category_info=cat_info, max_new_tokens=max_new_tokens,
                       return_logprobs=False, knn_fn=knn)
    return res["trace"]


def _resample(gen, seeds, strategy_id, q, cat_info, max_new_tokens, temperature=0.8):
    """A STOCHASTIC re-sample of the same prompt -- the honest 'immediate retry'.
    Must use temperature > 0, else re-sampling a greedy prompt reproduces the
    identical trace and recovers nothing (a trivially-weak baseline). We rebuild
    the same strategy prompt and run it through _run_model with sampling on."""
    prompt, _ = G._build_prompt(strategy_id, q, seeds, [], cat_info, None)
    traces, _ = gen._run_model(prompt, temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               num_return_sequences=1, return_logprobs=False)
    return traces[0]


def run_method(gen, manual, eval_problems, method, n, max_new_tokens, retrieval):
    """Run ONE method over the first n eval problems. Returns per-problem records.
    Resumable: reads any existing rows for this method and skips them."""
    strategy_id, immediate = METHODS[method]
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = K._retrieval_fn(retrieval)

    rows_path = OUT / f"baseline_{method}.jsonl"
    done = {}
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            done[r["id"]] = r
    f = open(rows_path, "a")

    problems = eval_problems[:n]
    correct = sum(r["correct"] for r in done.values())
    t0 = time.time()
    for i, prob in enumerate(problems, 1):
        item_id = f"p1_{i:05d}"
        if item_id in done:
            continue
        q, gold_sol, gold_ans = prob["question"], prob["answer"], prob["gold_answer"]
        cat_info = cat.categorize(q)

        trace = _solve_once(gen, seeds, knn, strategy_id, q, cat_info, max_new_tokens)
        comps, label, diag, scored = K.score_trace(q, trace, gold_sol, gold_ans)
        ok = bool(comps.correct)

        # immediate retry: if wrong, STOCHASTICALLY re-sample once (temp>0, NO
        # diagnosis, NO typed exemplars). This is the honest null for FADE:
        # "does typed diagnosis beat just sampling again?"
        retried_ok = False
        if immediate and not ok:
            trace2 = _resample(gen, seeds, strategy_id, q, cat_info, max_new_tokens)
            c2, _, _, _ = K.score_trace(q, trace2, gold_sol, gold_ans)
            retried_ok = bool(c2.correct)
        final_ok = ok or retried_ok

        rec = {"id": item_id, "correct": final_ok, "pass1_correct": ok,
               "category": cat_info["category_type"],
               "pred": comps.final_answer, "gold": gold_ans}
        f.write(json.dumps(rec) + "\n"); f.flush()
        correct += final_ok
        if i % 25 == 0 or i == len(problems):
            print(f"  [{method}] {i}/{len(problems)} | acc={correct/i:.1%} "
                  f"| {(time.time()-t0)/max(i,1):.1f}s/prob")
    return correct, len(problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--methods", nargs="+",
                    default=["few_shot_8", "immediate_retry"],
                    help="in-harness baselines to run (the fair comparison). "
                         "few_shot_8 = does retry help at all; immediate_retry = "
                         "does DIAGNOSIS help (the null). zero_shot/cot_few_shot/"
                         "self_consistency are available but usually CITED for "
                         "external context rather than re-run.")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--retrieval", choices=["knn", "3stage"], default="3stage")
    ap.add_argument("--exemplar-budget", type=int, default=8,
                    help="exemplars per prompt, enforced across all arms")
    ap.add_argument("--split", choices=["train","test"], default="train",
                    help="evaluate baselines on the same split as FADE")
    ap.add_argument("--seed-pool", type=int, default=300)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    args = ap.parse_args()

    K.MODEL_ID = args.model
    G.set_exemplar_budget(args.exemplar_budget)
    K.version_banner()
    print(f"  model: {K.MODEL_ID}")
    token = K.resolve_hf_token(args.secret_name)
    K.verify_identity_and_access(token)

    # SAME eval set for every method (cached; identical order)
    if args.split == "test":
        train_problems = K._load_split("train", args.seed_pool + 100)
        test_problems = K._load_split("test")
        manual, eval_problems = K.build_seeds_and_eval_order(
            train_problems, args.seed_pool, eval_split_problems=test_problems)
    else:
        problems = K.load_gsm8k(args.seed_pool + args.n + 100)
        manual, eval_problems = K.build_seeds_and_eval_order(problems, args.seed_pool)
    model, tok = K.load_model(token)
    gen = ExemplarGenerator(model=model, tokenizer=tok)

    print("\n" + "#" * 70)
    print(f"#  BASELINE COMPARISON | n={args.n} | SAME {args.n} problems each")
    print(f"#  model={K.MODEL_ID} | backend={backend_name()}")
    print("#" * 70)

    results = {}
    for method in args.methods:
        if method not in METHODS:
            print(f"  !! unknown method '{method}', skipping"); continue
        print(f"\n=== {method} ({STRATEGY_NAMES[METHODS[method][0]]}"
              f"{', +immediate retry' if METHODS[method][1] else ''}) ===")
        c, n = run_method(gen, manual, eval_problems, method, args.n,
                          args.max_new_tokens, args.retrieval)
        results[method] = {"correct": c, "n": n, "accuracy": round(c / n, 4)}

    # merge FADE's own result if a run_summary.json exists
    rs = OUT / "run_summary.json"
    if rs.exists():
        s = json.loads(rs.read_text())
        results["fade"] = {"correct": s.get("final_correct"),
                           "n": s.get("n_problems"),
                           "accuracy": s.get("final_accuracy")}

    (OUT / "baselines.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "baselines.csv", "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["method", "correct", "n", "accuracy"])
        for m, r in results.items():
            w.writerow([m, r["correct"], r["n"], r["accuracy"]])

    print("\n" + "=" * 60)
    print(f"  COMPARISON TABLE  (all on the same {args.n} problems)")
    print("=" * 60)
    print(f"  {'method':<20} {'acc':>7}   {'correct/n'}")
    for m, r in sorted(results.items(), key=lambda x: -(x[1]['accuracy'] or 0)):
        print(f"  {m:<20} {r['accuracy']:>7.1%}   {r['correct']}/{r['n']}")
    print("=" * 60)
    print(f"  written: {OUT/'baselines.json'} , {OUT/'baselines.csv'}")


if __name__ == "__main__":
    main()