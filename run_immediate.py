"""
run_immediate.py -- STANDALONE inline immediate-retry baseline.

Separate entry point so it does NOT modify kaggle_run.py (which may be running a
FADE job). It imports the shared helpers from kaggle_run and generation.

Definition (inline, per question):
    solve once (greedy).
    if correct -> if GOOD, add to pool; next question.
    if wrong   -> retry the SAME question IMMEDIATELY, up to `--retries` times
                  (blind re-sample at temp 0.8; same seed exemplars; NO diagnosis,
                  NO typed instruction). If a retry is correct -> if GOOD add to
                  pool, stop. If still wrong after all retries -> DISCARD (never
                  revisited).

No MEDIUM/BAD queues, no deferred phase, no second sweep. The pool is collected
(GOOD traces) but NOT used as retrieval exemplars -- every prompt uses the fixed
seeds only -- so this stays a clean no-diagnosis null baseline.

Usage:
    python run_immediate.py --n-problems 1319 --strategy 2 --split test \\
        --model meta-llama/Llama-2-7b-chat-hf --exemplar-budget 8 \\
        --max-new-tokens 320 --retries 2 --run-name immediate --secret-name HF_TOKEN
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import kaggle_run as K          # reuse everything; do not modify it
import generation as G
from generation import ExemplarGenerator
from classification import Label, TraceStore, STORE_ROOT
from categorizer import QuestionCategorizer
from similarity import backend_name
import config


def _resume_done_ids(results_path: Path) -> set:
    done = set()
    if results_path.exists():
        for line in open(results_path):
            try:
                r = json.loads(line)
                if r.get("phase") == "immediate" and r.get("id"):
                    done.add(r["id"])
            except Exception:
                continue
    if done:
        print(f"  resume: {len(done)} problems already done, continuing")
    return done


def run_immediate(gen, seeds, eval_problems, strategy_id, n_problems,
                  max_new_tokens, retries=2, show_every=50, run_name="immediate"):
    root = (STORE_ROOT.parent / f"store_{run_name}") if run_name else STORE_ROOT
    store = TraceStore(root=root)
    store_root = store.root
    print(f"  outputs -> {store_root}")

    cat = QuestionCategorizer()
    budget = G.EXEMPLAR_BUDGET or G.N_SHOTS
    problems = eval_problems[:n_problems]

    results_path = store_root / config.RESULTS_FILE
    done_ids = _resume_done_ids(results_path)
    results_f = open(results_path, "a")
    def log(rec): results_f.write(json.dumps(rec) + "\n"); results_f.flush()

    print("\n" + "#" * 70)
    print(f"#  IMMEDIATE RUN (inline retry x{retries}, discard-on-fail) | "
          f"n={len(problems)} | backend={backend_name()}")
    print(f"#  outputs -> store_{run_name}/")
    print("#" * 70)

    pool = []
    first_correct = final_correct = recovered = n_gen = 0
    t_start = time.time()

    for i, prob in enumerate(problems, 1):
        item_id = f"p1_{i:05d}"
        if item_id in done_ids:
            continue
        q, gold_sol, gold_ans = prob["question"], prob["answer"], prob["gold_answer"]
        cat_info = cat.categorize(q)

        # ---- initial attempt (greedy, seeds only) ----
        res = gen.generate(q, strategy_id, manual_exemplars=seeds, pool_exemplars=[],
                           category_info=cat_info, max_new_tokens=max_new_tokens,
                           return_logprobs=False, knn_fn=None)
        n_gen += 1
        trace = res["trace"]
        comps, label, diag, scored = K.score_trace(q, trace, gold_sol, gold_ans)
        ok = bool(comps.correct)
        attempts = 1
        if ok:
            first_correct += 1

        # ---- inline retries only if wrong ----
        while (not ok) and attempts <= retries:
            prompt = K.build_retry_prompt(q, seeds[:budget], "")     # blind, no instruction
            rtr, _ = gen._run_model(prompt, temperature=0.8,
                                    max_new_tokens=max_new_tokens,
                                    num_return_sequences=1, return_logprobs=False)
            n_gen += 1
            attempts += 1
            trace = rtr[0]
            comps, label, diag, scored = K.score_trace(q, trace, gold_sol, gold_ans)
            ok = bool(comps.correct)
            if ok:
                recovered += 1

        if ok:
            final_correct += 1
            if label is Label.GOOD:
                store.add(item_id, q, trace, gold_sol, gold_ans, label, comps)
                pool.append(item_id)
        # still wrong after `retries` -> discard, never revisited

        log({"phase": "immediate", "id": item_id, "eval_index": i - 1,
             "question": q, "gold_answer": gold_ans, "category": cat_info,
             "attempts": attempts, "correct": ok,
             "first_attempt_correct": (attempts == 1 and ok),
             "recovered": (ok and attempts > 1),
             "trace": trace, "signals": comps.signals(), "label": label.value})

        if show_every and (i % show_every == 0 or i == len(problems)):
            print(f"  {i}/{len(problems)} | first={first_correct} "
                  f"final={final_correct} recovered={recovered} "
                  f"pool={len(pool)} gens={n_gen}")

    n = len(problems)
    summary = {
        "status": "complete", "mode": "immediate_inline",
        "n_problems": n, "retries_per_problem": retries,
        "pass1_correct": first_correct, "pass1_accuracy": first_correct / max(n, 1),
        "final_correct": final_correct, "final_accuracy": final_correct / max(n, 1),
        "recovered_by_retry": recovered, "final_pool_size": len(pool),
        "iterations": [], "model": K.MODEL_ID, "split": K.SPLIT_TAG,
        "retry_mode": "immediate_inline", "exemplar_budget": budget,
        "generation_version": G.GENERATION_VERSION, "embedding_backend": backend_name(),
        "retrieval": "none(seeds only)",
        "pass1_diagnosis_distribution": {}, "per_category_accuracy": {},
        "cost": {"generations": n_gen, "wall_seconds": int(time.time() - t_start),
                 "sec_per_generation": round((time.time() - t_start) / max(n_gen, 1), 2),
                 "approx_tokens": 0,
                 "projected_full_gsm8k_gpu_hours": round(
                     (time.time() - t_start) / max(n, 1) * 7473 / 3600, 2)},
        "extraction_diagnostics": {},
    }
    (store_root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    print(f"\n  IMMEDIATE done | pass-1 {summary['pass1_accuracy']:.2%} | "
          f"final {summary['final_accuracy']:.2%} | recovered {recovered} | gens {n_gen}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Standalone inline immediate-retry baseline")
    ap.add_argument("--n-problems", type=int, default=1319)
    ap.add_argument("--strategy", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--run-name", default="immediate")
    ap.add_argument("--show-every", type=int, default=50)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    args = ap.parse_args()

    if args.model:
        K.MODEL_ID = args.model
    K.SPLIT_TAG = args.split
    G.set_exemplar_budget(args.exemplar_budget)
    K.version_banner()
    print(f"  model: {K.MODEL_ID} | immediate inline retry x{args.retries}")
    token = K.resolve_hf_token(args.secret_name)
    K.verify_identity_and_access(token)

    # SAME eval order as the other arms (train seeds, test eval) via the shared builder
    if args.split == "test":
        train = K._load_split("train", args.seed_pool + 100)
        test = K._load_split("test")
        manual, eval_problems = K.build_seeds_and_eval_order(
            train, args.seed_pool, eval_split_problems=test)
    else:
        problems = K.load_gsm8k(args.seed_pool + args.n_problems + 100)
        manual, eval_problems = K.build_seeds_and_eval_order(problems, args.seed_pool)

    seeds = manual.get(args.strategy, [])
    model, tok = K.load_model(token)
    gen = ExemplarGenerator(model=model, tokenizer=tok)

    run_immediate(gen, seeds, eval_problems, args.strategy, args.n_problems,
                  args.max_new_tokens, retries=args.retries,
                  show_every=args.show_every, run_name=args.run_name)


if __name__ == "__main__":
    main()