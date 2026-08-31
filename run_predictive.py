"""
run_predictive.py -- Phase 3 on the TEST split. Predict the failure BEFORE the
first attempt, preload the matched cure, and take ONE shot.

    question -> gold-free features -> predicted type -> preloaded cure -> 1 attempt

This is the deployable claim: no gold at inference, no prior failure to diagnose.
The exemplar pool and the predictor both come from the TRAIN run; the test split is
otherwise untouched.

TWO ARMS, same problems, same exemplar budget, one generation each:

    --arm fst        preloaded cure chosen by the predicted type
    --arm baseline   the same pool, plain 8-shot retrieval, no instruction

The baseline is what FST has to beat. Compare with compare_arms.py.

READ THIS BEFORE SPENDING GPU
-----------------------------
fst_probe.py already establishes, offline, that:
  * cure-routing is worth ~+9.5 pts when the type is KNOWN (oracle 17.4% vs 8.0%)
  * no model predicts the type above the 27.5% majority baseline
  * expected gain therefore brackets +2.6 to -3.2 pts
  * paired power to detect ~2 pts: 4% at n=100, 10% at n=500

So a null here is the expected outcome, and it will be a LOW-POWERED null. That is
still worth running if you want a measured result rather than a projection -- just
report the power limitation alongside it rather than claiming "no effect".

GOLD RULE: features read the question only; the pool is loaded through
TraceStore.exemplars(), which strips gold. generation.py's prompt-integrity assert
is active on every call.

USAGE
    python run_predictive.py --arm fst      --predictor fst.joblib \
        --pool-store stores/store_typed_train_audit2 --n-problems 500 \
        --store-dir stores/store_fst_test
    python run_predictive.py --arm baseline --pool-store ... --n-problems 500 \
        --store-dir stores/store_baseline_test
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
import generation as G
import kaggle_run as K
from categorizer import QuestionCategorizer
from classification import TraceStore, classify, needs_diagnosis, Label
from cure_bank import cure_for, describe
from diagnosis import diagnose_components


def load_pool(pool_store: str):
    """GOLD-STRIPPED exemplars from the train run's store."""
    store = TraceStore(root=Path(pool_store))
    pool = store.exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, \
            "GOLD LEAK: exemplars() returned a record carrying gold"
    return pool


def main():
    ap = argparse.ArgumentParser(description="FST on the test split")
    ap.add_argument("--arm", choices=["fst", "baseline"], required=True)
    ap.add_argument("--predictor", help="fst_type.joblib (required for --arm fst)")
    ap.add_argument("--gate", help="fst_recover.joblib -- rank questions by predicted "
                                   "recoverability and cure only the top --budget")
    ap.add_argument("--negatives", type=int, default=0,
                    help="k failure-mode warnings drawn from the wrong-trace bank "
                         "(0 = off). They go in the SYSTEM message, never in the "
                         "exemplar block -- see negatives.py.")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="fraction of problems to cure when --gate is given (default 1.0 = all)")
    ap.add_argument("--pool-store", required=True, help="the TRAIN store holding pool.jsonl")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--eval-offset", type=int, default=0)
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    a = ap.parse_args()

    if a.arm == "fst" and not a.predictor:
        raise SystemExit("--arm fst needs --predictor")

    K.MODEL_ID = a.model
    K.SPLIT_TAG = "test"
    G.set_exemplar_budget(a.exemplar_budget)
    K.version_banner()
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)

    # ---- data: seeds from TRAIN, evaluation on TEST -----------------------
    train_problems = K._load_split("train", a.seed_pool + 100)
    test_problems = K._load_split("test")
    manual, eval_problems = K.build_seeds_and_eval_order(
        train_problems, a.seed_pool, eval_split_problems=test_problems)
    seeds = manual.get(2, [])
    eval_problems = eval_problems[a.eval_offset:]
    problems = eval_problems[:a.n_problems]

    pool = load_pool(a.pool_store)
    print(f"\n  pool: {len(pool)} gold-stripped exemplars from {a.pool_store}")

    # ---- optional budgeted triage ----------------------------------------
    cure_ids = None
    if a.gate:
        if a.arm != "fst":
            raise SystemExit("--gate only applies to --arm fst")
        from fst_predictor import FSTPredictor as _FP
        g = _FP.load(a.gate)
        scores = g.risk_batch([p["question"] for p in problems])
        k = max(int(round(len(problems) * a.budget)), 1)
        keep = set(int(i) for i in scores.argsort()[::-1][:k])
        cure_ids = {f"t_{a.eval_offset + i + 1:05d}" for i in keep}
        print(f"  gate: {g.target} head, budget {a.budget:.0%} -> curing "
              f"{len(cure_ids)}/{len(problems)}; the rest get plain retrieval")
        print(f"        (gate AUROC on train: {g.train_report.get('auroc')})")

    negbank = None
    if a.negatives > 0:
        from negatives import NegativeBank
        negbank = NegativeBank.from_store(a.pool_store)
        n_neg = sum(len(v) for v in negbank.by_type.values())
        print(f"  negative bank: {n_neg} wrong traces, "
              f"{len(negbank.by_type)} types -> {a.negatives} warning(s) per prompt")
        if a.arm != "fst":
            print("  note: --negatives has no effect on the baseline arm (no type "
                  "is predicted, so there is nothing to warn about)")

    pred = None
    if a.arm == "fst":
        from fst_predictor import FSTPredictor
        pred = FSTPredictor.load(a.predictor)
        rep = pred.train_report or {}
        print(f"  predictor: {pred.model_name} | CV acc {rep.get('cv_accuracy')} "
              f"vs majority {rep.get('majority')} | min_conf {pred.min_confidence}")
        if rep.get("lift", 0) <= 0:
            print("  !! the predictor does not beat majority -- a null result here is"
                  "\n     the expected outcome, not a surprise.")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    cat = QuestionCategorizer()

    root = Path(a.store_dir); root.mkdir(parents=True, exist_ok=True)
    snap = {k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    snap["FADE_ARM"] = a.arm
    snap["FADE_N_NEGATIVES"] = a.negatives
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
            print(f"  RESUME: {len(done)} problems already scored")

    f = open(results_path, "a")
    correct = seen = 0
    pcount = Counter()
    t0 = time.time()

    print("\n" + "#" * 70)
    print(f"#  PREDICTIVE RUN | arm={a.arm.upper()} | n={len(problems)} | split=test")
    print("#" * 70)

    for i, prob in enumerate(problems, 1):
        pid = f"t_{a.eval_offset + i:05d}"
        if pid in done:
            continue
        q, gold_sol, gold_ans = prob["question"], prob["answer"], prob["gold_answer"]

        in_budget = (cure_ids is None) or (pid in cure_ids)
        ptype, conf = (None, None)
        if a.arm == "fst" and in_budget:
            ptype, conf = pred.predict(q)
        pcount[ptype or ("OUT_OF_BUDGET" if not in_budget else "ABSTAIN/none")] += 1

        if a.arm == "fst" and in_budget:
            positives, instruction = cure_for(
                ptype, pool, seeds, a.exemplar_budget,
                question=q, negatives=negbank, n_neg=a.negatives)
        else:
            from kaggle_run import select_generic_positives
            positives, instruction = select_generic_positives(
                pool, seeds, a.exemplar_budget, question=q), ""

        prompt, extra = K.build_retry_prompt(q, positives, instruction)
        traces, _ = gen._run_model(prompt, temperature=0.0,
                                   max_new_tokens=a.max_new_tokens,
                                   num_return_sequences=1, return_logprobs=False,
                                   extra_system=extra, problem=q)
        trace, happ = G.ensure_hash_line_flagged(
            traces[0], G.extract_final_answer(traces[0]))

        comps, label, diag, scored = K.score_trace(q, trace, gold_sol, gold_ans)
        correct += int(comps.correct); seen += 1

        f.write(json.dumps({
            "phase": "pass1", "id": pid, "eval_index": a.eval_offset + i - 1,
            "arm": a.arm, "question": q, "gold_solution": gold_sol,
            "gold_answer": gold_ans, "category": cat.categorize(q),
            "predicted_type": ptype, "predict_confidence": conf,
            "cure": (describe(ptype) if (a.arm == "fst" and in_budget)
                     else ("out-of-budget (plain retrieval)" if not in_budget
                           else "generic (baseline)")),
            "in_budget": in_budget,
            "instruction": instruction, "n_exemplars": len(positives),
            "n_negatives": (a.negatives if (a.arm == "fst" and in_budget) else 0),
            "trace": trace, "trace_scored": scored, "hash_appended": happ,
            "signals": comps.signals(), "label": label.value,
            "diagnosis": diag.ftype.value if diag else None,
        }) + "\n")
        f.flush()

        if a.show_every and (i % a.show_every == 0 or i == len(problems)):
            el = time.time() - t0
            print(f"  [{i}/{len(problems)}] acc {correct}/{seen} = "
                  f"{correct/max(seen,1):.1%} | {el/max(seen,1):.1f}s/prob"
                  + (f" | predictions {dict(pcount.most_common(4))}" if a.arm == "fst" else ""),
                  flush=True)
    f.close()

    # ---- summary ---------------------------------------------------------
    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    acc = sum(r["signals"]["correct"] for r in rows) / max(len(rows), 1)
    summary = {
        "arm": a.arm, "split": "test", "model": a.model,
        "n_problems": len(rows), "eval_offset": a.eval_offset,
        "eval_range": [a.eval_offset + 1, a.eval_offset + len(rows)],
        "accuracy": round(acc, 4),
        "correct": sum(r["signals"]["correct"] for r in rows),
        "pool_size": len(pool),
        "predicted_type_distribution": dict(Counter(r.get("predicted_type") for r in rows)),
        "predictor_report": (pred.train_report if pred else None),
        "cost": {"generations": seen, "sec_per_generation": round((time.time()-t0)/max(seen,1), 2)},
    }
    (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 66)
    print(f"  {a.arm.upper()} ARM — test split")
    print("=" * 66)
    print(f"  problems : {len(rows)}   range {summary['eval_range']}")
    print(f"  accuracy : {acc:.1%}  ({summary['correct']})")
    if a.arm == "fst":
        print(f"  predicted types: {summary['predicted_type_distribution']}")
    print(f"  outputs  : {root}")
    print("=" * 66)
    print("  Compare the two arms with:")
    print(f"    python compare_arms.py --typed <fst_store> --generic <baseline_store>")
    print("  (compare_arms is label-agnostic; 'typed' is just the left-hand arm.)")


if __name__ == "__main__":
    main()
