"""
run_mpv.py -- Metamorphic Program Verification over a FINISHED store.

Nothing is re-solved.  EDER, the deferred-retry pool and the stage-3 retries stay
exactly as they were.  This script only adds PROBES: for each question it writes up
to N metamorphic variants (one number changed), asks the frozen model each variant
ONCE, and stores the probe traces.  Everything that turns probes into a verdict or
a selection happens later, on CPU, in score_mpv.py -- and gold is opened only there.

    gen 0   (already done)  pass-1 trace, retry trace         <- the store
    cpu     make_probes(question, [pass-1, retry])            <- this script, free
    gen 1   model solves each probe question                  <- this script, GPU
    cpu     execute every candidate's chain on every probe,
            back-substitute the probe traces, select          <- score_mpv.py

COST
    one generation per probe.  --n-probes 1 on the 1,319-question test store is
    ~1,300 generations; --n-probes 2 is ~2,600.  At ~10 s/probe (320 tokens, the
    stopping criterion halts at '#### N') that is ~3.5 GPU-h and ~7 GPU-h.
    --only-flagged restricts to the rows that carry a retry (540 on stage 3).

RESUME
    probes are appended to <store>/mpv_probes.jsonl keyed by (id, probe index).
    Re-running skips what is done.  Pair it with autopush like every other run.

USAGE
    python run_mpv.py --store stores/store_ctctest_audit2_a1 --n-probes 2
    python run_mpv.py --store <dir> --limit 100          # a cheap pilot first
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import mpv
import generation as G
import kaggle_run as K
from extraction import extract_final_answer


def candidates_of(row: dict) -> list[str]:
    """The traces the store already holds for this question."""
    out = [row.get("trace") or ""]
    if row.get("retry_trace"):
        out.append(row["retry_trace"])
    return out


def prepare_all(rows, n_probes, factors):
    plan, reasons = [], {}
    for r in rows:
        traces = candidates_of(r)
        probes = mpv.make_probes(r["question"], traces, n_probes=n_probes, factors=factors)
        if not probes:
            reasons["no load-bearing number in any candidate chain"] = \
                reasons.get("no load-bearing number in any candidate chain", 0) + 1
            continue
        for j, p in enumerate(probes):
            plan.append((r, j, p))
    return plan, reasons


def main():
    ap = argparse.ArgumentParser(description="metamorphic probes over a finished store")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default=None, help="default <store>/mpv_probes.jsonl")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--n-probes", type=int, default=2)
    ap.add_argument("--factors", default="2,3", help="comma list of perturbation factors")
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N rows")
    ap.add_argument("--only-flagged", action="store_true",
                    help="only rows that carry a retry_trace")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--pool-store", required=True,
                    help="the typed TRAIN store whose pool the stage-3 run used")
    ap.add_argument("--retrieval", default="3stage")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--strategy", type=int, default=2)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true", help="plan only, no model")
    a = ap.parse_args()

    root = Path(a.store)
    rows = [json.loads(l) for l in open(root / "results.jsonl") if l.strip()]
    if a.only_flagged:
        rows = [r for r in rows if r.get("retry_trace")]
    if a.limit:
        rows = rows[:a.limit]
    factors = tuple(float(x) for x in a.factors.split(","))

    print("=" * 74)
    print(f"  MPV PROBES over {len(rows)} rows of {root.name}   n_probes={a.n_probes}  factors={factors}")
    print("=" * 74)

    plan, reasons = prepare_all(rows, a.n_probes, factors)
    n_q = len({r["id"] for r, _, _ in plan})
    print(f"\n  probes planned {len(plan)}  on {n_q}/{len(rows)} questions "
          f"({n_q/max(len(rows),1):.1%} testable; the rest ABSTAIN)")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    abstain: {k:<50}{v:>5}")
    if a.dry_run or not plan:
        return

    out_path = Path(a.out) if a.out else root / "mpv_probes.jsonl"
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                d = json.loads(ln); done.add((d["id"], d["probe_index"]))
            except Exception:
                pass
        print(f"  RESUME: {len(done)} probes already generated")

    K.MODEL_ID = a.model
    token = K.resolve_hf_token(a.secret_name)
    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    from categorizer import QuestionCategorizer
    cat = QuestionCategorizer()

    # The probe is asked the SAME way the original was: same strategy, same
    # frozen pool + seeds, same retrieval, same decoding.  Otherwise a
    # disagreement measures the prompt change, not the trace.  This mirrors
    # run_ctc_validate.py exactly.
    G.set_exemplar_budget(a.exemplar_budget)
    from classification import TraceStore
    pool = TraceStore(root=Path(a.pool_store)).exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
    train = K._load_split("train", a.seed_pool)
    manual, _ = K.build_seeds_and_eval_order(train, a.seed_pool)
    seeds = manual.get(a.strategy, [])
    knn = K._retrieval_fn(a.retrieval)
    print(f"  exemplars: pool {len(pool)} (gold-stripped) | seeds {len(seeds)} (top-up only)")

    f = open(out_path, "a")
    t0 = time.time(); n_gen = 0
    for i, (r, j, p) in enumerate(plan, 1):
        if (r["id"], j) in done:
            continue
        g = gen.generate(p.perturbed_question, a.strategy, manual_exemplars=seeds,
                         pool_exemplars=pool,
                         category_info=cat.categorize(p.perturbed_question),
                         max_new_tokens=a.max_new_tokens,
                         return_logprobs=False, knn_fn=knn)
        p.model_trace = g["trace"]
        p.model_answer = extract_final_answer(g["trace"])
        rec = p.to_dict()
        rec.update({"id": r["id"], "probe_index": j, "question": r["question"]})
        f.write(json.dumps(rec) + "\n"); f.flush()
        n_gen += 1
        if a.show_every and n_gen % a.show_every == 0:
            print(f"  [{i}/{len(plan)}] {(time.time()-t0)/n_gen:.1f}s/probe", flush=True)
    f.close()
    print(f"\n  done: {n_gen} new probes in {(time.time()-t0)/3600:.2f} h  -> {out_path}")
    print("  gold was never opened.  Score with:  python score_mpv.py --store", a.store)


if __name__ == "__main__":
    main()
