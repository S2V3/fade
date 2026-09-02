"""Enumerate every cure instead of predicting which one to apply.

The type head is right 59% of the time, so a matched cure reaches the problem
in a minority of cases. Enumeration removes the prediction from the critical
path: generate one candidate per failure type, then select among them. The
detector is still needed, but only to decide WHETHER to retry, which it does at
87.6% precision.

Two diversity sources are generated side by side so the experiment can tell
them apart:

    cures   one retry per distinct failure type (7 of the 8 differ; TR and
            UNCLASSIFIED share the no-cure path)
    temp    the same number of plain resamples at temperature > 0

Selection is then applied to each set under both policies, giving the 2x2 that
separates "is the diversity better" from "is the selector better".

USAGE
    python run_multicure.py --store <stage-3 store> --ctc <ctc2.joblib> \
        --pool-store <train store> --out-dir <dir> --n-problems 200
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

import config
import generation as G
import kaggle_run as K
import selector as SEL
from diagnosis import FailureType


def distinct_cures():
    """The failure types whose treatment actually differs."""
    from diagnosis import TYPED_INSTRUCTION, TYPED_PREFILL
    try:
        from diagnosis import TYPED_APPROACH
    except Exception:
        TYPED_APPROACH = {}
    out, seen = [], set()
    for t in FailureType:
        key = ((TYPED_INSTRUCTION.get(t) or "").strip()[:60],
               bool(TYPED_APPROACH.get(t)),
               bool((TYPED_PREFILL.get(t) or "").strip()))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser(description="enumerate cures, then select")
    ap.add_argument("--store", required=True, help="finished store with pass-1 traces")
    ap.add_argument("--ctc", required=True)
    ap.add_argument("--pool-store", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="retry every trace above this p_wrong, not a budget")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temp", type=float, default=0.8, help="for the resample arm")
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=10)
    a = ap.parse_args()

    if not hasattr(K, "apply_cure"):
        raise SystemExit("kaggle_run.py has no apply_cure() -- push the updated file")

    cures = distinct_cures()
    print("=" * 66)
    print(f"  {len(cures)} distinct cures: {', '.join(c.value for c in cures)}")
    print("=" * 66)

    K.MODEL_ID = a.model
    K.SPLIT_TAG = "test"
    G.set_exemplar_budget(a.exemplar_budget)
    token = K.resolve_hf_token(a.secret_name)

    rows = [json.loads(l) for l in open(Path(a.store) / "results.jsonl") if l.strip()]
    todo = [r for r in rows if r.get("p_wrong", 0) >= a.threshold][:a.n_problems]
    print(f"\n  {len(rows)} rows in the store, {len(todo)} predicted wrong "
          f"(p_wrong >= {a.threshold}), running {len(todo)}")

    from classification import TraceStore
    pool = TraceStore(root=Path(a.pool_store)).exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
    train = K._load_split("train", 2000)
    manual, _ = K.build_seeds_and_eval_order(train, 2000)
    seeds = manual.get(2, [])
    print(f"  pool {len(pool)} exemplars | seeds {len(seeds)}")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    from categorizer import QuestionCategorizer
    cat = QuestionCategorizer()
    knn = K._retrieval_fn("3stage")

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "candidates.jsonl"
    done = set()
    if path.exists():
        for ln in open(path):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
        if done:
            print(f"  RESUME: {len(done)} already generated")

    f = open(path, "a")
    t0 = time.time(); n = 0
    for i, r in enumerate(todo, 1):
        if r["id"] in done:
            continue
        q, first = r["question"], r["trace"]
        rec = {"id": r["id"], "question": q, "p_wrong": r["p_wrong"],
               "predicted_type": r.get("predicted_type"),
               "pass1_trace": first, "pass1_answer": r.get("final_answer"),
               "cures": [], "temps": []}

        # one candidate per distinct cure
        for ft in cures:
            tr, meta = K.apply_cure(
                gen, q, first, r.get("final_answer"), ft.value, pool, seeds,
                a.max_new_tokens, mode="typed", budget=a.exemplar_budget,
                rec_id=r["id"])
            rec["cures"].append({"type": ft.value, "trace": tr,
                                 "answer": G.extract_final_answer(tr)})

        # the compute-matched resample arm: same count, temperature diversity
        info = cat.categorize(q)
        for _ in range(len(cures)):
            res = gen.generate(q, 2, manual_exemplars=seeds, pool_exemplars=pool,
                               category_info=info, max_new_tokens=a.max_new_tokens,
                               return_logprobs=False, knn_fn=knn,
                               temperature=a.temp)
            tr = res["trace"]
            rec["temps"].append({"trace": tr, "answer": G.extract_final_answer(tr)})

        f.write(json.dumps(rec) + "\n"); f.flush()
        n += 1
        if a.show_every and n % a.show_every == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {el/n:.1f}s/problem "
                  f"({el/n/(2*len(cures)):.1f}s/generation)", flush=True)
    f.close()
    print(f"\n  wrote {path}")
    print(f"  {2 * len(cures)} candidates per problem "
          f"({len(cures)} cures + {len(cures)} resamples)")
    print("\n  score it with score_multicure.py, where gold is available.")


if __name__ == "__main__":
    main()
