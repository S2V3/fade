"""
run_baselines.py -- the two cheap baselines reviewers will ask for (E2, E3).

Adds generations to a FINISHED stage-2/stage-3 store. Nothing is re-solved and
gold is never opened; score_baselines.py does the scoring afterwards.

    arm        what it is                                         answers
    ---------  -------------------------------------------------  ---------------------
    resample   the pass-1 procedure run again, unchanged: same     E2  matched-cost
               strategy-2 prompt, same pool retrieval, same          self-consistency
               sampler (generation.py samples pass-1 at T=0.7).     E3a plain resample
               Disagreement with pass-1 is the self-consistency
               detector; its answer is the "just try again" repair.
    generic    kaggle_run.apply_cure(mode="generic"): question-    E3b retry without
               aware exemplars, NO diagnosis, NO instruction,        diagnosis
               greedy -- the paper's own control arm, run on EVERY
               question (not only flagged ones) so D is measured on
               every correct trace instead of 67 false positives.

Both arms run on every row, flagged or not. score_baselines.py reads R and D on
the flagged subset (what the break-even rule uses) AND on all rows (the stable
estimate).

USAGE
    python nm_patch.py run_baselines.py --store <stage-3 store> --pool-store <train store> \\
        --model Qwen/Qwen3.5-4B --arms resample,generic --limit 400
    --dry-run prints the plan and the provenance check without loading a model.

RESUME   keyed by (id, arm) in <store>/baselines.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import generation as G
import kaggle_run as K
from extraction import extract_final_answer

ARMS = ("resample", "generic")


def fingerprint(exs) -> str:
    return hashlib.sha1("␟".join(sorted(
        str(e.get("question", "")) for e in exs)).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description="E2/E3 baselines over a finished store")
    ap.add_argument("--store", required=True)
    ap.add_argument("--pool-store", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="resample,generic")
    ap.add_argument("--limit", type=int, default=0, help="first N rows by id (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--retrieval", default="3stage")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--strategy", type=int, default=2)
    ap.add_argument("--out", default=None, help="default <store>/baselines.jsonl")
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    ap.add_argument("--force", action="store_true",
                    help="run even if the pool fingerprint disagrees with the store's")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [x for x in arms if x not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {ARMS}")

    root = Path(a.store)
    rows = [json.loads(l) for l in open(root / "results.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["id"])
    if a.limit:
        rows = rows[:a.limit]
    out_path = Path(a.out) if a.out else root / "baselines.jsonl"
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                d = json.loads(ln)
                done.add((d["id"], d["arm"]))
            except Exception:
                pass
    todo = [(r, arm) for r in rows for arm in arms if (r["id"], arm) not in done]
    print("=" * 74)
    print(f"  BASELINES over {len(rows)} rows of {root.name} | arms {arms}")
    print(f"  done {len(done)} | to generate {len(todo)}")
    print("=" * 74)

    # ---- provenance: the SAME pool the store was generated with -------------
    from classification import TraceStore
    pool = TraceStore(root=Path(a.pool_store)).exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
    fp = fingerprint(pool)
    want_n, want_fp = None, None
    for fn, kn, kf in (("mpv_provenance.json", "pool_size", "pool_fp"),
                       ("run_summary.json", "pool_size", None)):
        try:
            blob = json.load(open(root / fn))
            want_n = want_n or blob.get(kn)
            want_fp = want_fp or (blob.get(kf) if kf else None)
        except Exception:
            pass
    print(f"  pool {len(pool)} / {fp}   store expects {want_n} / {want_fp or '?'}")
    if (want_n and int(want_n) != len(pool)) or (want_fp and want_fp != fp):
        msg = ("  STOP: this pool is not the one the store's pass-1 used, so the resample\n"
               "  would not be the same procedure as pass-1.")
        if not a.force:
            raise SystemExit(msg)
        print(msg.replace("STOP", "WARNING (--force)"))
    if a.dry_run or not todo:
        return

    K.MODEL_ID = a.model
    G.set_exemplar_budget(a.exemplar_budget)
    token = K.resolve_hf_token(a.secret_name)
    train = K._load_split("train", a.seed_pool)
    manual, _ = K.build_seeds_and_eval_order(train, a.seed_pool)
    seeds = manual.get(a.strategy, [])
    knn = K._retrieval_fn(a.retrieval)
    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    from categorizer import QuestionCategorizer
    cat = QuestionCategorizer()
    import torch

    prov = {"pool_size": len(pool), "pool_fp": fp, "seeds_fp": fingerprint(seeds),
            "model": a.model, "arms": arms, "max_new_tokens": a.max_new_tokens,
            "exemplar_budget": a.exemplar_budget, "retrieval": a.retrieval,
            "resample_rule": "generation.py strategy 2 (pass-1 procedure, T=0.7, top_p=0.95)",
            "generic_rule": "kaggle_run.apply_cure(mode='generic'), T=0 (greedy)"}
    (root / "baselines_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"  seeds {len(seeds)} / {prov['seeds_fp']}")

    f = open(out_path, "a")
    t0 = time.time(); n_gen = 0
    for i, (r, arm) in enumerate(todo, 1):
        q = r["question"]
        ts = time.time()
        # reproducible sampling: one seed per (question, arm)
        torch.manual_seed(zlib.crc32(f"{r['id']}|{arm}".encode()) & 0x7FFFFFFF)
        if arm == "resample":
            g = gen.generate(q, a.strategy, manual_exemplars=seeds, pool_exemplars=pool,
                             category_info=cat.categorize(q),
                             max_new_tokens=a.max_new_tokens,
                             return_logprobs=False, knn_fn=knn)
            trace, meta = g["trace"], {"hash_appended": g.get("hash_appended")}
        else:
            prev_ans = r.get("final_answer")
            if prev_ans is None:
                prev_ans = extract_final_answer(r.get("trace") or "")
            trace, meta = K.apply_cure(gen, q, r.get("trace") or "", prev_ans, None,
                                       pool, seeds, a.max_new_tokens, mode="generic",
                                       budget=a.exemplar_budget, negatives=None,
                                       n_neg=0, rec_id=r["id"])
        rec = {"id": r["id"], "arm": arm, "question": q, "trace": trace,
               "answer": extract_final_answer(trace),
               "n_exemplars": meta.get("n_exemplars"),
               "hash_appended": meta.get("hash_appended"),
               "sec": round(time.time() - ts, 2)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        n_gen += 1
        if a.show_every and (n_gen % a.show_every == 0 or i == len(todo)):
            print(f"  [{i}/{len(todo)}] {(time.time() - t0) / n_gen:.1f}s/gen", flush=True)
    f.close()
    print(f"\n  done: {n_gen} generations in {(time.time() - t0) / 3600:.2f} h -> {out_path}")
    print("  gold was never opened. Score with score_baselines.py.")


if __name__ == "__main__":
    main()
