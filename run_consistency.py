"""
run_consistency.py -- add the perturbation signal to a FINISHED stage-2/3 store.

WHY THIS AND NOT MORE TRAINING
------------------------------
Stage 1 predicted detect AUROC 0.776 on pass-1 traces. Stage 2 delivered 0.770 on
questions it had never seen. There is no generalisation gap, which means the model
is not underfitted, not overfitted, and not mistrained -- it is extracting all the
signal that STATIC FEATURES OF ONE TRACE contain. Retraining harder, adding folds,
tuning weights: all of it moves a number that is already at its ceiling for this
input. To predict better, CTC needs a DIFFERENT INPUT.

This is that input, and it already exists in ctc.py, unrun at scale.

THE MECHANISM (ctc.py, Counterfactual Trace Consistency)
--------------------------------------------------------
A trace that encodes a correct procedure is a FUNCTION of the question's numbers.

  1. extract the trace's equation chain                        no GPU
  2. find a question number the chain actually consumes        no GPU
  3. rewrite the question with that number changed             no GPU
  4. predict SYMBOLICALLY, from the trace's own chain, what
     the answer must become under that substitution            no GPU, sympy
  5. re-ask the model the perturbed question                   ONE generation
  6. compare the model's new answer to its own chain's prediction

  AGREE    -> the model executed a stable, inspectable procedure
  DISAGREE -> the first trace was not a procedure; it landed on a number

Step 4 is the hinge: the prediction comes from the model's OWN trace, resolved by
sympy. No second model, no token entropy, no trained verifier. Gold is never
touched, so this is legal on test exactly as it is on train.

WHY IT SHOULD ADD SOMETHING THE STATIC FEATURES CANNOT
------------------------------------------------------
On the 1,500-problem train run:

    failures WITH a false equation   351  (32.9%)   an arithmetic check sees these
    failures with NO false equation  717  (67.1%)   an arithmetic check is BLIND

The 717 are WP, SM, ST, NR, AL -- every step valid, the plan wrong. A wrong plan is
a DIFFERENT FUNCTION of the inputs than the right one, so it moves differently under
perturbation. The signal is orthogonal to arithmetic validity BY CONSTRUCTION, which
is precisely what the current feature set is missing.

COVERAGE, measured on train: a connected chain exists for 70.1%. CTC ABSTAINS on the
rest, and the abstain rate is reported rather than folded into a verdict.

WHAT THIS SCRIPT REPORTS
------------------------
Three AUROCs on the same rows, so the question "did it help?" has an answer rather
than a hope:

    p_wrong alone            what you have now
    disagreement alone       the new signal by itself
    blended                  and at which weight

If blended does not beat p_wrong alone, say so and drop it. That is a real result:
it would mean trace-level behaviour under perturbation carries nothing the static
features have not already captured.

COST
----
One extra generation per applicable row. On the 400-question stage-2 store that is
about 280 generations, ~1.2 GPU-h. Run it on the store you already have -- no
re-solving, no re-training.

USAGE
    python run_consistency.py --store stores/store_ctcval_train_audit2_a2
    python run_consistency.py --store <dir> --limit 200      # a cheap pilot first
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

import ctc as CTC
import generation as G
import kaggle_run as K


def main():
    ap = argparse.ArgumentParser(description="perturbation consistency over a finished store")
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default=None, help="default <store>/consistency.jsonl")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--limit", type=int, default=0, help="pilot on the first N rows")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    a = ap.parse_args()

    root = Path(a.store)
    rows = [json.loads(l) for l in open(root / "results.jsonl") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    print("=" * 74)
    print(f"  PERTURBATION CONSISTENCY over {len(rows)} finished rows")
    print("=" * 74)

    # ---- the free part first, so the GPU is only spent on what is applicable
    prepared = []
    reasons = {}
    for r in rows:
        res = CTC.prepare(r["question"], r["trace"], r.get("final_answer"),
                          factor=a.factor)
        if res.applicable:
            prepared.append((r, res))
        else:
            reasons[res.reason] = reasons.get(res.reason, 0) + 1
    print(f"\n  applicable {len(prepared)}/{len(rows)} = {len(prepared)/max(len(rows),1):.1%}"
          "   (the rest ABSTAIN -- reported, never guessed)")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    abstain: {k:<40}{v:>5}")
    if not prepared:
        raise SystemExit("  nothing applicable -- no chain reached a final answer")

    out_path = Path(a.out) if a.out else root / "consistency.jsonl"
    done = set()
    if out_path.exists():
        for ln in open(out_path):
            try:
                done.add(json.loads(ln)["id"])
            except Exception:
                pass
        print(f"  RESUME: {len(done)} already probed")

    K.MODEL_ID = a.model
    token = K.resolve_hf_token(a.secret_name)
    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    from categorizer import QuestionCategorizer
    cat = QuestionCategorizer()

    # The perturbed question is re-asked the SAME WAY the original was -- same
    # strategy, same exemplars, same decoding. A different prompt would make the
    # disagreement a measure of the prompt change, not of the trace.
    from classification import TraceStore
    pool = TraceStore(root=Path(a.store).parent /
                      "store_typed_train_audit2").exemplars() \
        if (Path(a.store).parent / "store_typed_train_audit2").exists() else []
    knn = K._retrieval_fn("3stage") if pool else None

    f = open(out_path, "a")
    t0 = time.time()
    for i, (r, res) in enumerate(prepared, 1):
        if r["id"] in done:
            continue
        gres = gen.generate(res.perturbed_question, 2, manual_exemplars=[],
                            pool_exemplars=pool,
                            category_info=cat.categorize(res.perturbed_question),
                            max_new_tokens=a.max_new_tokens,
                            return_logprobs=False, knn_fn=knn)
        ans = G.extract_final_answer(gres["trace"])
        agrees = CTC.judge(res, ans)
        rec = res.to_dict()
        rec.update({"id": r["id"], "agrees": bool(agrees),
                    "perturbed_trace": gres["trace"], "model_answer": ans})
        f.write(json.dumps(rec) + "\n"); f.flush()
        if a.show_every and i % a.show_every == 0:
            print(f"  [{i}/{len(prepared)}] {(time.time()-t0)/i:.1f}s/probe", flush=True)
    f.close()

    # ------------------------------------------------------------- analysis
    cons = {}
    for ln in open(out_path):
        if ln.strip():
            d = json.loads(ln); cons[d["id"]] = d
    scored = [r for r in rows if "correct" in r and r["id"] in cons]
    if not scored:
        print("\n  store has no gold labels (stage 3) -- probes written, "
              "nothing to evaluate against. Score them where gold exists.")
        return

    y = [0 if r["correct"] else 1 for r in scored]        # 1 = WRONG
    agree = [1.0 if cons[r["id"]]["agrees"] else 0.0 for r in scored]
    pw = [r["p_wrong"] for r in scored]
    n_ok = sum(1 for r, v in zip(scored, agree) if r["correct"] and v)
    n_bad = sum(1 for r, v in zip(scored, agree) if not r["correct"] and v)
    n_c = sum(1 for r in scored if r["correct"])
    print(f"\n  AGREEMENT RATE   correct traces {n_ok}/{n_c} = {n_ok/max(n_c,1):.1%}"
          f"   wrong traces {n_bad}/{len(scored)-n_c} = "
          f"{n_bad/max(len(scored)-n_c,1):.1%}")
    print("  A large gap here is the whole point -- it means the model's own")
    print("  procedure is stable when it is right and unstable when it is not.")

    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        raise SystemExit("sklearn missing")
    if len(set(y)) < 2:
        raise SystemExit("  one class only -- cannot compute AUROC")
    dis = [1.0 - v for v in agree]                        # disagree -> wrong
    a_pw = roc_auc_score(y, pw)
    a_di = roc_auc_score(y, dis)
    print(f"\n  {'signal':<34}{'AUROC':>8}")
    print(f"  {'p_wrong (static features)':<34}{a_pw:>8.3f}")
    print(f"  {'disagreement alone':<34}{a_di:>8.3f}")
    best = (None, -1)
    for w in [x / 20 for x in range(21)]:
        b = [w * d + (1 - w) * p for d, p in zip(dis, pw)]
        s = roc_auc_score(y, b)
        if s > best[1]:
            best = (w, s)
    print(f"  {'blended  w=%.2f on disagreement' % best[0]:<34}{best[1]:>8.3f}"
          f"   ({best[1]-a_pw:+.3f} over p_wrong)")

    if best[1] - a_pw < 0.01:
        print("\n  VERDICT: the perturbation signal adds nothing p_wrong did not")
        print("  already have. Report it as a measured null -- it is a real finding")
        print("  that behavioural instability is already visible in the static text.")
    else:
        print(f"\n  VERDICT: +{best[1]-a_pw:.3f} AUROC for one extra generation per")
        print("  question. Fold `disagrees` into ctc_features.trace_features() and")
        print("  retrain -- but note the cost: detection now needs 2 generations,")
        print("  so the budget arithmetic in run_ctc_validate.py must be redone.")
    json.dump({"auroc_p_wrong": a_pw, "auroc_disagree": a_di,
               "auroc_blend": best[1], "blend_w": best[0],
               "agree_rate_correct": n_ok / max(n_c, 1),
               "agree_rate_wrong": n_bad / max(len(scored) - n_c, 1),
               "n": len(scored), "applicable_rate": len(prepared) / max(len(rows), 1)},
              open(root / "consistency_report.json", "w"), indent=2)
    print(f"\n  -> {root/'consistency_report.json'}")


if __name__ == "__main__":
    main()
