"""
nm_smoke.py -- 10-minute go/no-go for a new model BEFORE any real GPU time is spent.

    python nm_patch.py nm_smoke.py --model Qwen/Qwen3.5-4B --n 12

Checks, in order, and exits non-zero on the first hard failure:
  1. the rendered chat prompt ends in an assistant turn with thinking CLOSED
     (Qwen: '<think>\\n\\n</think>'), and the live question survived the re-wrap
  2. generations are non-empty and finite (fp16 overflow shows up as empty or
     '!!!!' output -- if so, re-run with FADE_DTYPE=fp32)
  3. no '<think>' leaks into traces
  4. the '#### N' answer parses on >= 90% of traces
  5. accuracy is plausible for this model (warns, does not fail)
and prints seconds per generation, token lengths, and how close traces come to the
max_new_tokens ceilings the pipeline uses (400 train, 320 eval), so truncation (TR)
can be judged before it contaminates the failure taxonomy.

Uses the real prompt path: strategy 2, 8 exemplars, the train run's seeds (the pool
is empty at the start of a train run, so seeds fill all 8 slots -- exactly what the
first train questions see). Questions are GSM8K train[2000:2000+n], which no run in
the plan touches.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

import kaggle_run as K
import generation as G
from extraction import extract_final_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    K.MODEL_ID = a.model
    G.set_exemplar_budget(8)
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)
    train = K._load_split("train", a.seed_pool + a.n + 10)
    manual, _ = K.build_seeds_and_eval_order(train[:a.seed_pool], a.seed_pool)
    seeds = manual.get(2, [])
    probs = train[a.seed_pool:a.seed_pool + a.n]
    print(f"  seeds {len(seeds)} | smoke questions {len(probs)} (train[{a.seed_pool}:])")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)

    # ---- 1. what the model actually sees ------------------------------------
    prompt, _ = G._build_prompt(2, probs[0]["question"], seeds, [], None, None)
    enc = G._to_chat_prompt(tok, prompt) if G.CHAT_MODE else prompt
    G.assert_prompt_contains_question(enc, probs[0]["question"])
    tail = enc[-260:]
    print("\n  RENDERED PROMPT TAIL (must end in an assistant turn, thinking closed):")
    print("  " + tail.replace("\n", "\n  "))
    n_turns = enc.count("<|im_start|>user") or enc.count("[INST]")
    print(f"  user turns in prompt: {n_turns} (8 exemplars + the live question = 9)")
    think_open = ("<think>" in tail) and ("</think>" not in tail)
    if think_open:
        raise SystemExit("  FAIL: the generation prompt leaves <think> OPEN -- thinking is on.")
    n_prompt_tok = len(tok(enc).input_ids)
    print(f"  prompt tokens: {n_prompt_tok}")

    # ---- 2-5. generate -------------------------------------------------------
    rows = []
    for i, p in enumerate(probs, 1):
        t0 = time.time()
        res = gen.generate(p["question"], 2, manual_exemplars=seeds, pool_exemplars=[],
                           category_info=None, max_new_tokens=a.max_new_tokens,
                           return_logprobs=False, knn_fn=None)
        dt = time.time() - t0
        tr = res["trace"]
        ans = extract_final_answer(tr)
        ntok = len(tok.encode(tr))
        ok = ans is not None and abs(float(ans) - p["gold_answer"]) <= 1e-6 * max(1, abs(p["gold_answer"]))
        rows.append({"i": i, "sec": dt, "tokens": ntok, "answer": ans, "gold": p["gold_answer"],
                     "correct": ok, "empty": len(tr.strip()) < 5,
                     "think": "<think>" in tr or "</think>" in tr,
                     "hash_appended": res.get("hash_appended"),
                     "bang": tr.count("!") > 20})
        print(f"  [{i:>2}] {dt:5.1f}s {ntok:>4} tok  ans {ans!s:<10} gold {p['gold_answer']:<8g} "
              f"{'OK ' if ok else 'x  '}{'(#### appended)' if res.get('hash_appended') else ''}",
              flush=True)
        if i == 1:
            print("  first trace:\n    " + tr[:700].replace("\n", "\n    "))

    n = len(rows)
    secs = [r["sec"] for r in rows[1:]] or [rows[0]["sec"]]      # first call warms up
    toks = sorted(r["tokens"] for r in rows)
    summ = {
        "model": a.model, "dtype": os.environ.get("FADE_DTYPE", "fp16"),
        "n": n, "accuracy": sum(r["correct"] for r in rows) / n,
        "parse_rate": sum(r["answer"] is not None for r in rows) / n,
        "empty_rate": sum(r["empty"] or r["bang"] for r in rows) / n,
        "think_leak": sum(r["think"] for r in rows),
        "hash_appended_rate": sum(bool(r["hash_appended"]) for r in rows) / n,
        "sec_per_gen": st.mean(secs), "tokens_median": toks[n // 2],
        "tokens_max": toks[-1],
        "near_320": sum(t >= 300 for t in toks) / n,
        "near_400": sum(t >= 380 for t in toks) / n,
        "prompt_tokens": n_prompt_tok,
    }
    print("\n  SUMMARY")
    for k, v in summ.items():
        print(f"    {k:<20}{v:.3f}" if isinstance(v, float) else f"    {k:<20}{v}")

    s = summ["sec_per_gen"]
    print("\n  PROJECTED WALL-CLOCK per GPU lane at this speed")
    for name, gens in (("train shard (250 q + ~25% retried)", 250 * 1.35),
                       ("test shard (200 q + retries + 1 probe)", 200 * 2.1),
                       ("baselines shard (200 q x 2 arms)", 400),
                       ("held-out (200 q + retries + baselines)", 200 * 3.3)):
        print(f"    {name:<44}{gens * s / 3600:5.1f} h")

    verdict = "PASS"
    if summ["empty_rate"] >= 0.25:
        verdict = "FAIL: empty / degenerate output -- re-run with FADE_DTYPE=fp32"
    elif summ["think_leak"]:
        verdict = "FAIL: <think> leaked into traces -- thinking is not disabled"
    elif summ["parse_rate"] < 0.9:
        verdict = "FAIL: '####' answers parse on <90% of traces"
    elif summ["accuracy"] < 0.4:
        verdict = ("WARN: accuracy below 40% on easy train questions -- suspect "
                   "fp16 or the chat template before spending GPU")
    if summ["near_320"] > 0.15:
        print(f"\n  NOTE: {summ['near_320']:.0%} of traces reach 300+ tokens. The eval stages "
              "cap at 320 (as Llama-2 did). Consider --max-new-tokens 512 for BOTH models' "
              "new runs, or report the TR rate.")
    summ["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    out = a.out or f"/kaggle/working/nm_smoke_{summ['dtype']}.json"
    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
        print(f"  -> {out}")
    except Exception:
        pass
    if verdict.startswith("FAIL"):
        sys.exit(2)


if __name__ == "__main__":
    main()
