"""
kaggle_run.py -- the single entrypoint for FADE.

Everything routes through ExemplarGenerator (generation.py); there is no other
generation path. Two modes:

  --demo N        Solve the first N problems of the real eval order with the
                  strategy's real seeds. Prints, per question: the prompt stats
                  (tokens, protocol check, shot count), the model's reasoning
                  trace, gold, checkpoint/equation breakdowns, all signals, the
                  label, and the diagnosed error type + cure. NO storage, NO
                  retries -- a faithful preview of the main run.

  (default)       The full experiment on the SAME eval order: pass 1 over
                  --n-problems, then two deferred retry iterations (MEDIUM queue
                  drained fully, then BAD queue), everything scored / classified /
                  diagnosed / stored IN DETAIL, ending in the report.

The main run writes, under store/ :
  pool.jsonl / medium_queue.jsonl / bad_queue.jsonl   (the routed queues)
  results.jsonl        one record PER ATTEMPT (pass 1 + every retry), full detail
  run_summary.json     every aggregate metric, machine-readable
  report.md            the same, human-readable
  results.csv          per-problem final outcome, spreadsheet-ready
  config_snapshot.json every threshold / version used, for reproducibility
  attempts.jsonl       a terse append-only audit line per generation

Startup guards, in order: (1) generation version banner, (2) HF identity via
whoami() after purging any ambient token, (3) a gated-access probe on the Llama
repo -- token problems surface in a few lines, not after a 13 GB download.

The main run is RESUMABLE: on restart it reconstructs the pool/queues from
store/ and skips problems already completed in pass 1.

Usage on Kaggle (GPU on, internet on, HF token as a Kaggle Secret named HF_TOKEN):
    !git clone https://github.com/S2V3/fade.git && cd fade
    !python kaggle_run.py --demo 5 --strategy 2
    !python kaggle_run.py --n-problems 200 --strategy 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# --- make the repo importable regardless of CWD ------------------------------
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config
import generation as G
from generation import ExemplarGenerator, STRATEGY_NAMES, N_SHOTS
from categorizer import QuestionCategorizer
from exemplar_selector import ExemplarSelector, CATEGORY_TYPES
from components import compute_components
from classification import classify, CONSEQUENCE, needs_diagnosis, Label, TraceStore
from diagnosis import diagnose, TYPED_INSTRUCTION, TYPED_CURE, FailureType
from similarity import backend_name
from preprocessing import preprocess_problem, normalize_math

MODEL_ID = "meta-llama/Llama-2-7b-hf"   # overridable with --model
SPLIT_TAG = "train"                     # set from --split in main()
ARTIFACTS = REPO_DIR / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
FULL_GSM8K_TRAIN = 7473


# =============================================================================
# 0. STARTUP GUARDS
# =============================================================================

def version_banner() -> None:
    v = getattr(G, "GENERATION_VERSION", "<none>")
    print("=" * 70)
    print(f"  generation module version: {v}")
    if v != "v4-chat":
        print("  !! expected 'v4-chat' -- generation.py is stale/modified. Abort.")
        sys.exit(1)
    print(f"  {v} ACTIVE | N_SHOTS={N_SHOTS} "
          f"rep_penalty={G.REPETITION_PENALTY} no_repeat_ngram={G.NO_REPEAT_NGRAM}")
    print(f"  ban_strings={G.BAN_STRINGS} | stop_markers={G.STOP_MARKERS}")
    print(f"  preprocessing: normalize_traces={config.NORMALIZE_TRACES}")
    print("=" * 70)


def resolve_hf_token(secret_name: str) -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if var in os.environ:
            print(f"  purging ambient token {var} (using Kaggle Secret instead)")
            os.environ.pop(var, None)
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret(secret_name)
        if tok:
            print(f"  HF token loaded from Kaggle Secret '{secret_name}'")
            return tok.strip()
    except Exception as e:
        print(f"  (Kaggle Secret '{secret_name}' unavailable: {e})")
    tok = os.environ.get("FADE_HF_TOKEN")
    if tok:
        print("  HF token loaded from FADE_HF_TOKEN env var")
        return tok.strip()
    print("  no explicit token found; relying on any cached huggingface login")
    return None


def verify_identity_and_access(token: str | None) -> None:
    from huggingface_hub import login, whoami, HfApi
    if token:
        login(token=token, add_to_git_credential=False)
    try:
        who = whoami(token=token)
        print(f"  HF identity: {who.get('name', '<unknown>')} (type={who.get('type', '?')})")
    except Exception as e:
        print(f"  !! whoami() failed: {e}\n     A valid HF token is required.")
        sys.exit(1)
    try:
        info = HfApi().model_info(MODEL_ID, token=token)
        print(f"  gated-access probe OK: {MODEL_ID} reachable ({len(info.siblings)} files)")
    except Exception as e:
        print(f"  !! gated-access probe FAILED for {MODEL_ID}: {e}")
        print("     Approve access at https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print("     and ensure the token belongs to the approved account.")
        sys.exit(1)


# =============================================================================
# 1. MODEL + DATA
# =============================================================================

def load_model(token: str | None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()   # silence per-call gen warnings
    print(f"\nLoading {MODEL_ID} (fp16)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=token, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    chat = G.autodetect_chat_mode(tok, MODEL_ID)
    print(f"  loaded in {time.time() - t0:.0f}s on {next(model.parameters()).device} "
          f"| cuda={torch.cuda.is_available()}")
    print(f"  CHAT_MODE={chat} "
          f"({'chat template applied' if chat else 'plain completion prompts'})")
    return model, tok


GOLD_ANSWER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
ANNOTATION_RE = re.compile(r"<<[^>]*>>")


def gold_answer_value(answer_field: str) -> float | None:
    m = GOLD_ANSWER_RE.search(answer_field)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def strip_annotations(solution: str) -> str:
    return ANNOTATION_RE.sub("", solution).strip()


def _load_split(split: str, n_needed: int | None = None) -> list[dict]:
    """Load one GSM8K split, PREPROCESSED. Each item:
    {question(clean), answer(annotated,normalised), gold_answer(float)}.
    Both train and test carry the <<expr=result>> checkpoint annotations the
    signals need, so the instrument works identically on either."""
    from datasets import load_dataset
    ds, last_err = None, None
    for ds_id in ("openai/gsm8k", "gsm8k"):
        try:
            ds = load_dataset(ds_id, "main", split=split)
            print(f"  source: {ds_id} [{split}]")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"could not load GSM8K {split}: {last_err}")

    out, dropped = [], 0
    for row in ds:
        p = preprocess_problem({"question": row["question"], "answer": row["answer"]})
        ga = gold_answer_value(p["answer"])
        if ga is None:
            dropped += 1
            continue
        p["gold_answer"] = ga
        out.append(p)
        if n_needed is not None and len(out) >= n_needed:
            break
    print(f"  {split}: {len(out)} problems ready (preprocessed) | "
          f"dropped {dropped} unparseable")
    return out


def load_gsm8k(n_needed: int) -> list[dict]:
    """Back-compat: train only (seeds + eval both from train)."""
    print("\nLoading GSM8K (train split)...")
    return _load_split("train", n_needed)


# =============================================================================
# 2. SEEDS + EVAL ORDER  (cached; identical for every strategy)
# =============================================================================

def _interleave_by_category(seeds: list[dict]) -> list[dict]:
    """Reorder a strategy's seeds so the FIRST few span categories instead of
    being 5-of-one-category-then-5-of-the-next. Fixes the 8-shot bias: without
    this, few_shot's 8 shown shots were ~5 percentage + 3 monetary. Round-robin
    across categories preserves within-category richness order."""
    buckets = defaultdict(list)
    for ex in seeds:
        buckets[ex.get("category_type", "arithmetic")].append(ex)
    order, out = [c for c in CATEGORY_TYPES if buckets[c]], []
    while any(buckets[c] for c in order):
        for c in order:
            if buckets[c]:
                out.append(buckets[c].pop(0))
    return out


def build_seeds_and_eval_order(problems, seed_pool_size=300, eval_split_problems=None):
    """Seeds always come from the first `seed_pool_size` TRAIN problems.

    If `eval_split_problems` is given (the TEST split), evaluation runs over those
    instead of the train remainder -- so seeds are from train, eval is on test,
    and the two are automatically disjoint (different splits). This is the
    reviewer-proof setup: published GSM8K numbers are on the 1,319 test problems,
    and FADE's streaming pool builds itself from those same test problems as it
    goes (no separate 'training' pass, no weight updates).

    If `eval_split_problems` is None, behaviour is unchanged: eval = train
    remainder after seed selection (single-split mode)."""
    tag = "test" if eval_split_problems is not None else "train"
    cache = ARTIFACTS / f"seeds_evalorder_pool{seed_pool_size}_{tag}.json"
    if cache.exists():
        print(f"\nLoading cached seeds + eval order from {cache.name}")
        blob = json.loads(cache.read_text())
        return ({int(k): v for k, v in blob["manual_exemplars"].items()},
                blob["eval_problems"])

    print(f"\nSelecting seeds from the first {seed_pool_size} TRAIN problems...")
    pool = problems[:seed_pool_size]
    selector = ExemplarSelector(pool_size=seed_pool_size)
    manual, _ = selector.select(pool, QuestionCategorizer(), ground_truth_key="answer")

    for sid, seeds in manual.items():
        for ex in seeds:
            gold_sol = ex.get("ground_truth", "")
            ex["gold_solution"] = gold_sol
            ex["trace"] = strip_annotations(gold_sol)
            ex["gold_answer"] = gold_answer_value(gold_sol)
        manual[sid] = _interleave_by_category(seeds)

    selected_qs = {ex["question"] for seeds in manual.values() for ex in seeds}

    if eval_split_problems is not None:
        # eval on TEST; seeds are from TRAIN -> disjoint by construction, but
        # guard anyway against the rare identical-question overlap across splits.
        eval_problems = [p for p in eval_split_problems
                         if p["question"] not in selected_qs]
    else:
        eval_problems = [p for p in pool if p["question"] not in selected_qs]
        eval_problems += problems[seed_pool_size:]

    eval_qs = {p["question"] for p in eval_problems}
    overlap = selected_qs & eval_qs
    assert not overlap, f"seed/eval overlap ({len(overlap)}) -- exclusion broken"
    print(f"  seed/eval disjoint OK | {len(selected_qs)} seeds ({tag} eval), "
          f"{len(eval_problems)} eval problems")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "manual_exemplars": {str(k): v for k, v in manual.items()},
        "eval_problems": eval_problems}))
    print(f"  cached to {cache.name}")
    return manual, eval_problems


# =============================================================================
# 3. SCORING one generated trace  (preprocess -> components -> label -> cascade)
# =============================================================================

def score_trace(question, trace, gold_solution, gold_answer):
    """Normalise the trace + gold before the instrument reads them (recovers
    unicode-math equations/values), then score. Returns (comps, label, diag,
    scored_trace) -- scored_trace is what the instrument actually parsed."""
    q = question
    tr = normalize_math(trace) if config.NORMALIZE_TRACES else trace
    gs = normalize_math(gold_solution) if config.NORMALIZE_TRACES else gold_solution
    comps = compute_components(q, tr, gs, gold_answer)
    label = classify(comps)
    diag = None
    if needs_diagnosis(label):
        diag = diagnose(comps.E, comps.V, comps.n_steps, comps.n_checkpoints,
                        comps.G, comps.coherent, n_equations=comps.n_equations)
    return comps, label, diag, tr


# =============================================================================
# 4. DEMO MODE
# =============================================================================

def _retrieval_fn(mode: str):
    """Which retrieval arm to use. 'knn' = similarity-only (the naive-
    accumulation baseline); '3stage' = relevance -> complexity re-rank ->
    diversity (design doc Layer-1)."""
    sel = ExemplarSelector()
    return sel.get_knn_exemplars if mode == "knn" else sel.get_exemplars_3stage


def run_demo(gen, manual, eval_problems, strategy_id, n, max_new_tokens,
             retrieval="3stage"):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = _retrieval_fn(retrieval)
    print("\n" + "#" * 70)
    print(f"#  DEMO | strategy {strategy_id}={STRATEGY_NAMES[strategy_id]} | "
          f"{n} problems | backend={backend_name()}")
    print("#" * 70)

    for i, prob in enumerate(eval_problems[:n], 1):
        q, gold_sol, gold_ans = prob["question"], prob["answer"], prob["gold_answer"]
        cat_info = cat.categorize(q)
        prompt, _ = G._build_prompt(strategy_id, q, seeds, [], cat_info, knn)
        n_tok = len(gen.tokenizer.encode(prompt)) if gen.tokenizer else len(prompt.split())
        n_shots = max(prompt.count("Question:") - 1, 0)

        print("\n" + "=" * 70)
        print(f"[{i}/{n}]  category={cat_info['category_type']}/"
              f"{cat_info['complexity']}  op={cat_info['main_operation']}")
        print("=" * 70)
        print(f"QUESTION:\n  {q}")
        print(f"\nPROMPT: {n_tok} tokens | protocol_header={'#### <number>' in prompt} | "
              f"shots={n_shots}")

        t0 = time.time()
        res = gen.generate(q, strategy_id, manual_exemplars=seeds, pool_exemplars=[],
                           category_info=cat_info, max_new_tokens=max_new_tokens,
                           return_logprobs=False, knn_fn=knn)
        dt = time.time() - t0
        trace = res["trace"]

        print(f"\nMODEL TRACE ({res['num_tokens']} tok, {dt:.1f}s):")
        print("\n".join("  " + ln for ln in trace.splitlines()) or "  <empty>")
        print("\nGOLD:\n  " + "\n  ".join(strip_annotations(gold_sol).splitlines()))

        comps, label, diag, scored = score_trace(q, trace, gold_sol, gold_ans)
        from extraction import (extract_gold_checkpoints, extract_values,
                                value_in, extract_equations)
        cps = extract_gold_checkpoints(normalize_math(gold_sol))
        tvals = extract_values(scored)
        hitmiss = " ".join(f"{c:g}:{'HIT' if value_in(c, tvals) else 'MISS'}" for c in cps)
        eqs = extract_equations(scored)
        eqbreak = " ".join("T" if e.is_true else "F" for e in eqs) or "none"

        print(f"\nCHECKPOINTS: {hitmiss or 'none'}")
        print(f"EQUATIONS ({len(eqs)}): {eqbreak}")
        print(f"SIGNALS: E={comps.E:.2f} V={comps.V:.2f} R={comps.R:.2f} "
              f"A={comps.A:.2f} G={comps.G:.2f} coherent={comps.coherent}")
        print(f"COUNTS:  misses={comps.misses} bad_eqs={comps.bad_eqs} "
              f"s={comps.n_steps} s_hat={comps.n_checkpoints}  "
              f"pred={comps.final_answer} gold={gold_ans} correct={comps.correct}")
        print(f"LABEL:   {label.value}  ->  {CONSEQUENCE[label]}")
        if diag:
            print(f"DIAGNOSIS: {diag.ftype.value}  ({diag.reason})")
            print(f"   typed cure:        {TYPED_CURE[diag.ftype]}")
            print(f"   typed instruction: {TYPED_INSTRUCTION[diag.ftype] or '(generic)'}")
        else:
            print("DIAGNOSIS: (GOOD -- enters the pool, not diagnosed)")

    print("\n" + "#" * 70)
    print("#  DEMO COMPLETE -- no storage, no retries")
    print("#" * 70)


# =============================================================================
# 5. FULL RUN  (pass 1 + two deferred retry iterations, detailed storage, resume)
# =============================================================================

def _typed_candidates(records, ftype):
    """Filter a list of exemplar records by the design's structural rules for a
    failure type (section 4). Works on both GOOD-pool records and seed records
    (seeds carry question/trace; signals may be absent, so rules degrade
    gracefully to question/trace features)."""
    def n_q_numbers(rec):
        return len(re.findall(r"\d+(?:\.\d+)?", rec.get("question", "")))

    def naming(rec):
        t = (rec.get("question", "") + " " + rec.get("trace", "")).lower()
        return any(w in t for w in ("how many", "how much", "find", "we need",
                                    "what is", "total"))

    def n_eq(rec):
        # equations stated in the trace (proxy when signals absent)
        return rec.get("signals", {}).get("n_equations",
                       rec.get("trace", "").count("="))

    def n_steps(rec):
        return rec.get("signals", {}).get(
            "n_steps", len([l for l in rec.get("trace", "").splitlines()
                            if l.strip() and not l.strip().startswith("####")]))

    out = []
    for rec in records:
        s = rec.get("signals", {})
        V = s.get("V", 1.0)                     # seeds are gold -> treat as sound
        shat = s.get("n_checkpoints", rec.get("complexity", n_steps(rec)))
        if ftype == FailureType.NR and n_eq(rec) <= 2 and naming(rec):
            out.append(rec)
        elif ftype == FailureType.SM and n_q_numbers(rec) >= 3 and n_eq(rec) <= 2 and naming(rec):
            out.append(rec)
        elif ftype == FailureType.CE and V >= 0.85 and n_eq(rec) >= 1:
            out.append(rec)
        elif ftype == FailureType.ST and n_steps(rec) >= max(shat, 1):
            out.append(rec)
    return out


def select_typed_positives(pool, ftype, seeds, k):
    """Typed positives for a diagnosed failure, drawn from the GOOD pool FIRST
    (the model's own successes), then TOPPED UP FROM SEEDS (gold exemplars) that
    match the same structural type. This is the design doc's specified cold-start
    fallback -- without it, an empty early pool makes typed retry identical to
    generic and the headline comparison cannot run. Seeds are gold, so they
    trivially satisfy the structural criteria; using them as typed positives is
    honest and reported, not gold-answer leakage (the retry stays hint-free)."""
    cand = _typed_candidates(pool, ftype)
    if len(cand) < k:                          # top up with type-matched seeds
        have = {rec.get("question") for rec in cand}
        for rec in _typed_candidates(seeds, ftype):
            if rec.get("question") not in have:
                cand.append(rec)
                if len(cand) >= k:
                    break
    if len(cand) < k:                          # last resort: any seeds (generic)
        have = {rec.get("question") for rec in cand}
        for rec in seeds:
            if rec.get("question") not in have:
                cand.append(rec)
                if len(cand) >= k:
                    break
    return cand[:k]


def select_generic_positives(pool, seeds, k):
    """Generic (untyped) positives: similarity-agnostic recent GOOD, topped up
    from seeds. This is the ABLATION control for typed positives -- same count,
    same retry, but exemplars NOT matched to the diagnosed type."""
    cand = list(pool)[-k:]
    if len(cand) < k:
        have = {rec.get("question") for rec in cand}
        for rec in seeds:
            if rec.get("question") not in have:
                cand.append(rec)
                if len(cand) >= k:
                    break
    return cand[:k]


def build_retry_prompt(problem, positives, instruction):
    """Hint-free retry: positives + (optional) one-line instruction + problem.
    NEVER the gold answer, NEVER 'you were wrong'."""
    block = G._fmt_exemplars(positives, len(positives))
    instr = (instruction + "\n") if instruction else ""
    return f"{G._HEADER}{block}{instr}Question: {problem}\nSolution:"


def retry_once(gen, rec, pool, seeds, max_new_tokens, mode="typed", budget=8):
    """One hint-free retry. `mode` selects the arm:
        typed     -- typed positives (by diagnosis) + typed instruction  [FADE]
        generic   -- untyped positives, no typed instruction             [ablation]
        immediate -- NO exemplar change, NO instruction, just re-sample   [null]
    All three show the SAME number of exemplars (budget) and use the SAME retry
    budget, so the arms differ by exactly one factor each."""
    ftype = (FailureType(rec["diagnosis"]) if rec.get("diagnosis")
             else FailureType.UNCLASSIFIED)
    if mode == "typed":
        positives = select_typed_positives(pool, ftype, seeds, budget)
        instruction = TYPED_INSTRUCTION.get(ftype, "")
        temperature = 0.0
    elif mode == "generic":
        positives = select_generic_positives(pool, seeds, budget)
        instruction = ""
        temperature = 0.0
    else:  # immediate -- blind re-sample of the original few-shot prompt
        positives = seeds[:budget]
        instruction = ""
        temperature = 0.8                      # must be >0 or greedy repeats

    prompt = build_retry_prompt(rec["question"], positives, instruction)
    traces, _ = gen._run_model(prompt, temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               num_return_sequences=1, return_logprobs=False)
    trace = traces[0]
    comps, label, diag, _ = score_trace(rec["question"], trace,
                                        rec["gold_solution"], rec["gold_answer"])
    gen_tokens = len(gen.tokenizer.encode(trace)) if gen.tokenizer else len(trace.split())
    return trace, comps, label, diag, gen_tokens, len(positives)


def _detail_record(**kw):
    return {k: v for k, v in kw.items()}


def _show_batch_report(batch, start_i, n_total, running):
    """A diagnostic batch report: per-problem lines PLUS the three things you
    actually need to decide whether to keep running -- is the model reasoning,
    is extraction losing answers, and is the pool growing."""
    n = len(batch)
    print("\n" + "=" * 82)
    print(f"  BATCH {start_i}-{start_i + n - 1} of {n_total}   "
          f"|   running accuracy {running['correct']}/{running['seen']} "
          f"= {running['correct'] / max(running['seen'], 1):.1%}")
    print("=" * 82)
    print(f"  {'#':>5} {'category':<10} {'ok':<3} {'label':<14} {'diag':<12} "
          f"{'pred':>7} {'gold':>7} {'E':>4} {'V':>4} {'note':<22}")
    for b in batch:
        # per-problem NOTE: the single most useful thing about this trace
        note = ""
        if b["correct"]:
            note = "correct"
        elif b["gold_in_trace"]:
            note = "** answer in trace, LOST"      # extraction problem
        elif b["pred"] is not None and b["gold"] and abs(b["pred"] - b["gold"]) / abs(b["gold"]) < 0.02:
            note = "rounding (<2%)"
        elif not b["has_hash"]:
            note = "no #### emitted"
        elif b["V"] >= 0.8:
            note = "sound work, wrong ans"
        else:
            note = "reasoning error"
        ok = "Y" if b["correct"] else "."
        print(f"  {b['i']:>5} {b['cat']:<10} {ok:<3} {b['label']:<14} "
              f"{(b['diag'] or '-'):<12} {str(b['pred']):>7} {b['gold']:>7g} "
              f"{b['E']:>4.2f} {b['V']:>4.2f} {note:<22}")

    bc = sum(b["correct"] for b in batch)
    lost = sum(b["gold_in_trace"] and not b["correct"] for b in batch)
    nohash = sum(not b["has_hash"] for b in batch)
    rounding = sum(b["pred"] is not None and b["gold"] and not b["correct"]
                   and abs(b["pred"] - b["gold"]) / abs(b["gold"]) < 0.02 for b in batch)
    print("  " + "-" * 80)
    print(f"  batch: {bc}/{n} correct | "
          f"extraction-lost: {lost} | rounding: {rounding} | no-####: {nohash}/{n}")
    print(f"  batch means: E={sum(b['E'] for b in batch)/n:.2f} "
          f"V={sum(b['V'] for b in batch)/n:.2f} "
          f"G={sum(b['G'] for b in batch)/n:.2f} | "
          f"labels {dict(Counter(b['label'] for b in batch))}")
    print(f"  pool={running['pool']} medium={running['medium']} bad={running['bad']}  "
          f"(pool = GOOD traces available as typed positives)")
    # cumulative health flags -- the lines that tell you to STOP and fix
    seen = max(running["seen"], 1)
    if running["gold_present_wrong"] / seen > 0.08:
        print(f"  [!] EXTRACTION: {running['gold_present_wrong']}/{running['seen']} "
              f"({running['gold_present_wrong']/seen:.0%}) had the gold value in-trace "
              f"but scored wrong -- fixable accuracy being lost.")
    if running.get("nohash_total", 0) / seen > 0.5:
        print(f"  [!] FORMAT: {running['nohash_total']}/{running['seen']} traces emitted "
              f"no '####' -- consider strategy 16 or stronger format enforcement.")
    if running["pool"] < 0.03 * seen:
        print(f"  [!] POOL STARVATION: only {running['pool']} GOOD traces -- typed "
              f"positives will fall back to generic; typed-vs-generic not yet measurable.")
    print("=" * 82)


def run_full(gen, manual, eval_problems, strategy_id, n_problems,
             max_new_tokens, iterations=2, show_every=5, retrieval="3stage",
             retry_mode="typed", run_name=None):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = _retrieval_fn(retrieval)
    from classification import STORE_ROOT as _SR
    _root = (_SR.parent / f"store_{run_name}") if run_name else _SR
    store = TraceStore(root=_root)
    store_root = store.root
    print(f"  outputs -> {store_root}")
    results_path = store_root / config.RESULTS_FILE
    attempts_path = store_root / "attempts.jsonl"

    problems = eval_problems[:n_problems]
    print("\n" + "#" * 70)
    print(f"#  FULL RUN | strategy {strategy_id}={STRATEGY_NAMES[strategy_id]} "
          f"| n={len(problems)} | backend={backend_name()} | retrieval={retrieval}")
    print("#" * 70)

    # ---- resume: reconstruct pool + queues from store/ -------------------
    pool = store.load_pool()
    pool_ids = {r["id"] for r in pool}
    medium = [r for r in store.load_queue("medium") if r["id"] not in pool_ids]
    bad = [r for r in store.load_queue("bad") if r["id"] not in pool_ids]
    done_ids = pool_ids | {r["id"] for r in medium} | {r["id"] for r in bad}
    if done_ids:
        print(f"  RESUME: {len(done_ids)} problems already scored "
              f"(pool={len(pool)} medium={len(medium)} bad={len(bad)})")

    results_f = open(results_path, "a")
    attempts_f = open(attempts_path, "a")

    def log_detail(rec):
        results_f.write(json.dumps(rec) + "\n"); results_f.flush()

    def log_attempt(**kw):
        attempts_f.write(json.dumps(kw) + "\n"); attempts_f.flush()

    diag_dist = Counter()
    per_cat = defaultdict(lambda: [0, 0])
    pass1_correct = 0
    n_gen = n_tok = 0
    # extraction diagnostics -- so ONE run reveals whether accuracy is lost
    # between the model writing an answer and the ladder recording it
    dx_has_hash = dx_highE_wrong = dx_goldpresent_wrong = 0
    batch_rows = []          # rows for the rolling batch report
    t_start = time.time()

    # ---------------- PASS 1 ----------------
    for i, prob in enumerate(problems, 1):
        item_id = f"p1_{i:05d}"
        q, gold_sol, gold_ans = prob["question"], prob["answer"], prob["gold_answer"]
        cat_info = cat.categorize(q)

        if item_id in done_ids:                        # resume skip
            continue

        prompt, _ = G._build_prompt(strategy_id, q, seeds, pool, cat_info, knn)
        p_tok = len(gen.tokenizer.encode(prompt)) if gen.tokenizer else len(prompt.split())
        t0 = time.time()
        res = gen.generate(q, strategy_id, manual_exemplars=seeds, pool_exemplars=pool,
                           category_info=cat_info, max_new_tokens=max_new_tokens,
                           return_logprobs=False, knn_fn=knn)
        dt = time.time() - t0
        n_gen += 1; n_tok += res["num_tokens"]
        trace = res["trace"]
        comps, label, diag, scored = score_trace(q, trace, gold_sol, gold_ans)

        store.add(item_id, q, trace, gold_sol, gold_ans, label, comps,
                  diagnosis=diag.ftype.value if diag else None,
                  diagnosis_reason=diag.reason if diag else None)
        log_detail(_detail_record(
            phase="pass1", id=item_id, eval_index=i - 1, question=q,
            gold_solution=gold_sol, gold_answer=gold_ans,
            category=cat_info, prompt_tokens=p_tok, gen_seconds=round(dt, 2),
            gen_tokens=res["num_tokens"], trace=trace, trace_scored=scored,
            signals=comps.signals(), label=label.value,
            diagnosis=diag.ftype.value if diag else None,
            diagnosis_reason=diag.reason if diag else None))
        log_attempt(id=item_id, pass_=1, label=label.value,
                    correct=comps.correct, diagnosis=diag.ftype.value if diag else None)

        rec = {"id": item_id, "question": q, "trace": trace,
               "gold_solution": gold_sol, "gold_answer": gold_ans,
               "label": label.value, "diagnosis": diag.ftype.value if diag else None}
        if label is Label.GOOD:
            pool.append({**rec, "signals": comps.signals(),
                         "complexity": comps.n_checkpoints})
        elif label in (Label.MEDIUM_CORRECT, Label.MEDIUM_WRONG):
            medium.append(rec)
        else:
            bad.append(rec)

        pass1_correct += int(comps.correct)
        if diag:
            diag_dist[diag.ftype.value] += 1
        per_cat[cat_info["category_type"]][1] += 1
        per_cat[cat_info["category_type"]][0] += int(comps.correct)
        # extraction diagnostics
        if "####" in trace:
            dx_has_hash += 1
        if comps.E >= 0.8 and not comps.correct:
            dx_highE_wrong += 1
        if not comps.correct:
            from extraction import extract_values
            tvals = extract_values(scored)
            if any(abs(v - gold_ans) < 1e-6 for v in tvals):
                dx_goldpresent_wrong += 1
        # per-row extraction check for the batch report
        _gold_in = False
        if not comps.correct:
            from extraction import extract_values
            _tv = extract_values(scored)
            _gold_in = any(abs(v - gold_ans) < 1e-6 for v in _tv)
        batch_rows.append({
            "i": i, "cat": cat_info["category_type"], "correct": comps.correct,
            "label": label.value, "diag": diag.ftype.value if diag else None,
            "pred": comps.final_answer, "gold": gold_ans,
            "E": comps.E, "V": comps.V, "G": comps.G,
            "tok": res["num_tokens"], "has_hash": "####" in trace,
            "gold_in_trace": _gold_in,
        })
        if show_every and (i % show_every == 0 or i == len(problems)):
            _show_batch_report(
                batch_rows, i - len(batch_rows) + 1, len(problems),
                {"correct": pass1_correct, "seen": i, "pool": len(pool),
                 "medium": len(medium), "bad": len(bad),
                 "gold_present_wrong": dx_goldpresent_wrong,
                 "nohash_total": i - dx_has_hash})
            batch_rows = []
        elif i % 10 == 0 or i == len(problems):
            print(f"  pass1 {i}/{len(problems)} | pool={len(pool)} "
                  f"medium={len(medium)} bad={len(bad)} correct={pass1_correct}")

    # count pass-1 correct across ALL problems (including resumed) from store buckets
    solved = set()
    for r in pool:
        if r["id"].startswith("p1_"):
            solved.add(r["id"])
    # a resumed medium/bad correct trace is still 'correct' -- recover from signals
    for bucket in ("medium", "bad"):
        for r in store.load_queue(bucket):
            if r.get("signals", {}).get("correct"):
                solved.add(r["id"])
    pass1_correct = len(solved)
    pass1_acc = pass1_correct / len(problems) if problems else 0.0

    # ---------------- DEFERRED RETRY ITERATIONS ----------------
    iter_reports = []
    for it in range(1, iterations + 1):
        report = {"iter": it}
        for origin, queue in (("MEDIUM", medium), ("BAD", bad)):
            still, trans, recov_by_diag = [], Counter(), Counter()
            newly_correct = 0
            for rec in queue:
                trace, comps, label, diag, gtok, npos = retry_once(
                    gen, rec, pool, seeds, max_new_tokens,
                    mode=retry_mode, budget=G.EXEMPLAR_BUDGET or G.N_SHOTS)
                n_gen += 1; n_tok += gtok
                prev_diag = rec.get("diagnosis")
                trans[label.value] += 1
                became_good = label is Label.GOOD
                is_new = comps.correct and rec["id"] not in solved
                if is_new:
                    newly_correct += 1
                    solved.add(rec["id"])
                    if prev_diag:
                        recov_by_diag[prev_diag] += 1
                log_detail(_detail_record(
                    phase="retry", iter=it, origin=origin, id=rec["id"],
                    question=rec["question"], gold_answer=rec["gold_answer"],
                    prev_label=rec["label"], prev_diagnosis=prev_diag,
                    typed_positives=npos, instruction=TYPED_INSTRUCTION.get(
                        FailureType(prev_diag) if prev_diag else FailureType.UNCLASSIFIED, ""),
                    gen_tokens=gtok, trace=trace, signals=comps.signals(),
                    label=label.value, diagnosis=diag.ftype.value if diag else None,
                    became_good=became_good, newly_correct=is_new))
                log_attempt(id=rec["id"], iter=it, origin=origin, label=label.value,
                            correct=comps.correct, prev_diagnosis=prev_diag)
                new = {**rec, "trace": trace, "label": label.value,
                       "diagnosis": diag.ftype.value if diag else None}
                if became_good:
                    store.add(rec["id"], rec["question"], trace, rec["gold_solution"],
                              rec["gold_answer"], label, comps)
                    pool.append({**new, "signals": comps.signals(),
                                 "complexity": comps.n_checkpoints})
                else:
                    still.append(new)
            report[origin] = {"retried": len(queue), "transitions": dict(trans),
                              "newly_correct": newly_correct,
                              "recovery_by_prior_diagnosis": dict(recov_by_diag)}
            queue[:] = still

        m_rec, m_n = report["MEDIUM"]["newly_correct"], max(report["MEDIUM"]["retried"], 1)
        b_rec, b_n = report["BAD"]["newly_correct"], max(report["BAD"]["retried"], 1)
        report["retry_worthiness_gap"] = round(m_rec / m_n - b_rec / b_n, 4)
        iter_reports.append(report)
        print(f"  iter {it}: MEDIUM +{m_rec}/{report['MEDIUM']['retried']}  "
              f"BAD +{b_rec}/{report['BAD']['retried']}  "
              f"gap={report['retry_worthiness_gap']:+.3f}  pool={len(pool)}")

    results_f.close(); attempts_f.close()

    # ---------------- WRITE SUMMARY / REPORT / CSV / SNAPSHOT ----------------
    final_correct = len(solved)
    final_acc = final_correct / len(problems) if problems else 0.0
    wall = time.time() - t_start
    sec_per_gen = wall / max(n_gen, 1)
    per_cat_out = {c: {"correct": v[0], "total": v[1],
                       "acc": round(v[0] / v[1], 4) if v[1] else None}
                   for c, v in sorted(per_cat.items())}

    summary = {
        "strategy_id": strategy_id, "strategy_name": STRATEGY_NAMES[strategy_id],
        "generation_version": G.GENERATION_VERSION, "embedding_backend": backend_name(),
        "retrieval": retrieval, "model": MODEL_ID, "split": SPLIT_TAG,
        "retry_mode": retry_mode, "exemplar_budget": G.EXEMPLAR_BUDGET,
        "n_problems": len(problems),
        "pass1_correct": pass1_correct, "pass1_accuracy": round(pass1_acc, 4),
        "final_correct": final_correct, "final_accuracy": round(final_acc, 4),
        "recovered_by_retry": final_correct - pass1_correct,
        "final_pool_size": len(pool),
        "pass1_diagnosis_distribution": dict(diag_dist),
        "extraction_diagnostics": {
            "traces_with_hash": dx_has_hash,
            "highE_but_wrong": dx_highE_wrong,
            "gold_in_trace_but_wrong": dx_goldpresent_wrong,
        },
        "iterations": iter_reports,
        "per_category_accuracy": per_cat_out,
        "cost": {"generations": n_gen, "approx_tokens": n_tok,
                 "sec_per_generation": round(sec_per_gen, 2), "wall_seconds": round(wall),
                 "projected_full_gsm8k_gpu_hours": round(sec_per_gen * FULL_GSM8K_TRAIN / 3600, 2)},
    }
    (store_root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))

    snapshot = {k: getattr(config, k) for k in dir(config)
                if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    (store_root / config.CONFIG_SNAPSHOT_FILE).write_text(json.dumps(snapshot, indent=2))

    _write_report(store_root / config.REPORT_FILE, summary)
    _write_csv(store_root / config.CSV_FILE, results_path)

    # ---------------- CONSOLE REPORT ----------------
    print("\n" + "=" * 70)
    print("  FADE FULL-RUN REPORT")
    print("=" * 70)
    print(f"  strategy         : {strategy_id} = {STRATEGY_NAMES[strategy_id]}")
    print(f"  problems         : {len(problems)}")
    print(f"  pass-1 accuracy  : {pass1_acc:.1%} ({pass1_correct})")
    print(f"  final accuracy   : {final_acc:.1%} ({final_correct})")
    print(f"  recovered        : {final_correct - pass1_correct}")
    print(f"  final pool size  : {len(pool)}")
    print(f"  pass-1 diagnoses : {dict(diag_dist)}")
    n_p = max(len(problems), 1)
    print("  extraction diagnostics (does the ladder capture the model's answer?):")
    print(f"    traces with '####'         : {dx_has_hash}/{len(problems)} "
          f"({dx_has_hash / n_p:.0%})")
    print(f"    E>=0.8 but scored WRONG    : {dx_highE_wrong}  "
          f"(did the work, answer not recorded)")
    print(f"    gold value in trace, WRONG : {dx_goldpresent_wrong}  "
          f"(answer present but mis-extracted)")
    for r in iter_reports:
        print(f"  iter {r['iter']}: MEDIUM {r['MEDIUM']['newly_correct']}/{r['MEDIUM']['retried']} "
              f"BAD {r['BAD']['newly_correct']}/{r['BAD']['retried']} "
              f"gap={r['retry_worthiness_gap']:+.3f}")
    print("  per-category:")
    for c, v in per_cat_out.items():
        if v["total"]:
            print(f"    {c:<12} {v['correct']}/{v['total']} = {v['acc']:.1%}")
    print(f"  cost: {n_gen} gens, {sec_per_gen:.1f} s/gen, {wall:.0f}s wall, "
          f"~{summary['cost']['projected_full_gsm8k_gpu_hours']} GPU-h projected")
    print("=" * 70)
    print(f"  outputs written under: {store_root}")
    for f in (config.RESULTS_FILE, config.SUMMARY_FILE, config.REPORT_FILE,
              config.CSV_FILE, config.CONFIG_SNAPSHOT_FILE):
        print(f"    - {f}")
    print("=" * 70)


def _write_report(path, s):
    """A clean, complete, presentation-ready run report. Tables only where they
    earn it; every number tied to a claim; no filler."""
    L = []
    a = s
    ex = a.get("extraction_diagnostics", {})
    it = a["iterations"]
    # headline
    L.append(f"# FADE — GSM8K run report\n")
    L.append(f"**Model** `{a.get('model','?')}` · **frozen** (no weight updates)  ")
    L.append(f"**Strategy** {a['strategy_id']} = `{a['strategy_name']}` · "
             f"**Retrieval** `{a.get('retrieval','?')}` · "
             f"**Eval split** `{a.get('split','?')}`  ")
    L.append(f"**Generation** `{a['generation_version']}` · "
             f"**Embeddings** `{a['embedding_backend']}`\n")

    # 1. headline results
    L.append("## 1. Headline results\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Problems evaluated | {a['n_problems']} |")
    L.append(f"| **Pass-1 accuracy** (first attempt) | **{a['pass1_accuracy']:.1%}** "
             f"({a['pass1_correct']}/{a['n_problems']}) |")
    L.append(f"| **Final accuracy** (after retry) | **{a['final_accuracy']:.1%}** "
             f"({a['final_correct']}/{a['n_problems']}) |")
    L.append(f"| Recovered by retry | +{a['recovered_by_retry']} "
             f"({a['recovered_by_retry']/max(a['n_problems'],1):.1%} of all problems) |")
    L.append(f"| GOOD traces in pool | {a['final_pool_size']} |\n")

    # 2. the headline experiment: retry-worthiness
    L.append("## 2. Retry-worthiness — does the classifier predict which "
             "failures are worth retrying?\n")
    L.append("MEDIUM-labelled failures should recover far more often than BAD "
             "ones. If the gap is ~0, the label is not a property.\n")
    L.append("| Iter | MEDIUM recovered | BAD recovered | Gap (MEDIUM−BAD) |")
    L.append("|---|---|---|---|")
    for r in it:
        m, b = r["MEDIUM"], r["BAD"]
        mr = m["newly_correct"]/max(m["retried"],1)
        br = b["newly_correct"]/max(b["retried"],1)
        L.append(f"| {r['iter']} | {m['newly_correct']}/{m['retried']} ({mr:.0%}) "
                 f"| {b['newly_correct']}/{b['retried']} ({br:.0%}) "
                 f"| **{r['retry_worthiness_gap']:+.3f}** |")
    L.append("")

    # 3. recovery by diagnosed type
    by_diag = {}
    for r in it:
        for origin in ("MEDIUM", "BAD"):
            for d, c in r[origin].get("recovery_by_prior_diagnosis", {}).items():
                by_diag[d] = by_diag.get(d, 0) + c
    if by_diag:
        L.append("## 3. Recovery by diagnosed failure type\n")
        L.append("| Diagnosis | Recovered on retry |")
        L.append("|---|---|")
        for d, c in sorted(by_diag.items(), key=lambda x: -x[1]):
            L.append(f"| {d} | {c} |")
        L.append("")

    # 4. pass-1 label + diagnosis distribution
    L.append("## 4. Pass-1 distribution\n")
    L.append("| Diagnosis (non-GOOD traces) | Count |")
    L.append("|---|---|")
    tot = sum(a["pass1_diagnosis_distribution"].values()) or 1
    for k, v in sorted(a["pass1_diagnosis_distribution"].items(),
                       key=lambda x: -x[1]):
        L.append(f"| {k} | {v} ({v/tot:.0%}) |")
    L.append("")

    # 5. per-category accuracy
    L.append("## 5. Accuracy by problem category\n")
    L.append("| Category | Correct / Total | Accuracy |")
    L.append("|---|---|---|")
    for c, v in sorted(a["per_category_accuracy"].items(),
                       key=lambda x: -(x[1]["acc"] or 0)):
        if v["total"]:
            L.append(f"| {c} | {v['correct']}/{v['total']} | {v['acc']:.1%} |")
    L.append("")

    # 6. extraction health (proves accuracy isn't lost to parsing)
    if ex:
        L.append("## 6. Extraction health\n")
        L.append("Confirms the accuracy number reflects the model, not parsing bugs.\n")
        L.append("| Check | Value |")
        L.append("|---|---|")
        L.append(f"| Traces emitting `####` | {ex.get('traces_with_hash','?')}"
                 f"/{a['n_problems']} |")
        L.append(f"| Did the work (E≥0.8) but scored wrong | {ex.get('highE_but_wrong','?')} |")
        L.append(f"| Gold value in trace but scored wrong | "
                 f"{ex.get('gold_in_trace_but_wrong','?')} |")
        L.append("")

    # 7. cost
    c = a["cost"]
    L.append("## 7. Cost\n")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Generations | {c['generations']} |")
    L.append(f"| Seconds / generation | {c['sec_per_generation']} |")
    L.append(f"| Wall time | {c['wall_seconds']} s ({c['wall_seconds']/3600:.1f} h) |")
    L.append(f"| Projected full-GSM8K-train | {c['projected_full_gsm8k_gpu_hours']} GPU-h |")
    L.append("")

    L.append("---")
    L.append("*Reproduce: config in `config_snapshot.json`; per-attempt detail in "
             "`results.jsonl`; per-problem outcomes in `results.csv`.*")
    path.write_text("\n".join(L) + "\n")


def _write_csv(path, results_path):
    """Per-problem FINAL outcome, spreadsheet-ready: reduce results.jsonl to one
    row per problem (pass-1 label + whether any attempt got it right)."""
    rows = {}
    if not results_path.exists():
        return
    with open(results_path) as f:
        for ln in f:
            r = json.loads(ln)
            pid = r["id"]
            if r["phase"] == "pass1":
                rows[pid] = {
                    "id": pid, "eval_index": r.get("eval_index"),
                    "category": r["category"]["category_type"],
                    "complexity": r["category"]["complexity"],
                    "pass1_label": r["label"],
                    "pass1_diagnosis": r.get("diagnosis") or "",
                    "pass1_correct": r["signals"]["correct"],
                    "final_correct": r["signals"]["correct"],
                    "recovered": False,
                    "E": r["signals"]["E"], "V": r["signals"]["V"],
                    "G": r["signals"]["G"], "misses": r["signals"]["misses"],
                    "bad_eqs": r["signals"]["bad_eqs"]}
            else:  # retry row: update final outcome
                if pid in rows and r.get("newly_correct"):
                    rows[pid]["final_correct"] = True
                    rows[pid]["recovered"] = True
    if not rows:
        return
    cols = list(next(iter(rows.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows.values():
            w.writerow(row)


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    global MODEL_ID
    ap = argparse.ArgumentParser(description="FADE Kaggle entrypoint")
    ap.add_argument("--demo", type=int, default=0,
                    help="run demo on the first N eval problems (no storage/retries)")
    ap.add_argument("--n-problems", type=int, default=200,
                    help="full-run problem count (ignored when --demo is set)")
    ap.add_argument("--strategy", type=int, default=2,
                    help="strategy id 0-15 (default 2 = few_shot / 8-shot)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--secret-name", default="HF_TOKEN",
                    help="name of the Kaggle Secret holding the HF token")
    ap.add_argument("--model", default=None,
                    help="HF model id (e.g. meta-llama/Llama-2-7b-chat-hf). "
                         "Chat/instruct models auto-enable the chat template.")
    ap.add_argument("--retry-mode", choices=["typed", "generic", "immediate"],
                    default="typed",
                    help="typed = FADE (positives by diagnosis + typed "
                         "instruction); generic = untyped positives (ablation: "
                         "does TYPING help?); immediate = blind re-sample "
                         "(null: does retry-at-all help?). All at the same "
                         "exemplar budget and retry count.")
    ap.add_argument("--exemplar-budget", type=int, default=8,
                    help="exemplars shown per prompt, ENFORCED across all arms "
                         "for count parity (few_shot, cot, fade_best, retries). "
                         "The confound-killer: gains can't be 'more shots'.")
    ap.add_argument("--retrieval", choices=["knn", "3stage"], default="3stage",
                    help="'knn' = similarity-only baseline arm; '3stage' = "
                         "relevance + complexity re-rank + diversity")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="'train' = seeds+eval both from train (default); "
                         "'test' = seeds from train, EVALUATE on the 1319 test "
                         "problems (directly comparable to published numbers)")
    ap.add_argument("--eval-offset", type=int, default=0,
                    help="skip the first K eval problems (run the dataset in "
                         "chunks across Kaggle sessions: 0, then 500, then ...)")
    ap.add_argument("--seed-pool", type=int, default=2000,
                    help="seeds selected from the first K TRAIN "
                         "problems (train only, never test)")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--run-name", default=None,
                    help="write outputs to store_<run-name>/ instead of store/. "
                         "Use distinct names to run arms in PARALLEL notebooks "
                         "(e.g. --run-name typed vs --run-name immediate) without "
                         "clobbering each other. They share artifacts/ (same "
                         "eval order), so the comparison stays on identical "
                         "problems.")
    ap.add_argument("--show-every", type=int, default=5,
                    help="print the live trace + scoring every N problems "
                         "during the run (0 = off). Interrupt anytime; the run "
                         "is resumable from store/.")
    args = ap.parse_args()

    global SPLIT_TAG
    if args.model:
        MODEL_ID = args.model
    SPLIT_TAG = args.split
    G.set_exemplar_budget(args.exemplar_budget)
    version_banner()
    print(f"  exemplar budget: {args.exemplar_budget} (enforced across all arms)")
    print(f"  model: {MODEL_ID}")
    token = resolve_hf_token(args.secret_name)
    verify_identity_and_access(token)

    eval_need = args.demo if args.demo else args.n_problems
    if args.split == "test":
        # seeds from train (need seed_pool + margin), eval on the full test split
        print("\nLoading GSM8K (train for seeds, test for eval)...")
        train_problems = _load_split("train", args.seed_pool + 100)
        test_problems = _load_split("test")           # all 1319
        manual, eval_problems = build_seeds_and_eval_order(
            train_problems, args.seed_pool, eval_split_problems=test_problems)
    else:
        problems = load_gsm8k(args.seed_pool + eval_need + 100)
        manual, eval_problems = build_seeds_and_eval_order(problems, args.seed_pool)
    model, tok = load_model(token)
    gen = ExemplarGenerator(model=model, tokenizer=tok)

    if args.eval_offset:
        eval_problems = eval_problems[args.eval_offset:]
        print(f"  eval-offset: skipping first {args.eval_offset} eval problems")
    if args.demo:
        run_demo(gen, manual, eval_problems, args.strategy, args.demo,
                 args.max_new_tokens, retrieval=args.retrieval)
    else:
        run_full(gen, manual, eval_problems, args.strategy, args.n_problems,
                 args.max_new_tokens, iterations=args.iterations,
                 show_every=args.show_every, retrieval=args.retrieval,
                 retry_mode=args.retry_mode, run_name=args.run_name)


if __name__ == "__main__":
    main()