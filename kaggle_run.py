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


def load_gsm8k(n_needed: int) -> list[dict]:
    """Stream GSM8K train in fixed order, PREPROCESSED. Each item:
    {question(clean), answer(annotated,normalised), gold_answer(float)}.
    Drops any row whose gold answer doesn't parse (kept in a counter)."""
    from datasets import load_dataset
    print("\nLoading GSM8K (train split)...")
    ds, last_err = None, None
    for ds_id in ("openai/gsm8k", "gsm8k"):
        try:
            ds = load_dataset(ds_id, "main", split="train")
            print(f"  source: {ds_id}")
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"could not load GSM8K: {last_err}")

    out, dropped = [], 0
    for row in ds:
        p = preprocess_problem({"question": row["question"], "answer": row["answer"]})
        ga = gold_answer_value(p["answer"])
        if ga is None:
            dropped += 1
            continue
        p["gold_answer"] = ga
        out.append(p)
        if len(out) >= n_needed:
            break
    print(f"  {len(out)} problems ready (preprocessed) | dropped {dropped} unparseable")
    return out


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


def build_seeds_and_eval_order(problems, seed_pool_size=300):
    cache = ARTIFACTS / f"seeds_evalorder_pool{seed_pool_size}.json"
    if cache.exists():
        print(f"\nLoading cached seeds + eval order from {cache.name}")
        blob = json.loads(cache.read_text())
        return ({int(k): v for k, v in blob["manual_exemplars"].items()},
                blob["eval_problems"])

    print(f"\nSelecting seeds from the first {seed_pool_size} problems...")
    pool = problems[:seed_pool_size]
    selector = ExemplarSelector(pool_size=seed_pool_size)
    manual, _ = selector.select(pool, QuestionCategorizer(), ground_truth_key="answer")

    for sid, seeds in manual.items():
        for ex in seeds:                       # attach clean trace + gold answer
            gold_sol = ex.get("ground_truth", "")
            ex["gold_solution"] = gold_sol
            ex["trace"] = strip_annotations(gold_sol)
            ex["gold_answer"] = gold_answer_value(gold_sol)
        manual[sid] = _interleave_by_category(seeds)

    selected_qs = {ex["question"] for seeds in manual.values() for ex in seeds}
    eval_problems = [p for p in pool if p["question"] not in selected_qs]
    eval_problems += problems[seed_pool_size:]

    # hard guarantee: no problem is ever both an exemplar and a test item
    eval_qs = {p["question"] for p in eval_problems}
    overlap = selected_qs & eval_qs
    assert not overlap, f"seed/eval overlap ({len(overlap)}) -- exclusion broken"
    print(f"  seed/eval disjoint OK | {len(selected_qs)} seeds, {len(eval_problems)} eval")

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

def run_demo(gen, manual, eval_problems, strategy_id, n, max_new_tokens):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = ExemplarSelector().get_knn_exemplars
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

def select_typed_positives(pool, ftype, k=2):
    """Curate typed positive exemplars from the GOOD pool by the structural rules
    in the design (section 4). Generic fallback (recent GOOD) when a typed pool
    has < 3 members -- the cold-start behaviour the ablation wants."""
    def n_q_numbers(rec):
        return len(re.findall(r"\d+(?:\.\d+)?", rec["question"]))

    def naming(rec):
        t = (rec["question"] + " " + rec.get("trace", "")).lower()
        return any(w in t for w in ("how many", "how much", "find", "we need",
                                    "what is", "total"))

    cand = []
    for rec in pool:
        s = rec.get("signals", {})
        neq, nst = s.get("n_equations", 0), s.get("n_steps", 0)
        shat, V = s.get("n_checkpoints", rec.get("complexity", 0)), s.get("V", 0.0)
        if ftype == FailureType.NR and neq <= 2 and naming(rec):
            cand.append(rec)
        elif ftype == FailureType.SM and n_q_numbers(rec) >= 3 and neq <= 2 and naming(rec):
            cand.append(rec)
        elif ftype == FailureType.CE and V >= 0.85 and neq >= 1:
            cand.append(rec)
        elif ftype == FailureType.ST and nst >= max(shat, 1):
            cand.append(rec)
    if len(cand) < 3:
        cand = list(pool)[-max(k, 3):]
    return cand[:k]


def build_retry_prompt(problem, positives, instruction):
    """Hint-free typed retry: typed positives + one-line typed instruction + the
    problem. NEVER the gold answer, NEVER 'you were wrong'."""
    block = G._fmt_exemplars(positives, len(positives))
    instr = (instruction + "\n") if instruction else ""
    return f"{G._HEADER}{block}{instr}Question: {problem}\nSolution:"


def retry_once(gen, rec, pool, max_new_tokens):
    ftype = (FailureType(rec["diagnosis"]) if rec.get("diagnosis")
             else FailureType.UNCLASSIFIED)
    positives = select_typed_positives(pool, ftype, k=2)
    instruction = TYPED_INSTRUCTION.get(ftype, "")
    prompt = build_retry_prompt(rec["question"], positives, instruction)
    traces, _ = gen._run_model(prompt, temperature=0.0, max_new_tokens=max_new_tokens,
                               num_return_sequences=1, return_logprobs=False)
    trace = traces[0]
    comps, label, diag, _ = score_trace(rec["question"], trace,
                                        rec["gold_solution"], rec["gold_answer"])
    gen_tokens = len(gen.tokenizer.encode(trace)) if gen.tokenizer else len(trace.split())
    return trace, comps, label, diag, gen_tokens, len(positives)


def _detail_record(**kw):
    return {k: v for k, v in kw.items()}


def _show_batch_report(batch, start_i, n_total, running):
    """Print a REPORT over the last N problems (not one sample). Shows every
    problem in the batch side by side so patterns are visible mid-run: which
    labels/diagnoses dominate, whether answers are being extracted, whether E is
    moving. Interrupt the run here if something looks wrong -- progress is
    already flushed to store/ and the same command resumes."""
    n = len(batch)
    print("\n" + "=" * 78)
    print(f"  BATCH REPORT | problems {start_i}-{start_i + n - 1} of {n_total}")
    print("=" * 78)
    print(f"  {'#':>4} {'category':<11} {'ok':<5} {'label':<14} {'diag':<13} "
          f"{'pred':>8} {'gold':>8} {'E':>5} {'V':>5} {'G':>5} {'tok':>5}")
    for b in batch:
        print(f"  {b['i']:>4} {b['cat']:<11} {str(b['correct']):<5} {b['label']:<14} "
              f"{(b['diag'] or '-'):<13} {str(b['pred']):>8} {b['gold']:>8g} "
              f"{b['E']:>5.2f} {b['V']:>5.2f} {b['G']:>5.2f} {b['tok']:>5}")
    bc = sum(b["correct"] for b in batch)
    print(f"  -- batch: {bc}/{n} correct | "
          f"labels: {dict(Counter(b['label'] for b in batch))}")
    print(f"  -- batch diagnoses: {dict(Counter(b['diag'] for b in batch if b['diag']))}")
    print(f"  -- batch means: E={sum(b['E'] for b in batch)/n:.2f} "
          f"V={sum(b['V'] for b in batch)/n:.2f} "
          f"G={sum(b['G'] for b in batch)/n:.2f} "
          f"tok={sum(b['tok'] for b in batch)/n:.0f} | "
          f"with '####': {sum(b['has_hash'] for b in batch)}/{n}")
    print(f"  -- RUNNING: {running['correct']}/{running['seen']} correct "
          f"({running['correct'] / max(running['seen'], 1):.1%}) | "
          f"pool={running['pool']} medium={running['medium']} bad={running['bad']}")
    # the one line that says whether accuracy is being LOST vs never earned
    if running["gold_present_wrong"]:
        print(f"  -- WARNING: {running['gold_present_wrong']} traces so far contain "
              f"the gold value but scored WRONG (extraction may be losing answers)")
    print("=" * 78)


def run_full(gen, manual, eval_problems, strategy_id, n_problems,
             max_new_tokens, iterations=2, show_every=5):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = ExemplarSelector().get_knn_exemplars
    store = TraceStore()
    store_root = store.root
    results_path = store_root / config.RESULTS_FILE
    attempts_path = store_root / "attempts.jsonl"

    problems = eval_problems[:n_problems]
    print("\n" + "#" * 70)
    print(f"#  FULL RUN | strategy {strategy_id}={STRATEGY_NAMES[strategy_id]} "
          f"| n={len(problems)} | backend={backend_name()}")
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
        batch_rows.append({
            "i": i, "cat": cat_info["category_type"], "correct": comps.correct,
            "label": label.value, "diag": diag.ftype.value if diag else None,
            "pred": comps.final_answer, "gold": gold_ans,
            "E": comps.E, "V": comps.V, "G": comps.G,
            "tok": res["num_tokens"], "has_hash": "####" in trace,
        })
        if show_every and (i % show_every == 0 or i == len(problems)):
            _show_batch_report(
                batch_rows, i - len(batch_rows) + 1, len(problems),
                {"correct": pass1_correct, "seen": i, "pool": len(pool),
                 "medium": len(medium), "bad": len(bad),
                 "gold_present_wrong": dx_goldpresent_wrong})
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
                    gen, rec, pool, max_new_tokens)
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
    L = []
    L.append(f"# FADE run report -- {s['strategy_name']} (strategy {s['strategy_id']})\n")
    L.append(f"- generation version: `{s['generation_version']}`  |  embedding backend: `{s['embedding_backend']}`")
    L.append(f"- problems: **{s['n_problems']}**")
    L.append(f"- pass-1 accuracy: **{s['pass1_accuracy']:.1%}** ({s['pass1_correct']})")
    L.append(f"- final accuracy: **{s['final_accuracy']:.1%}** ({s['final_correct']})")
    L.append(f"- recovered by retry: **{s['recovered_by_retry']}**")
    L.append(f"- final pool size: {s['final_pool_size']}\n")
    L.append("## Pass-1 diagnosis distribution")
    L.append("| type | count |\n|---|---|")
    for k, v in s["pass1_diagnosis_distribution"].items():
        L.append(f"| {k} | {v} |")
    L.append("\n## Deferred retry iterations")
    L.append("| iter | MEDIUM recovered | BAD recovered | retry-worthiness gap |\n|---|---|---|---|")
    for r in s["iterations"]:
        L.append(f"| {r['iter']} | {r['MEDIUM']['newly_correct']}/{r['MEDIUM']['retried']} "
                 f"| {r['BAD']['newly_correct']}/{r['BAD']['retried']} "
                 f"| {r['retry_worthiness_gap']:+.3f} |")
    L.append("\n## Per-category accuracy")
    L.append("| category | correct/total | acc |\n|---|---|---|")
    for c, v in s["per_category_accuracy"].items():
        if v["total"]:
            L.append(f"| {c} | {v['correct']}/{v['total']} | {v['acc']:.1%} |")
    c = s["cost"]
    L.append("\n## Cost")
    L.append(f"- generations: {c['generations']}  |  ~tokens: {c['approx_tokens']}")
    L.append(f"- {c['sec_per_generation']} s/generation  |  wall {c['wall_seconds']}s")
    L.append(f"- projected full-GSM8K: **{c['projected_full_gsm8k_gpu_hours']} GPU-h**")
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
    ap.add_argument("--seed-pool", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--show-every", type=int, default=5,
                    help="print the live trace + scoring every N problems "
                         "during the run (0 = off). Interrupt anytime; the run "
                         "is resumable from store/.")
    args = ap.parse_args()

    if args.model:
        MODEL_ID = args.model
    version_banner()
    print(f"  model: {MODEL_ID}")
    token = resolve_hf_token(args.secret_name)
    verify_identity_and_access(token)

    eval_need = args.demo if args.demo else args.n_problems
    problems = load_gsm8k(args.seed_pool + eval_need + 100)
    manual, eval_problems = build_seeds_and_eval_order(problems, args.seed_pool)
    model, tok = load_model(token)
    gen = ExemplarGenerator(model=model, tokenizer=tok)

    if args.demo:
        run_demo(gen, manual, eval_problems, args.strategy, args.demo, args.max_new_tokens)
    else:
        run_full(gen, manual, eval_problems, args.strategy, args.n_problems,
                 args.max_new_tokens, iterations=args.iterations,
                 show_every=args.show_every)


if __name__ == "__main__":
    main()