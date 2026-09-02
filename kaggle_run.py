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
from classification import (classify, CONSEQUENCE, needs_diagnosis, Label,
                            TraceStore, sanitize_exemplar)
from diagnosis import (diagnose, diagnose_components, TYPED_INSTRUCTION,
                       TYPED_CURE, FailureType)
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
    if v != "v5-hashline":
        print("  !! expected 'v5-hashline' -- generation.py is stale/modified. Abort.")
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


def _score_seed(ex: dict) -> None:
    """[AUDIT D16] Give every seed REAL `signals`, in place.

    THE BUG THIS FIXES. `_typed_candidates` reads `V = rec["signals"].get("V", 1.0)`.
    Seeds carried NO `signals`, so V defaulted to 1.0 and EVERY seed passed CE's
    `V >= 0.85` test. With the pool empty or tiny (it peaked at 133 = 8.9% of
    eval), typed positives for CE were therefore just "the first 8 seeds" --
    byte-identical to `select_generic_positives`. Measured overlap with an empty
    pool: CE 8/8 (IDENTICAL ARMS), SM 3/8, NR 1/8.

    CE was the largest typed bucket (296 of 1,367 diagnoses), so the headline
    typed-vs-generic comparison was null BY CONSTRUCTION for 22% of all
    diagnoses -- independently of the D1 prompt bug.

    A seed's trace is its own gold solution with annotations stripped, so scoring
    it against its own gold is exact and costs no GPU (35 traces).
    """
    gold_sol = ex.get("gold_solution") or ""
    trace = ex.get("trace") or ""
    ga = ex.get("gold_answer")
    if not gold_sol or not trace or ga is None:
        return
    try:
        c = compute_components(ex.get("question", ""), normalize_math(trace),
                               normalize_math(gold_sol), ga)
        ex["signals"] = c.signals()
        ex["complexity"] = c.n_checkpoints      # int, not the categorizer string
    except Exception as e:
        print(f"  [warn] could not score seed: {e}")


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
        manual = {int(k): v for k, v in blob["manual_exemplars"].items()}
        # [AUDIT D16] Caches written before this fix carry NO `signals`, which is
        # exactly what made every seed pass CE's V >= 0.85 test and collapsed
        # typed positives onto generic. Score them on load, so an existing
        # artifacts/ file cannot silently reintroduce the bug.
        n_scored = 0
        for seeds in manual.values():
            for ex in seeds:
                if not ex.get("signals"):
                    _score_seed(ex)
                    n_scored += bool(ex.get("signals"))
        if n_scored:
            print(f"  [D16] scored {n_scored} cached seeds (no GPU) so typed "
                  f"positives are distinguishable from generic")
        return manual, blob["eval_problems"]

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
            _score_seed(ex)                 # [AUDIT D16] see below
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
    # [AUDIT SELF-REVIEW] With polish retries off (D20) a CORRECT trace's
    # diagnosis has no consumer, and 60 such traces were inflating the
    # UNCLASSIFIED rate for a non-reason (they cannot fire WP or AL, both of
    # which require `not correct`). config.DIAGNOSE_CORRECT = True restores it.
    _skip = comps.correct and not getattr(config, "DIAGNOSE_CORRECT", False) \
        and not config.RETRY_POLISH_MEDIUM_CORRECT
    if needs_diagnosis(label) and not _skip:
        # [AUDIT] diagnose_components passes E_eff and the new signals
        # (A, A_struct, n_symbolic, truncated, correct) so a caller cannot
        # silently fall back to the pre-audit rule set by forgetting an argument.
        diag = diagnose_components(comps)
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

    def n_symbolic(rec):
        """[AUDIT D12] '='-lines carrying a genuine algebraic variable."""
        s = rec.get("signals") or {}
        if isinstance(s.get("n_symbolic"), int):
            return s["n_symbolic"]
        from extraction import symbolic_line_count
        return symbolic_line_count(rec.get("trace", ""))

    def _int_shat(rec, s):
        """[AUDIT D17] `complexity` is an INT (s_hat) on pool records but the
        CATEGORIZER'S STRING ('simple') on seed records, so the old
        `max(shat, 1)` raised
            TypeError: '>' not supported between instances of 'int' and 'str'
        the moment ST looked at seeds. It survived the 1,500-problem run only
        because ST fired 7 times and the pool happened to cover it. Fixing D14
        makes ST fire far more often, so this WOULD have crashed."""
        v = s.get("n_checkpoints")
        if isinstance(v, (int, float)) and v:
            return int(v)
        c = rec.get("complexity")
        if isinstance(c, (int, float)) and c:
            return int(c)
        return max(n_steps(rec), 1)

    out = []
    for rec in records:
        s = rec.get("signals") or {}
        # [AUDIT D16] With seeds now scored (see _score_seed) this default is a
        # genuine fallback rather than the thing that collapsed typed==generic.
        V = s.get("V", 1.0)
        shat = _int_shat(rec, s)
        if ftype == FailureType.NR and n_eq(rec) <= 2 and naming(rec):
            out.append(rec)
        elif ftype == FailureType.SM and n_q_numbers(rec) >= 3 and n_eq(rec) <= 2 and naming(rec):
            out.append(rec)
        elif ftype == FailureType.CE and V >= 0.85 and n_eq(rec) >= 1:
            out.append(rec)
        elif ftype == FailureType.ST and n_steps(rec) >= max(shat, 1):
            out.append(rec)
        # ---- new types [AUDIT D11/D12] -------------------------------------
        elif ftype == FailureType.AL:
            # cure for algebraic non-execution: exemplars that are PURELY
            # numeric (no symbols) and visibly compute a value on every line
            if n_symbolic(rec) == 0 and n_eq(rec) >= 2 and V >= 0.85:
                out.append(rec)
        elif ftype == FailureType.WP:
            # cure for a wrong plan: exemplars that NAME the target quantity and
            # carry several ordered steps -- planning, not arithmetic hygiene
            if naming(rec) and n_steps(rec) >= 3 and n_eq(rec) >= 2:
                out.append(rec)
        elif ftype == FailureType.TR:
            # TR is a generation failure; it has no cure. Callers should
            # re-generate. Fall through to generic if it ever gets here.
            pass
    return out


def _type_fit_score(rec, ftype) -> float:
    """[AUDIT D16 / retrieval upgrade 2] How WELL does this exemplar exemplify the
    cure for `ftype`? Higher is better.

    Eligibility alone is not enough. `_typed_candidates` is a boolean filter, and
    both selectors then took the FIRST k of their list in seed order -- so
    whenever the eligible set was most of the seeds (CE 31/35, WP 24/35, ST 35/35)
    the typed and generic prompts were nearly the same 8 exemplars regardless of
    the diagnosis. Ranking by type-fit is what actually makes the typed arm a
    different treatment rather than a differently-labelled one.
    """
    s = rec.get("signals") or {}
    trace = rec.get("trace", "")
    q = rec.get("question", "")
    n_eq = s.get("n_equations", trace.count("="))
    n_st = s.get("n_steps", len([l for l in trace.splitlines() if l.strip()]))
    shat = s.get("n_checkpoints") or 1
    V = s.get("V", 1.0)
    n_sym = s.get("n_symbolic", 0)
    text = (q + " " + trace).lower()

    if ftype == FailureType.NR:
        # simplest complete demonstration: fewest operations, names its target
        return -float(n_eq) + (1.0 if any(w in text for w in
                               ("how many", "how much", "we need", "find")) else 0.0)
    if ftype == FailureType.SM:
        # comprehension: many given quantities, few computations
        return len(re.findall(r"\d+(?:\.\d+)?", q)) - 1.5 * float(n_eq)
    if ftype == FailureType.CE:
        # arithmetic hygiene: flawless arithmetic, plenty of it
        return 3.0 * float(V) + 0.5 * float(n_eq)
    if ftype == FailureType.ST:
        # decomposition: MORE written steps than the problem strictly requires
        return float(n_st) - float(shat)
    if ftype == FailureType.AL:
        # numeric-only: zero symbols, every line lands on a number
        return -5.0 * float(n_sym) + float(n_eq) + 2.0 * float(V)
    if ftype == FailureType.WP:
        # planning: names the target quantity, several ordered steps
        return (2.0 if any(w in text for w in ("how many", "how much", "total",
                                               "we need", "find")) else 0.0) \
               + 0.5 * float(n_st)
    return 0.0


def _retrieve(question, candidates, k, ftype=None):
    """[AUDIT D39] Question-aware retrieval for the RETRY path.

    Routes through the same 3-stage retriever pass-1 uses (relevance ->
    complexity re-rank -> MMR diversity), with the structural-shape blend and the
    type-match bonus. Passing ftype makes it prefer exemplars that exemplify the
    cure for that failure; passing None makes it the untyped control. Both arms
    therefore get topical relevance and differ only by the thing under test.
    """
    try:
        from exemplar_selector import ExemplarSelector
        sel = ExemplarSelector(pool_size=len(candidates) or 1)
        ft = ftype.value if hasattr(ftype, "value") else ftype

        def _order(xs):
            # [AUDIT D42] most relevant LAST, nearest the live question.
            return list(reversed(xs)) if getattr(
                config, "RETRY_RELEVANCE_LAST", False) else list(xs)

        # [AUDIT D55c] BOTH arms build the SAME shortlist, with the same size and
        # no ftype. get_exemplars_3stage ends in an MMR diversity pass whose
        # result depends on how many exemplars are being chosen, so asking for 8
        # and asking for 32 give different top-8s. Generic was getting top-8-of-8
        # and typed the first-8-of-32, which is why typed still trailed by 3-14%
        # even after the shortlist stopped being type-biased. One shortlist, one
        # procedure, and the arms differ only in what they do with it.
        over = max(k, int(k * getattr(config, "RETRY_TYPEFIT_OVERSAMPLE", 4)))
        short = list(sel.get_exemplars_3stage(
            question, candidates, min(over, len(candidates))))

        if ft is None:
            return _order(short[:k])

        # [AUDIT D40] Over-sample by relevance, then RE-RANK by type fit.
        #
        # _typed_candidates passes 100% of the pool for ST and CE, 87% for WP and
        # 82% for AL, so filtering-then-retrieving leaves the typed arm identical
        # to generic apart from the instruction. Blending a continuous type-fit
        # score into the ranking makes typing discriminate even where the boolean
        # filter does not.
        # Typing is applied in exactly ONE place: the re-rank of the tail below.
        if len(short) <= k:
            return _order(short[:k])

        fits = [_type_fit_score(r, ftype) for r in short]
        lo, hi = min(fits), max(fits)
        span = (hi - lo) or 1.0
        n = len(short)
        w = float(getattr(config, "RETRY_W_TYPEFIT", 0.45))
        # relevance is encoded as position in `short` (0 = most relevant)
        order = sorted(range(n), key=lambda i: -(
            (1.0 - w) * (1.0 - i / max(n - 1, 1)) + w * ((fits[i] - lo) / span)))

        # [AUDIT D55] Keep the top-FLOOR by pure relevance no matter what the
        # re-rank says, so typed always contains generic's best exemplars and
        # typing only decides the remaining slots. Without this the blend threw
        # away 14-30% of topical relevance on every type -- typed was handing the
        # model worse material than the control it was supposed to beat.
        floor = max(0, min(int(getattr(config, "RETRY_RELEVANCE_FLOOR", 0)), k))
        base, typed_picks = [], []
        if floor:
            base = list(range(min(floor, n)))            # guaranteed by relevance
        seen = set(base)
        for i in order:                                  # blended: relevance x type-fit
            if len(base) + len(typed_picks) >= k:
                break
            if i not in seen:
                typed_picks.append(i); seen.add(i)

        if getattr(config, "RETRY_LAYERED_ORDER", False):
            # [AUDIT D63] Keep the two groups SEPARATE and in order: the
            # relevance-retrieved exemplars first (what a generic arm would show),
            # then the type-selected ones, then -- added by the caller -- the cure
            # demonstration, nearest the live question. Without this grouping the
            # blend interleaves them and there is no layering for the system
            # message's "the last examples show the right approach" to point at.
            base_sorted = sorted(base, reverse=True)     # ascending relevance
            return [short[i] for i in base_sorted + typed_picks][:k]
        return _order([short[i] for i in (base + typed_picks)[:k]])
    except Exception as e:
        print(f"  [D39] retrieval unavailable ({type(e).__name__}: {e}), "
              f"falling back to question-blind selection")
        return []


def select_typed_positives(pool, ftype, seeds, k, question=None):
    """Typed positives for a diagnosed failure, drawn from the GOOD pool FIRST
    (the model's own successes), then TOPPED UP FROM SEEDS (gold exemplars) that
    match the same structural type. This is the design doc's specified cold-start
    fallback -- without it, an empty early pool makes typed retry identical to
    generic and the headline comparison cannot run. Seeds are gold, so they
    trivially satisfy the structural criteria; using them as typed positives is
    honest and reported, not gold-answer leakage (the retry stays hint-free)."""
    # [AUDIT D39] question-aware first: rank the TYPE-ELIGIBLE pool by relevance
    # to THIS question, not by type-fit alone. Without it every retry of a given
    # type saw the same 8 exemplars.
    cand = []
    if question and getattr(config, "RETRY_QUESTION_AWARE", False) and pool:
        # [AUDIT D49] Retrieve over the WHOLE pool by default. The hard filter cost
        # SM 42% of its topical relevance while being a no-op for ST and CE; the
        # type-fit re-rank inside _retrieve supplies the typing continuously.
        if getattr(config, "RETRY_TYPE_HARD_FILTER", False):
            eligible = _typed_candidates(pool, ftype)
            cand = _retrieve(question, eligible if len(eligible) >= k else list(pool),
                             k, ftype)
        else:
            cand = _retrieve(question, list(pool), k, ftype)
    if not cand:
        # [AUDIT] RANK, don't just filter -- see _type_fit_score.
        cand = sorted(_typed_candidates(pool, ftype),
                      key=lambda r: _type_fit_score(r, ftype), reverse=True)
    if len(cand) < k:                          # top up with type-matched seeds
        have = {rec.get("question") for rec in cand}
        for rec in sorted(_typed_candidates(seeds, ftype),
                          key=lambda r: _type_fit_score(r, ftype), reverse=True):
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
    out = cand[:k]

    # [AUDIT D41] Swap ONE slot for a demonstration of the cure -- replaced, never
    # added, so typed and generic still show exactly k exemplars.
    #
    # [AUDIT D47] It must displace the LEAST relevant exemplar, and which end that
    # is depends on RETRY_RELEVANCE_LAST. D42 reversed the block so the most
    # relevant sits nearest the question; D41 was still overwriting the last slot,
    # so the demo was displacing the BEST-matched exemplar every time (measured:
    # similarity 0.169, rank 1 of 8, thrown away). With the ordering reversed the
    # least relevant is at the FRONT, so the demo goes there -- it also primes the
    # procedure early, while the most relevant exemplar keeps the slot adjacent to
    # the live question.
    if getattr(config, "TYPED_APPROACH_DEMO", False):
        from diagnosis import approach_demo
        demo = approach_demo(ftype)
        if demo is not None and len(out) >= 1:
            if getattr(config, "RETRY_LAYERED_ORDER", False):
                # [AUDIT D63] general -> type-matched -> demonstration -> question.
                # _retrieve already returns the shared, relevance-ranked exemplars
                # ahead of the type-selected ones, so only the demo has to move to
                # the end, where it sits closest to the live question.
                out = out[:k - 1] + [demo]
            elif getattr(config, "RETRY_RELEVANCE_LAST", False):
                out = [demo] + out[1:k]
            else:
                out = out[:k - 1] + [demo]
    return out


def select_generic_positives(pool, seeds, k, question=None):
    """Generic (untyped) positives -- the ABLATION control for typed positives:
    same count, same retry, same retrieval, but NOT matched to the diagnosed type.

    [AUDIT D39] This used to take the LAST k of the pool, identically for every
    problem. Both arms are now retrieved by question similarity so the ablation
    isolates typing rather than comparing two topically irrelevant prompts."""
    cand = []
    if (question and getattr(config, "RETRY_QUESTION_AWARE", False)
            and getattr(config, "GENERIC_QUESTION_AWARE", True) and pool):
        cand = _retrieve(question, list(pool), k, ftype=None)
    if not cand:
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
    """Hint-free retry: positives + problem. NEVER the gold answer, NEVER
    'you were wrong'. Returns (prompt, extra_system).

    The typed instruction is NO LONGER inlined between the exemplar block and the
    live question. Doing that broke generation._EX_BLOCK_RE's '\\n\\nQuestion:'
    lookahead in chat mode: the live question got absorbed into the last
    exemplar's answer and the model was handed an EXEMPLAR to solve. Measured
    cost of that bug on the 1,500-problem train run -- grounding on typed retries
    collapsed 0.89 -> 0.12, and NR/SM recovered 0 of 325. The instruction now
    travels separately and is routed into the system message by
    _run_model(extra_system=...)."""
    block = G._fmt_exemplars(positives, len(positives))
    prompt = f"{G._HEADER}{block}Question: {problem}\nSolution:"
    return prompt, (instruction or "")


def apply_cure(gen, question, prev_trace, prev_answer, diagnosis, pool, seeds,
               max_new_tokens, mode="typed", budget=8, negatives=None, n_neg=0,
               rec_id=""):
    """One hint-free retry. `mode` selects the arm:
        typed     -- typed positives (by diagnosis) + typed instruction  [FADE]
        typed+neg -- typed, PLUS a failure-mode warning in the system msg [FADE+]
        generic   -- untyped positives, no typed instruction             [ablation]
        immediate -- NO exemplar change, NO instruction, just re-sample   [null]
    All arms show the SAME number of exemplars (budget) and use the SAME retry
    budget, so they differ by exactly one factor each.

    `typed+neg` is where the two-sided cure should be tested FIRST. Here the type
    is KNOWN -- it came from diagnosing an actual failed trace -- so the arm
    measures the warning itself. Stacking negatives on the FST path instead would
    confound them with a predictor that does not beat its majority baseline, and a
    null there would be uninterpretable.

    The warning goes into `instruction`, which build_retry_prompt routes to the
    system message. It never enters the exemplar block. See negatives.py.
    """
    ftype = (FailureType(diagnosis) if diagnosis
             else FailureType.UNCLASSIFIED)
    # [AUDIT D52] Abstention is right for the taxonomy, wrong for the cure. The
    # RECORDED diagnosis stays whatever the cascade said -- the abstain rate the
    # paper reports is unchanged -- but an UNCLASSIFIED failure with obvious trace
    # evidence gets the matched treatment instead of the generic one. 74 of 88
    # UNCLASSIFIED failures route this way (66 CE, 8 AL).
    _cure_note = ""
    if mode in ("typed", "typed+neg"):
        from diagnosis import cure_type_for
        ftype, _cure_note = cure_type_for(ftype, prev_trace)
    if mode in ("typed", "typed+neg"):
        positives = select_typed_positives(pool, ftype, seeds, budget,
                                           question=question)
        instruction = TYPED_INSTRUCTION.get(ftype, "")
        temperature = 0.0
        # [AUDIT D63] Name the failure mode in the SYSTEM message -- the one place
        # a per-attempt directive can go without touching the exemplar parser.
        if (instruction and getattr(config, "RETRY_NAME_FAILURE_MODE", False)):
            from diagnosis import FAILURE_MODE_PHRASE
            _ph = FAILURE_MODE_PHRASE.get(ftype, "")
            if _ph:
                instruction = (
                    f"A previous attempt at this question {_ph}. "
                    "The last examples above show the right approach for that.\n\n"
                    + instruction)
        if mode == "typed+neg" and negatives is not None and n_neg > 0:
            w = negatives.warning_for(question, ftype.value, k=n_neg)
            if w:
                instruction = (instruction + "\n\n" + w) if instruction else w
    elif mode == "generic":
        # PURITY: the control arm gets retrieval and nothing else. No ftype, no
        # type-fit re-rank, no instruction, no failure-mode warning. Anything
        # typed leaking in here makes the ablation meaningless.
        positives = select_generic_positives(pool, seeds, budget,
                                             question=question)
        instruction = ""
        temperature = 0.0
        assert not instruction, "generic arm must carry no instruction"
    else:  # immediate -- blind re-sample of the original few-shot prompt
        positives = seeds[:budget]
        instruction = ""
        temperature = 0.8                      # must be >0 or greedy repeats

    # [AUDIT D13] TR = the model's output was TRUNCATED, not wrong. There is
    # nothing to cure: re-generate with the ordinary prompt and no instruction.
    if ftype is FailureType.TR:
        instruction = ""
        temperature = 0.8                       # must be > 0 or greedy repeats

    # [AUDIT D46/D61] NON-GENERATIVE cure. When the failed trace is
    # self-inconsistent -- it reports a value its own arithmetic contradicts --
    # the corrected number is recoverable without gold and without generating.
    #
    # [D61] It is a FALLBACK, not a pre-emption. Returning it and skipping the
    # generative retry looked like a saving and was a loss: repair's precision is
    # 24%, so most repaired problems got a still-wrong answer AND forfeited the
    # generative attempt. Measured on the completed runs, scoring the union of
    # attempts as the harness already does:
    #     generic, plain retry               39.5%  (161 recovered)
    #     repair PRE-EMPTS the retry         40.9%  (182)
    #     repair as a FALLBACK, both kept    42.3%  (202)
    _repaired_pass1 = None
    if (mode in ("typed", "typed+neg")
            and getattr(config, "TYPED_SYMBOLIC_REPAIR", False)
            and ftype.value in getattr(config, "SYMBOLIC_REPAIR_TYPES", ())):
        import symbolic_repair as _SR
        _fixed = _SR.repair((prev_trace or ""),
                            prev_answer)
        if _fixed and not getattr(config, "SYMBOLIC_REPAIR_AS_FALLBACK", True):
            _trace, _ans, _note = _fixed        # legacy pre-emption, for the ablation
            return _trace, {"n_resampled": 0, "accept_ok": True,
                            "accept_reason": _note, "prefill": "",
                            "symbolic_repair": True, "symbolic_repair_post": False,
                            "cure_type": ftype.value, "cure_note": _cure_note,
                            "n_unchanged_resamples": 0, "instruction_sent": "",
                            "n_exemplars": len(positives), "hash_appended": False}
        _repaired_pass1 = _fixed[0] if _fixed else None

    prompt, extra_system = build_retry_prompt(question, positives, instruction)

    # [AUDIT D43] Typed-only prefill: end the prompt mid-sentence so the model has
    # to complete it. Generic and immediate get "" and are byte-identical to before.
    prefill = ""
    if mode in ("typed", "typed+neg"):
        from diagnosis import prefill_for
        prefill = prefill_for(ftype, question)
        if prefill:
            prompt = prompt + " " + prefill

    traces, _ = gen._run_model(prompt, temperature=temperature,
                               max_new_tokens=max_new_tokens,
                               num_return_sequences=1, return_logprobs=False,
                               extra_system=extra_system,
                               problem=question)   # [AUDIT D2] integrity
    trace = traces[0]
    if prefill:
        # Stitch the prefill back on, or every downstream signal (n_steps, the
        # equation chain, G_score) would read a trace missing its opening words,
        # and the stored trace would not be what the model actually produced.
        head = prefill.strip()
        if head and not trace.lstrip().startswith(head[:24]):
            trace = head + (" " if not head.endswith("\n") else "") + trace.lstrip()

    # ARM PARITY: pass-1 goes through gen.generate(), which ends with
    # _ensure_hash_line() -- appending a canonical '#### N' when the model finished
    # in prose. Retries call _run_model directly and used to skip it, so pass-1
    # carried a '####' line 97.7% of the time versus 60.6% for retries. Retries
    # were being scored through a strictly weaker answer path than pass-1, which
    # depressed measured recovery in every arm. Apply the same normalisation here.
    trace, _hash_appended = G.ensure_hash_line_flagged(
        trace, G.extract_final_answer(trace))   # [AUDIT D34] measured, not inferred

    # [AUDIT D44] Type-conditioned acceptance test. GOLD-FREE -- it reads only the
    # question and the new trace, so it is legal on any split. If the cure did not
    # take, resample once and keep whichever candidate passes.
    n_resampled = 0
    accept_ok, accept_why = True, ""
    if mode in ("typed", "typed+neg"):
        from diagnosis import cure_took
        accept_ok, accept_why = cure_took(ftype, question, trace, prev_trace)
        if not accept_ok:
            # [AUDIT D50] TR failed by running out of room; resampling at the same
            # ceiling just truncates again.
            _mnt = max_new_tokens
            if ftype is FailureType.TR:
                _mnt = int(max_new_tokens * float(
                    getattr(config, "ACCEPT_TR_TOKEN_BOOST", 1.0)))
            alt, _ = gen._run_model(
                prompt, temperature=float(getattr(config, "ACCEPT_RESAMPLE_TEMP", 0.7)),
                max_new_tokens=_mnt, num_return_sequences=1,
                return_logprobs=False, extra_system=extra_system,
                problem=question)
            alt_trace = alt[0]
            if prefill:
                _h = prefill.strip()
                if _h and not alt_trace.lstrip().startswith(_h[:24]):
                    alt_trace = _h + (" " if not _h.endswith("\n") else "") + alt_trace.lstrip()
            alt_trace, _ = G.ensure_hash_line_flagged(
                alt_trace, G.extract_final_answer(alt_trace))
            n_resampled = 1
            alt_ok, _ = cure_took(ftype, question, alt_trace, prev_trace)
            # [AUDIT D45] A resample must pass the test AND not compute less than
            # the candidate it replaces. Otherwise "no false equation" is won by
            # stating no equations, which is what collapsed CE on the first re-run.
            if alt_ok and getattr(config, "ACCEPT_REJECT_DEGENERATE", True):
                from components import computational_steps as _cs
                if _cs(alt_trace) < _cs(trace):
                    alt_ok = False
                    accept_why = (accept_why or "") + " | resample rejected: computed less"
            if alt_ok:                       # the resample passed; the first did not
                trace, accept_ok, accept_why = alt_trace, True, ""
    elif mode == "generic" and getattr(config, "GENERIC_MATCH_RESAMPLE", False):
        # COMPUTE PARITY: generic has no test, so it resamples at the same RATE,
        # chosen at random, and keeps the second draw. Without this typed gets more
        # generations and a reviewer can attribute the win to compute rather than
        # to the diagnosis.
        import random as _rnd
        rate = float(getattr(config, "GENERIC_RESAMPLE_RATE", 0.0))
        if rate > 0 and _rnd.Random(hash(rec_id) & 0xFFFFFFFF).random() < rate:
            alt, _ = gen._run_model(
                prompt, temperature=float(getattr(config, "ACCEPT_RESAMPLE_TEMP", 0.7)),
                max_new_tokens=max_new_tokens, num_return_sequences=1,
                return_logprobs=False, extra_system=extra_system,
                problem=question)
            trace, _ = G.ensure_hash_line_flagged(alt[0], G.extract_final_answer(alt[0]))
            n_resampled = 1

    # [AUDIT D60] If the retry reproduced the failed answer, the cure did nothing.
    # Unlike the D44 acceptance test this needs no gold and is not a correlate of
    # failure -- an unchanged answer IS the failure, observed directly. A candidate
    # worth 0% is replaced, so the trade cannot lose.
    n_unchanged = 0
    if (mode in ("typed", "typed+neg")
            and getattr(config, "RETRY_RESAMPLE_IF_UNCHANGED", False)):
        prev_ans = prev_answer
        for _ in range(int(getattr(config, "UNCHANGED_MAX_RESAMPLES", 1))):
            cur = G.extract_final_answer(trace)
            if prev_ans is None or cur is None:
                break
            try:
                same = abs(float(cur) - float(prev_ans)) <= 1e-6 * max(1.0, abs(float(prev_ans)))
            except (TypeError, ValueError):
                same = str(cur) == str(prev_ans)
            if not same:
                break
            alt, _ = gen._run_model(
                prompt,
                temperature=float(getattr(config, "UNCHANGED_RESAMPLE_TEMP", 0.8)),
                max_new_tokens=max_new_tokens, num_return_sequences=1,
                return_logprobs=False, extra_system=extra_system,
                problem=question)
            alt_trace = alt[0]
            if prefill:
                _h = prefill.strip()
                if _h and not alt_trace.lstrip().startswith(_h[:24]):
                    alt_trace = _h + (" " if not _h.endswith("\n") else "") + alt_trace.lstrip()
            alt_trace, _ = G.ensure_hash_line_flagged(
                alt_trace, G.extract_final_answer(alt_trace))
            trace = alt_trace
            n_unchanged += 1

    # [AUDIT D46b] Repair the RETRY OUTPUT too, not only the failed pass-1 trace.
    # The retry can be self-inconsistent in its own right. Measured on the 738
    # completed retries: repairing pass-1 alone fixes 31 the generative cure
    # missed; repairing the retry output as well fixes 44. +6.0 points instead of
    # +4.2, for no extra generation.
    if (mode in ("typed", "typed+neg")
            and getattr(config, "TYPED_SYMBOLIC_REPAIR", False)
            and ftype.value in getattr(config, "SYMBOLIC_REPAIR_TYPES", ())):
        import symbolic_repair as _SR
        _r = _SR.repair(trace, G.extract_final_answer(trace))
        _post_repair = bool(_r)
        if _r:
            trace = _r[0]
    else:
        _post_repair = False

    # [AUDIT D61] Gold-free choice between the generated candidate and the
    # repaired pass-1: prefer whichever states no false equation, generation wins
    # ties. Repair only takes over when the generation contradicts itself.
    _used_repaired = False
    if _repaired_pass1 is not None:
        from extraction import extract_equations as _ee, trace_body as _tb
        if (any(not e.is_true for e in _ee(_tb(trace)))
                and not any(not e.is_true for e in _ee(_tb(_repaired_pass1)))):
            trace = _repaired_pass1
            _used_repaired = True

    
    # [AUDIT D45] Record what the acceptance test did. The first re-run logged
    # nothing, so there was no way to tell from results.jsonl whether the test had
    # fired at all -- the damage had to be inferred from equation counts.
    # NOTE: this dict is the WHOLE record of what the cure did, so every flag
    # must be set here. An earlier version wrote symbolic_repair_post above and
    # had it silently wiped by a later assignment -- the repair ran but was never
    # recorded.
    return trace, {"n_resampled": n_resampled, "accept_ok": bool(accept_ok),
                   "accept_reason": accept_why, "prefill": prefill,
                   "symbolic_repair": _used_repaired,
                   "symbolic_repair_post": _post_repair,
                   "cure_type": ftype.value, "cure_note": _cure_note,
                   "n_unchanged_resamples": n_unchanged,
                   # [AUDIT D54] the instruction ACTUALLY SENT, including the
                   # failure-mode warning appended in typed+neg.
                   "instruction_sent": instruction,
                   "n_exemplars": len(positives),
                   "hash_appended": bool(_hash_appended)}


def retry_once(gen, rec, pool, seeds, max_new_tokens, mode="typed", budget=8,
               negatives=None, n_neg=0):
    """Scored wrapper over apply_cure() -- the TRAIN path, where gold exists.

    [AUDIT D76] The cure used to live in this function, which meant the only way
    to run it was to hand over a record carrying gold_solution and gold_answer.
    Stage 2 and stage 3 have neither, so run_ctc_validate.py had reimplemented a
    SUBSET of it: exemplars and an instruction, but no UNCLASSIFIED reroute, no
    failure-mode naming, no prefill, no acceptance resample, no unchanged-answer
    resample and no symbolic repair. It was measuring a weaker pipeline than the
    one the 42.0% typed result came from, and would have understated CTC.

    Everything except the final score_trace is gold-free, so the cure now lives
    in apply_cure() and BOTH paths call it. This function adds only scoring.
    """
    trace, meta = apply_cure(
        gen, rec["question"], rec.get("trace"),
        (rec.get("signals") or {}).get("final_answer"),
        rec.get("diagnosis"), pool, seeds, max_new_tokens, mode=mode,
        budget=budget, negatives=negatives, n_neg=n_neg, rec_id=rec.get("id", ""))
    comps, label, diag, _ = score_trace(rec["question"], trace,
                                        rec["gold_solution"], rec["gold_answer"])
    gen_tokens = (len(gen.tokenizer.encode(trace)) if gen.tokenizer
                  else len(trace.split()))
    retry_once.last_accept = meta
    return (trace, comps, label, diag, gen_tokens, meta["n_exemplars"],
            meta["hash_appended"])


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
             retry_mode="typed", n_negatives=0, negatives_store=None,
             run_name=None, eval_offset=0,
             store_dir=None):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = _retrieval_fn(retrieval)
    # [AUDIT D38] store_dir lets the caller put the store OUTSIDE the repo clone.
    # It used to live at REPO/store_<run_name>, and the Kaggle notebook begins by
    # deleting and re-cloning REPO -- so re-running the notebook from the top
    # DESTROYED an in-progress run. Hours of generation were lost that way, with
    # only the last 10-minute checkpoint surviving on GitHub.
    from classification import STORE_ROOT as _SR
    if store_dir:
        _root = Path(store_dir)
    else:
        _root = (_SR.parent / f"store_{run_name}") if run_name else _SR
    store = TraceStore(root=_root)
    store_root = store.root
    print(f"  outputs -> {store_root}")
    results_path = store_root / config.RESULTS_FILE
    attempts_path = store_root / "attempts.jsonl"

    # [AUDIT D35] Stamp the store with the CODE VERSION *immediately*, not at the
    # end. A store is restored and resumed long before it is finished, and a
    # resume has no other way to tell which code produced the data it is about to
    # continue. Without this stamp a pre-audit `store_<run>_inprogress` on the
    # results branch was silently restored into an audit-2 run: 1,500 problems
    # looked "already done", ZERO generations happened, and the harness reported
    # the old run's numbers (old taxonomy, ST=7, no AL/WP/TR) as if they were new.
    _snap_early = {k: getattr(config, k) for k in dir(config)
                   if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    (store_root / config.CONFIG_SNAPSHOT_FILE).write_text(json.dumps(_snap_early, indent=2))

    problems = eval_problems[:n_problems]
    print("\n" + "#" * 70)
    print(f"#  FULL RUN | strategy {strategy_id}={STRATEGY_NAMES[strategy_id]} "
          f"| n={len(problems)} | backend={backend_name()} | retrieval={retrieval}")
    _ARM_DESC = {
        "typed": "typed positives + typed instruction [FADE]",
        "typed+neg": "typed positives + instruction + failure-mode warning [FADE+]",
        "generic": "untyped positives, no instruction [generic ablation]",
        "immediate": "blind re-sample, no diagnosis [immediate null]",
    }
    print(f"#  RETRY MODE = {retry_mode.upper()}  ({_ARM_DESC.get(retry_mode, '?')})")

    negbank = None
    if retry_mode == "typed+neg":
        if n_negatives <= 0:
            raise SystemExit("--retry-mode typed+neg needs --negatives k (try 2)")
        from negatives import NegativeBank
        src = negatives_store or str(store.root)
        negbank = NegativeBank.from_store(src)
        have = sum(len(v) for v in negbank.by_type.values())
        print(f"#  NEGATIVE BANK = {have} wrong traces from {src}, "
              f"{n_negatives} warning(s) per retry prompt")
        if have < 50:
            print("#  !! the bank is nearly empty -- point --negatives-store at a")
            print("#     FINISHED train store, or this arm is identical to `typed`.")
    elif n_negatives:
        print(f"#  note: --negatives {n_negatives} ignored "
              f"(only --retry-mode typed+neg uses it)")
    print(f"#  outputs -> store_{run_name if run_name else '(default)'}/")
    print("#" * 70)

    # ---- resume: reconstruct pool + queues from store/ -------------------
    pool = store.exemplars()          # gold-stripped view: exemplars never carry gold
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
    dx_native_hash = 0   # [AUDIT D34] measured at generation time
    batch_rows = []          # rows for the rolling batch report
    t_start = time.time()

    # ---------------- PASS 1 ----------------
    for i, prob in enumerate(problems, 1):
        # [AUDIT D22] ABSOLUTE eval index, not the position within this chunk.
        # The id used to be f"p1_{i:05d}" where i restarted at 1 for every run, so
        # two chunks launched with different --eval-offset produced the SAME ids
        # for DIFFERENT questions. Merging their results.jsonl files then silently
        # interleaved unrelated problems -- which is exactly the failure recorded
        # in the project report ("merge swept in a different arm: 1,319 rows, same
        # positional ids, different questions"). Chunked runs are only safe with
        # a globally unique id.
        item_id = f"p1_{eval_offset + i:05d}"
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
                  diagnosis_reason=diag.reason if diag else None,
                  diagnosis_confidence=diag.confidence if diag else None)
        log_detail(_detail_record(
            phase="pass1", id=item_id, eval_index=eval_offset + i - 1, question=q,
            gold_solution=gold_sol, gold_answer=gold_ans,
            category=cat_info, prompt_tokens=p_tok, gen_seconds=round(dt, 2),
            gen_tokens=res["num_tokens"], trace=trace, trace_scored=scored,
            hash_appended=res.get("hash_appended"),   # [AUDIT D34]
            signals=comps.signals(), label=label.value,
            diagnosis=diag.ftype.value if diag else None,
            diagnosis_reason=diag.reason if diag else None,
            diagnosis_confidence=diag.confidence if diag else None))
        log_attempt(id=item_id, pass_=1, label=label.value,
                    correct=comps.correct, diagnosis=diag.ftype.value if diag else None)

        rec = {"id": item_id, "question": q, "trace": trace,
               "gold_solution": gold_sol, "gold_answer": gold_ans,
               "label": label.value, "diagnosis": diag.ftype.value if diag else None}
        if label is Label.GOOD:
            pool.append(sanitize_exemplar({**rec, "signals": comps.signals(),
                                           "complexity": comps.n_checkpoints}))
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
        if res.get("hash_appended") is False:
            dx_native_hash += 1
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

    # ---- [AUDIT D29] REBUILD AGGREGATES FROM results.jsonl, not from this
    # session's loop. diag_dist / per_cat / dx_* are only incremented inside the
    # pass-1 loop, which RESUME SKIPS. So a resumed run -- i.e. every chunk after
    # the first, and every restart after a Kaggle timeout -- wrote a summary with
    # an EMPTY diagnosis distribution, 0% '####' coverage and no per-category
    # table, and then overwrote the good summary from the first session with it.
    # Verified by simulation: run twice, and the second report shows
    # "pass-1 diagnoses : {}" and "traces with '####' : 0/8 (0%)".
    if results_path.exists():
        _dd, _pc = Counter(), defaultdict(lambda: [0, 0])
        _hash = _highE = _goldwrong = _native = 0
        _seen_ids = set()
        with open(results_path) as _f:
            for _ln in _f:
                try:
                    _r = json.loads(_ln)
                except Exception:
                    continue
                if _r.get("phase") != "pass1" or _r.get("id") in _seen_ids:
                    continue
                _seen_ids.add(_r.get("id"))
                _sig = _r.get("signals") or {}
                if _r.get("diagnosis"):
                    _dd[_r["diagnosis"]] += 1
                _cat = (_r.get("category") or {}).get("category_type", "unknown")
                _pc[_cat][1] += 1
                _pc[_cat][0] += int(bool(_sig.get("correct")))
                if "####" in (_r.get("trace") or ""):
                    _hash += 1
                if _r.get("hash_appended") is False:
                    _native += 1
                if (_sig.get("E") or 0) >= 0.8 and not _sig.get("correct"):
                    _highE += 1
                if not _sig.get("correct"):
                    from extraction import extract_values as _ev
                    _tv = _ev(_r.get("trace_scored") or _r.get("trace") or "")
                    if any(abs(_v - (_r.get("gold_answer") or 0)) < 1e-6 for _v in _tv):
                        _goldwrong += 1
        if _seen_ids:
            diag_dist = _dd
            per_cat = _pc
            dx_has_hash, dx_highE_wrong, dx_goldpresent_wrong = _hash, _highE, _goldwrong
            dx_native_hash = _native
            print(f"  aggregates rebuilt from {len(_seen_ids)} logged pass-1 rows "
                  f"(so a resumed chunk reports the FULL picture, not just this session)")

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
    # Write a PRELIMINARY summary now (pass-1 done) so that if the session is cut
    # off during the long retry phase, at least the pass-1 result survives. It is
    # overwritten with the full summary at the end.
    try:
        (store_root / config.SUMMARY_FILE).write_text(json.dumps({
            "status": "pass1_complete_retries_pending",
            "n_problems": len(problems),
            "pass1_correct": pass1_correct,
            "pass1_accuracy": pass1_acc,
            "final_correct": pass1_correct, "final_accuracy": pass1_acc,
            "recovered_by_retry": 0, "final_pool_size": len(pool),
            "iterations": [], "model": MODEL_ID, "split": SPLIT_TAG,
            "retry_mode": retry_mode,
            "pass1_diagnosis_distribution": dict(diag_dist),
            "per_category_accuracy": {c: {"correct": v[0], "total": v[1],
                "acc": v[0]/v[1] if v[1] else 0.0} for c, v in per_cat.items()},
        }, indent=2))
        print(f"  wrote preliminary summary (pass-1: {pass1_acc:.1%}) "
              f"-- retries starting; final summary will overwrite it")
    except Exception as _e:
        print(f"  (could not write preliminary summary: {_e})")

    # ---- [AUDIT D20] retry-queue economics -------------------------------
    # MEDIUM_CORRECT consumed 149 of 1,367 retry generations (11%) and recovered
    # 0 -- IMPOSSIBLE by construction, because `newly_correct` requires the id NOT
    # to be in `solved` and a correct trace already is. It yielded 8 pool entries.
    # Worse: retrying a trace the model got RIGHT made it WRONG 114/149 times
    # (76.5%), which is independent evidence that the retry path was a regression.
    if not config.RETRY_POLISH_MEDIUM_CORRECT:
        _dropped = [r for r in medium if r.get("label") == Label.MEDIUM_CORRECT.value]
        medium[:] = [r for r in medium if r.get("label") != Label.MEDIUM_CORRECT.value]
        if _dropped:
            print(f"  [D20] skipping {len(_dropped)} MEDIUM_CORRECT polish retries "
                  f"(0 possible recoveries); budget goes to wrong answers. "
                  f"Set config.RETRY_POLISH_MEDIUM_CORRECT = True to restore.")

    # ---- [AUDIT D3] retry resume ------------------------------------------
    # Pass 1 resumed, but the retry loop did not: a disconnect mid-retry
    # re-generated every retry from scratch. Replay what is already logged.
    logged_retries: dict[tuple, dict] = {}
    if results_path.exists():
        with open(results_path) as _f:
            for _ln in _f:
                try:
                    _r = json.loads(_ln)
                except Exception:
                    continue
                if _r.get("phase") == "retry":
                    logged_retries[(_r.get("id"), _r.get("iter"),
                                    _r.get("origin"))] = _r
    if logged_retries:
        print(f"  RESUME: {len(logged_retries)} retries already logged -- "
              f"replaying them without the GPU")

    iter_reports = []
    pool_entries_total = 0
    for it in range(1, iterations + 1):
        report = {"iter": it}
        for origin, queue in (("MEDIUM", medium), ("BAD", bad)):
            still, trans, recov_by_diag = [], Counter(), Counter()
            newly_correct = 0
            pool_entries = 0
            for rec in queue:
                prev = logged_retries.get((rec["id"], it, origin))
                if prev is not None:
                    # [AUDIT D3] replay: no generation, identical bookkeeping
                    _sig = prev.get("signals") or {}
                    trans[prev.get("label", "?")] += 1
                    if prev.get("newly_correct") and rec["id"] not in solved:
                        newly_correct += 1
                        solved.add(rec["id"])
                        if rec.get("diagnosis"):
                            recov_by_diag[rec["diagnosis"]] += 1
                    if prev.get("became_good"):
                        pool_entries += 1
                        pool.append(sanitize_exemplar({
                            **rec, "trace": prev.get("trace", rec["trace"]),
                            "label": prev.get("label"), "signals": _sig,
                            "complexity": _sig.get("n_checkpoints")}))
                    else:
                        still.append({**rec, "trace": prev.get("trace", rec["trace"]),
                                      "label": prev.get("label"),
                                      "diagnosis": prev.get("diagnosis")})
                    continue

                trace, comps, label, diag, gtok, npos, _happ = retry_once(
                    gen, rec, pool, seeds, max_new_tokens,
                    mode=retry_mode, budget=G.EXEMPLAR_BUDGET or G.N_SHOTS,
                    negatives=negbank, n_neg=n_negatives)
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
                # [AUDIT D54] Take the instruction from what retry_once ACTUALLY
                # sent. Re-deriving it from prev_diagnosis was wrong twice over:
                #   * a D52 reroute (cascade said UNCLASSIFIED, cure was CE) logged
                #     UNCLASSIFIED's empty instruction while CE's was really sent --
                #     which is why rerouted rows showed a cure_type of CE and no
                #     cure text at all;
                #   * typed+neg appends a failure-mode warning to the instruction,
                #     and none of it was ever recorded.
                _la = getattr(retry_once, "last_accept", None) or {}
                _logged_instr = _la.get("instruction_sent")
                if _logged_instr is None:            # older code path
                    _logged_instr = (TYPED_INSTRUCTION.get(
                        FailureType(prev_diag) if prev_diag else FailureType.UNCLASSIFIED, "")
                        if retry_mode in ("typed", "typed+neg") else "")
                # [AUDIT SELF-REVIEW] STORE BEFORE LOG. The retry path logged to
                # results.jsonl first and only then called store.add, so a crash
                # between the two left a `became_good` row in results.jsonl with
                # no corresponding pool entry on disk -- and the D3 replay would
                # then rebuild the in-memory pool from a record the store never
                # had. Writing the store first makes "logged => stored" an
                # invariant the resume can rely on. store.add is idempotent by id.
                if label is Label.GOOD:
                    store.add(rec["id"], rec["question"], trace, rec["gold_solution"],
                              rec["gold_answer"], label, comps,
                              diagnosis=diag.ftype.value if diag else None,
                              diagnosis_reason=diag.reason if diag else None,
                              diagnosis_confidence=diag.confidence if diag else None)
                log_detail(_detail_record(
                    phase="retry", iter=it, origin=origin, id=rec["id"],
                    retry_mode=retry_mode,
                    **{k: v for k, v in
                       (getattr(retry_once, "last_accept", None) or {}).items()
                       if k != "instruction_sent"},
                    question=rec["question"], gold_answer=rec["gold_answer"],
                    prev_label=rec["label"], prev_diagnosis=prev_diag,
                    typed_positives=npos, instruction=_logged_instr,
                    hash_appended=_happ,
                    gen_tokens=gtok, trace=trace, signals=comps.signals(),
                    label=label.value, diagnosis=diag.ftype.value if diag else None,
                    became_good=became_good, newly_correct=is_new))
                log_attempt(id=rec["id"], iter=it, origin=origin, label=label.value,
                            correct=comps.correct, prev_diagnosis=prev_diag)
                new = {**rec, "trace": trace, "label": label.value,
                       "diagnosis": diag.ftype.value if diag else None}
                if became_good:
                    pool_entries += 1
                    # (store.add already happened above, before log_detail)
                    pool.append(sanitize_exemplar({**new, "signals": comps.signals(),
                                                   "complexity": comps.n_checkpoints}))
                else:
                    still.append(new)
            pool_entries_total += pool_entries
            report[origin] = {"retried": len(queue), "transitions": dict(trans),
                              "newly_correct": newly_correct,
                              # [AUDIT D20] polish retries can only ever be judged
                              # on this, never on newly_correct
                              "pool_entries": pool_entries,
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
        "retry_mode": retry_mode, "n_negatives": n_negatives,
        "exemplar_budget": G.EXEMPLAR_BUDGET,
        "n_problems": len(problems),
        "eval_offset": eval_offset,
        "eval_range": [eval_offset + 1, eval_offset + len(problems)],
        "pass1_correct": pass1_correct, "pass1_accuracy": round(pass1_acc, 4),
        "final_correct": final_correct, "final_accuracy": round(final_acc, 4),
        "recovered_by_retry": final_correct - pass1_correct,
        "pool_entries_from_retry": pool_entries_total,
        "final_pool_size": len(pool),
        "pass1_diagnosis_distribution": dict(diag_dist),
        "extraction_diagnostics": {
            "traces_with_hash": dx_has_hash,
            "hash_natively_emitted": dx_native_hash,
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
    ap.add_argument("--negatives", type=int, default=0,
                    help="k failure-mode warnings per retry prompt (needs "
                         "--retry-mode typed+neg). They go in the system message, "
                         "never in the exemplar block.")
    ap.add_argument("--negatives-store",
                    help="store to draw wrong traces from (default: --store-dir, "
                         "i.e. this run's own accumulating failures)")
    ap.add_argument("--retry-mode",
                    choices=["typed", "typed+neg", "generic", "immediate"],
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
    ap.add_argument("--build-artifacts-only", action="store_true",
                    help="prime the shared eval order (artifacts/) WITHOUT "
                         "loading the model, then exit. Run once, save "
                         "artifacts/, and reuse it for every parallel arm so all "
                         "arms evaluate identical problems.")
    ap.add_argument("--run-name", default=None,
                    help="write outputs to store_<run-name>/ instead of store/. "
                         "Use distinct names to run arms in PARALLEL notebooks "
                         "(e.g. --run-name typed vs --run-name immediate) without "
                         "clobbering each other. They share artifacts/ (same "
                         "eval order), so the comparison stays on identical "
                         "problems.")
    ap.add_argument("--store-dir", default=None,
                    help="[AUDIT D38] absolute path for this run's store. Put it "
                         "OUTSIDE any directory you delete between sessions "
                         "(the Kaggle notebook re-clones the repo every time).")
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

    if args.build_artifacts_only:
        # Prime the shared eval order WITHOUT loading the model. Run this once,
        # save artifacts/ as a Dataset, then every parallel arm restores it so
        # all arms evaluate the identical problems in the identical order.
        print(f"\nartifacts/ primed ({len(eval_problems)} eval problems, "
              f"{sum(len(v) for v in manual.values())} seeds). "
              f"No model loaded. Save artifacts/ and reuse it for every arm.")
        return

    model, tok = load_model(token)
    gen = ExemplarGenerator(model=model, tokenizer=tok)

    if args.eval_offset:
        eval_problems = eval_problems[args.eval_offset:]
        print(f"  eval-offset: skipping first {args.eval_offset} eval problems "
              f"(ids will start at p1_{args.eval_offset + 1:05d})")
    if args.demo:
        run_demo(gen, manual, eval_problems, args.strategy, args.demo,
                 args.max_new_tokens, retrieval=args.retrieval)
    else:
        run_full(gen, manual, eval_problems, args.strategy, args.n_problems,
                 args.max_new_tokens, iterations=args.iterations,
                 show_every=args.show_every, retrieval=args.retrieval,
                 retry_mode=args.retry_mode, n_negatives=args.negatives,
                 negatives_store=args.negatives_store, run_name=args.run_name,
                 eval_offset=args.eval_offset,   # [AUDIT D22] globally unique ids
                 store_dir=args.store_dir)       # [AUDIT D38] survives a re-clone


if __name__ == "__main__":
    main()