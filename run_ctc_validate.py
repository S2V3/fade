"""
run_ctc_validate.py -- STAGE 2. Does CTC work on questions it has never seen?

THE RUN
-------
400 fresh TRAIN questions (offset 1500 -- the pool-building run stopped there).

ONE QUESTION AT A TIME -- attempt, judge, cure, next:

  gen 1     model solves THIS question                   -> trace, answer
  cpu       CTC reads (question, trace), GOLD-FREE:
              p_wrong        is this trace wrong?
              predicted_type which of the eight failures?
  gen 2     if flagged: the matched cure, solved again, right now
  score     gold is opened LAST, and changes nothing above it

Then the next question. No step reads any other problem, which is what makes
stage 3 the same code with the scoring removed.

GOLD IS REQUIRED HERE, AND THAT IS NOT A CONTRADICTION
------------------------------------------------------
There is no way to say whether CTC was right without knowing whether the trace was
actually wrong and what actually went wrong with it. Both labels come from gold:
`correct` from the answer, `true_type` from the cascade's diagnosis. So this stage
opens gold and must.

The rule is about ORDER, not abstinence. Gold is read in step 4, after the cure has
already been chosen and run, and nothing it produces is fed back. Drop step 4
entirely (--no-score) and steps 1-3 emit byte-identical traces -- that is the
property that makes the same script legal on test, and the assertion below the loop
checks it rather than trusting it.

This is the full pipeline rehearsed on train, where gold exists to score it. Stage 3
is the identical pipeline on test with the scoring removed.

WHY THE RETRY IS INCLUDED BY DEFAULT
------------------------------------
Detection accuracy alone does not tell you whether the pipeline helps. Two rates
decide that, and they pull in opposite directions:

    a TRUE positive  gains ~16.1%  (measured: 172 of 1068 retries recovered)
    a FALSE positive costs ~76.5%  (pre-audit run: retrying an already-correct
                                    trace made it WRONG 114 of 149 times)

The 76.5% is a PRIOR, not a fresh measurement -- [AUDIT D20] removed those retries
from the current run because they could not possibly recover anything. So this stage
re-measures it: every flagged trace is retried, including the ones that were already
right, and the damage rate comes out of the data instead of out of a footnote.

Skip it with --no-retry-flagged to get detection numbers only (~1.9 GPU-h instead of
~3.1), but then the threshold for stage 3 has to be chosen on a prior.

WHAT STAGE 1 SAID (CTC v2, 2,568 traces, GroupKFold by question)
----------------------------------------------------------------
    detect  AUROC 0.756
    type    acc 0.627 vs majority 0.216   lift +0.411
    recall  TR 98.9%  NR 95.3%  AL 87.1%  WP 69.2%  ST 62.8%
            UNCL 51.7%  CE 42.5%  SM 38.2%

BUT THOSE ARE AVERAGES OVER TWO DIFFERENT DISTRIBUTIONS, AND THIS STAGE ONLY
MEETS ONE OF THEM. A question contributes both a pass-1 trace and a retry trace.
Here CTC only ever reads a FIRST attempt. Sliced out of fold by phase:

    slice    n      wrong%   detect AUROC   type acc
    pass-1   1500   71.2%    0.776          0.598
    retry    1068   81.5%    0.683          0.679
    both     2568   75.5%    0.751          0.635

So expect detection ABOVE the headline and typing BELOW it. Compare against the
pass-1 row, not the headline.

Note the detector's accuracy LOSES to the majority baseline while its AUROC is
0.78. That is not a contradiction: 71% of first attempts are wrong, so "always say
wrong" is a strong accuracy baseline, and the detector's value is in RANKING. Two
scorecards are therefore printed separately at the end: whether the JUDGE was right
(precision, recall, AUROC, type accuracy) and whether the MODEL got more QUESTIONS
right (pass 1 vs final, McNemar). They are different measurements and can move in
opposite directions.

HOW THE FLAG THRESHOLD IS CHOSEN
--------------------------------
Not by a fixed 0.5. A flag pays only when precision beats damage/(recovery+damage)
= 82.6% on the priors, and on stage-1 pass-1 traces the budget curve is:

    budget   thr    precision   expected net over 1068
      10%   0.90       97.3%          +20.4
      20%   0.83       95.7%          +36.3
      40%   0.68       90.7%          +44.7   <- default
      65%   ~0.50      ~83%            ~+5    <- what a fixed 0.5 would have done
     100%   0.13       71.2%         -158.5

A quantile needs the whole distribution, which a deployed system does not have --
question 1 must be decided before question 2 is attempted. So ctc_train2.py measures
the budget-to-cut mapping out of fold and ships it inside the model, and 40% becomes
"p_wrong >= 0.683", a rule ONE question can be judged against.

The loop is therefore: attempt one question, analyse that trace, cure it there and
then if it is flagged, move to the next. Nothing about one problem's decision depends
on any other problem, so stage 3 runs identically with the scoring removed.

THE TYPE CONFIDENCE GATE
------------------------
Type accuracy is 0.598 on pass-1 traces, but that mean hides a usable signal:

    gate   coverage   accuracy
    0.50      68.4%      66.6%   <- default
    0.70      33.3%      79.2%
    0.90      13.3%      91.5%

Below the gate the predicted type is not trusted and the GENERIC cure is used --
a measured 15.1% recovery instead of a near-coin-flip typed one. --type-conf-gate 0
disables it, which is the ablation the paper needs.

GOLD RULE
---------
Generation, detection and cure selection read only the question and the model's own
trace. Gold is opened afterwards, to score. --no-score drops it entirely, which is
exactly how stage 3 runs on test.

USAGE
    python run_ctc_validate.py --ctc stores/ctc2/ctc2.joblib \
        --pool-store stores/store_typed_train_audit2 \
        --store-dir stores/store_ctcval_d65 --n-problems 400 --eval-offset 1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import config
import generation as G
import kaggle_run as K
from categorizer import QuestionCategorizer
from cure_bank import describe

# [AUDIT D76] the retry here must be the SAME cure the typed arm runs. If this
# kaggle_run.py predates the apply_cure() extraction, stage 2 would silently fall
# back to a weaker retry and report a number that does not describe the pipeline.
if not hasattr(K, "apply_cure"):
    raise SystemExit(
        "  kaggle_run.py has no apply_cure() -- it is the pre-D76 version.\n"
        "  Stage 2 would run a retry MISSING the UNCLASSIFIED reroute, the\n"
        "  failure-mode naming, the prefill, both resamples and symbolic repair,\n"
        "  i.e. a weaker pipeline than the one the typed result came from.\n"
        "  Push the updated kaggle_run.py before spending GPU.")

RECOVERY_PRIOR = 0.161
DAMAGE_PRIOR = 0.765


def _close(a, b, tol=1e-6):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def main():
    ap = argparse.ArgumentParser(description="CTC stage 2 -- validate on fresh train questions")
    ap.add_argument("--ctc", required=True, help="ctc.joblib from stage 1")
    ap.add_argument("--pool-store", required=True, help="train store holding the exemplar pool")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--n-problems", type=int, default=400)
    ap.add_argument("--eval-offset", type=int, default=1500)
    # [AUDIT D81] STAGE 3. --split test draws the QUESTIONS from the test split
    # while the SEEDS and the POOL stay exactly as the train run built them --
    # they are the model's own successes and are part of the system, not part of
    # the evaluation. Only the questions must be unseen.
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="where the QUESTIONS come from (default train). "
                         "Seeds and pool always come from the train run.")
    ap.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--retrieval", choices=["knn", "3stage"], default="3stage",
                    help="retrieval arm, must match the train run (default 3stage)")
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    # [AUDIT D67] a FIXED 0.5 was the old default and it is close to break-even.
    # Measured on the stage-1 pass-1 traces (prior recovery 16.1%, prior damage
    # 76.5%, so a flag pays only above 82.6% precision):
    #
    #     budget   thr    precision   expected net over 1068
    #       10%   0.90       97.3%          +20.4
    #       20%   0.83       95.7%          +36.3
    #       40%   0.68       90.7%          +44.7   <- optimum
    #       50%   0.60       88.0%          +37.4
    #       65%   ~0.50      ~83%            ~+5    <- the OLD default
    #       70%   0.48       82.1%           -5.0
    #      100%   0.13       71.2%         -158.5
    #
    # Flagging by BUDGET rather than by a fixed probability also survives a shift
    # in calibration between splits, which a fixed cut does not, and it is the
    # form stage 3 needs: "retry the worst 40%" is a GPU budget, not a guess.
    ap.add_argument("--flag-budget", type=float, default=0.40,
                    help="flag this fraction, worst p_wrong first (default 0.40, "
                         "the measured optimum). Set 0 to use --threshold instead.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed p_wrong cut. Overrides --flag-budget when given. "
                         "Swept afterwards either way.")
    # [AUDIT D68] type accuracy is 0.598 on pass-1 traces overall, but rises with
    # confidence -- measured out-of-fold on stage-1 pass-1 wrong traces:
    #
    #     gate   coverage   accuracy
    #     0.00     100.0%      59.8%
    #     0.50      68.4%      66.6%   <- default
    #     0.70      33.3%      79.2%
    #     0.90      13.3%      91.5%
    #
    # Below the gate the type is not trusted and the GENERIC cure is used, which
    # is a measured 15.1% recovery rather than a coin-flip typed cure. This also
    # gives the paper its cleanest ablation: gate at 0 to disable it.
    ap.add_argument("--type-conf-gate", type=float, default=0.50,
                    help="below this type_confidence, use the generic cure "
                         "(default 0.50). 0 disables the gate.")
    ap.add_argument("--negatives", dest="negatives", action="store_true", default=True,
                    help="attach the failure-mode warning, as the typed arm does")
    ap.add_argument("--no-negatives", dest="negatives", action="store_false")
    ap.add_argument("--n-neg", type=int, default=2)
    ap.add_argument("--retry-flagged", dest="retry_flagged", action="store_true", default=True)
    ap.add_argument("--no-retry-flagged", dest="retry_flagged", action="store_false")
    ap.add_argument("--no-score", action="store_true", help="never read gold (stage-3 mode)")
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--show-every", type=int, default=25)
    a = ap.parse_args()

    K.MODEL_ID = a.model
    K.SPLIT_TAG = a.split
    if a.split == "test" and not a.no_score:
        print("\n  NOTE: --split test WITH scoring. Legal -- gold is opened only")
        print("  after every decision -- but stage 3 is normally --no-score, and")
        print("  the two must agree. Run --no-score first, then this, and diff the")
        print("  traces: they must be identical or the gold rule was broken.")
    G.set_exemplar_budget(a.exemplar_budget)
    K.version_banner()
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)

    # [AUDIT D65] load v1 OR v2. joblib pickles by module reference, so the
    # defining module must be importable before the unpickle -- whichever one
    # wrote the file. Sniffing the class afterwards keeps this script usable
    # against an old checkpoint without editing it.
    import importlib
    for _m in ("ctc_train", "ctc_train2"):
        try:
            importlib.import_module(_m)
        except Exception:
            pass
    import joblib
    ctc_model = joblib.load(a.ctc)
    ctc_version = 2 if type(ctc_model).__name__ == "CTC2" else 1
    for _need in ("predict", "report"):
        if not hasattr(ctc_model, _need):
            raise SystemExit(f"  {a.ctc} is not a CTC model (no .{_need})")
    rep = getattr(ctc_model, "report", None) or {}
    # [AUDIT D69] v1 called it cv_accuracy, v2 calls it accuracy. Read whichever
    # is present instead of printing nan and looking like a failed load.
    _t = rep.get("type", {})
    _tacc = _t.get("accuracy", _t.get("cv_accuracy"))
    print(f"\n  CTC v{ctc_version} loaded: detect AUROC "
          f"{rep.get('detect',{}).get('auroc')} | type acc {_tacc} "
          f"(majority {_t.get('majority')})")
    if ctc_version == 1:
        print("  !! v1 model. v2 (ctc2.joblib) is stronger and its evaluation is\n"
              "     leakage-free -- prefer it unless you are comparing the two.")

    # [AUDIT D70] stage 1 reports ONE number over pass-1 and retry traces mixed
    # together, but this stage only ever shows CTC a FIRST attempt. Measured
    # out-of-fold on the stage-1 data, sliced by the phase of the trace:
    #
    #     slice    n      wrong%   detect AUROC   type acc
    #     pass-1   1500   71.2%    0.776          0.598
    #     retry    1068   81.5%    0.683          0.679
    #     both     2568   75.5%    0.751          0.635
    #
    # Expect detection ABOVE the 0.756 headline and typing BELOW the 0.627
    # headline. Neither is a bug -- the mixed average is not what this stage
    # measures. (Training on pass-1 rows alone was tried: detect +0.006,
    # 95% CI [-0.004, +0.016], and type -0.008. Not worth halving the data.)
    print("  expected on pass-1 traces: detect ~0.78, type ~0.60")

    from classification import TraceStore
    pool = TraceStore(root=Path(a.pool_store)).exemplars()
    for r in pool:
        assert "gold_solution" not in r and "gold_answer" not in r, "GOLD LEAK in pool"
    print(f"  pool: {len(pool)} gold-stripped exemplars")

    # ---- eval order, WITHOUT re-selecting seeds ---------------------------
    # The exemplar source for this stage is the POOL -- the 334 GOOD traces the
    # model produced for itself. Seeds are not re-chosen here; the train run's
    # seeds are reused verbatim, and only ever top up a prompt when retrieval
    # cannot fill the budget from the pool.
    #
    # build_seeds_and_eval_order() short-circuits to its cached artifact when one
    # exists, so calling it with the ORIGINAL load size returns the train run's
    # seeds and eval ordering with NO selection pass. (An earlier version of this
    # file deleted that cache when the eval list was too short, which forced a
    # fresh selection -- exactly what we do not want.)
    need = a.eval_offset + a.n_problems
    # Seeds ALWAYS come from train, on both splits. They were chosen once by the
    # train run and are part of the frozen system; re-selecting them on test
    # would be fitting the prompt to the evaluation.
    train = K._load_split("train", a.seed_pool)
    manual, eval_problems = K.build_seeds_and_eval_order(train, a.seed_pool)
    seeds = manual.get(2, [])
    seed_qs = {ex.get("question") for v in manual.values() for ex in v}
    n_cached = len(eval_problems)

    if a.split == "test":
        # [AUDIT D81] The QUESTIONS are the test split, in dataset order. No seed
        # selection, no reordering, no exclusion -- there is nothing to exclude,
        # since no test question ever entered the pool or the seeds.
        eval_problems = K._load_split("test", None)
        n_cached = len(eval_problems)
        print(f"\n  TEST SPLIT: {len(eval_problems)} questions, dataset order")
        _sq = {p_["question"] for p_ in eval_problems} & seed_qs
        if _sq:
            raise SystemExit(f"  {len(_sq)} SEED questions appear in test -- stop")
        if a.n_problems > len(eval_problems) - a.eval_offset:
            a.n_problems = len(eval_problems) - a.eval_offset
            need = a.eval_offset + a.n_problems
            print(f"  n-problems clipped to {a.n_problems} (the split's size)")

    # Seed selection removes the chosen problems from the eval list (2000 - 268 =
    # 1732), so [1500:1900] would silently return 232, not 400. EXTEND the tail
    # with train problems beyond the seed pool -- an append, never a re-selection.
    if n_cached < need and a.split == "train":
        extra = K._load_split("train", a.seed_pool + (need - n_cached) + 800)
        have = {p_["question"] for p_ in eval_problems} | seed_qs
        for p_ in extra[a.seed_pool:]:
            if p_["question"] in have:
                continue
            eval_problems.append(p_); have.add(p_["question"])
            if len(eval_problems) >= need:
                break
        print(f"  eval order extended {n_cached} -> {len(eval_problems)} "
              f"(appended from train[{a.seed_pool}:], no seeds re-selected)")

    problems = eval_problems[a.eval_offset:need]
    print(f"\n  seeds        {len(seed_qs)} reused from the train run (NOT re-selected)")
    print(f"  eval order   {len(eval_problems)} problems")
    print(f"  slice        [{a.eval_offset}:{need}] -> {len(problems)} problems")
    if len(problems) < a.n_problems:
        raise SystemExit(
            f"  ONLY {len(problems)} PROBLEMS AVAILABLE, {a.n_problems} requested.\n"
            "  Refusing to run a short experiment that would silently report a\n"
            "  smaller n. Lower --n-problems, or raise the load margin above.")

    # ---- are these really unseen? -----------------------------------------
    # eval_problems is a stable ordering, and the train run consumed [0:1500].
    # Verify the questions we are about to use do NOT appear in the pool store's
    # pass-1 rows. If they do, the ordering shifted and this is not a held-out set.
    seen = set()
    _rp = Path(a.pool_store) / "results.jsonl"
    if _rp.exists():
        for ln in open(_rp):
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("phase") == "pass1" and r.get("question"):
                seen.add(r["question"])
    # On test this is a formality -- the pool was built from train -- but a
    # formality that costs nothing and would catch a mis-set --split.
    overlap = sum(1 for p_ in problems if p_["question"] in seen)
    print(f"  overlap with the {len(seen)} questions the train run already saw: {overlap}")
    if overlap:
        raise SystemExit(
            f"  {overlap} of these {len(problems)} problems were ALREADY SOLVED in\n"
            "  the train run. They are not held out, and CTC would be scored on\n"
            "  its own training distribution. Check --eval-offset.")

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)
    cat = QuestionCategorizer()
    knn = K._retrieval_fn(a.retrieval)

    # ---- PROVENANCE CHECK -------------------------------------------------
    # generation._build_prompt draws exemplars from the POOL via knn_fn and only
    # tops up from the seeds when the pool returns too few. Omit knn_fn and it
    # silently takes EVERY exemplar from the seeds -- the 334-entry pool is passed
    # in and never read. That is [AUDIT D15] all over again ("retrieval never
    # ran"), and it would invalidate the whole claim: the point is that the model's
    # OWN successes become the exemplars, not that hand-written seeds do.
    _pq = {r.get("question") for r in pool}
    _probe = G._build_prompt(2, problems[0]["question"], seeds, pool,
                             cat.categorize(problems[0]["question"]), knn)[0]
    _from_pool = sum(1 for q_ in _pq if q_ and q_[:60] in _probe)
    print(f"\n  EXEMPLAR POLICY")
    print(f"    pool   {len(pool)} GOOD traces, retrieved by similarity -- the source")
    print(f"    seeds  {len(seeds)} from the train run -- top-up only, never re-selected")
    print(f"    pool is FROZEN: correct traces from THIS run are not added to it")
    print(f"  retrieval: {a.retrieval} | exemplars traced to the POOL: "
          f"{_from_pool}/{a.exemplar_budget}")
    if pool and _from_pool == 0:
        raise SystemExit(
            "  RETRIEVAL IS NOT RUNNING: a non-empty pool contributed zero\n"
            "  exemplars, so every one came from the seeds. Do not spend GPU.")
    if _from_pool < a.exemplar_budget:
        print(f"  note: {a.exemplar_budget - _from_pool} topped up from seeds "
              "(pool had too few matches for this question)")

    root = Path(a.store_dir); root.mkdir(parents=True, exist_ok=True)
    snap = {k: getattr(config, k) for k in dir(config)
            if k.isupper() and isinstance(getattr(config, k), (int, float, str, bool))}
    snap["FADE_ARM"] = "ctc_validate" if a.split == "train" else "ctc_test"
    snap["FADE_SPLIT"] = a.split
    snap["FADE_CTC_FLAG_BUDGET"] = a.flag_budget
    snap["FADE_CTC_THRESHOLD"] = -1.0 if a.threshold is None else a.threshold
    snap["FADE_CTC_TYPE_GATE"] = a.type_conf_gate
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
            print(f"  RESUME: {len(done)} already done")

    def _solve(q):
        """Pass 1 only. The retry goes through K.apply_cure(), which owns the
        whole typed treatment -- see [AUDIT D76] in pass B below."""
        info = cat.categorize(q)
        # knn_fn=knn is REQUIRED -- without it the pool is ignored entirely.
        res = gen.generate(q, 2, manual_exemplars=seeds, pool_exemplars=pool,
                           category_info=info, max_new_tokens=a.max_new_tokens,
                           return_logprobs=False, knn_fn=knn)
        return res["trace"], G.extract_final_answer(res["trace"])

    # [AUDIT D66] the negative bank the typed arm uses. Built from the POOL store,
    # never from this run -- these questions are held out and must stay that way.
    neg = None
    if a.negatives and a.n_neg > 0:
        try:
            from negatives import NegativeBank
            neg = NegativeBank.from_store(Path(a.pool_store))
            _nn = sum(len(v) for v in getattr(neg, "by_type", {}).values())
            print(f"  negatives: {_nn} wrong traces over "
                  f"{len(getattr(neg, 'by_type', {}))} types, from the POOL store "
                  "(never from this run's held-out questions)")
            if _nn == 0:
                print("  !! empty negative bank -- warnings will be silently absent")
                neg = None
        except Exception as e:
            print(f"  negatives unavailable ({e}); continuing without them")

    # ---- the cut, decided BEFORE the loop and applied per question --------
    # [AUDIT D78] ONE QUESTION AT A TIME: solve it, judge it, retry it there and
    # then, move on. The previous version ran two passes so the cut could be a
    # quantile of the whole p_wrong distribution -- but that is not a pipeline
    # anyone can deploy. At test time question 1 must be decided before question 2
    # is attempted; there is no distribution yet to take a quantile of.
    #
    # So the budget is converted to a FIXED p_wrong cut using the calibration
    # ctc_train2.py measured out of fold on pass-1 traces and shipped inside the
    # model [D77]. "Retry the worst 40%" becomes "retry p_wrong >= 0.683", which
    # is a rule a single question can be judged against.
    _cal = (rep.get("detect", {}) or {}).get("budget_thresholds") or {}
    if a.threshold is not None:
        thr = float(a.threshold); how = f"fixed --threshold {thr:.3f}"
    elif a.flag_budget and a.flag_budget > 0:
        key = f"{a.flag_budget:.1f}"
        if key in _cal:
            thr = float(_cal[key])
            _pr = ((rep.get("detect", {}) or {}).get("budget_precision") or {}).get(key)
            how = (f"--flag-budget {a.flag_budget:.0%} -> p_wrong >= {thr:.3f} "
                   f"(stage-1 calibration on "
                   f"{rep['detect'].get('calibration_slice','?')} traces"
                   + (f", precision {_pr:.1%})" if _pr else ")"))
        else:
            thr = 0.683
            how = (f"--flag-budget {a.flag_budget:.0%} -> p_wrong >= {thr:.3f} "
                   "(FALLBACK: this model shipped no calibration table -- retrain "
                   "with the D77 ctc_train2.py to calibrate on your own data)")
    else:
        thr = 0.5; how = "fallback 0.50"
    print(f"\n  FLAG RULE: {how}")
    print("  Applied per question, the moment its trace exists. Nothing about the")
    print("  cut depends on the other problems, so stage 3 runs identically.")

    # =====================================================================
    # THE LOOP -- solve, detect, cure, next
    # =====================================================================
    f = open(results_path, "a")
    t0 = time.time()
    n_seen = n_flag = n_retried = n_gated = 0
    rdone = {}

    print("\n" + "#" * 70)
    print(f"#  n={len(problems)} | type gate {a.type_conf_gate:.2f} | "
          f"retry={a.retry_flagged}")
    print("#" * 70)

    for i, prob in enumerate(problems, 1):
        pid = f"cv_{a.eval_offset + i:05d}"
        if pid in done:
            continue
        q = prob["question"]

        # 1. attempt
        trace, ans = _solve(q)

        # 2. analyse -- GOLD-FREE, reads only the question and this trace
        pred = ctc_model.predict(q, trace)
        flagged = pred["p_wrong"] >= thr
        n_seen += 1; n_flag += flagged

        row = {"id": pid, "eval_index": a.eval_offset + i - 1, "question": q,
               "trace": trace, "final_answer": ans,
               "p_wrong": pred["p_wrong"], "predicted_type": pred["predicted_type"],
               "type_confidence": pred["type_confidence"],
               "flagged": bool(flagged), "threshold": thr,
               # [AUDIT D71] CTC2.predict returns no "features" key. The old
               # row["features"] = pred["features"] raised KeyError on the first
               # problem, before a single trace was written.
               "features": pred.get("features")}

        # 3. cure, here and now, before the next question is touched
        if flagged and a.retry_flagged:
            # [AUDIT D68] trust the type only above the gate. Below it the
            # predicted type is near a coin flip, and a mismatched cure is worse
            # than the generic one -- so drop the type entirely, the same
            # abstention the cascade uses for UNCLASSIFIED.
            gated = pred["type_confidence"] < a.type_conf_gate
            use_type = None if gated else pred["predicted_type"]
            n_gated += gated; n_retried += 1

            # [AUDIT D76] THE SAME CURE THE TYPED ARM RUNS, not a subset of it.
            # This used to be cure_for() + build_retry_prompt() + one greedy
            # generation, which gave the exemplars and the instruction but
            # silently dropped the UNCLASSIFIED reroute (D52), the failure-mode
            # naming (D63), the prefill (D43/D51), the acceptance resample (D44),
            # the unchanged-answer resample (D60) and symbolic repair (D46/D61).
            # Stage 2 was measuring a WEAKER pipeline than the one that produced
            # the 42.0% typed result, so it would have understated CTC.
            #
            # apply_cure() is that pipeline with only the final score_trace()
            # removed, so it is legal on a split with no gold. The order inside
            # is: question-aware retrieval over the pool -> type-fit re-rank ->
            # layered ordering (most relevant first, type-matched last, nearest
            # the question) -> approach demo -> instruction, failure-mode naming
            # and negatives into the SYSTEM message -> prefill -> generate.
            mode = ("generic" if gated
                    else ("typed+neg" if (neg and a.n_neg > 0) else "typed"))
            trace2, meta = K.apply_cure(
                gen, q, trace, ans, use_type, pool, seeds, a.max_new_tokens,
                mode=mode, budget=a.exemplar_budget, negatives=neg,
                n_neg=a.n_neg, rec_id=pid)
            row.update({
                "retry_trace": trace2,
                "retry_answer": G.extract_final_answer(trace2),
                "cure_type": ("GENERIC(gated)" if gated else meta.get("cure_type")),
                "type_gated": bool(gated), "cure_mode": mode,
                "cure": describe(use_type) if use_type else "generic",
                "instruction": meta.get("instruction_sent", ""),
                "prefill": meta.get("prefill", ""),
                "cure_note": meta.get("cure_note", ""),
                "n_resampled": meta.get("n_resampled", 0),
                "n_unchanged_resamples": meta.get("n_unchanged_resamples", 0),
                "symbolic_repair": meta.get("symbolic_repair", False),
                "symbolic_repair_post": meta.get("symbolic_repair_post", False),
                "accept_ok": meta.get("accept_ok", True),
                "n_exemplars": meta.get("n_exemplars", 0)})

        # 4. score -- the ONLY place gold is opened, and it changes nothing above
        if not a.no_score:
            gold = prob["gold_answer"]
            comps, label, diag, _ = K.score_trace(q, trace, prob["answer"], gold)
            row.update({"gold_answer": gold, "gold_solution": prob["answer"],
                        "correct": bool(comps.correct),
                        "true_type": (diag.ftype.value if diag else None),
                        "label": label.value, "signals": comps.signals()})
            if "retry_trace" in row:
                c2, _l2, _d2, _ = K.score_trace(q, row["retry_trace"],
                                                prob["answer"], gold)
                row["retry_correct"] = bool(c2.correct)
            row["final_correct"] = bool(comps.correct or row.get("retry_correct"))

        f.write(json.dumps(row) + "\n"); f.flush()
        if "retry_trace" in row:
            rdone[pid] = row

        # [AUDIT D80] The first few questions print one line each, so the
        # interleaving is visible in the log rather than taken on trust: every
        # question is solved, judged and cured before the next is touched. If you
        # ever see two SOLVE lines with no decision between them, something is
        # batching and this loop is not doing what it claims.
        if n_seen <= 3:
            _d = ("flagged -> cured" if row.get("retry_trace")
                  else ("flagged, retry off" if row.get("flagged") else "clean, moving on"))
            print(f"    {pid}  p_wrong {pred['p_wrong']:.3f}  "
                  f"{pred['predicted_type']:<13} conf {pred['type_confidence']:.2f}"
                  f"  -> {_d}", flush=True)
            if n_seen == 3:
                print("    (per-question lines stop here; totals every "
                      f"{a.show_every or '--'} below)")

        if a.show_every and (i % a.show_every == 0 or i == len(problems)):
            el = time.time() - t0
            # `i` walks the whole list; `n_seen` counts only what THIS session
            # generated. On a resume they differ, and an earlier version printed
            # them side by side with no labels -- "[60/120] flagged 13/13" reads
            # like a bug when it is simply 47 problems restored from a checkpoint.
            _resumed = i - n_seen
            print(f"  [{i}/{len(problems)}]"
                  + (f" ({_resumed} resumed + {n_seen} new)" if _resumed else "")
                  + f" | flagged {n_flag}/{n_seen} ({n_flag/max(n_seen,1):.0%} of new)"
                  f" | retried {n_retried} | gated {n_gated}"
                  f" | {el/max(n_seen,1):.1f}s/prob", flush=True)
    f.close()

    rows = [json.loads(l) for l in open(results_path) if l.strip()]
    rows.sort(key=lambda r: r["id"])
    for r in rows:                      # resumed rows are already complete
        if "retry_trace" in r:
            rdone.setdefault(r["id"], r)
    n_flag = sum(bool(r.get("flagged")) for r in rows)
    print(f"\n  flagged {n_flag}/{len(rows)} = {n_flag/max(len(rows),1):.1%}"
          f"   retried {len(rdone)}")

    # ---- the gold-free property, checked rather than asserted in prose ----
    # Every flag must be reproducible from p_wrong and the threshold ALONE. If any
    # row's decision disagrees with that pure function, something gold-derived
    # reached the decision, and the same script would NOT be legal on test.
    _bad = [r["id"] for r in rows
            if bool(r.get("flagged")) != bool(r["p_wrong"] >= r.get("threshold", thr))]
    if _bad:
        raise SystemExit(
            f"  GOLD-FREE VIOLATION: {len(_bad)} rows have a flag that is not a\n"
            f"  function of p_wrong and the threshold (e.g. {_bad[:3]}). The retry\n"
            "  decision depended on something else -- do not report this run.")
    _leak = [r["id"] for r in rows
             if any(k in (r.get("features") or {}) for k in
                    ("gold_answer", "gold_solution", "correct", "true_type"))]
    if _leak:
        raise SystemExit(f"  GOLD LEAK in CTC features on {len(_leak)} rows")
    print(f"  gold-free check: all {len(rows)} flags reproduce from p_wrong >= "
          f"{thr:.3f}, no gold in features")

    summary = {"arm": ("ctc_validate" if a.split == "train" else "ctc_test"),
               "split": a.split, "n": len(rows), "threshold": thr,
               "flag_rule": how, "flag_budget": a.flag_budget,
               "type_conf_gate": a.type_conf_gate, "ctc_version": ctc_version,
               "n_flagged": n_flag,
               "eval_offset": a.eval_offset, "model": a.model,
               "retry_flagged": a.retry_flagged, "retrieval": a.retrieval,
               "pool_size": len(pool), "exemplars_from_pool_probe": _from_pool}

    # One merged file for downstream tools -- identical to results.jsonl now that
    # each row is written complete, kept so existing readers do not break.
    with open(root / "merged.jsonl", "w") as mf:
        for r in rows:
            mf.write(json.dumps(r) + "\n")

    # ---- did the typed machinery actually fire? --------------------------
    # Each of these was silently absent from stage 2 before D76. Printing the
    # counts is how you tell that the retry here IS the typed arm, rather than
    # trusting that it is.
    _all_r = list(rdone.values())
    if _all_r:
        _typed_r = [x for x in _all_r if not x.get("type_gated")]
        print("\n  CURE MACHINERY (typed retries only, n=%d)" % len(_typed_r))
        for lbl, key in (("prefill used", "prefill"),
                         ("failure mode named", "instruction"),
                         ("UNCLASSIFIED rerouted", "cure_note")):
            n_ = sum(1 for x in _typed_r if x.get(key))
            print(f"    {lbl:<24}{n_:>5}/{len(_typed_r)}")
        for lbl, key in (("acceptance resample", "n_resampled"),
                         ("unchanged-answer resample", "n_unchanged_resamples")):
            n_ = sum(1 for x in _typed_r if x.get(key))
            print(f"    {lbl:<24}{n_:>5}/{len(_typed_r)}")
        for lbl, key in (("symbolic repair (pass 1)", "symbolic_repair"),
                         ("symbolic repair (retry)", "symbolic_repair_post")):
            n_ = sum(1 for x in _all_r if x.get(key))
            print(f"    {lbl:<24}{n_:>5}/{len(_all_r)}")
        # Only warn when a prefill was EXPECTED. TR, ST, CE and UNCLASSIFIED carry
        # none by design, so "0 prefills" is correct for a run whose cures were all
        # of those -- a dry run that predicted ST for every problem tripped an
        # earlier version of this warning and sent me hunting a bug in working code.
        try:
            from diagnosis import TYPED_PREFILL
            _has_pf = {k.value for k, v in TYPED_PREFILL.items() if v}
        except Exception:
            _has_pf = set()
        _expect_pf = [x for x in _typed_r if x.get("cure_type") in _has_pf]
        if _typed_r and not _expect_pf:
            print("    (no prefill expected -- every cure type here carries none)")
        elif _expect_pf and not any(x.get("prefill") for x in _expect_pf):
            print(f"    !! {len(_expect_pf)} retries had a type that DOES carry a "
                  "prefill and none fired -- check TYPED_PREFILL_ENABLED")

    if a.no_score:
        (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
        print(f"\n  scoring disabled. outputs: {root}")
        return

    scored = [r for r in rows if "correct" in r]
    y = [not r["correct"] for r in scored]            # True = actually WRONG
    p = [r["p_wrong"] for r in scored]

    print("\n" + "=" * 74)
    print("  1 · DETECTION -- can CTC tell a wrong trace from a right one?")
    print("=" * 74)
    base = sum(y) / max(len(y), 1)
    print(f"  n={len(scored)}   actually wrong {sum(y)} ({base:.1%})")
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    print(f"  AUROC {auroc:.3f}    (stage-1 CV, mixed phases: "
          f"{rep.get('detect',{}).get('auroc', float('nan')):.3f} | "
          f"on pass-1 traces only: 0.776)")
    print("  0.776 is the like-for-like comparison -- every trace here is a first")
    print("  attempt, and stage 1's headline averages first attempts with retries.")
    summary["auroc"] = auroc

    # The sweep is run TWICE: once on the priors that chose this run's budget,
    # and once on the recovery and damage rates this run actually measured. If
    # the two disagree about where the optimum sits, the prior was wrong and the
    # measured curve is what stage 3 should use.
    _ret = [r for r in scored if "retry_correct" in r]
    _tp_ret = [r for r in _ret if not r["correct"]]
    _fp_ret = [r for r in _ret if r["correct"]]
    gain_m = (sum(1 for r in _tp_ret if r["retry_correct"]) / len(_tp_ret)
              if _tp_ret else None)
    cost_m = (sum(1 for r in _fp_ret if not r["retry_correct"]) / len(_fp_ret)
              if _fp_ret else None)

    def _sweep(gain, cost, label):
        print("\n" + "-" * 74)
        print(f"  THRESHOLD SWEEP -- {label}"
              f"   (recovery {gain:.1%}, damage {cost:.1%})")
        print(f"  a flag pays only above precision {cost/(gain+cost):.1%}")
        print("-" * 74)
        print(f"  {'budget':>7}{'thr':>6}{'flagged':>9}{'TP':>6}{'FP':>6}"
              f"{'precision':>11}{'recall':>9}{'expected net':>14}")
        # rank order, matching the live cut in D74 -- a `>= quantile` sweep
        # silently reports the tied group instead of the budget.
        srt = sorted(scored, key=lambda r: (-r["p_wrong"], r["id"]))
        best_ = None
        for bud in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            kb = max(1, min(len(srt), int(round(bud * len(srt)))))
            sel = srt[:kb]
            t_ = sel[-1]["p_wrong"]
            tp = sum(1 for r in sel if not r["correct"])
            fp = len(sel) - tp
            prec = tp / max(len(sel), 1)
            net = tp * gain - fp * cost
            if best_ is None or net > best_[2]:
                best_ = (bud, t_, net, prec)
            print(f"  {bud:>7.0%}{t_:>6.2f}{len(sel):>9}{tp:>6}{fp:>6}{prec:>11.1%}"
                  f"{tp/max(sum(y),1):>9.1%}{net:>+14.1f}")
        print(f"\n  best: budget {best_[0]:.0%} (p_wrong >= {best_[1]:.2f}), "
              f"precision {best_[3]:.1%}, net {best_[2]:+.1f} problems")
        return best_

    b_prior = _sweep(RECOVERY_PRIOR, DAMAGE_PRIOR, "on the PRIORS that chose this run")
    summary["best_budget_prior"] = b_prior[0]
    if gain_m is not None and cost_m is not None and (gain_m + cost_m) > 0:
        b_meas = _sweep(gain_m, cost_m, "on the rates THIS RUN measured")
        summary["best_budget_measured"] = b_meas[0]
        if abs(b_meas[0] - a.flag_budget) > 0.15:
            print(f"\n  !! the measured optimum ({b_meas[0]:.0%}) is far from the "
                  f"budget used ({a.flag_budget:.0%}).")
            print("     Stage 3 should use the measured one. This run is still valid;")
            print("     it simply did not sit at its own optimum.")
    else:
        print("\n  (no retries scored, so the measured curve cannot be drawn)")

    print("\n" + "=" * 74)
    print("  2 · TYPE -- when CTC says wrong, does it name the right failure?")
    print("=" * 74)
    tw = [r for r in scored if not r["correct"] and r.get("true_type")]
    hit = sum(1 for r in tw if r["predicted_type"] == r["true_type"])
    print(f"  wrong traces with a cascade diagnosis: {len(tw)}")
    print(f"  CTC's type == cascade's type: {hit}/{len(tw)} = {hit/max(len(tw),1):.1%}")
    _s1 = _tacc if _tacc is not None else float("nan")
    print(f"     (stage-1 CV, mixed phases: {_s1:.1%} | on pass-1 traces only: 59.8%)")
    by = defaultdict(lambda: [0, 0])
    for r in tw:
        by[r["true_type"]][1] += 1
        by[r["true_type"]][0] += (r["predicted_type"] == r["true_type"])
    print(f"\n  {'true type':<16}{'n':>5}{'recall':>9}")
    for t, (c, n) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"  {t:<16}{n:>5}{c/max(n,1):>9.1%}")
    summary["type_accuracy"] = hit / max(len(tw), 1)

    # ---- what does it say instead? ---------------------------------------
    types = sorted({r["true_type"] for r in tw} | {r["predicted_type"] for r in tw})
    if len(types) > 1:
        print(f"\n  CONFUSION  (rows = cascade's type, cols = CTC's guess)")
        print("  " + " " * 14 + "".join(f"{t[:4]:>6}" for t in types))
        for a_ in types:
            m_ = [r for r in tw if r["true_type"] == a_]
            if not m_:
                continue
            print(f"  {a_:<14}" + "".join(
                f"{sum(1 for r in m_ if r['predicted_type'] == b_):>6}" for b_ in types))

    # ---- the type accuracy that actually drives a cure --------------------
    # Only FLAGGED traces get a cure, so that subset is what matters for the
    # pipeline. Type accuracy over all wrong traces includes ones CTC never
    # flagged and therefore never routed.
    fw = [r for r in tw if r.get("flagged")]
    if fw:
        h2 = sum(1 for r in fw if r["predicted_type"] == r["true_type"])
        print(f"\n  among WRONG AND FLAGGED (the ones that actually got a cure):")
        print(f"    {h2}/{len(fw)} = {h2/len(fw):.1%} routed to the right cure")
        summary["type_accuracy_flagged"] = h2 / len(fw)

    # ---- DOES THE TYPE MATTER? -------------------------------------------
    # The question the paper needs answered. Among retried-and-truly-wrong
    # traces, split recovery by whether CTC named the type correctly. If the two
    # rates are the same, the type head is decoration and the gain (if any) comes
    # from retrying with pool exemplars at all -- the same question the generic
    # arm asks of EDER, asked here of CTC.
    if a.retry_flagged:
        rr = [r for r in tw if "retry_correct" in r]
        # [AUDIT D73] split on the cure ACTUALLY APPLIED, not on predicted_type.
        # With the confidence gate on, a low-confidence prediction is overridden
        # and the generic cure is used, so predicted_type no longer describes what
        # the model was given. Comparing against it would credit or blame a cure
        # that never ran.
        ungated = [r for r in rr if not r.get("type_gated")]
        right_t = [r for r in ungated if r["predicted_type"] == r["true_type"]]
        wrong_t = [r for r in ungated if r["predicted_type"] != r["true_type"]]

        # ---- IS THE GATE EARNING ITS PLACE? ------------------------------
        gated = [r for r in rr if r.get("type_gated")]
        if gated or ungated:
            print(f"\n  THE CONFIDENCE GATE (typed above {a.type_conf_gate:.2f}, "
                  "generic below)")
            g_rec = (sum(bool(r["retry_correct"]) for r in gated) / len(gated)
                     if gated else None)
            u_rec = (sum(bool(r["retry_correct"]) for r in ungated) / len(ungated)
                     if ungated else None)
            print(f"    gated to generic   : "
                  f"{'--' if g_rec is None else format(g_rec, '>6.1%')}  n={len(gated)}")
            print(f"    kept the typed cure: "
                  f"{'--' if u_rec is None else format(u_rec, '>6.1%')}  n={len(ungated)}")
            summary.update({"recovery_gated_generic": g_rec,
                            "recovery_ungated_typed": u_rec,
                            "n_gated": len(gated), "n_ungated": len(ungated)})
            if not ungated:
                print(f"\n    EVERY flagged trace fell below the gate, so the typed")
                print(f"    cure never ran and this is a GENERIC arm wearing CTC's")
                print(f"    name. Nothing below about type routing means anything.")
                print("    Lower --type-conf-gate -- stage 1 put ~68% of pass-1")
                print("    traces above 0.50 -- or report this as the result: the")
                print("    type head was never confident enough to be used.")
            elif not gated:
                print("\n    Nothing was gated -- this run is equivalent to "
                      "--type-conf-gate 0.")
            else:
                print("    These are different populations -- low-confidence traces")
                print("    are harder -- so this is NOT a controlled comparison. The")
                print("    clean ablation is a second run with --type-conf-gate 0.")
        if right_t and wrong_t:
            a_rec = sum(bool(r["retry_correct"]) for r in right_t) / len(right_t)
            b_rec = sum(bool(r["retry_correct"]) for r in wrong_t) / len(wrong_t)
            print(f"\n  DOES NAMING THE TYPE CORRECTLY HELP?")
            print(f"    cure matched the true type : {a_rec:>6.1%}  n={len(right_t)}")
            print(f"    cure was mismatched        : {b_rec:>6.1%}  n={len(wrong_t)}")
            print(f"    difference                 : {(a_rec-b_rec)*100:>+5.1f} pts")
            try:
                from stats import fisher_exact_2x2
                pv = fisher_exact_2x2([
                    [sum(bool(r["retry_correct"]) for r in right_t),
                     len(right_t) - sum(bool(r["retry_correct"]) for r in right_t)],
                    [sum(bool(r["retry_correct"]) for r in wrong_t),
                     len(wrong_t) - sum(bool(r["retry_correct"]) for r in wrong_t)]])
                print(f"    Fisher exact p             : {pv:.4f}")
                summary["type_matters_p"] = pv
            except Exception:
                pass
            print("\n    If these two rates are the same, the type head is not earning")
            print("    its place and the gain is 'retry with pool exemplars', not")
            print("    'matched cure'. That is a different claim -- say which one.")
            summary.update({"recovery_type_matched": a_rec,
                            "recovery_type_mismatched": b_rec,
                            "n_type_matched": len(right_t),
                            "n_type_mismatched": len(wrong_t)})
        else:
            print("\n  (cannot split recovery by type correctness -- one side is empty)")

    if a.retry_flagged:
        print("\n" + "=" * 74)
        print("  3 · THE PIPELINE -- did it actually help?")
        print("=" * 74)
        p1c = sum(1 for r in scored if r["correct"])
        fin = sum(1 for r in scored if r.get("final_correct"))
        ret = [r for r in scored if "retry_correct" in r]
        tp_ret = [r for r in ret if not r["correct"]]
        fp_ret = [r for r in ret if r["correct"]]
        recovered = sum(1 for r in tp_ret if r["retry_correct"])
        broken = sum(1 for r in fp_ret if not r["retry_correct"])
        print(f"  pass-1 accuracy      {p1c}/{len(scored)} = {p1c/len(scored):.1%}")
        print(f"  final  accuracy      {fin}/{len(scored)} = {fin/len(scored):.1%}"
              f"   ({(fin-p1c)/len(scored):+.1%})")
        print(f"\n  retried {len(ret)}   true positives {len(tp_ret)}  false positives {len(fp_ret)}")
        print(f"    recovered from the true positives : {recovered}/{len(tp_ret)} = "
              f"{recovered/max(len(tp_ret),1):.1%}   (prior {RECOVERY_PRIOR:.1%})")
        print(f"    MEASURED DAMAGE on false positives: {broken}/{len(fp_ret)} = "
              f"{broken/max(len(fp_ret),1):.1%}   (prior {DAMAGE_PRIOR:.1%})")
        print("\n  The damage rate above is the number the 76.5% prior was standing in")
        print("  for. Use it, not the prior, to pick the stage-3 threshold.")
        summary.update({"pass1_accuracy": p1c/len(scored),
                        "final_accuracy": fin/len(scored),
                        "n_retried": len(ret), "n_true_pos": len(tp_ret),
                        "n_false_pos": len(fp_ret), "recovered": recovered,
                        "broken": broken,
                        "measured_damage": broken/max(len(fp_ret), 1),
                        "measured_recovery": recovered/max(len(tp_ret), 1)})

    # =====================================================================
    # TWO SCORECARDS -- they answer different questions, keep them apart
    # =====================================================================
    # [AUDIT D79] "accuracy" was ambiguous in every earlier report. CTC being
    # right about a trace and the MODEL being right about a question are separate
    # measurements, and they can move in opposite directions: a detector that
    # flags perfectly still gains nothing if the cure cannot fix what it found,
    # and a cure that works is wasted on traces the detector never flags.
    print("\n" + "=" * 74)
    print("  SCORECARD 1 · CTC -- was the JUDGE right?")
    print("=" * 74)
    _tp = sum(1 for r in scored if r.get("flagged") and not r["correct"])
    _fp = sum(1 for r in scored if r.get("flagged") and r["correct"])
    _fn = sum(1 for r in scored if not r.get("flagged") and not r["correct"])
    _tn = sum(1 for r in scored if not r.get("flagged") and r["correct"])
    _det_acc = (_tp + _tn) / max(len(scored), 1)
    _maj = max(sum(y), len(scored) - sum(y)) / max(len(scored), 1)
    print(f"  DOES AN ERROR EXIST?      at the cut actually used ({thr:.3f})")
    print(f"    TP {_tp:>4}   FP {_fp:>4}   FN {_fn:>4}   TN {_tn:>4}")
    print(f"    precision {_tp/max(_tp+_fp,1):>6.1%}   "
          f"recall {_tp/max(_tp+_fn,1):>6.1%}   "
          f"accuracy {_det_acc:>6.1%}  (majority {_maj:.1%})")
    print(f"    AUROC {auroc:.3f}  -- the threshold-free number, and the one to")
    print(f"    quote: accuracy loses to majority whenever most traces are wrong.")
    print("\n  WHICH ERROR IS IT?        among wrong traces with a diagnosis")
    if tw:
        _tc = Counter(r["true_type"] for r in tw)
        _typ_maj = _tc.most_common(1)[0][1] / len(tw)
        print(f"    over ALL wrong traces        {hit}/{len(tw)} = "
              f"{hit/len(tw):>6.1%}   (majority {_typ_maj:.1%})")
        summary["ctc_type_majority"] = _typ_maj
    else:
        print("    (no wrong trace carried a cascade diagnosis)")
    if fw:
        print(f"    over the ones CTC FLAGGED    {h2}/{len(fw)} = {h2/len(fw):>6.1%}"
              "   <- the only ones a cure was chosen for")
    _ung = [r for r in scored if r.get("retry_trace") and not r.get("type_gated")
            and r.get("true_type")]
    if _ung:
        _uh = sum(1 for r in _ung if r["predicted_type"] == r["true_type"])
        print(f"    over the ones it ACTED on    {_uh}/{len(_ung)} = "
              f"{_uh/len(_ung):>6.1%}   <- past the confidence gate")
    summary.update({"ctc_detect_precision": _tp/max(_tp+_fp,1),
                    "ctc_detect_recall": _tp/max(_tp+_fn,1),
                    "ctc_detect_accuracy": _det_acc,
                    "ctc_detect_majority": _maj})

    print("\n" + "=" * 74)
    print("  SCORECARD 2 · THE MODEL -- did it get more QUESTIONS right?")
    print("=" * 74)
    _p1 = sum(1 for r in scored if r["correct"])
    _fin = sum(1 for r in scored if r.get("final_correct"))
    print(f"  pass 1   {_p1:>4}/{len(scored)} = {_p1/max(len(scored),1):>6.1%}")
    print(f"  final    {_fin:>4}/{len(scored)} = {_fin/max(len(scored),1):>6.1%}"
          f"   ({(_fin-_p1)/max(len(scored),1):+.1%}, {_fin-_p1:+d} problems)")
    _b = sum(1 for r in scored if not r["correct"] and r.get("final_correct"))
    _c = sum(1 for r in scored if r["correct"] and not r.get("final_correct"))
    print(f"  recovered {_b}   broken {_c}   net {_b - _c:+d}")
    try:
        # PAIRED -- same questions before and after, so McNemar, never a
        # two-proportion z-test. See stats.mcnemar's docstring.
        from stats import mcnemar
        _mc = mcnemar({r["id"]: bool(r["correct"]) for r in scored},
                      {r["id"]: bool(r.get("final_correct")) for r in scored})
        print(f"  McNemar exact p = {_mc['p']:.4f}"
              + ("   SIGNIFICANT" if _mc["p"] < 0.05 else "   not significant"))
        summary["mcnemar_p"] = _mc["p"]
    except Exception as _e:
        print(f"  (McNemar unavailable: {_e})")
    summary.update({"recovered": _b, "broken": _c})
    print("\n  These two scorecards are independent. A perfect judge still gains")
    print("  nothing if the cure cannot fix what it found, and a cure that works")
    print("  is wasted on traces the judge never flags. Report both.")

    (root / config.SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 74)
    print("  GATE FOR STAGE 3")
    print("=" * 74)
    if a.retry_flagged and summary.get("final_accuracy", 0) > summary.get("pass1_accuracy", 1):
        print("  The pipeline IMPROVED accuracy on unseen questions. Take it to test,")
        print(f"  using the threshold that maximises the MEASURED net gain above.")
    elif auroc == auroc and auroc >= 0.65:
        print("  Detection generalises, but the retry did not pay off at this")
        print("  threshold. Re-run the sweep against the MEASURED damage rate and")
        print("  pick a stricter threshold before going to test.")
    else:
        print("  Detection did not generalise. Report it with the FST negative --")
        print("  two honest routes to the same conclusion is a real analysis.")
    print(f"\n  outputs: {root}")


if __name__ == "__main__":
    main()
