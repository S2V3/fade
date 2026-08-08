"""
run_stability.py -- IS THE FAILURE TYPE A PROPERTY OF THE QUESTION AT ALL?

This is the experiment that decides whether FST is worth building, and it is
strictly more informative than the gate that preceded it.

THE PROBLEM IT SOLVES
---------------------
fst_gate.py asked: "can gold-free features of the question predict the failure
type?" On the 1,500-problem train run the answer was NO (chi2 p=0.29, Cramer's
V=0.09, and no classifier -- category lookup, logistic regression, random forest,
or TF-IDF over the raw question text -- beat the majority-class baseline).

A negative like that has two very different explanations, and a reviewer WILL ask
which one you are reporting:

  (a) the features are too weak  -> get better features, FST may still work;
  (b) the TARGET is not stable   -> no feature can ever work, and the null is a
                                    property of the phenomenon, not of your setup.

You cannot tell (a) from (b) by adding more features. You tell them apart by
asking the SAME question k times and measuring how often the diagnosis agrees
with itself. That self-agreement is the reliability ceiling: no predictor of any
kind can exceed the reproducibility of the thing it predicts.

Preliminary evidence for (b) already exists in the train run: of the 713 pass-1
UNCLASSIFIED traces, 38% re-diagnosed as a DIFFERENT type on their single retry.
This script measures it properly, with k samples and a real agreement statistic.

WHAT IT DOES
------------
For each of N questions: sample the frozen model k times at temperature T with
the identical prompt, score and diagnose every sample, then report

  1. Fleiss' kappa over the FULL label space {CORRECT, NR, ST, SM, CE,
     UNCLASSIFIED} across the k samples  -- the headline number.
  2. Fleiss' kappa over {NR, ST, SM, CE, UNCLASSIFIED} restricted to the WRONG
     samples of each question -- "given that it failed, does it fail the same
     WAY?" This is the target FST actually predicts.
  3. Cohen's kappa on the first two samples -- the simplest, most quotable form.
  4. Modal-type share and per-question entropy -- how concentrated the types are.
  5. accuracy@1 vs pass@k -- the elicitation gap FADE exists to close, measured
     on the same samples for free.

HOW TO READ IT
--------------
  kappa >= 0.4  the type is reasonably stable -> the gate's NO-GO is about weak
                FEATURES. Build better features and re-run the gate.
  kappa <  0.2  the type is close to a coin flip on re-sampling -> no
                question-only predictor can work, and you have a null WITH a
                measured mechanism. That is a better result than a marginal GO,
                and it is the finding to write up.

GOLD RULE: gold is used only to SCORE and DIAGNOSE, never rendered into a prompt.
Prompts here are the ordinary strategy prompts, built by the same code path as a
normal run.

USAGE
    python run_stability.py --n 300 --k 5 --temperature 0.8
    python run_stability.py --analyze-only        # re-analyse an existing file

Resumable: re-running skips questions already complete in stability.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import config
import generation as G
import kaggle_run as K
import stats as S
from categorizer import QuestionCategorizer
from classification import Label

REPO = Path(__file__).resolve().parent
FULL_SPACE = ("CORRECT", "NR", "ST", "SM", "CE", "UNCLASSIFIED")
TYPED_SPACE = ("NR", "ST", "SM", "CE", "UNCLASSIFIED")


def _sample_label(comps, label, diag) -> str:
    """One sample -> one categorical rating in the FULL label space.

    A correct sample is 'CORRECT' regardless of its diagnosis: for the stability
    question, 'got it right' is a distinct outcome from any failure mode, and
    collapsing it into a failure type would manufacture disagreement.
    """
    if comps.correct:
        return "CORRECT"
    return diag.ftype.value if diag else "UNCLASSIFIED"


# ===========================================================================
# GENERATION
# ===========================================================================
def run_samples(gen, manual, eval_problems, strategy_id, n_problems, k,
                temperature, max_new_tokens, out_path, retrieval="3stage"):
    cat = QuestionCategorizer()
    seeds = manual.get(strategy_id, [])
    knn = K._retrieval_fn(retrieval)

    done = set()
    if out_path.exists():
        for line in open(out_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
        if done:
            print(f"  RESUME: {len(done)} questions already sampled")

    f = open(out_path, "a")
    problems = eval_problems[:n_problems]
    t0 = time.time()

    for i, prob in enumerate(problems, 1):
        qid = f"s_{i:05d}"
        if qid in done:
            continue
        q = prob["question"]
        gold_sol, gold_ans = prob["answer"], prob["gold_answer"]
        cat_info = cat.categorize(q)

        # ONE prompt, built exactly as a normal run builds it, reused for every
        # sample. Any variation across samples is therefore decoding stochasticity
        # and nothing else -- which is the whole point.
        prompt, _ = G._build_prompt(strategy_id, q, seeds, [], cat_info, knn)

        samples = []
        for _s in range(k):
            traces, _ = gen._run_model(prompt, temperature=temperature,
                                       max_new_tokens=max_new_tokens,
                                       num_return_sequences=1,
                                       return_logprobs=False)
            trace = G._ensure_hash_line(traces[0], G.extract_final_answer(traces[0]))
            comps, label, diag, _ = K.score_trace(q, trace, gold_sol, gold_ans)
            samples.append({
                "trace": trace,
                "rating": _sample_label(comps, label, diag),
                "correct": bool(comps.correct),
                "label": label.value,
                "diagnosis": diag.ftype.value if diag else None,
                "diagnosis_confidence": diag.confidence if diag else None,
                "signals": comps.signals(),
            })

        rec = {"id": qid, "question": q, "gold_answer": gold_ans,
               "category": cat_info, "k": k, "temperature": temperature,
               "samples": samples}
        f.write(json.dumps(rec) + "\n")
        f.flush()

        if i % 10 == 0 or i == len(problems):
            el = time.time() - t0
            print(f"  [{i}/{len(problems)}] {el/max(i,1):.1f}s/question "
                  f"| last ratings: {[s['rating'] for s in samples]}")
    f.close()


# ===========================================================================
# ANALYSIS
# ===========================================================================
def _entropy(counts) -> float:
    n = sum(counts.values())
    if n <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n, 2) for c in counts.values() if c)
    return h


def analyze(path: Path, write_files: bool = True):
    """Print the report and, unless disabled, ALSO write it to disk next to the
    raw samples: stability_summary.json (machine-readable) and
    stability_report.md (human-readable). Console output dies with the Kaggle
    session; files get checkpointed to GitHub."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("no rows -- run the sampling phase first")
        return
    k = rows[0]["k"]
    summary: dict = {"n_questions": len(rows), "k": k,
                     "temperature": rows[0]["temperature"]}
    print("=" * 72)
    print(f"STABILITY OF THE FAILURE TYPE   ({len(rows)} questions x k={k} samples, "
          f"T={rows[0]['temperature']})")
    print("=" * 72)

    # ---- 1. headline: full label space -----------------------------------
    full = [[s["rating"] for s in r["samples"]] for r in rows]
    kf = S.fleiss_kappa(full, FULL_SPACE)
    print("\n[1] Fleiss' kappa over {CORRECT, NR, ST, SM, CE, UNCLASSIFIED}")
    print(f"    kappa = {kf['kappa']:.4f}   ({kf['interpretation']})")
    print(f"    observed agreement {kf['P_observed']:.4f} vs chance {kf['P_expected']:.4f}")
    print(f"    marginals: {kf['marginals']}")

    # ---- 2. the target FST predicts: type GIVEN failure -------------------
    wrong = [[s["rating"] for s in r["samples"] if not s["correct"]] for r in rows]
    kw = S.fleiss_kappa(wrong, TYPED_SPACE)
    n_usable = sum(1 for w in wrong if len(w) >= 2)
    print(f"\n[2] Fleiss' kappa over failure TYPE, wrong samples only "
          f"({n_usable} questions with >=2 wrong samples)")
    if kw.get("note"):
        print(f"    {kw['note']}")
    else:
        print(f"    kappa = {kw['kappa']:.4f}   ({kw['interpretation']})")
        print(f"    observed {kw['P_observed']:.4f} vs chance {kw['P_expected']:.4f}")
        print(f"    marginals: {kw['marginals']}")
    print("    ^ THIS is the ceiling on any question-only failure-type predictor.")

    # ---- 3. simplest form: first two samples ------------------------------
    a = [r["samples"][0]["rating"] for r in rows]
    b = [r["samples"][1]["rating"] for r in rows if len(r["samples"]) > 1]
    kc = S.cohen_kappa(a[:len(b)], b)
    print(f"\n[3] Cohen's kappa, sample 1 vs sample 2 (n={kc['n']})")
    print(f"    kappa = {kc['kappa']:.4f}   ({kc['interpretation']})   "
          f"raw agreement {kc['p_observed']:.3f}")

    # ---- 4. concentration --------------------------------------------------
    modal_shares, ents, all_same = [], [], 0
    for r in rows:
        c = Counter(s["rating"] for s in r["samples"])
        modal_shares.append(c.most_common(1)[0][1] / len(r["samples"]))
        ents.append(_entropy(c))
        all_same += (len(c) == 1)
    print(f"\n[4] Per-question concentration")
    print(f"    mean modal-rating share : {sum(modal_shares)/len(modal_shares):.3f} "
          f"(1.0 = perfectly stable, {1/len(FULL_SPACE):.2f} = uniform)")
    print(f"    mean entropy            : {sum(ents)/len(ents):.3f} bits "
          f"(0 = stable, {math.log(len(FULL_SPACE),2):.2f} = uniform)")
    print(f"    questions where all {k} samples agree : {all_same}/{len(rows)} "
          f"({all_same/len(rows):.1%})")

    # ---- 5. elicitation gap (free from the same samples) ------------------
    acc1 = sum(r["samples"][0]["correct"] for r in rows) / len(rows)
    passk = sum(any(s["correct"] for s in r["samples"]) for r in rows) / len(rows)
    maj = 0
    for r in rows:
        answers = [s["signals"].get("final_answer") for s in r["samples"]
                   if s["signals"].get("final_answer") is not None]
        if answers:
            top = Counter(answers).most_common(1)[0][0]
            maj += abs(top - r["gold_answer"]) < 1e-6
    print(f"\n[5] Elicitation gap on the same samples")
    print(f"    accuracy@1        {acc1:.3f}")
    print(f"    majority@{k}       {maj/len(rows):.3f}")
    print(f"    pass@{k}           {passk:.3f}    <- the headroom FADE is trying to reach")

    # ---- verdict ----------------------------------------------------------
    kk = kw.get("kappa", float("nan"))
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if kk != kk:
        verdict, advice = "UNDEFINED", (
            "Not enough wrong samples to measure agreement. Increase --n or --k.")
    elif kk < 0.20:
        verdict, advice = "UNSTABLE", (
            f"kappa={kk:.3f}. The failure type is close to a coin flip on re-sampling "
            f"the SAME question, so no question-only feature can predict it. The "
            f"gate's NO-GO is a property of the phenomenon, not of your feature set. "
            f"Write the null: report this kappa as the ceiling, and do NOT build "
            f"fst_predictor.py.")
    elif kk < 0.40:
        verdict, advice = "WEAKLY STABLE", (
            f"kappa={kk:.3f}. Some reproducible signal, fair at best. A predictor "
            f"could in principle reach ~{kk:.2f} agreement -- report the ceiling "
            f"alongside the gate, and only build the predictor if that margin is "
            f"worth the claim.")
    else:
        verdict, advice = "STABLE", (
            f"kappa={kk:.3f}. The type IS reproducible, so the gate's NO-GO points at "
            f"weak FEATURES, not an unpredictable target. Improve the features (the "
            f"categorizer collapses to 5 of its 8 values) and re-run fst_gate.py "
            f"before writing any null.")
    print(f"{verdict}. {advice}")
    caveat = ("Measured at T={:.2f}. Stability is temperature-dependent; a greedy "
              "(T=0) run is deterministic and would trivially give kappa=1, which is "
              "why this uses the same sampling regime a real deployment would."
              .format(rows[0]["temperature"]))
    print("\nCAVEAT: " + caveat)

    # ---- persist ----------------------------------------------------------
    summary.update({
        "kappa_full_space": kf["kappa"], "kappa_full_detail": kf,
        "kappa_failure_type": kw.get("kappa"), "kappa_failure_detail": kw,
        "n_questions_with_2plus_wrong": n_usable,
        "cohen_kappa_s1_vs_s2": kc,
        "mean_modal_share": sum(modal_shares) / len(modal_shares),
        "mean_entropy_bits": sum(ents) / len(ents),
        "all_samples_agree": all_same,
        "accuracy_at_1": acc1, "majority_at_k": maj / len(rows), "pass_at_k": passk,
        "verdict": verdict, "advice": advice, "caveat": caveat,
    })
    if not write_files:
        return summary

    out_json = path.with_name("stability_summary.json")
    out_md = path.with_name("stability_report.md")
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    out_md.write_text(_render_md(summary))
    print(f"\nwrote {out_json.name} and {out_md.name} next to {path.name}")
    print("  ^ these are the files to checkpoint/push. Console output dies with "
          "the session; these do not.")
    return summary


def _render_md(s: dict) -> str:
    kt = s.get("kappa_failure_type")
    kt_s = f"{kt:.4f}" if isinstance(kt, float) and kt == kt else "undefined"
    return f"""# FADE — failure-type stability

{s['n_questions']} questions x k={s['k']} samples at T={s['temperature']}.

## Headline

| quantity | value |
|---|---|
| Fleiss kappa, full label space | {s['kappa_full_space']:.4f} |
| **Fleiss kappa, failure TYPE (wrong samples only)** | **{kt_s}** |
| Cohen kappa, sample 1 vs sample 2 | {s['cohen_kappa_s1_vs_s2']['kappa']:.4f} |
| questions with >=2 wrong samples | {s['n_questions_with_2plus_wrong']} |
| mean modal-rating share | {s['mean_modal_share']:.3f} |
| mean entropy | {s['mean_entropy_bits']:.3f} bits |
| all {s['k']} samples agree | {s['all_samples_agree']}/{s['n_questions']} |

The bolded row is the ceiling on any question-only failure-type predictor.

## Elicitation gap

| | |
|---|---|
| accuracy@1 | {s['accuracy_at_1']:.3f} |
| majority@{s['k']} | {s['majority_at_k']:.3f} |
| pass@{s['k']} | {s['pass_at_k']:.3f} |

## Verdict

**{s['verdict']}.** {s['advice']}

*Caveat: {s['caveat']}*
"""


def main():
    ap = argparse.ArgumentParser(description="failure-type stability / Fleiss kappa")
    ap.add_argument("--n", type=int, default=300, help="questions to sample")
    ap.add_argument("--k", type=int, default=5, help="samples per question")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--strategy", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="run this on TRAIN -- test stays frozen for the final run")
    ap.add_argument("--seed-pool", type=int, default=2000)
    ap.add_argument("--retrieval", choices=["knn", "3stage"], default="3stage")
    ap.add_argument("--exemplar-budget", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--secret-name", default="HF_TOKEN")
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()

    out = Path(a.out) if a.out else (REPO / "store" / "stability.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    if a.analyze_only:
        analyze(out)
        return

    if a.model:
        K.MODEL_ID = a.model
    G.set_exemplar_budget(a.exemplar_budget)
    K.version_banner()
    token = K.resolve_hf_token(a.secret_name)
    K.verify_identity_and_access(token)

    if a.split == "test":
        train_problems = K._load_split("train", a.seed_pool + 100)
        test_problems = K._load_split("test")
        manual, eval_problems = K.build_seeds_and_eval_order(
            train_problems, a.seed_pool, eval_split_problems=test_problems)
    else:
        problems = K.load_gsm8k(a.seed_pool + a.n + 100)
        manual, eval_problems = K.build_seeds_and_eval_order(problems, a.seed_pool)

    model, tok = K.load_model(token)
    gen = G.ExemplarGenerator(model=model, tokenizer=tok)

    print("\n" + "#" * 70)
    print(f"#  STABILITY RUN | n={a.n} questions x k={a.k} samples | T={a.temperature}")
    print(f"#  {a.n * a.k} generations total | split={a.split}")
    print("#" * 70)

    run_samples(gen, manual, eval_problems, a.strategy, a.n, a.k,
                a.temperature, a.max_new_tokens, out, retrieval=a.retrieval)
    print(f"\n  wrote {out}")
    analyze(out)


if __name__ == "__main__":
    main()