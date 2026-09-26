"""
score_baselines.py -- CPU only. Opens gold once, at the end, after every generation.

Reads a finished store (pass-1 + shipped retries), its baselines.jsonl
(run_baselines.py) and mpv_probes.jsonl (run_mpv.py), and reports:

  E3  REPAIR RATES  R (wrong -> right) and D (right -> wrong) for each repair method
                    -- shipped cure, plain resample, generic retry -- on the FLAGGED
                    rows (what the break-even rule uses) and on ALL rows (stable
                    estimate), with exact 95% intervals, pi* = D/(R+D), the observed
                    net of replacing flagged answers, and the cap R x N_wrong.
  E2  SELF-CONSISTENCY vs PROBE  on the same testable rows: precision when the
                    second sample agrees vs when the probe supports; odds ratios;
                    and, the key cut, disagreement / refutation / static-flag rate
                    by TRUE failure type against each method's rate on correct traces.
                    Detection AUROC for p_wrong alone and blended (fixed w=0.5) with
                    each signal, with paired bootstrap CIs for the gain.
  E1  HELD-OUT FORECAST  (--heldout-store) R, D and precision estimated on held-out
                    train questions, used to forecast the test net BEFORE looking at
                    it; bootstrap interval; whether the observed net falls inside.
  E4  bootstrap CIs everywhere; --cluster template resamples GSM-Symbolic templates
                    instead of rows, since instances of one template are not
                    independent.

Writes <store>/baselines_report.json and <store>/rd_points.csv (for nm_tables.py).

USAGE
    python score_baselines.py --store <merged test store> --split test --limit 400 \\
        [--gold-jsonl p1_gold.jsonl --cluster template] [--heldout-store <stage-2 store>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import mpv
from extraction import extract_final_answer

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- statistics
def cp_ci(k, n, alpha=0.05):
    """Clopper-Pearson exact interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def auroc(scores, wrong):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(wrong)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, float)))


def boot_idx(n, clusters, B):
    """Yield bootstrap index arrays, resampling clusters when given."""
    if clusters is None:
        for _ in range(B):
            yield RNG.integers(0, n, n)
        return
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups)
    for _ in range(B):
        pick = RNG.integers(0, len(keys), len(keys))
        yield np.fromiter((i for k in pick for i in groups[keys[k]]), int)


def pct(xs, lo=2.5, hi=97.5):
    xs = [x for x in xs if x == x]
    if not xs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(xs, lo)), float(np.percentile(xs, hi)))


def fisher(a, b, c, d):
    from scipy.stats import fisher_exact
    orr, p = fisher_exact([[a, b], [c, d]])
    return float(orr), float(p)


def same(x, y):
    if x is None or y is None:
        return False
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return False
    return abs(x - y) <= 1e-6 * max(1.0, abs(y))


# ---------------------------------------------------------------- loading
def load_rows(store, limit):
    rows = [json.loads(l) for l in open(Path(store) / "results.jsonl") if l.strip()]
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def load_baselines(path):
    out = defaultdict(dict)
    if Path(path).exists():
        for ln in open(path):
            if ln.strip():
                d = json.loads(ln)
                out[d["id"]][d["arm"]] = d
    return out


def load_probes(path):
    """-> ({id: [Probe]}, {id: [raw dict]}). Same keying as score_mpv.py: one probe per
    (target, new value); probes whose Q' answer did not parse carry no evidence."""
    probes, raw = defaultdict(dict), defaultdict(dict)
    if Path(path).exists():
        for ln in open(path):
            if not ln.strip():
                continue
            d = json.loads(ln)
            if d.get("model_answer") is None:
                continue
            key = (round(float(d["target_value"]), 6), round(float(d["new_value"]), 6))
            probes[d["id"]][key] = mpv.Probe.from_dict(d)
            raw[d["id"]][key] = d
    return ({k: [v[kk] for kk in sorted(v)] for k, v in probes.items()},
            {k: [v[kk] for kk in sorted(v)] for k, v in raw.items()})


def gold_map(split, gold_jsonl):
    if gold_jsonl:
        g = {}
        for ln in open(gold_jsonl):
            if ln.strip():
                d = json.loads(ln)
                g[d["question"]] = d
        return g
    import kaggle_run as K
    return {p["question"]: p for p in K._load_split(split, None)}


class Scorer:
    """Gold-side scoring through the harness's own score_trace, cached."""
    def __init__(self):
        import kaggle_run as K
        self.K = K
        self.cache = {}

    def correct(self, q, trace, g):
        key = ("c", q, trace)
        if key not in self.cache:
            self.cache[key] = bool(self.K.score_trace(q, trace, g["answer"], g["gold_answer"])[0].correct)
        return self.cache[key]

    def true_type(self, q, trace, g):
        comps, _l, diag, _ = self.K.score_trace(q, trace, g["answer"], g["gold_answer"])
        if comps.correct:
            return "CORRECT"
        ft = getattr(diag, "ftype", None)
        return str(getattr(ft, "value", ft) or "UNCLASSIFIED")


# ---------------------------------------------------------------- per-row table
def build_table(rows, base, probes, gold, S):
    T = []
    for r in rows:
        q = r["question"]
        g = gold.get(q)
        if g is None and "gold_answer" in r and "gold_solution" in r:
            g = {"answer": r["gold_solution"], "gold_answer": r["gold_answer"]}
        if g is None:
            continue
        a1 = extract_final_answer(r["trace"])
        t = {"id": r["id"], "q": q, "c1": S.correct(q, r["trace"], g), "a1": a1,
             "type": S.true_type(q, r["trace"], g),
             "p_wrong": r.get("p_wrong"), "flagged": bool(r.get("flagged")),
             "template": g.get("template_id"), "gold_solution": g["answer"]}
        if r.get("retry_trace"):
            t["shipped"] = S.correct(q, r["retry_trace"], g)
        b = base.get(r["id"], {})
        for arm in ("resample", "generic"):
            if arm in b:
                t[arm] = S.correct(q, b[arm]["trace"], g)
                t[arm + "_ans"] = b[arm].get("answer")
        pr = probes.get(r["id"])
        if pr:
            # same definitions as score_mpv.py, so numbers match the paper's Table 2:
            # testable = detection score exists; supported = score < 1 (>=1 probe agreed)
            t["has_probe"] = True
            ms = mpv.detection_score(q, r["trace"], a1, pr)
            if ms is not None:
                t["mpv_score"] = ms
                t["supported"] = ms < 1.0
        T.append(t)
    return T


# ---------------------------------------------------------------- E3 repair rates
def repair_rates(T, method, subset):
    rows = [t for t in T if method in t and (t["flagged"] if subset == "flagged" else True)]
    wrong = [t for t in rows if not t["c1"]]
    right = [t for t in rows if t["c1"]]
    rec = sum(t[method] for t in wrong)
    brk = sum(not t[method] for t in right)
    R = rec / len(wrong) if wrong else float("nan")
    D = brk / len(right) if right else float("nan")
    return {"n": len(rows), "n_wrong": len(wrong), "n_right": len(right),
            "recovered": rec, "broken": brk, "R": R, "R_ci": cp_ci(rec, len(wrong)),
            "D": D, "D_ci": cp_ci(brk, len(right)),
            "pi_star": (D / (R + D)) if (R == R and D == D and R + D > 0) else float("nan"),
            "net_if_replaced": rec - brk}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gold-jsonl", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--baselines", default=None)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--heldout-store", default=None,
                    help="a SCORED stage-2 store (train split). Its baselines.jsonl is used if present.")
    ap.add_argument("--cluster", choices=["none", "template"], default="none")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--label", default=None, help="model/dataset label for the tables")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.store)
    label = a.label or root.name
    rows = load_rows(root, a.limit)
    leak = [k for r in rows for k in ("gold_answer", "correct", "true_type") if k in r]
    if a.split == "test" and leak:
        raise SystemExit(f"GOLD IN A TEST RUN OUTPUT: {set(leak)}")
    base = load_baselines(a.baselines or root / "baselines.jsonl")
    probes, raw_probes = load_probes(a.probes or root / "mpv_probes.jsonl")
    gold = gold_map(a.split, a.gold_jsonl) if a.split == "test" else {}
    S = Scorer()
    T = build_table(rows, base, probes, gold, S)
    n = len(T)
    if n == 0:
        raise SystemExit("no scorable rows -- wrong --split / --gold-jsonl?")
    clusters = [t["template"] for t in T] if a.cluster == "template" else None
    R = {"label": label, "store": root.name, "n": n}
    print("=" * 78)
    print(f"  {label}   n={n}   baselines on {sum(1 for t in T if 'resample' in t or 'generic' in t)} rows"
          f"   probes on {sum(1 for t in T if t.get('has_probe'))} rows")
    print("=" * 78)

    # ---- accuracy + detector ------------------------------------------------
    c1 = np.array([t["c1"] for t in T])
    acc = c1.mean()
    accs = [c1[ix].mean() for ix in boot_idx(n, clusters, a.boot)]
    R["pass1"] = {"correct": int(c1.sum()), "acc": float(acc), "ci": pct(accs),
                  "n_wrong": int(n - c1.sum())}
    print(f"\n  pass-1 accuracy {c1.sum()}/{n} = {acc:.1%}  95% CI [{pct(accs)[0]:.1%}, {pct(accs)[1]:.1%}]")
    have_pw = [t for t in T if t["p_wrong"] is not None]
    if have_pw:
        sc = [t["p_wrong"] for t in have_pw]; yw = [0 if t["c1"] else 1 for t in have_pw]
        au = auroc(sc, yw)
        idx_all = np.arange(len(have_pw))
        aus = [auroc(np.asarray(sc)[ix], np.asarray(yw)[ix])
               for ix in boot_idx(len(have_pw), [clusters[i] for i in idx_all] if clusters else None, a.boot)]
        fl = [t for t in T if t["flagged"]]
        tp = sum(not t["c1"] for t in fl); fp = len(fl) - tp
        R["detector"] = {"auroc": au, "auroc_ci": pct(aus), "n_flagged": len(fl),
                         "tp": tp, "fp": fp, "precision": tp / max(len(fl), 1),
                         "precision_ci": cp_ci(tp, len(fl)),
                         "recall": tp / max(n - c1.sum(), 1)}
        print(f"  static detector AUROC {au:.3f} [{pct(aus)[0]:.3f}, {pct(aus)[1]:.3f}]"
              f" | flagged {len(fl)} ({len(fl)/n:.1%}) TP {tp} FP {fp}"
              f" precision {tp/max(len(fl),1):.1%}")
    no_false_eq = None
    try:
        from extraction import extract_equations, trace_body
        wr = [t for t in T if not t["c1"]]
        rowmap = {r["id"]: r for r in rows}
        k = sum(1 for t in wr if not any(not e.is_true for e in
                                          extract_equations(trace_body(rowmap[t["id"]]["trace"]))))
        no_false_eq = k / max(len(wr), 1)
        R["share_wrong_no_false_equation"] = no_false_eq
        print(f"  wrong traces with NO false equation: {k}/{len(wr)} = {no_false_eq:.1%}")
    except Exception as e:
        print(f"  (no-false-equation share unavailable: {e})")

    # ---- the probe on its own (needs no baselines) ---------------------------
    tt = [t for t in T if "supported" in t]
    n_cov = sum(1 for t in T if t.get("has_probe"))
    R["probe"] = {"coverage": n_cov / n, "questions_with_probe": n_cov, "testable": len(tt)}
    if tt:
        s_c = sum(1 for t in tt if t["supported"] and t["c1"])
        s_n = sum(1 for t in tt if t["supported"])
        r_c = sum(1 for t in tt if not t["supported"] and t["c1"])
        r_n = len(tt) - s_n
        b = sum(t["c1"] for t in tt) / len(tt)
        orr, p = fisher(s_c, s_n - s_c, r_c, r_n - r_c)
        R["probe"].update({"supported_n": s_n, "supported_correct": s_c, "refuted_n": r_n,
                           "refuted_correct": r_c, "base_rate": b,
                           "support_precision": s_c / max(s_n, 1),
                           "support_precision_ci": cp_ci(s_c, s_n),
                           "odds_ratio": orr, "fisher_p": p,
                           "lift": (s_c / max(s_n, 1)) / max(b, 1e-9),
                           "false_refutation": (sum(1 for t in tt if t["c1"] and not t["supported"])
                                                / max(sum(t["c1"] for t in tt), 1))})
        print(f"\n  PROBE  coverage {n_cov}/{n} = {n_cov/n:.1%} (scorable {len(tt)}) | supported {s_c}/{s_n} correct "
              f"= {s_c/max(s_n,1):.1%} vs base {b:.1%} (OR {orr:.1f}, p={p:.1e}, lift "
              f"{R['probe']['lift']:.2f}x) | refutes {R['probe']['false_refutation']:.1%} of correct traces")
        # alpha: the model's accuracy on Q' -- scoring only, from the gold annotations
        try:
            import ctc as CTC
            hit = tot = 0
            for t in tt:
                for d in raw_probes.get(t["id"], []):
                    gq = CTC.perturbed_gold(t["gold_solution"], d["target_value"], d["new_value"])
                    if gq is None:
                        continue
                    tot += 1
                    hit += mpv._close(d.get("model_answer"), gq)
            if tot:
                R["probe"]["alpha_accuracy_on_Qprime"] = hit / tot
                R["probe"]["alpha_n"] = tot
                print(f"  model accuracy on perturbed questions (alpha) {hit}/{tot} = {hit/tot:.1%}"
                      f"   vs pass-1 {acc:.1%}")
        except Exception as e:
            print(f"  (alpha unavailable: {e})")

    # ---- E3 -------------------------------------------------------------------
    print("\n  E3  REPAIR RATES   R = wrong->right,  D = right->wrong,  pi* = D/(R+D)")
    print(f"  {'method':<10}{'subset':<9}{'n':>5}{'R':>8}{'95% CI':>17}{'D':>8}{'95% CI':>17}"
          f"{'pi*':>8}{'net':>6}")
    R["repair"] = {}
    rd_rows = []
    for m in ("shipped", "resample", "generic"):
        for sub in ("flagged", "all"):
            if m == "shipped" and sub == "all":
                continue
            rr = repair_rates(T, m, sub)
            if rr["n"] == 0:
                continue
            R["repair"][f"{m}/{sub}"] = rr
            print(f"  {m:<10}{sub:<9}{rr['n']:>5}{rr['R']:>8.1%}"
                  f"  [{rr['R_ci'][0]:5.1%},{rr['R_ci'][1]:5.1%}]{rr['D']:>8.1%}"
                  f"  [{rr['D_ci'][0]:5.1%},{rr['D_ci'][1]:5.1%}]{rr['pi_star']:>8.1%}"
                  f"{rr['net_if_replaced']:>+6d}")
            rd_rows.append({"label": label, "method": m, "subset": sub, "n": rr["n"],
                            "R": rr["R"], "R_lo": rr["R_ci"][0], "R_hi": rr["R_ci"][1],
                            "D": rr["D"], "D_lo": rr["D_ci"][0], "D_hi": rr["D_ci"][1],
                            "pi_star": rr["pi_star"],
                            "pi_achieved": R.get("detector", {}).get("precision")})
    nw = n - int(c1.sum())
    caps = {}
    for key, rr in R["repair"].items():
        if key.endswith("/all") and rr["R"] == rr["R"]:
            caps[key.split("/")[0]] = {"problems": rr["R"] * nw, "points": rr["R"] * nw / n}
    if "shipped/flagged" in R["repair"]:
        rr = R["repair"]["shipped/flagged"]
        caps["shipped(flagged R, all failures)"] = {"problems": rr["R"] * nw, "points": rr["R"] * nw / n}
    R["cap_perfect_detector"] = caps
    for k, v in caps.items():
        print(f"  cap with a PERFECT detector, {k}: {v['problems']:.0f} problems (+{v['points']:.1%})")

    # ---- E2 -------------------------------------------------------------------
    both = [t for t in T if "resample_ans" in t]
    if both:
        print("\n  E2  SELF-CONSISTENCY (1 extra sample) vs METAMORPHIC PROBE")
        R["e2"] = {}
        for scope, rows_ in (("all", both),
                             ("probe_testable", [t for t in both if "supported" in t])):
            if not rows_:
                continue
            ag = [same(t["a1"], t["resample_ans"]) for t in rows_]
            a_c = sum(1 for t, g in zip(rows_, ag) if g and t["c1"])
            a_n = sum(ag)
            d_c = sum(1 for t, g in zip(rows_, ag) if not g and t["c1"])
            d_n = len(rows_) - a_n
            b = sum(t["c1"] for t in rows_) / len(rows_)
            orr, p = fisher(a_c, a_n - a_c, d_c, d_n - d_c)
            e = {"n": len(rows_), "agree_n": a_n, "agree_correct": a_c,
                 "disagree_n": d_n, "disagree_correct": d_c, "base_rate": b,
                 "agree_precision": a_c / max(a_n, 1), "odds_ratio": orr, "fisher_p": p,
                 "lift": (a_c / max(a_n, 1)) / max(b, 1e-9)}
            print(f"    [{scope}] n={len(rows_)}  agree {a_c}/{a_n} correct = {e['agree_precision']:.1%}"
                  f"   disagree {d_c}/{d_n} = {d_c/max(d_n,1):.1%}   base {b:.1%}"
                  f"   OR {orr:.1f} p={p:.1e}  lift {e['lift']:.2f}x")
            if scope == "probe_testable":
                s_c = sum(1 for t in rows_ if t["supported"] and t["c1"])
                s_n = sum(1 for t in rows_ if t["supported"])
                r_c = sum(1 for t in rows_ if not t["supported"] and t["c1"])
                r_n = len(rows_) - s_n
                orr2, p2 = fisher(s_c, s_n - s_c, r_c, r_n - r_c)
                e["probe"] = {"supported_n": s_n, "supported_correct": s_c,
                              "refuted_n": r_n, "refuted_correct": r_c,
                              "support_precision": s_c / max(s_n, 1), "odds_ratio": orr2,
                              "fisher_p": p2, "lift": (s_c / max(s_n, 1)) / max(b, 1e-9)}
                print(f"    [probe, same rows] supported {s_c}/{s_n} = {s_c/max(s_n,1):.1%}"
                      f"   refuted {r_c}/{r_n} = {r_c/max(r_n,1):.1%}   OR {orr2:.1f} p={p2:.1e}"
                      f"  lift {e['probe']['lift']:.2f}x")
            R["e2"][scope] = e

        # the key cut: by true failure type, on the probe-testable rows
        tt = [t for t in both if "supported" in t]
        if tt:
            print("\n    rate at which each verifier calls a trace WRONG, by TRUE type (probe-testable rows)")
            print(f"    {'type':<14}{'n':>5}{'2nd sample disagrees':>22}{'probe refutes':>15}{'static flags':>14}")
            by = defaultdict(list)
            for t in tt:
                by[t["type"]].append(t)
            R["e2"]["by_type"] = {}
            for ty, xs in sorted(by.items(), key=lambda kv: (kv[0] == "CORRECT", -len(kv[1]))):
                dis = sum(not same(t["a1"], t["resample_ans"]) for t in xs) / len(xs)
                ref = sum(not t["supported"] for t in xs) / len(xs)
                flg = sum(t["flagged"] for t in xs) / len(xs)
                R["e2"]["by_type"][ty] = {"n": len(xs), "disagree": dis, "refute": ref, "flag": flg}
                print(f"    {ty:<14}{len(xs):>5}{dis:>22.1%}{ref:>15.1%}{flg:>14.1%}")
            print("    (CORRECT row = each verifier's false-alarm rate; compare every other row to it)")

            # detection AUROC with fixed w=0.5 blends, paired bootstrap
            tp_ = [t for t in tt if t["p_wrong"] is not None and t.get("mpv_score") is not None]
            if tp_:
                y = np.array([0 if t["c1"] else 1 for t in tp_])
                pw = np.array([t["p_wrong"] for t in tp_])
                dis = np.array([0.0 if same(t["a1"], t["resample_ans"]) else 1.0 for t in tp_])
                ms = np.array([t["mpv_score"] for t in tp_], float)
                sigs = {"p_wrong": pw, "disagree": dis, "mpv": ms,
                        "p_wrong+disagree": 0.5 * pw + 0.5 * dis,
                        "p_wrong+mpv": 0.5 * pw + 0.5 * ms,
                        "p_wrong+mpv+disagree": (pw + ms + dis) / 3}
                cl = [t["template"] for t in tp_] if clusters else None
                base_au = auroc(pw, y)
                R["e2"]["auroc"] = {}
                print("\n    detection AUROC on the same rows (blends fixed a priori at equal weight)")
                for k, v in sigs.items():
                    au = auroc(v, y)
                    diffs = []
                    for ix in boot_idx(len(tp_), cl, a.boot):
                        diffs.append(auroc(v[ix], y[ix]) - auroc(pw[ix], y[ix]))
                    lo, hi = pct(diffs)
                    R["e2"]["auroc"][k] = {"auroc": au, "gain_vs_p_wrong": au - base_au,
                                           "gain_ci": (lo, hi)}
                    print(f"      {k:<22}{au:.3f}   gain vs p_wrong {au - base_au:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # ---- E1 -------------------------------------------------------------------
    if a.heldout_store:
        print("\n  E1  HELD-OUT FORECAST  (estimate on held-out train, predict this store)")
        H_rows = load_rows(a.heldout_store, 0)
        H_base = load_baselines(Path(a.heldout_store) / "baselines.jsonl")
        H = build_table(H_rows, H_base, {}, {}, S)
        R["e1"] = {"heldout_store": Path(a.heldout_store).name, "heldout_n": len(H)}
        hf = [t for t in H if t["flagged"]]
        pi_h = sum(not t["c1"] for t in hf) / max(len(hf), 1)
        fl = [t for t in T if t["flagged"]]
        tp_t = sum(not t["c1"] for t in fl); fp_t = len(fl) - tp_t
        for m in ("shipped", "resample", "generic"):
            sub_h = "flagged" if m == "shipped" else "all"
            hr = repair_rates(H, m, sub_h)
            if hr["n"] == 0 or not fl or not all(m in t for t in fl):
                continue
            obs = sum((not t["c1"]) and t[m] for t in fl) - sum(t["c1"] and not t[m] for t in fl)
            full, semi = [], []
            Hw = [t for t in (hf if sub_h == "flagged" else H) if m in t]
            Hf = hf
            for _ in range(a.boot):
                hs = [Hw[i] for i in RNG.integers(0, len(Hw), len(Hw))]
                w_ = [t for t in hs if not t["c1"]]; r_ = [t for t in hs if t["c1"]]
                Rb = sum(t[m] for t in w_) / max(len(w_), 1)
                Db = sum(not t[m] for t in r_) / max(len(r_), 1) if r_ else hr["D"]
                hb = [Hf[i] for i in RNG.integers(0, len(Hf), len(Hf))] if Hf else []
                pib = sum(not t["c1"] for t in hb) / max(len(hb), 1) if hb else pi_h
                full.append(len(fl) * (pib * Rb - (1 - pib) * Db))
                semi.append(tp_t * Rb - fp_t * Db)
            pf = len(fl) * (pi_h * hr["R"] - (1 - pi_h) * hr["D"])
            ps = tp_t * hr["R"] - fp_t * hr["D"]
            lo_f, hi_f = pct(full); lo_s, hi_s = pct(semi)
            R["e1"][m] = {"heldout_R": hr["R"], "heldout_D": hr["D"], "heldout_pi": pi_h,
                          "heldout_n_wrong": hr["n_wrong"], "heldout_n_right": hr["n_right"],
                          "forecast_full": pf, "forecast_full_ci": (lo_f, hi_f),
                          "forecast_semi": ps, "forecast_semi_ci": (lo_s, hi_s),
                          "observed": obs, "inside_full": lo_f <= obs <= hi_f,
                          "inside_semi": lo_s <= obs <= hi_s}
            print(f"    {m:<9} held-out R {hr['R']:.1%} (n={hr['n_wrong']}) D {hr['D']:.1%} (n={hr['n_right']})"
                  f" pi {pi_h:.1%} | forecast {pf:+.1f} [{lo_f:+.1f}, {hi_f:+.1f}]"
                  f"  (with test pi: {ps:+.1f} [{lo_s:+.1f}, {hi_s:+.1f}])"
                  f" | observed {obs:+d} -> {'INSIDE' if lo_f <= obs <= hi_f else 'OUTSIDE'}")

    out = Path(a.out) if a.out else root / "baselines_report.json"
    json.dump(R, open(out, "w"), indent=2, default=float)
    with open(root / "rd_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rd_rows[0].keys()) if rd_rows else ["label"])
        w.writeheader(); w.writerows(rd_rows)
    print(f"\n  -> {out}\n  -> {root / 'rd_points.csv'}")


if __name__ == "__main__":
    main()
