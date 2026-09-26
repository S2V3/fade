"""
nm_tables.py -- turn per-store reports into the camera-ready tables, figure and macros.

Each store must already have baselines_report.json (score_baselines.py) and,
for the accuracy arms, mpv_report.json (score_mpv.py).

    python nm_tables.py --out nm_reports \\
        --run "Llama-2-7B|GSM8K=stores/nm_llama2_test_gsm8k" \\
        --run "Llama-2-7B|GSM-Sym P1=stores/nm_llama2_test_p1" \\
        --run "Qwen3.5-4B|GSM8K=stores/nm_qwen35_test_gsm8k" \\
        --run "Qwen3.5-4B|GSM-Sym P1=stores/nm_qwen35_test_p1"

Writes into --out:
    main_table.md / .csv   one column per (model, dataset): the paper's Table 1
    arms_table.md          accuracy arms with McNemar p (from mpv_report.json)
    by_type.md             who calls a trace wrong, by true failure type
    rd_plane.png / .pdf    the R-D plane with break-even lines (Figure 2)
    paper_macros.tex       every number as a LaTeX macro, so text never drifts
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def fmt_pct(x, d=1):
    return "--" if x is None or x != x else f"{100 * x:.{d}f}%"


def fmt(x, d=3):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def macro_name(*parts):
    s = "".join(p.title() for p in parts)
    s = re.sub(r"[^A-Za-z]", "", s.replace("1", "One").replace("2", "Two")
               .replace("3", "Three").replace("4", "Four").replace("5", "Five")
               .replace("7", "Seven").replace("8", "Eight"))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"Model|Dataset=store_dir"')
    ap.add_argument("--out", default="nm_reports")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in a.run:
        lab, path = spec.split("=", 1)
        model, ds = (lab.split("|", 1) + [""])[:2]
        p = Path(path)
        rep = json.load(open(p / "baselines_report.json"))
        mrep = json.load(open(p / "mpv_report.json")) if (p / "mpv_report.json").exists() else {}
        runs.append({"model": model, "dataset": ds, "label": lab, "rep": rep, "mrep": mrep})

    # ---------------------------------------------------------- main table
    def row(name, f):
        return [name] + [f(r) for r in runs]

    R = lambda r, k: get(r["rep"], "repair", k, default={})
    lines = [
        row("n questions", lambda r: str(r["rep"]["n"])),
        row("Pass-1 accuracy", lambda r: fmt_pct(get(r["rep"], "pass1", "acc"))),
        row("Wrong traces with no false equation", lambda r: fmt_pct(r["rep"].get("share_wrong_no_false_equation"))),
        row("Static detector AUROC", lambda r: fmt(get(r["rep"], "detector", "auroc"))),
        row("Flagged / precision (pi)", lambda r: f"{get(r['rep'],'detector','n_flagged', default='--')} / "
                                                   f"{fmt_pct(get(r['rep'],'detector','precision'))}"),
        row("Probe coverage", lambda r: fmt_pct(get(r["rep"], "probe", "coverage"))),
        row("Model accuracy on Q' (alpha)", lambda r: fmt_pct(get(r["rep"], "probe", "alpha_accuracy_on_Qprime"))),
        row("Probe support precision vs base",
            lambda r: f"{fmt_pct(get(r['rep'],'probe','support_precision'))} vs {fmt_pct(get(r['rep'],'probe','base_rate'))}"),
        row("Probe odds ratio / lift",
            lambda r: f"{fmt(get(r['rep'],'probe','odds_ratio'),1)} / {fmt(get(r['rep'],'probe','lift'),2)}x"),
        row("Probe refutes correct traces", lambda r: fmt_pct(get(r["rep"], "probe", "false_refutation"))),
        row("2nd sample: agree precision vs base",
            lambda r: f"{fmt_pct(get(r['rep'],'e2','probe_testable','agree_precision'))} vs "
                      f"{fmt_pct(get(r['rep'],'e2','probe_testable','base_rate'))}"),
        row("Shipped cure R / D (flagged)",
            lambda r: f"{fmt_pct(R(r,'shipped/flagged').get('R'))} / {fmt_pct(R(r,'shipped/flagged').get('D'))}"),
        row("pi* (shipped) vs achieved pi",
            lambda r: f"{fmt_pct(R(r,'shipped/flagged').get('pi_star'))} vs {fmt_pct(get(r['rep'],'detector','precision'))}"),
        row("Plain resample R / D (all)",
            lambda r: f"{fmt_pct(R(r,'resample/all').get('R'))} / {fmt_pct(R(r,'resample/all').get('D'))}"),
        row("Generic retry R / D (all)",
            lambda r: f"{fmt_pct(R(r,'generic/all').get('R'))} / {fmt_pct(R(r,'generic/all').get('D'))}"),
        row("Best arm vs pass-1 (McNemar p)", lambda r: best_arm(r["mrep"])),
    ]
    head = ["Metric"] + [r["label"] for r in runs]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(x) + " |" for x in lines]
    (out / "main_table.md").write_text("\n".join(md) + "\n")
    with open(out / "main_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(lines)
    print("\n".join(md))

    # ---------------------------------------------------------- arms
    arm_names = ["pass1", "replace", "replace_mpv_veto", "select_symbolic", "select_mpv", "select_mpv_backsub"]
    md = ["| arm | " + " | ".join(r["label"] for r in runs) + " |", "|---|" + "---|" * len(runs)]
    for k in arm_names:
        cells = []
        for r in runs:
            v = r["mrep"].get(k)
            cells.append("--" if not v else f"{v['correct']}/{v['n']} {100*v['acc']:.1f}% "
                                            f"(p={v['vs_pass1']['p']:.2g})")
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    (out / "arms_table.md").write_text("\n".join(md) + "\n")

    # ---------------------------------------------------------- by type
    md = []
    for r in runs:
        bt = get(r["rep"], "e2", "by_type") or {}
        if not bt:
            continue
        md += [f"\n**{r['label']}**\n", "| true type | n | 2nd sample disagrees | probe refutes | static flags |",
               "|---|---|---|---|---|"]
        for ty, v in bt.items():
            md.append(f"| {ty} | {v['n']} | {fmt_pct(v['disagree'])} | {fmt_pct(v['refute'])} | {fmt_pct(v['flag'])} |")
    (out / "by_type.md").write_text("\n".join(md) + "\n")

    # ---------------------------------------------------------- macros
    mac = ["% generated by nm_tables.py -- do not edit by hand"]
    for r in runs:
        base = macro_name(r["model"], r["dataset"])
        vals = {"Acc": get(r["rep"], "pass1", "acc"), "Auroc": get(r["rep"], "detector", "auroc"),
                "Prec": get(r["rep"], "detector", "precision"),
                "NoFalseEq": r["rep"].get("share_wrong_no_false_equation"),
                "SupPrec": get(r["rep"], "probe", "support_precision"),
                "Base": get(r["rep"], "probe", "base_rate"), "OddsR": get(r["rep"], "probe", "odds_ratio"),
                "Cov": get(r["rep"], "probe", "coverage"),
                "R": R(r, "shipped/flagged").get("R"), "D": R(r, "shipped/flagged").get("D"),
                "PiStar": R(r, "shipped/flagged").get("pi_star")}
        for k, v in vals.items():
            if v is None or v != v:
                continue
            txt = f"{v:.3f}" if k in ("Auroc",) else (f"{v:.1f}" if k == "OddsR" else f"{100*v:.1f}\\%")
            mac.append(f"\\newcommand{{\\{base}{k}}}{{{txt}}}")
    (out / "paper_macros.tex").write_text("\n".join(mac) + "\n")

    # ---------------------------------------------------------- R-D plane
    try:
        rd_plane(runs, out)
    except Exception as e:
        print(f"  (R-D figure skipped: {e})")
    print(f"\n  -> {out}/main_table.md, arms_table.md, by_type.md, paper_macros.tex, rd_plane.png")


def best_arm(m):
    best = None
    for k in ("replace", "replace_mpv_veto", "select_symbolic", "select_mpv", "select_mpv_backsub"):
        v = m.get(k)
        if v and (best is None or v["correct"] > best[1]["correct"]):
            best = (k, v)
    if not best or "pass1" not in m:
        return "--"
    d = best[1]["correct"] - m["pass1"]["correct"]
    return f"{best[0]} {d:+d} (p={best[1]['vs_pass1']['p']:.2g})"


def rd_plane(runs, out):
    """R on x, D on y, one panel per dataset, colour = model, marker = repair method.
    Dashed line per model: break-even D = R*pi/(1-pi) at that model's achieved pi.
    Points ABOVE a model's line lose accuracy when its flags are replaced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
    MARK = {"shipped": ("o", "diagnosis cure (flagged)"),
            "resample": ("s", "plain resample"),
            "generic": ("^", "retry, no diagnosis")}
    datasets = list(dict.fromkeys(r["dataset"] for r in runs))
    models = list(dict.fromkeys(r["model"] for r in runs))
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.0),
                             sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        ax.set_facecolor("white")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        xs = [i / 100 for i in range(0, 101)]
        for r in [r for r in runs if r["dataset"] == ds]:
            pi = get(r["rep"], "detector", "precision")
            if pi and 0 < pi < 1:
                ax.plot(xs, [x * pi / (1 - pi) for x in xs], ls="--", lw=1.2,
                        color=color[r["model"]], alpha=0.8)
            for key, (mk, _) in MARK.items():
                rr = get(r["rep"], "repair", f"{key}/flagged" if key == "shipped" else f"{key}/all")
                if not rr or rr.get("R") != rr.get("R"):
                    continue
                ax.errorbar(rr["R"], rr["D"],
                            xerr=[[rr["R"] - rr["R_ci"][0]], [rr["R_ci"][1] - rr["R"]]],
                            yerr=[[rr["D"] - rr["D_ci"][0]], [rr["D_ci"][1] - rr["D"]]],
                            fmt=mk, ms=6, color=color[r["model"]], ecolor=color[r["model"]],
                            elinewidth=0.8, capsize=0, mec="white", mew=1.0)
        ax.plot(xs, [x * 0.99 / 0.01 for x in xs], ls=":", lw=1, color=INK2)
        rmax = max([0.5] + [get(r["rep"], "repair", k, "R_ci", default=[0, 0])[1] or 0
                            for r in runs if r["dataset"] == ds
                            for k in ("shipped/flagged", "resample/all", "generic/all")])
        ax.set_xlim(0, min(1.0, rmax + 0.05)); ax.set_ylim(0, 1.02)
        ax.set_title(ds, fontsize=9, color=INK)
        ax.set_xlabel("R: retry repairs a wrong trace", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
    axes[0][0].set_ylabel("D: retry breaks a correct trace", fontsize=8, color=INK2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=color[m], lw=0, marker="o", ms=6, label=m) for m in models]
    h += [Line2D([], [], color=INK2, lw=0, marker=mk, ms=6, label=lab) for mk, lab in MARK.values()]
    h += [Line2D([], [], color=INK2, ls="--", lw=1.2, label="break-even at achieved precision"),
          Line2D([], [], color=INK2, ls=":", lw=1, label="break-even at 99% precision")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out / "rd_plane.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "rd_plane.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
