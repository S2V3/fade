"""
nm_data.py -- the second dataset for the camera-ready: GSM-Symbolic (Apple, 2024).

WHY GSM-SYMBOLIC
----------------
A 2025-26 model scores ~90% on GSM8K, which leaves too few failures on a
subsample to analyse. GSM-Symbolic regenerates GSM8K test questions from
templates with new names and numbers; P1 adds one clause and P2 two, so a newer
model fails more often. Rows keep GSM8K's '#### N' answer format, so the
existing parser and scorer work unchanged, and the canary string makes it less
likely to be in a 2026 model's training data than GSM8K itself.

WHAT THIS MODULE DOES
---------------------
    load_symbolic(variant)  rows in EXACTLY the dict shape kaggle_run._load_split
                            returns: {question, answer, gold_answer} + metadata
    annotate(solution)      GSM-Symbolic solutions carry no <<expr=result>>
                            calculator annotations. The instrument reads gold
                            checkpoints from those, so we synthesise them from
                            every pure-arithmetic 'expr = result' in the solution.
                            Only SCORING reads them (correct / true_type), never
                            the model. Equations written with units in them
                            ('12 inches * 7 = 84') are skipped, so checkpoint
                            counts are a LOWER bound -- per-type tables on this
                            dataset are approximate and the paper must say so.
    export_gold_jsonl()     rows for score_mpv.py --gold-jsonl

ORDER
-----
Rows are sorted by (instance, template id). The first N therefore cover every
template before any template repeats: N=400 on P1 is instances 0-3 of all 100
templates. Instances of one template are NOT independent, so bootstrap CIs on
this dataset should resample templates (score_baselines.py --cluster template).

GOLD RULE
---------
Nothing here is ever shown to the model. Gold fields are consumed only by the
scoring scripts, exactly as for GSM8K.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

VARIANTS = {
    # name                 HF config   GitHub file
    "gsm_symbolic":       ("main",     "GSM_symbolic.jsonl"),
    "gsm_symbolic_p1":    ("p1",       "GSM_p1.jsonl"),
    "gsm_symbolic_p2":    ("p2",       "GSM_p2.jsonl"),
}
HF_ID = "apple/GSM-Symbolic"
RAW_URL = ("https://raw.githubusercontent.com/apple/ml-gsm-symbolic/main/"
           "generated_data/{fname}")

GOLD_RE = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")
# A pure-arithmetic left side: numbers, operators, parentheses, spaces, '$', '%'.
# At least one operator is required, so '= 12' after prose never matches.
_NUM = r"\$?\d[\d,]*(?:\.\d+)?%?"
_OP = r"[\+\-\*/x×÷]"
EQ_RE = re.compile(
    rf"(?<![\w.])(\(?\s*{_NUM}(?:\s*\)?\s*{_OP}\s*\(?\s*{_NUM}\s*\)?)+)\s*=\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def _clean_num(s: str) -> str:
    return s.replace("$", "").replace(",", "").replace("%", "").strip()


def annotate(solution: str) -> tuple[str, int]:
    """Insert GSM8K-style '<<expr=result>>' after every pure-arithmetic equation.
    -> (annotated_solution, n_annotations). Idempotent on already-annotated text."""
    if "<<" in solution:
        return solution, solution.count("<<")
    out, last, n = [], 0, 0
    for m in EQ_RE.finditer(solution):
        lhs, rhs = m.group(1), m.group(2)
        expr = re.sub(r"\s+", "", lhs)
        expr = expr.replace("×", "*").replace("÷", "/").replace("x", "*")
        expr = _clean_num(expr)
        res = _clean_num(rhs)
        if not re.fullmatch(r"[\d\.\+\-\*/\(\)]+", expr):
            continue
        # place the annotation right before the result, as GSM8K does
        rs = m.start(2)
        out.append(solution[last:rs])
        out.append(f"<<{expr}={res}>>")
        last = rs
        n += 1
    out.append(solution[last:])
    return "".join(out), n


def _read_rows(variant: str) -> list[dict]:
    cfg, fname = VARIANTS[variant]
    # 1. a local copy (Kaggle dataset, or a file you uploaded)
    local = os.environ.get("FADE_SYMBOLIC_DIR")
    if local and (Path(local) / fname).exists():
        return [json.loads(l) for l in open(Path(local) / fname) if l.strip()]
    # 2. Hugging Face
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_ID, cfg, split="test")
        print(f"  source: {HF_ID} [{cfg}/test]")
        return [dict(r) for r in ds]
    except Exception as e:
        print(f"  ({HF_ID} unavailable: {str(e)[:120]}) -- trying GitHub")
    # 3. the authors' GitHub release
    import urllib.request
    url = RAW_URL.format(fname=fname)
    with urllib.request.urlopen(url, timeout=120) as r:
        txt = r.read().decode("utf-8")
    print(f"  source: {url}")
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def load_symbolic(variant: str = "gsm_symbolic_p1", n: int | None = None) -> list[dict]:
    """Rows in kaggle_run._load_split's shape, template-balanced order."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    from preprocessing import preprocess_problem, normalize_math
    raw = _read_rows(variant)
    raw.sort(key=lambda r: (int(r.get("instance", 0)), int(r.get("id", 0))))
    out, dropped, n_ann = [], 0, 0
    for r in raw:
        m = GOLD_RE.search(r["answer"])
        if not m:
            dropped += 1
            continue
        try:
            ga = float(m.group(1).replace(",", ""))
        except ValueError:
            dropped += 1
            continue
        ans, k = annotate(normalize_math(r["answer"]))
        n_ann += k
        p = preprocess_problem({"question": r["question"], "answer": ans})
        p["gold_answer"] = ga
        p["template_id"] = int(r.get("id", -1))
        p["instance"] = int(r.get("instance", -1))
        p["original_id"] = r.get("original_id")
        p["dataset"] = variant
        p["annotations_synthesised"] = True
        out.append(p)
        if n is not None and len(out) >= n:
            break
    print(f"  {variant}: {len(out)} problems ready | dropped {dropped} | "
          f"{n_ann / max(len(out), 1):.2f} synthesised checkpoints per solution")
    return out


def export_gold_jsonl(rows: list[dict], path: str) -> str:
    """Write rows for score_mpv.py / score_baselines.py --gold-jsonl."""
    with open(path, "w") as f:
        for p in rows:
            f.write(json.dumps({"question": p["question"], "answer": p["answer"],
                                "gold_answer": p["gold_answer"],
                                "template_id": p.get("template_id")}) + "\n")
    return path


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="inspect / export GSM-Symbolic")
    ap.add_argument("--variant", default="gsm_symbolic_p1", choices=sorted(VARIANTS))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--export", default=None, help="write a --gold-jsonl file here")
    a = ap.parse_args()
    rows = load_symbolic(a.variant, a.n)
    from extraction import extract_gold_checkpoints
    ck = [len(extract_gold_checkpoints(p["answer"])) for p in rows]
    print(f"  templates covered: {len({p['template_id'] for p in rows})}")
    print(f"  checkpoints per solution: mean {sum(ck)/max(len(ck),1):.2f}, "
          f"zero on {sum(c == 0 for c in ck)}/{len(ck)}")
    print("  example:\n   Q:", rows[0]["question"][:200], "\n   A:", rows[0]["answer"][:300])
    if a.export:
        export_gold_jsonl(rows, a.export)
        print(f"  wrote {a.export}")
