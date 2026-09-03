"""
mpv_offline_check.py -- zero-GPU validation of the MPV executor on a TRAIN store.

For every pass-1 trace with gold: build the probes, execute the trace's program on
the perturbed question, and compare with GSM8K's own <<...>> annotations executed
on the same perturbation (ctc.perturbed_gold -- SCORING ONLY, never at inference).

Reports P(program prediction == gold') for correct vs wrong traces.  A large gap
is the precondition for MPV to work at all: it says a correct trace's chain IS a
faithful program of the question (extractor fidelity), and a wrong trace's is not.

Measured on store_typed_train_audit2_d63 pass-1 (1,500):
    correct traces   499 match / 143 mismatch   -> 77.7%
    wrong traces      97 match / 1232 mismatch  ->  7.3%
    testable         995/1500 (66%)

USAGE
    python mpv_offline_check.py --store stores/store_typed_train_audit2
"""
from __future__ import annotations
import argparse, json, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mpv, ctc as CTC
from extraction import extract_final_answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--n-probes", type=int, default=2)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.store) / "results.jsonl") if l.strip()]
    p1 = [r for r in rows if r.get("phase") == "pass1" and r.get("gold_solution")]
    st, fid = collections.Counter(), collections.Counter()
    for r in p1:
        q, t = r["question"], r["trace"]
        ans = extract_final_answer(t); c = bool(r["signals"]["correct"])
        probes = mpv.make_probes(q, [t], n_probes=a.n_probes)
        st["n"] += 1; st["testable"] += bool(probes)
        for p in probes:
            pred = mpv.execute(t, p.subs, ans)
            g = CTC.perturbed_gold(r["gold_solution"], p.target_value, p.new_value)
            if pred is None or g is None:
                fid[("abstain", c)] += 1; continue
            fid[("match" if mpv._close(pred, g) else "mismatch", c)] += 1
    print(f"  pass-1 traces {st['n']}   testable {st['testable']} ({st['testable']/max(st['n'],1):.1%})")
    print("  program prediction vs GOLD answer of the perturbed question:")
    for c in (True, False):
        m, mm, ab = fid[("match", c)], fid[("mismatch", c)], fid[("abstain", c)]
        print(f"    {'correct' if c else 'wrong  '} traces: match {m:>5}  mismatch {mm:>5}  abstain {ab:>4}"
              f"   -> P(pred == gold') = {m/max(m+mm,1):.3f}")


if __name__ == "__main__":
    main()
