"""
fork_pass1_store.py -- rewind a FINISHED run back to its post-pass-1 state, so a
second retry arm can be run on the SAME first attempts without paying for them
again. NO GPU. Seconds.

WHY THIS EXISTS
---------------
The headline EDER number is:

    pass-1 28.8%  ->  final 40.3%      (+11.5 points, 172 recovered)

but nothing in that comparison isolates the DIAGNOSIS. The +11.5 could just as
easily be the gain from retrying at all with a better exemplar pool. The control
that settles it is a deferred retry with NO diagnosis and NO typed instruction --
same problems, same pool, same retry budget, same number of exemplars -- and the
only way it is a fair test is if both arms retry the IDENTICAL first attempts.

Re-running pass-1 to get there would cost ~7 GPU-h and would not even reproduce
the same traces exactly. So instead we fork the finished store:

    keep    results.jsonl  ->  pass-1 rows only
            pool.jsonl     ->  only the entries that existed at end of pass-1
            medium_queue.jsonl, bad_queue.jsonl  ->  verbatim (pass-1 state)
            config_snapshot.json
    drop    every phase == "retry" row, and the derived summary files

kaggle_run's resume rebuilds `done_ids` from pool + queues, so a forked store
reports all N problems already scored and SKIPS pass-1 entirely. It replays
retries from results.jsonl, so dropping those rows makes it generate fresh ones
under whatever --retry-mode you pass.

WHAT THE POOL FILTER IS FOR
---------------------------
The pool grows DURING retry: a rescued problem becomes GOOD and is added. On the
reference run the pool ended at 334 but only 255 of those came from pass-1. Handing
the control arm all 334 would give it 79 exemplars the typed arm did not have when
it started retrying -- a silent advantage, and the comparison would be worthless.
The filter keeps exactly the 255.

USAGE
    python fork_pass1_store.py --src stores/store_typed_train_audit2 \
                               --dst stores/store_deferred_only_audit2
    python kaggle_run.py --retry-mode generic --store-dir stores/store_deferred_only_audit2 ...
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

KEEP_VERBATIM = ("medium_queue.jsonl", "bad_queue.jsonl", "config_snapshot.json")
DROP = ("run_summary.json", "report.md", "results.csv")


def _read(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def main():
    ap = argparse.ArgumentParser(description="rewind a finished store to post-pass-1")
    ap.add_argument("--src", required=True, help="the FINISHED store to fork from")
    ap.add_argument("--dst", required=True, help="new store dir for the control arm")
    ap.add_argument("--arm", default="deferred-only",
                    help="label stamped into the forked config snapshot")
    ap.add_argument("--limit", type=int,
                    help="keep only the first K pass-1 problems (by eval order). "
                         "Use it to buy a cheaper paired run: 1068 retries is "
                         "~5 GPU-h, 400 is ~1.9 h. compare_arms pairs on shared "
                         "ids, so a subset is still a valid paired test -- just "
                         "lower-powered.")
    ap.add_argument("--force", action="store_true", help="overwrite --dst if it exists")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    if not (src / "results.jsonl").exists():
        raise SystemExit(f"no results.jsonl in {src}")
    if dst.exists():
        if not a.force:
            raise SystemExit(f"{dst} already exists -- pass --force to overwrite")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # ---- results.jsonl: pass-1 only --------------------------------------
    rows = _read(src / "results.jsonl")
    p1 = [r for r in rows if r.get("phase") == "pass1"]
    n_retry = len(rows) - len(p1)
    if not p1:
        raise SystemExit("source store has no pass-1 rows")
    # Computed from the FULL pass-1 set, BEFORE --limit truncates it. The pool is
    # the retrieval resource and must match what the typed arm had at retry time.
    p1_good_all = {r["id"] for r in p1 if r.get("label") == "GOOD"}
    if a.limit:
        p1.sort(key=lambda r: r.get("eval_index", 0))
        p1 = p1[:a.limit]
        print(f"  LIMIT: keeping the first {len(p1)} pass-1 problems for retry")
    with open(dst / "results.jsonl", "w") as f:
        for r in p1:
            f.write(json.dumps(r) + "\n")

    # ---- attempts.jsonl: pass-1 only, if present -------------------------
    lim_ids = {r["id"] for r in p1} if a.limit else None
    att = _read(src / "attempts.jsonl")
    if att:
        keep = [r for r in att if r.get("phase") in (None, "pass1")
                and (lim_ids is None or r.get("id") in lim_ids)]
        with open(dst / "attempts.jsonl", "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")

    # ---- pool.jsonl: entries that existed at END OF PASS-1 ---------------
    p1_good = p1_good_all
    pool = _read(src / "pool.jsonl")
    # NOTE: --limit deliberately does NOT shrink the pool. The typed arm ran all
    # of pass-1 before it retried anything, so at retry time it had the FULL
    # post-pass-1 pool (255 on the reference run). Filtering the pool to the
    # limited slice as well would hand the control arm 48 exemplars against the
    # typed arm's 255 -- a confound bigger than the effect being measured. The
    # limit reduces which problems are RETRIED, never the resource they retrieve
    # from.
    kept_pool = [r for r in pool if r.get("id") in p1_good]
    with open(dst / "pool.jsonl", "w") as f:
        for r in kept_pool:
            f.write(json.dumps(r) + "\n")

    for name in KEEP_VERBATIM:
        if not (src / name).exists():
            continue
        if lim_ids is not None and name.endswith("_queue.jsonl"):
            with open(dst / name, "w") as f:
                for r in _read(src / name):
                    if r.get("id") in lim_ids:
                        f.write(json.dumps(r) + "\n")
        else:
            shutil.copy2(src / name, dst / name)

    # stamp the arm so compare_arms and any later audit can tell them apart
    snap_p = dst / "config_snapshot.json"
    if snap_p.exists():
        try:
            snap = json.loads(snap_p.read_text())
            snap["FADE_FORKED_FROM"] = str(src)
            snap["FADE_ARM"] = a.arm
            snap_p.write_text(json.dumps(snap, indent=2))
        except Exception:
            pass

    med, bad = _read(dst / "medium_queue.jsonl"), _read(dst / "bad_queue.jsonl")
    pool_ids = {r["id"] for r in kept_pool}
    done = pool_ids | {r["id"] for r in med} | {r["id"] for r in bad}
    to_retry = sum(1 for r in med + bad
                   if r.get("label") in ("MEDIUM_WRONG", "BAD_WRONG"))

    print("=" * 68)
    print(f"  FORKED  {src}  ->  {dst}")
    print("=" * 68)
    print(f"  pass-1 rows kept          {len(p1)}")
    print(f"  retry rows dropped        {n_retry}")
    print(f"  pool  {len(pool)} -> {len(kept_pool)}   "
          f"({len(pool)-len(kept_pool)} were added during retry, removed)")
    if lim_ids is not None:
        print(f"  pool kept at FULL post-pass-1 size on purpose -- --limit reduces")
        print(f"  which problems are retried, not what they retrieve from")
    print(f"  medium queue              {len(med)}   {dict(Counter(r.get('label') for r in med))}")
    print(f"  bad queue                 {len(bad)}   {dict(Counter(r.get('label') for r in bad))}")
    print(f"\n  resume will report        {len(done)} problems already scored")
    print(f"  -> pass-1 SKIPPED, {to_retry} problems go to fresh retries")

    if lim_ids is not None:
        print("  (done > kept rows is expected under --limit: the pool spans the whole")
        print("   source run, while only the limited slice is queued for retry)")
    elif len(done) != len(p1):
        print(f"\n  !! {len(done)} reconstructed != {len(p1)} pass-1 rows. The queues and")
        print("     the results log disagree; check the source store before running.")
    if kept_pool and any("gold_solution" in r for r in kept_pool):
        print("  note: pool.jsonl carries gold on disk (by design) -- TraceStore.exemplars()")
        print("        strips it before anything reaches a prompt.")
    print("\n  next:")
    print(f"    python kaggle_run.py --retry-mode generic --store-dir {dst} \\")
    print(f"        --run-name <name> --n-problems {len(p1)} --model <model>")


if __name__ == "__main__":
    main()
