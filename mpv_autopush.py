"""
mpv_autopush.py -- start autopush for a run that grows `mpv_probes.jsonl`
instead of `results.jsonl`.

THE BUG THIS EXISTS TO FIX
--------------------------
`autopush.AutoPush.push_now()` uses TWO different row counts and they must agree:

    n    = _rows(self.store)              "is there anything new to push?"
    kept = _safe_copy(store, snap)[0]     "how many rows did I just push?"  -> _last_pushed

`_rows` and `_safe_copy` both count **results.jsonl**.  An MPV run never touches
results.jsonl -- it appends to mpv_probes.jsonl.  Patching only `_rows` (the
obvious fix) makes the two disagree, and the observed failure is:

    [autopush ctctest_audit2_a1] pushed 1319 rows          <- kept  = results.jsonl
    [autopush ctctest_audit2_a1] snapshot has 69 rows but 1319 were already
                                 pushed -- refusing to shrink the checkpoint

`_last_pushed` was set to 1319 from results.jsonl while `n` came back as 69 probe
rows, so the shrink guard fired on every subsequent cycle and the run checkpointed
exactly once.  Nothing is lost -- the first push carried the whole store tree,
including mpv_probes.jsonl -- but there are no further checkpoints, which is
precisely the protection the guard exists to provide.

Both functions must be patched together.  That is all this module does.

USAGE (replaces `autopush.start(...)` in the notebook)
    import mpv_autopush
    ap = mpv_autopush.start(STORE, RUN, GH, minutes=10)
    try:
        ...run...
    finally:
        ap.stop()
"""
from __future__ import annotations

import os

import autopush

COUNT_FILE = "mpv_probes.jsonl"
_patched = False


def probe_rows(store) -> int:
    f = os.path.join(str(store), COUNT_FILE)
    if not os.path.exists(f):
        return 0
    n = 0
    with open(f, errors="replace") as fh:
        for ln in fh:
            if ln.strip():
                n += 1
    return n


def patch(count_file: str = COUNT_FILE) -> None:
    """Make autopush count `count_file` everywhere it currently counts
    results.jsonl.  Idempotent."""
    global _patched, COUNT_FILE
    COUNT_FILE = count_file
    if _patched:
        return
    original_safe_copy = autopush._safe_copy

    def _safe_copy(src, dst):
        kept, dropped = original_safe_copy(src, dst)   # still sanitises every *.jsonl
        return probe_rows(dst), dropped                # ... but report OUR row count

    autopush._rows = probe_rows
    autopush._safe_copy = _safe_copy
    _patched = True
    print(f"  [mpv_autopush] autopush now checkpoints on {COUNT_FILE} row growth")


def start(store, run, token, minutes: int = 10, verbose: bool = True,
          count_file: str = COUNT_FILE):
    patch(count_file)
    return autopush.start(store, run, token, minutes=minutes, verbose=verbose)


def push_now(store, run, token, final: bool = True, count_file: str = COUNT_FILE):
    """One-shot push, e.g. after scoring writes mpv_report.json."""
    patch(count_file)
    return autopush.AutoPush(store, run, token).push_now(final=final)
