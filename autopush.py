"""
autopush.py -- push the store to GitHub every N minutes WHILE a run is going.

WHY
---
The notebooks only pushed in the `finally:` of the run cell. That covers a clean
finish and a stopped cell, but not a lost KERNEL -- a session timeout, toggling the
Internet switch (which restarts the kernel), a closed tab, "Stop session". One
generic run reached 150 problems and lost all of it that way.

This runs a daemon thread beside the run. Every `minutes` it snapshots the store and
pushes. Nothing is lost beyond the last interval.

THE PART THAT NEEDS CARE
------------------------
kaggle_run.py APPENDS to results.jsonl continuously, so a naive copy can catch a
half-written final line. The copy would then be corrupt, and -- worse -- it would
push a corrupt store over a good one. `_safe_copy` therefore rewrites results.jsonl
keeping only lines that parse as JSON, and reports how many it dropped (normally 0
or 1: the line being written at that instant).

It also refuses to push a snapshot with FEWER rows than the last one it pushed, so
a truncated read can never overwrite a larger good checkpoint.

The repo is cloned ONCE and reused, so a 10-minute cadence costs a copy and a small
push, not a fresh clone each time.

USAGE (in the run cell)
    import autopush
    ap = autopush.start(STORE, RUN, GH, minutes=10)
    try:
        ...run...
    finally:
        ap.stop()            # pushes one final time before returning
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

BRANCH = "results"
REPO_URL = "github.com/S2V3/fade.git"


def _rows(p):
    f = os.path.join(p, "results.jsonl")
    return sum(1 for _ in open(f)) if os.path.exists(f) else 0


def _sanitise(path):
    """Drop any partially-written JSON line. -> (kept, dropped)"""
    if not os.path.exists(path):
        return 0, 0
    good, dropped = [], 0
    for ln in open(path, errors="replace"):
        if not ln.strip():
            continue
        try:
            json.loads(ln)
            good.append(ln if ln.endswith("\n") else ln + "\n")
        except Exception:
            dropped += 1
    with open(path, "w") as out:
        out.writelines(good)
    return len(good), dropped


def _safe_copy(src, dst):
    """Copy a store, dropping any partially-written trailing JSON line.

    [AUDIT D75] every *.jsonl is sanitised, not just results.jsonl. Stage 2
    writes retries.jsonl in a second pass, and a snapshot taken mid-append used
    to push a torn last line. The resume path tolerates it -- that retry is
    simply redone -- but a checkpoint should never carry known-corrupt bytes.
    """
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    for name in sorted(os.listdir(dst)):
        if name.endswith(".jsonl") and name != "results.jsonl":
            k, d = _sanitise(os.path.join(dst, name))
            if d:
                print(f"    [autopush] {name}: dropped {d} torn line(s)")
    return _sanitise(os.path.join(dst, "results.jsonl"))


class AutoPush:
    def __init__(self, store, run, token, minutes=10, verbose=True):
        self.store = str(store)
        self.run = run
        self.auth = f"https://{token}@{REPO_URL}"
        self.interval = max(60, int(minutes * 60))
        self.verbose = verbose
        self.work = f"/kaggle/working/_autopush_{run}"
        self.snap = f"/kaggle/working/_autosnap_{run}"
        self._stop = threading.Event()
        self._thread = None
        self._last_pushed = 0
        self._n_pushes = 0
        self._ready = False

    # ------------------------------------------------------------- internals
    def _clone_once(self):
        if self._ready and os.path.isdir(os.path.join(self.work, ".git")):
            return True
        shutil.rmtree(self.work, ignore_errors=True)
        r = subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                            self.auth, self.work], capture_output=True, text=True)
        if r.returncode:
            r = subprocess.run(["git", "clone", "--depth", "1", self.auth, self.work],
                               capture_output=True, text=True)
            if r.returncode:
                self._log(f"clone failed: {r.stderr.strip()[:160]}")
                return False
            subprocess.run(["git", "checkout", "-b", BRANCH], cwd=self.work,
                           capture_output=True, text=True)
        for c in (["git", "config", "user.email", "fade@kaggle"],
                  ["git", "config", "user.name", "fade"]):
            subprocess.run(c, cwd=self.work, capture_output=True, text=True)
        self._ready = True
        return True

    def _log(self, msg):
        if self.verbose:
            print(f"  [autopush {self.run}] {msg}", flush=True)

    def push_now(self, final=False):
        if not os.path.isdir(self.store):
            return False
        n = _rows(self.store)
        if n == 0:
            return False
        if n < self._last_pushed and not final:
            self._log(f"snapshot has {n} rows but {self._last_pushed} were already "
                      f"pushed -- refusing to shrink the checkpoint")
            return False
        if n == self._last_pushed and not final:
            return False
        if not self._clone_once():
            return False

        kept, dropped = _safe_copy(self.store, self.snap)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        names = [f"store_{self.run}_inprogress"]
        if final:
            names.append(f"store_{self.run}_final_{stamp}")
        for nm in names:
            dest = os.path.join(self.work, "results", nm)
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(self.snap, dest)

        subprocess.run(["git", "add", "-A"], cwd=self.work, capture_output=True, text=True)
        subprocess.run(["git", "commit", "-m", f"{self.run} {stamp} ({kept} rows)"],
                       cwd=self.work, capture_output=True, text=True)
        p = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=self.work,
                           capture_output=True, text=True)
        if p.returncode:
            subprocess.run(["git", "pull", "--rebase", "origin", BRANCH], cwd=self.work,
                           capture_output=True, text=True)
            p = subprocess.run(["git", "push", "-u", "origin", BRANCH], cwd=self.work,
                               capture_output=True, text=True)
        if p.returncode == 0:
            self._last_pushed = kept
            self._n_pushes += 1
            self._log(f"pushed {kept} rows"
                      + (f" (dropped {dropped} partial line)" if dropped else "")
                      + (" [FINAL]" if final else ""))
            return True
        self._log(f"PUSH FAILED: {(p.stderr or '').strip()[:160]} -- retrying next cycle")
        self._ready = False          # force a fresh clone next time
        return False

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.push_now()
            except Exception as e:
                self._log(f"error: {e!r} -- continuing")

    # ---------------------------------------------------------------- public
    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log(f"started, every {self.interval//60} min -> {self.store}")
        return self

    def stop(self, final=True):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        ok = self.push_now(final=final)
        self._log(f"stopped after {self._n_pushes} checkpoint(s); "
                  f"final push {'ok' if ok else 'FAILED -- run rescue_push.py'}")
        return ok


def start(store, run, token, minutes=10, verbose=True):
    return AutoPush(store, run, token, minutes, verbose).start()
