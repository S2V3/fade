"""
nm_store.py -- store plumbing for the camera-ready runs: restore, checkpoint, shard, merge.

Why not autopush.py: it counts ONE file (results.jsonl). A chained job writes
results.jsonl, then mpv_probes.jsonl, then baselines.jsonl into the same store,
and mpv_autopush exists only because counting the wrong file silently stopped
checkpoints once already. Here the measure of progress is the total number of
JSON lines across EVERY *.jsonl in the store, which only ever grows, whatever
stage is running. One shared clone of the results branch serves every pusher in
the notebook, guarded by a lock, so two GPU lanes never run git at the same time.

    restore(run, dest, token)        newest/biggest results/store_<run>_* -> dest
    restore_dir(name, dest, token)   an exact results/<name> directory -> dest
    Pusher(store, run, token).start() / .stop()
    merge(shard_dirs, dest)          concatenate shard stores (dedup by key)
    subset(src, dest, n)             first n rows by id, and everything keyed to them
"""
from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

REPO_URL = "github.com/S2V3/fade.git"
BRANCH = "results"
WORK = Path(os.environ.get("NM_WORK", "/kaggle/working"))
CLONE = WORK / "_nm_results"
_LOCK = threading.Lock()

# rows are de-duplicated on these keys when shards are merged
DEDUP_KEYS = {
    "results.jsonl": lambda d: (d.get("id"), d.get("phase"), d.get("iter"),
                                (d.get("trace") or "")[:80]),
    "mpv_probes.jsonl": lambda d: (d.get("id"), d.get("perturbed_question")),
    "baselines.jsonl": lambda d: (d.get("id"), d.get("arm")),
}


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def jsonl_lines(store) -> int:
    """Progress measure: non-empty lines over every *.jsonl in the store."""
    n = 0
    p = Path(store)
    if not p.is_dir():
        return 0
    for f in p.glob("*.jsonl"):
        with open(f, errors="replace") as fh:
            n += sum(1 for ln in fh if ln.strip())
    return n


def _sanitise_copy(src, dst):
    """Copy a store, dropping torn (half-written) JSON lines from every *.jsonl."""
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    dropped = 0
    for f in Path(dst).glob("*.jsonl"):
        good = []
        for ln in open(f, errors="replace"):
            if not ln.strip():
                continue
            try:
                json.loads(ln)
                good.append(ln if ln.endswith("\n") else ln + "\n")
            except Exception:
                dropped += 1
        f.write_text("".join(good))
    return dropped


def _clone(token, refresh=False):
    auth = f"https://{token}@{REPO_URL}" if token else f"https://{REPO_URL}"
    if (CLONE / ".git").is_dir():
        if refresh:
            _run(["git", "fetch", "--depth", "1", "origin", BRANCH], cwd=CLONE)
            _run(["git", "reset", "--hard", f"origin/{BRANCH}"], cwd=CLONE)
        return CLONE
    shutil.rmtree(CLONE, ignore_errors=True)
    r = _run(["git", "clone", "--depth", "1", "--branch", BRANCH, auth, str(CLONE)])
    if r.returncode:
        raise RuntimeError(f"clone of {BRANCH} failed: {r.stderr.strip()[:200]}")
    _run(["git", "config", "user.email", "fade@kaggle"], cwd=CLONE)
    _run(["git", "config", "user.name", "fade"], cwd=CLONE)
    return CLONE


def restore_dir(name, dest, token, refresh=True) -> bool:
    """Copy results/<name> (an exact directory name) to dest."""
    with _LOCK:
        c = _clone(token, refresh=refresh)
        src = c / "results" / name
        if not src.is_dir():
            print(f"  [nm_store] results/{name} not on the branch")
            return False
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
    print(f"  [nm_store] restored results/{name} -> {dest} ({jsonl_lines(dest)} lines)")
    return True


def restore(run, dest, token, refresh=True) -> int:
    """Restore the largest results/store_<run>_* checkpoint into dest, unless the
    local copy is already at least as large. Returns the line count now in dest."""
    local = jsonl_lines(dest)
    with _LOCK:
        c = _clone(token, refresh=refresh)
        res = c / "results"
        cands = []
        if res.is_dir():
            for d in res.iterdir():
                if d.is_dir() and d.name.startswith(f"store_{run}_"):
                    cands.append((jsonl_lines(d), d.name, d))
        cands.sort(reverse=True)
        if cands and cands[0][0] > local:
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(cands[0][2], dest)
            print(f"  [nm_store] {run}: restored {cands[0][1]} ({cands[0][0]} lines; local had {local})")
            return cands[0][0]
    print(f"  [nm_store] {run}: local {local} lines, "
          f"{'nothing larger on the branch' if cands else 'nothing on the branch'}")
    return local


class Pusher:
    """Checkpoint a store to results/store_<run>_inprogress every `minutes`."""

    def __init__(self, store, run, token, minutes=10, verbose=True):
        self.store, self.run, self.token = str(store), run, token
        self.interval = max(60, int(minutes * 60))
        self.verbose = verbose
        self._stop = threading.Event()
        self._thread = None
        self.last = 0
        self.snap = WORK / f"_nm_snap_{run}"

    def _log(self, m):
        if self.verbose:
            print(f"  [push {self.run}] {m}", flush=True)

    def push_now(self, final=False) -> bool:
        n = jsonl_lines(self.store)
        if n == 0 or (n <= self.last and not final):
            return False
        with _LOCK:
            dropped = _sanitise_copy(self.store, self.snap)
            n = jsonl_lines(self.snap)
            if n < self.last and not final:
                self._log(f"snapshot {n} < pushed {self.last}; refusing to shrink")
                return False
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            names = [f"store_{self.run}_inprogress"] + ([f"store_{self.run}_final_{stamp}"] if final else [])
            p = None
            for attempt in range(3):
                # refresh to the remote tip first: another notebook may have pushed
                try:
                    c = _clone(self.token, refresh=True)
                except Exception as e:
                    self._log(f"clone failed: {e}")
                    return False
                for nm in names:
                    d = c / "results" / nm
                    shutil.rmtree(d, ignore_errors=True)
                    shutil.copytree(self.snap, d)
                _run(["git", "add", "-A", "results"], cwd=c)
                _run(["git", "commit", "-m", f"{self.run} {stamp} ({n} lines)"], cwd=c)
                p = _run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=c)
                if p.returncode == 0:
                    break
        if p is not None and p.returncode == 0:
            self.last = n
            self._log(f"pushed {n} lines" + (f" (dropped {dropped} torn)" if dropped else "")
                      + (" [FINAL]" if final else ""))
            return True
        self._log(f"PUSH FAILED: {(p.stderr or '').strip()[:160]}")
        return False

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.push_now()
            except Exception as e:
                self._log(f"error {e!r}")

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._log(f"every {self.interval // 60} min -> results/store_{self.run}_inprogress")
        return self

    def stop(self, final=True):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.push_now(final=final)


def merge(shards, dest) -> Path:
    """Concatenate shard stores into dest. *.jsonl rows are de-duplicated by the
    keys above; *.json files come from the first shard, and any disagreement in a
    provenance file between shards is reported (it would mean the shards did not
    run the same system)."""
    shards = [Path(s) for s in shards if Path(s).is_dir()]
    if not shards:
        raise SystemExit("merge: no shard directories exist")
    dest = Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    names = sorted({f.name for s in shards for f in s.glob("*.jsonl")})
    for nm in names:
        keyf = DEDUP_KEYS.get(nm)
        seen, rows = set(), []
        for s in shards:
            f = s / nm
            if not f.exists():
                continue
            for ln in open(f, errors="replace"):
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                k = keyf(d) if keyf else ln
                if k in seen:
                    continue
                seen.add(k)
                rows.append(d)
        if nm == "results.jsonl":
            rows.sort(key=lambda d: str(d.get("id")))
        with open(dest / nm, "w") as out:
            for d in rows:
                out.write(json.dumps(d) + "\n")
    for f in shards[0].iterdir():
        if f.is_file() and not f.name.endswith(".jsonl"):
            shutil.copy2(f, dest / f.name)
    for prov in ("mpv_provenance.json", "baselines_provenance.json"):
        blobs = []
        for s in shards:
            try:
                b = json.load(open(s / prov))
                blobs.append({k: b.get(k) for k in ("pool_size", "pool_fp", "seeds_fp", "model")})
            except Exception:
                pass
        if len({json.dumps(b, sort_keys=True) for b in blobs}) > 1:
            print(f"  !! [merge] {prov} differs between shards: {blobs}")
    (dest / "nm_merge.json").write_text(json.dumps(
        {"shards": [s.name for s in shards],
         "lines": {s.name: jsonl_lines(s) for s in shards}}, indent=2))
    print(f"  [merge] {len(shards)} shards -> {dest} ({jsonl_lines(dest)} lines)")
    return dest


def subset(src, dest, n) -> Path:
    """First n rows (by id) of a stage-3 store, and every jsonl row keyed to them."""
    src, dest = Path(src), Path(dest)
    rows = [json.loads(l) for l in open(src / "results.jsonl") if l.strip()]
    rows.sort(key=lambda d: d["id"])
    keep = {r["id"] for r in rows[:n]}
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    for f in src.iterdir():
        if f.suffix == ".jsonl":
            with open(dest / f.name, "w") as out:
                for ln in open(f):
                    if ln.strip() and json.loads(ln).get("id") in keep:
                        out.write(ln if ln.endswith("\n") else ln + "\n")
        elif f.is_file():
            shutil.copy2(f, dest / f.name)
    (dest / "nm_subset.json").write_text(json.dumps({"src": src.name, "n": len(keep)}))
    print(f"  [subset] {src.name}: first {len(keep)} rows -> {dest}")
    return dest


def push_dir(src, name, token, message=None) -> bool:
    """Copy any directory to results/<name> and push it (reports, figures)."""
    with _LOCK:
        c = _clone(token, refresh=True)
        d = c / "results" / name
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(src, d)
        _run(["git", "add", "-A", "results"], cwd=c)
        _run(["git", "commit", "-m", message or f"{name} {datetime.datetime.now():%Y%m%d_%H%M%S}"], cwd=c)
        p = _run(["git", "push", "origin", f"HEAD:{BRANCH}"], cwd=c)
    print(f"  [nm_store] push results/{name}: {'ok' if p.returncode == 0 else p.stderr.strip()[:160]}")
    return p.returncode == 0
