"""
export_results.py -- after a run, package everything in store/ so it survives the
Kaggle session: a local zip you can download, and (optionally) a commit+push to a
GitHub results branch/folder so the outputs live in the same repo you cloned from.

Usage (download only -- always works):
    python export_results.py

Usage (also push to GitHub):
    python export_results.py --push \\
        --repo https://github.com/S2V3/fade.git \\
        --branch results --token <GITHUB_PAT>
On Kaggle put the PAT in a Secret and read it in the notebook; never hard-code it.

What it writes:
    /kaggle/working/fade_results_<timestamp>.zip   (download this)
    (optional) commits store/ into <repo>@<branch>/results/<timestamp>/
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
STORE = REPO / "store"
ARTIFACTS = REPO / "artifacts"


def _stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _summarise() -> str:
    """One-line headline for the commit message / manifest, if a summary exists."""
    rs = STORE / "run_summary.json"
    if not rs.exists():
        return "FADE results (no run_summary.json)"
    s = json.loads(rs.read_text())
    return (f"FADE {s.get('model','?')} {s.get('strategy_name','?')} "
            f"split={s.get('split','?')} n={s.get('n_problems','?')} "
            f"pass1={s.get('pass1_accuracy','?')} final={s.get('final_accuracy','?')}")


def make_zip(dest_dir: Path) -> Path:
    if not STORE.exists():
        sys.exit("no store/ directory -- run kaggle_run.py first")
    stamp = _stamp()
    staging = REPO / f"_export_{stamp}"
    staging.mkdir(exist_ok=True)
    shutil.copytree(STORE, staging / "store", dirs_exist_ok=True)
    if ARTIFACTS.exists():                       # needed for resume reproducibility
        shutil.copytree(ARTIFACTS, staging / "artifacts", dirs_exist_ok=True)
    (staging / "MANIFEST.txt").write_text(
        _summarise() + "\n\nfiles:\n" +
        "\n".join(sorted(p.name for p in STORE.iterdir())) + "\n")
    zip_base = dest_dir / f"fade_results_{stamp}"
    shutil.make_archive(str(zip_base), "zip", staging)
    shutil.rmtree(staging, ignore_errors=True)
    zip_path = zip_base.with_suffix(".zip")
    print(f"  wrote {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")
    print("  -> download it from the Kaggle output panel, or save as a Dataset")
    return zip_path


def _run(cmd, cwd, check=True):
    print("  $", " ".join(c if "token" not in c.lower() else "***" for c in cmd))
    return subprocess.run(cmd, cwd=cwd, check=check,
                          capture_output=True, text=True)


def push_to_github(repo_url: str, branch: str, token: str) -> None:
    if not token:
        sys.exit("--push needs --token (a GitHub Personal Access Token)")
    stamp = _stamp()
    work = REPO / f"_gitpush_{stamp}"
    work.mkdir(exist_ok=True)
    # authenticated URL (token injected; never printed)
    auth_url = repo_url.replace("https://", f"https://{token}@")
    try:
        # shallow clone just the target branch (create it if missing)
        r = _run(["git", "clone", "--depth", "1", "--branch", branch,
                  auth_url, str(work / "repo")], cwd=REPO, check=False)
        if r.returncode != 0:                    # branch doesn't exist yet
            _run(["git", "clone", "--depth", "1", auth_url, str(work / "repo")], cwd=REPO)
            _run(["git", "checkout", "-b", branch], cwd=work / "repo")
        dest = work / "repo" / "results" / stamp
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(STORE, dest / "store", dirs_exist_ok=True)
        (dest / "MANIFEST.txt").write_text(_summarise() + "\n")
        _run(["git", "config", "user.email", "fade-bot@local"], cwd=work / "repo")
        _run(["git", "config", "user.name", "fade-bot"], cwd=work / "repo")
        _run(["git", "add", "-A"], cwd=work / "repo")
        _run(["git", "commit", "-m", f"results {stamp}: {_summarise()}"], cwd=work / "repo")
        _run(["git", "push", "-u", "origin", branch], cwd=work / "repo")
        print(f"  pushed to {repo_url} @ {branch}/results/{stamp}/")
    except subprocess.CalledProcessError as e:
        print("  git failed:\n", e.stdout, e.stderr)
        print("  (the local zip was still written -- use that.)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/kaggle/working",
                    help="where to write the zip (default Kaggle output dir)")
    ap.add_argument("--push", action="store_true", help="also push to GitHub")
    ap.add_argument("--repo", default="https://github.com/S2V3/fade.git")
    ap.add_argument("--branch", default="results")
    ap.add_argument("--token", default=None, help="GitHub PAT (use a Kaggle Secret)")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print("Exporting store/ ...")
    print(" ", _summarise())
    make_zip(dest)
    if args.push:
        push_to_github(args.repo, args.branch, args.token)
    print("done.")


if __name__ == "__main__":
    main()