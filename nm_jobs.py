"""
nm_jobs.py -- every GPU job for the camera-ready, as lanes that run side by side.

A Kaggle T4x2 session has two GPUs. A 4B model in fp16 fits on one, so each
notebook runs TWO lanes in parallel (CUDA_VISIBLE_DEVICES=0 and =1). Llama-2-7B
needs both GPUs, so its notebook runs one lane over both.

    notebook        lanes                                                    starts
    --------------  -------------------------------------------------------  -----------------
    NB1 qwen        GPU0: train shard 0 -> [wait shard 1] -> prepare          now
                          (merge, CTC, budget, fidelity) -> held-out chain
                          -> GSM-Sym P1 shard 1 chain
                    GPU1: train shard 1 -> [wait prepare] -> P1 shard 0 chain
    NB2 llama       GPU0+1: GSM8K baselines (first N of the existing          now
                          stage-3 store) -> GSM-Sym P1 chain
    NB3 qwen test   GPU0: GSM8K test shard 0 chain                            when NB1 prints
                    GPU1: GSM8K test shard 1 chain                            PREPARE DONE
    NB4 score       CPU: merge shards, score_mpv, score_baselines, tables     when runs finish

A "chain" is: run_ctc_validate.py (pass-1 + detector + flagged cure, gold-free)
-> run_mpv.py (probes) -> run_baselines.py (resample + generic retry). Every step
resumes from its own output, and every store is checkpointed to the results branch
every 10 minutes, so a dead session costs at most 10 minutes: re-run the same
notebook and each lane restores and continues.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import nm_store as NS                                   # noqa: E402

WORK = NS.WORK
STORES = WORK / "stores"
LOGS = WORK / "nm_logs"
MARKS = WORK / "_nm_markers"

LLAMA = "meta-llama/Llama-2-7b-chat-hf"
# the exact directories the paper's Llama-2 numbers came from (verified by the
# pool fingerprint b44be787fea3 recorded in the stage-3 store's mpv_provenance.json)
LLAMA_POOL_DIR = "store_typed_train_audit2_d40_inprogress"
LLAMA_POOL_FP = "b44be787fea3"
LLAMA_CTC_DIR = "ctc2_audit2_latest"
LLAMA_TEST_DIR = "store_ctctest_audit2_a1_final_20260903_144740"
LLAMA_HELDOUT_DIR = "store_ctcval_train_audit2_a2_final_20260902_050134"


@dataclass
class Cfg:
    model: str = "Qwen/Qwen3.5-4B"
    dtype: str = "fp16"                 # from nm_smoke.py
    n_train: int = 500                  # GSM8K train, split over 2 shards
    n_heldout: int = 200                # GSM8K train, after the train slice (E1)
    n_test: int = 400                   # first N GSM8K test questions (paired with Llama-2)
    n_p1: int = 400                     # GSM-Symbolic P1, 4 instances x 100 templates
    p1_variant: str = "gsm_symbolic_p1"
    # Generation caps. Llama-2 ran at 400 (train) / 320 (eval) and rarely hit them.
    # Qwen3.5 writes 2-3x longer traces: at 400, 28% of its train traces hit the cap
    # and were right 13% of the time -- truncation, not reasoning. The rule we keep
    # across models is "the cap must not bind", so non-Llama models default to 1024.
    train_max_tok: int | None = None    # None -> 400 for Llama-2, 1024 otherwise
    eval_max_tok: int | None = None     # None -> 320 for Llama-2, 1024 otherwise
    run_version: str = "v2"             # v1 = the 26 Sep Qwen run (3072-token prompt cut, 400 cap)
    n_probes: int = 1                   # the paper's results used ~1 probe per question
    flag_budget: float | None = None    # None -> nm_pick_budget.py decides (Qwen)
    llama_flag_budget: float = 0.40     # what Llama-2 stage 3 shipped
    gh_token: str | None = None
    hf_secret: str = "HF_TOKEN"
    push_minutes: int = 10

    def __post_init__(self):
        legacy = "llama-2" in self.model.lower()
        if self.train_max_tok is None:
            self.train_max_tok = 400 if legacy else 1024
        if self.eval_max_tok is None:
            self.eval_max_tok = 320 if legacy else 1024

    @property
    def tag(self) -> str:
        m = self.model.lower()
        if "qwen3.5" in m:
            base = "qwen35_" + m.split("-")[-1].replace(".", "")          # qwen35_4b
        else:
            base = "".join(c for c in m.split("/")[-1] if c.isalnum())[:16]
        return base if self.run_version in ("", "v1") else f"{base}_{self.run_version}"


# =============================================================================
# plumbing
# =============================================================================
def mark(name):
    MARKS.mkdir(parents=True, exist_ok=True)
    (MARKS / name).write_text(time.strftime("%H:%M:%S"))


def is_marked(name):
    return (MARKS / name).exists()


class Lane(threading.Thread):
    def __init__(self, name, gpus, steps, env=None):
        super().__init__(daemon=True)
        self.name_, self.gpus, self.steps = name, gpus, steps
        self.env = dict(os.environ)
        self.env["CUDA_VISIBLE_DEVICES"] = gpus
        self.env["PYTHONUNBUFFERED"] = "1"
        self.env.update(env or {})
        self.ok, self.error = True, None
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = open(LOGS / f"{name}.log", "a")

    def say(self, msg):
        line = f"[{self.name_} {time.strftime('%H:%M')}] {msg}"
        print(line, flush=True)
        self.log.write(line + "\n"); self.log.flush()

    def sh(self, cmd, env=None, quiet_prefixes=()):
        env_ = dict(self.env); env_.update(env or {})
        self.say(f"$ {cmd}")
        p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env_, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        for ln in p.stdout:
            ln = ln.rstrip()
            self.log.write(ln + "\n")
            if ln and not ln.startswith(quiet_prefixes):
                print(f"  {self.name_}| {ln}", flush=True)
        p.wait()
        self.log.flush()
        if p.returncode:
            raise RuntimeError(f"exit {p.returncode}: {cmd[:120]}")
        return p.returncode

    def run(self):
        t0 = time.time()
        for st in self.steps:
            try:
                st(self)
            except Exception as e:
                self.ok, self.error = False, repr(e)
                self.say(f"STOPPED on {getattr(st, '__name__', st)}: {e}")
                mark(f"FAILED_{self.name_}")      # so a lane waiting on this one stops too
                break
        self.say(f"lane finished ({'ok' if self.ok else 'FAILED'}) in {(time.time()-t0)/3600:.2f} h")


def run_lanes(lanes):
    for p in (MARKS.glob("FAILED_*") if MARKS.exists() else []):
        p.unlink()                                  # a re-run starts with a clean slate
    for ln in lanes:
        ln.start()
    for ln in lanes:
        ln.join()
    bad = [ln.name_ for ln in lanes if not ln.ok]
    print("\n" + "=" * 70)
    print("  ALL LANES OK" if not bad else f"  FAILED LANES: {bad} -- read {LOGS}/<lane>.log; "
          "re-running this cell resumes every step")
    print("=" * 70)
    return not bad


def step(fn):
    """Decorator so each step shows up by name in the lane log."""
    return fn


def with_push(run, store, cfg, body):
    """Restore <run> from the branch, checkpoint it while `body` runs, push final."""
    def _s(lane):
        NS.restore(run, store, cfg.gh_token)
        pu = NS.Pusher(store, run, cfg.gh_token, cfg.push_minutes).start() if cfg.gh_token else None
        try:
            body(lane)
        finally:
            if pu:
                pu.stop()
    _s.__name__ = f"run[{run}]"
    return _s


def wait_for(name, poll=30):
    def _s(lane):
        lane.say(f"waiting for {name} ...")
        while not is_marked(name):
            failed = [p.name for p in MARKS.glob("FAILED_*")] if MARKS.exists() else []
            if failed:
                raise RuntimeError(f"not waiting for {name}: {failed} failed -- fix and re-run the cell")
            time.sleep(poll)
        lane.say(f"{name} is done, continuing")
    _s.__name__ = f"wait[{name}]"
    return _s


def py(lane, script, args, env=None):
    launcher = os.environ.get("NM_LAUNCHER", "nm_patch.py")   # tests swap in a mock
    return lane.sh(f"python -u {launcher} {script} {args}", env=env)


# =============================================================================
# Qwen: train shards and prepare
# =============================================================================
def train_shard(cfg, k):
    run = f"nm_{cfg.tag}_train_s{k}"
    store = STORES / f"store_{run}"
    half = cfg.n_train // 2
    def body(lane):
        py(lane, "kaggle_run.py",
           f"--retry-mode typed+neg --negatives 2 --store-dir {store} --run-name {run} "
           f"--n-problems {half} --eval-offset {k * half} --model {cfg.model} --iterations 1 "
           f"--max-new-tokens {cfg.train_max_tok} --show-every 25 --secret-name {cfg.hf_secret}")
        mark(run)
    return with_push(run, store, cfg, body)


def paths(cfg):
    return {"train": STORES / f"store_nm_{cfg.tag}_train",
            "ctc": STORES / f"nm_{cfg.tag}_ctc"}


def prepare(cfg):
    """CPU: merge the two train shards, train the detector, pick the budget,
    and run the executor-fidelity check. Pushes the merged store and the detector."""
    def _s(lane):
        P = paths(cfg)
        shards = [STORES / f"store_nm_{cfg.tag}_train_s{k}" for k in (0, 1)]
        NS.merge(shards, P["train"])
        P["ctc"].mkdir(parents=True, exist_ok=True)
        lane.sh(f"python -u ctc_data.py --stores {P['train']} --out {P['ctc']}/ctc_dataset.jsonl")
        lane.sh(f"python -u ctc_train2.py --data {P['ctc']}/ctc_dataset.jsonl --out-dir {P['ctc']}")
        lane.sh(f"python -u nm_pick_budget.py --ctc-report {P['ctc']}/ctc2_report.json "
                f"--train-store {P['train']} --out {P['ctc']}/budget.json")
        lane.sh(f"python -u mpv_offline_check.py --store {P['train']} --n-probes 2 "
                f"| tee {P['ctc']}/mpv_offline_check.txt")
        (P["ctc"] / "nm_cfg.json").write_text(json.dumps(
            {"model": cfg.model, "dtype": cfg.dtype, "n_train": cfg.n_train,
             "n_heldout": cfg.n_heldout, "n_test": cfg.n_test, "n_p1": cfg.n_p1,
             "p1_variant": cfg.p1_variant, "train_max_tok": cfg.train_max_tok,
             "eval_max_tok": cfg.eval_max_tok, "n_probes": cfg.n_probes,
             "run_version": cfg.run_version}, indent=2))
        if cfg.gh_token:
            NS.Pusher(P["train"], f"nm_{cfg.tag}_train", cfg.gh_token).push_now(final=True)
            NS.Pusher(P["ctc"], f"nm_{cfg.tag}_ctc", cfg.gh_token).push_now(final=True)
        mark(f"nm_{cfg.tag}_prepare")
        lane.say("PREPARE DONE -- NB3 (Qwen GSM8K test) can start now")
    _s.__name__ = "prepare"
    return _s


def ensure_prepared(cfg):
    """For NB3: restore the merged train store and detector from the branch."""
    def _s(lane):
        P = paths(cfg)
        if not (P["ctc"] / "ctc2.joblib").exists():
            NS.restore(f"nm_{cfg.tag}_ctc", P["ctc"], cfg.gh_token)
        if NS.jsonl_lines(P["train"]) == 0:
            NS.restore(f"nm_{cfg.tag}_train", P["train"], cfg.gh_token)
        if not (P["ctc"] / "ctc2.joblib").exists():
            raise RuntimeError("no detector on the branch yet -- wait for NB1 'PREPARE DONE'")
        mark(f"nm_{cfg.tag}_prepare")
    _s.__name__ = "ensure_prepared"
    return _s


def budget_of(cfg):
    if cfg.flag_budget is not None:
        return cfg.flag_budget
    b = json.load(open(paths(cfg)["ctc"] / "budget.json"))
    return b["budget"]


# =============================================================================
# chains
# =============================================================================
def chain(cfg, run, *, model, pool, ctc, split, offset, n, dataset="gsm8k",
          budget=None, score=False, probes=True, baselines=True):
    store = STORES / f"store_{run}"
    max_tok = 320 if model == LLAMA else cfg.eval_max_tok
    env = {"FADE_DTYPE": cfg.dtype if model != LLAMA else "fp16",
           "FADE_TEST_DATASET": dataset}

    def body(lane):
        b = budget if budget is not None else budget_of(cfg)
        lane.say(f"{run}: split={split} dataset={dataset} offset={offset} n={n} budget={b}")
        py(lane, "run_ctc_validate.py",
           f"--ctc {ctc} --pool-store {pool} --store-dir {store} --n-problems {n} "
           f"--split {split} --eval-offset {offset} {'' if score else '--no-score'} "
           f"--flag-budget {b} --type-conf-gate 0.5 --retry-flagged --retrieval 3stage "
           f"--model {model} --max-new-tokens {max_tok} --show-every 25 "
           f"--secret-name {cfg.hf_secret}", env=env)
        if probes:
            py(lane, "run_mpv.py",
               f"--store {store} --pool-store {pool} --model {model} --n-probes {cfg.n_probes} "
               f"--max-new-tokens {max_tok} --secret-name {cfg.hf_secret}", env=env)
        if baselines:
            py(lane, "run_baselines.py",
               f"--store {store} --pool-store {pool} --model {model} --arms resample,generic "
               f"--max-new-tokens {max_tok} --secret-name {cfg.hf_secret}", env=env)
        mark(run)
    return with_push(run, store, cfg, body)


def qwen_test_shard(cfg, k, dataset):
    P = paths(cfg)
    dname = "gsm8k" if dataset == "gsm8k" else "p1"
    n = cfg.n_test if dataset == "gsm8k" else cfg.n_p1
    half = n // 2
    return chain(cfg, f"nm_{cfg.tag}_test_{dname}_s{k}", model=cfg.model, pool=P["train"],
                 ctc=P["ctc"] / "ctc2.joblib", split="test", offset=k * half, n=half,
                 dataset=dataset)


def qwen_heldout(cfg):
    P = paths(cfg)
    return chain(cfg, f"nm_{cfg.tag}_heldout", model=cfg.model, pool=P["train"],
                 ctc=P["ctc"] / "ctc2.joblib", split="train", offset=cfg.n_train,
                 n=cfg.n_heldout, score=True, probes=False)


# =============================================================================
# Llama-2
# =============================================================================
def llama_assets(cfg):
    def _s(lane):
        pool, ctc = STORES / "llama2_pool", STORES / "llama2_ctc"
        if not pool.exists():
            NS.restore_dir(LLAMA_POOL_DIR, pool, cfg.gh_token)
        if not ctc.exists():
            NS.restore_dir(LLAMA_CTC_DIR, ctc, cfg.gh_token)
        from classification import TraceStore
        import hashlib
        ex = TraceStore(root=pool).exemplars()
        fp = hashlib.sha1("␟".join(sorted(str(e.get("question", "")) for e in ex))
                          .encode()).hexdigest()[:12]
        if fp != LLAMA_POOL_FP:
            raise RuntimeError(f"Llama-2 pool fingerprint {fp} != {LLAMA_POOL_FP}")
        lane.say(f"Llama-2 pool {len(ex)} exemplars, fingerprint {fp} matches stage 3")
    _s.__name__ = "llama_assets"
    return _s


def llama_gsm8k_baselines(cfg):
    run = "nm_llama2_test_gsm8k"
    store = STORES / f"store_{run}"
    pool = STORES / "llama2_pool"

    def body(lane):
        if NS.jsonl_lines(store) == 0 or not (store / "results.jsonl").exists():
            tmp = STORES / "_llama_stage3_full"
            NS.restore_dir(LLAMA_TEST_DIR, tmp, cfg.gh_token)
            NS.subset(tmp, store, cfg.n_test)
        py(lane, "run_baselines.py",
           f"--store {store} --pool-store {pool} --model {LLAMA} --arms resample,generic "
           f"--max-new-tokens 320 --secret-name {cfg.hf_secret}")
        mark(run)
    return with_push(run, store, cfg, body)


def llama_p1(cfg):
    return chain(cfg, "nm_llama2_test_p1", model=LLAMA, pool=STORES / "llama2_pool",
                 ctc=STORES / "llama2_ctc" / "ctc2.joblib", split="test", offset=0,
                 n=cfg.n_p1, dataset=cfg.p1_variant, budget=cfg.llama_flag_budget)


# =============================================================================
# the notebooks
# =============================================================================
def nb1_lanes(cfg):
    env = {"FADE_DTYPE": cfg.dtype}
    if cfg.dtype == "fp32":          # the model needs both GPUs: one lane, same steps in order
        return [Lane("gpu01", "0,1", [train_shard(cfg, 0), train_shard(cfg, 1), prepare(cfg),
                                      qwen_heldout(cfg), qwen_test_shard(cfg, 0, cfg.p1_variant),
                                      qwen_test_shard(cfg, 1, cfg.p1_variant)], env)]
    lane0 = Lane("gpu0", "0", [train_shard(cfg, 0), wait_for(f"nm_{cfg.tag}_train_s1"),
                               prepare(cfg), qwen_heldout(cfg),
                               qwen_test_shard(cfg, 1, cfg.p1_variant)], env)
    lane1 = Lane("gpu1", "1", [train_shard(cfg, 1), wait_for(f"nm_{cfg.tag}_prepare"),
                               qwen_test_shard(cfg, 0, cfg.p1_variant)], env)
    return [lane0, lane1]


def nb2_lanes(cfg):
    return [Lane("llama", "0,1", [llama_assets(cfg), llama_gsm8k_baselines(cfg), llama_p1(cfg)])]


def nb3_lanes(cfg):
    env = {"FADE_DTYPE": cfg.dtype}
    if cfg.dtype == "fp32":
        return [Lane("gpu01", "0,1", [ensure_prepared(cfg), qwen_test_shard(cfg, 0, "gsm8k"),
                                      qwen_test_shard(cfg, 1, "gsm8k")], env)]
    return [Lane("gpu0", "0", [ensure_prepared(cfg), qwen_test_shard(cfg, 0, "gsm8k")], env),
            Lane("gpu1", "1", [wait_for(f"nm_{cfg.tag}_prepare"),
                               qwen_test_shard(cfg, 1, "gsm8k")], env)]


def cfg_from_branch(gh_token, model_hint="Qwen/Qwen3.5-4B", **overrides):
    """For NB3/NB4: rebuild the Cfg NB1 actually used (model, dtype, sizes) from the
    nm_cfg.json that prepare() pushed next to the detector."""
    import dataclasses
    tag = Cfg(model=model_hint).tag
    d = STORES / f"nm_{tag}_ctc"
    STORES.mkdir(parents=True, exist_ok=True)
    if not (d / "nm_cfg.json").exists():
        NS.restore(f"nm_{tag}_ctc", d, gh_token)
    if not (d / "nm_cfg.json").exists():
        raise SystemExit(f"no nm_cfg.json for {model_hint} on the branch -- has NB1 printed PREPARE DONE?")
    blob = json.load(open(d / "nm_cfg.json"))
    fields = {f.name for f in dataclasses.fields(Cfg)}
    blob = {k: v for k, v in blob.items() if k in fields}
    blob.update(overrides)
    cfg = Cfg(gh_token=gh_token, **blob)
    print(f"  config from NB1: {blob}")
    return cfg


def stream(cmd, env=None):
    """Run a shell command from the notebook and stream its output into the cell."""
    env_ = dict(os.environ); env_.update(env or {})
    p = subprocess.Popen(cmd, shell=True, cwd=REPO, env=env_, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    for ln in p.stdout:
        print(ln, end="", flush=True)
    return p.wait()


def pick_model(candidates=(("Qwen/Qwen3.5-4B", "fp16"), ("Qwen/Qwen3.5-4B", "fp32"),
                           ("Qwen/Qwen3-4B-Instruct-2507", "fp16")), n=12):
    """Smoke-test candidates in order; return the first (model, dtype) that passes."""
    for m, d in candidates:
        print("\n" + "#" * 70 + f"\n#  SMOKE TEST  {m}  {d}\n" + "#" * 70, flush=True)
        rc = stream(f"python -u nm_patch.py nm_smoke.py --model {m} --n {n} "
                    f"--out {WORK}/nm_smoke_{d}.json",
                    env={"CUDA_VISIBLE_DEVICES": "0" if d != "fp32" else "0,1", "FADE_DTYPE": d})
        if rc == 0:
            print(f"\n  -> using {m} in {d}")
            return m, d
        print(f"\n  {m} {d} failed (exit {rc}); trying the next candidate")
    raise SystemExit("no candidate passed the smoke test -- read the output above")


# =============================================================================
# scoring (CPU)
# =============================================================================
def score_all(cfg, out="nm_reports", with_llama=True):
    """Restore every store, merge shards, score, and build the tables."""
    class _L:                      # a minimal lane for shell output
        name_ = "score"
        def sh(self, cmd, env=None):
            env_ = dict(os.environ); env_.update(env or {})
            print(f"$ {cmd}", flush=True)
            r = subprocess.run(cmd, shell=True, cwd=REPO, env=env_)
            if r.returncode:
                print(f"  !! exit {r.returncode}")
            return r.returncode
    L = _L()
    g = cfg.gh_token
    P = paths(cfg)
    STORES.mkdir(parents=True, exist_ok=True)
    NS.restore(f"nm_{cfg.tag}_ctc", P["ctc"], g)
    runs = []

    def merged(name, n_shards=2):
        shards = []
        for k in range(n_shards):
            d = STORES / f"store_{name}_s{k}"
            NS.restore(f"{name}_s{k}", d, g)
            shards.append(d)
        dest = STORES / f"store_{name}"
        NS.merge(shards, dest)
        return dest

    # gold for GSM-Symbolic, written once
    p1_gold = STORES / f"{cfg.p1_variant}_gold.jsonl"
    L.sh(f"python -u nm_data.py --variant {cfg.p1_variant} --n {cfg.n_p1} --export {p1_gold}")

    q_test = merged(f"nm_{cfg.tag}_test_gsm8k")
    q_p1 = merged(f"nm_{cfg.tag}_test_p1")
    q_ho = STORES / f"store_nm_{cfg.tag}_heldout"
    NS.restore(f"nm_{cfg.tag}_heldout", q_ho, g)
    ctc_q = P["ctc"] / "ctc2.joblib"
    jobs = [(f"{cfg.model.split('/')[-1]}|GSM8K", q_test, ctc_q, "--split test", q_ho, "none"),
            (f"{cfg.model.split('/')[-1]}|GSM-Sym P1", q_p1, ctc_q,
             f"--split test --gold-jsonl {p1_gold}", None, "template")]
    if with_llama:
        l_test = STORES / "store_nm_llama2_test_gsm8k"
        l_p1 = STORES / "store_nm_llama2_test_p1"
        l_ho = STORES / "llama2_heldout"
        NS.restore("nm_llama2_test_gsm8k", l_test, g)
        NS.restore("nm_llama2_test_p1", l_p1, g)
        if not l_ho.exists():
            NS.restore_dir(LLAMA_HELDOUT_DIR, l_ho, g)
        if not (STORES / "llama2_ctc").exists():
            NS.restore_dir(LLAMA_CTC_DIR, STORES / "llama2_ctc", g)
        ctc_l = STORES / "llama2_ctc" / "ctc2.joblib"
        jobs = [("Llama-2-7B|GSM8K", l_test, ctc_l, "--split test", l_ho, "none"),
                ("Llama-2-7B|GSM-Sym P1", l_p1, ctc_l, f"--split test --gold-jsonl {p1_gold}",
                 None, "template")] + jobs
    for label, store, ctc, split_args, ho, cl in jobs:
        if not (store / "results.jsonl").exists():
            print(f"  -- {label}: no store yet, skipped")
            continue
        launcher = os.environ.get("NM_LAUNCHER", "nm_patch.py")
        L.sh(f"python -u {launcher} score_mpv.py --store {store} --ctc {ctc} {split_args}")
        L.sh(f"python -u {launcher} score_baselines.py --store {store} {split_args} --label '{label}' "
             f"--cluster {cl}" + (f" --heldout-store {ho}" if ho and ho.exists() else ""))
        if (store / "baselines_report.json").exists():
            runs.append(f'--run "{label}={store}"')
    if runs:
        L.sh(f"python -u nm_tables.py --out {WORK / out} " + " ".join(runs))
    return WORK / out
