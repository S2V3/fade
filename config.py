"""
Every tunable in one place. Priors, not commitments: coarse thresholds get
decided by targeted grading near the boundary, then frozen. Fine decimals
are never tuned at these sample sizes -- which is why classification runs
on COUNTS (misses, bad_eqs), not weighted sums. Q is gone entirely: only
its components are computed (components.py).

Shared by: extraction.py, similarity.py, components.py, classification.py,
diagnosis.py, run_phase0.py, inspect_trace.py
"""
from pathlib import Path

# ------------------------------------------------ count-based classification
GOOD_R_MIN = 0.6            # GOOD additionally requires R >= 0.6
MEDIUM_MISSES_MAX = 1       # MEDIUM (both kinds): at most one missed checkpoint
MEDIUM_WRONG_BAD_EQS_MAX = 1  # MEDIUM-wrong tolerates at most one false equation
G_MIN = 0.5                 # grounding floor: MEDIUM-wrong AND the SM guard

# ---------------------------------------------------------------- R (redundancy)
R_VARIANT = "dup_fraction"  # calibration picks among the three, once
DUP_SIM_THRESHOLD = 0.90    # cosine above this = near-duplicate step

# ------------------------------------------------------------ value matching
VALUE_MATCH_TOL = 1e-6

# ------------------------------------------------- diagnostic cascade
# Order: NR -> ST -> abstain -> SM -> CE -> UNCLASSIFIED
#
# NR (as actually implemented in diagnosis.diagnose): E < NR_E_MAX AND the
# trace states NO verifiable arithmetic at all (n_equations == 0). That is
# the ONLY NR condition today. The V/G/coherence "broken engagement signals"
# below were part of an earlier NR design and are NOT currently consulted by
# NR -- NR_V_MAX is retained only so importers don't break and as a reserved
# prior for a future graded NR. Do not assume it gates anything right now.
NR_E_MAX = 0.30            # NR: E < 0.30 (with n_equations == 0)
NR_V_MAX = 0.40            # RESERVED / currently unused by diagnose() (see note above)

# ST -- step omission. Fires only when the trace is short RELATIVE to the
# expected step count AND execution fidelity is ALSO low. The E gate
# (ST_E_MAX) is new: genuine omission misses checkpoints (low E); a short
# trace that still hit most checkpoints is compact-correct work, not
# omission. Without the gate, terse chat traces were labelled ST regardless
# of E, starving CE and biasing every FST cell built on the ST label.
ST_LEN_RATIO = 0.6         # ST: s < 0.6 * s_hat
ST_MIN_SHAT = 2            # ST: only for problems with >= 2 gold checkpoints
                           # NOTE: diagnosis.py prose previously said ">= 3".
                           # The CODE value has always been 2; docs were
                           # aligned DOWN to the code (behaviour unchanged).
                           # If 3 was the intent, change this to 3 and re-run
                           # the dev histogram -- flagged, not silently picked.
ST_E_MAX = 0.60            # ST additionally requires E < 0.60 (execution also low).
                           # PRIOR, not calibrated: tune on dev with the
                           # diagnosis histogram. Lower = tighter ST (more CE
                           # coverage). Prefer switching this to an A-based
                           # bound once diagnose() is passed A.

SM_E_MAX = 0.30            # SM: E < 0.30 (plus V, length, grounding, coherence)
CE_E_MIN = 0.30            # CE: E >= 0.30 -- NO upper bound: hitting 75-100% of
                           # checkpoints with valid arithmetic is the PUREST CE
V_MIN = 0.40               # SM and CE require V > 0.40
V_MAX = 0.85               # CE requires V < 0.85
FULL_LEN_RATIO = 0.7       # SM and CE require s >= 0.7 * s_hat
ABSTAIN_MARGIN = 0.05      # |E - 0.30| < margin -> UNCLASSIFIED (strict '<');
                           # 0 recovers the raw rules

# ------------------------------------------------------------ trace storage
# GOOD -> pool (the Phase-1 exemplar source; the ONLY way in)
# MEDIUM_* -> medium queue (retried first, end-of-pass)
# BAD_*    -> bad queue (retried after every MEDIUM retry)
STORE_ROOT = Path(__file__).parent / "store"
POOL_FILE = "pool.jsonl"
MEDIUM_QUEUE_FILE = "medium_queue.jsonl"
BAD_QUEUE_FILE = "bad_queue.jsonl"

# ------------------------------------------------------------ preprocessing
NORMALIZE_TRACES = True    # fold unicode math glyphs before scoring
                           # (recovers equations/values a base model writes
                           # with -,x,/ instead of ASCII; never changes value)

# --------------------------------------------------- detailed run outputs
# Written by kaggle_run.py's full mode into STORE_ROOT, alongside the queues.
RESULTS_FILE = "results.jsonl"     # one record PER ATTEMPT (pass1 + retries)
SUMMARY_FILE = "run_summary.json"  # all aggregate metrics, machine-readable
REPORT_FILE = "report.md"          # the same, human-readable
CSV_FILE = "results.csv"           # per-problem final outcome, spreadsheet-ready
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"  # every threshold used, for repro