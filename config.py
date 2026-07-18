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
GOOD_R_MIN = 0.6              # GOOD additionally requires R >= 0.6
MEDIUM_MISSES_MAX = 1         # MEDIUM (both kinds): at most one missed checkpoint
MEDIUM_WRONG_BAD_EQS_MAX = 1  # MEDIUM-wrong tolerates at most one false equation
G_MIN = 0.5                   # grounding floor: MEDIUM-wrong AND the NR/SM guards

# ---------------------------------------------------------------- R (redundancy)
R_VARIANT = "dup_fraction"    # calibration picks among the three, once
DUP_SIM_THRESHOLD = 0.90      # cosine above this = near-duplicate step

# ------------------------------------------------------------ value matching
VALUE_MATCH_TOL = 1e-6

# ------------------------------------------------- diagnostic cascade
# Order: NR -> ST -> abstain -> SM -> CE -> UNCLASSIFIED
NR_E_MAX = 0.30        # NR: E < 0.30 AND at least one broken engagement signal
NR_V_MAX = 0.40        #     broken signal 1: V <= 0.40
                       #     broken signal 2: G < G_MIN
                       #     broken signal 3: not coherent
ST_LEN_RATIO = 0.6     # ST: s < 0.6 * s_hat
ST_MIN_SHAT = 2        # ST: only for problems with >= 3 gold checkpoints
SM_E_MAX = 0.30        # SM: E < 0.30 (plus V, length, grounding, coherence)
CE_E_MIN = 0.30        # CE: E >= 0.30 -- NO upper bound: hitting 75-100% of
                       # checkpoints with valid arithmetic is the PUREST CE
V_MIN = 0.40           # SM and CE require V > 0.40
V_MAX = 0.85           # CE requires V < 0.85
FULL_LEN_RATIO = 0.7   # SM and CE require s >= 0.7 * s_hat
ABSTAIN_MARGIN = 0.05  # |E - 0.30| < margin -> UNCLASSIFIED (strict '<');
                       # 0 recovers the raw rules

# ------------------------------------------------------------ trace storage
# GOOD -> pool (the Phase-1 exemplar source; the ONLY way in)
# MEDIUM_* -> medium queue (retried first, end-of-pass)
# BAD_*   -> bad queue     (retried after every MEDIUM retry)
STORE_ROOT = Path(__file__).parent / "store"
POOL_FILE = "pool.jsonl"
MEDIUM_QUEUE_FILE = "medium_queue.jsonl"
BAD_QUEUE_FILE = "bad_queue.jsonl"