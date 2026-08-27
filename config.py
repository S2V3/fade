"""
Every tunable in one place. Priors, not commitments: coarse thresholds get
decided by targeted grading near the boundary, then frozen. Fine decimals
are never tuned at these sample sizes -- which is why classification runs
on COUNTS (misses, bad_eqs), not weighted sums. Q is gone entirely: only
its components are computed (components.py).

Shared by: extraction.py, similarity.py, components.py, classification.py,
diagnosis.py, kaggle_run.py, inspect_trace.py

AUDIT REVISION (see FADE_AUDIT.md). Changed defaults are marked [AUDIT Dnn].
Every one of them is reversible by restoring the value named in its comment.
"""
from pathlib import Path

# Bumped whenever the scoring/taxonomy changes in a way that invalidates old runs.
# The Kaggle notebook REFUSES TO RUN if the cloned repo does not report this.
FADE_CODE_VERSION = "audit-2"

# ------------------------------------------------ count-based classification
# [AUDIT D7] R is a DEAD signal: separation between correct and wrong traces is
# +0.010 (E is +0.615, A is +0.397). It nevertheless gated the pool and rejected
# at least one perfect trace because MiniLM scored two structurally-parallel math
# steps ("In April..." / "In May...") as near-duplicates. R is still COMPUTED and
# LOGGED as a diagnostic; it no longer decides anything.
# Restore the old behaviour with GOOD_REQUIRE_R = True.
GOOD_REQUIRE_R = False
GOOD_R_MIN = 0.6            # only consulted when GOOD_REQUIRE_R is True

# [AUDIT D8] A (step alignment) is the second-strongest signal and was read by
# NOTHING. A pool exemplar should demonstrably have COMPUTED its checkpoints, not
# merely mentioned them. Set GOOD_A_MIN = 0.0 to disable.
GOOD_A_MIN = 0.5

MEDIUM_MISSES_MAX = 1       # MEDIUM (both kinds): at most one missed checkpoint
MEDIUM_WRONG_BAD_EQS_MAX = 1  # MEDIUM-wrong tolerates at most one false equation
G_MIN = 0.5                 # grounding floor: MEDIUM-wrong AND the SM guard

# ---------------------------------------------------------------- R (redundancy)
R_VARIANT = "dup_fraction"  # calibration picks among the three, once
# [AUDIT D7] 0.90 flags legitimately parallel reasoning steps as duplicates.
# Raised, and a duplicate pair must now also share a numeric value
# (components.R_score) -- parallel PHRASING with different numbers is not
# redundancy. Restore with DUP_SIM_THRESHOLD = 0.90 / R_REQUIRE_SHARED_VALUE=False.
DUP_SIM_THRESHOLD = 0.97
R_REQUIRE_SHARED_VALUE = True

# ------------------------------------------------- step segmentation
# split_steps() (similarity.py) defines s (model step count, read by the
# cascade) AND R's step list.
# [AUDIT / NON-ISSUE] Measured on the 1,500-problem run: only 33 traces fail the
# full_length gate and s/s_hat averages 2.74 -- the model is 2.7x MORE verbose
# than gold, not terser. Step undercount is NOT a real problem here. Left off.
STEP_SPLIT_SENTENCES = False

# ------------------------------------------------------------ value matching
VALUE_MATCH_TOL = 1e-6

# [AUDIT D5] extract_values used to emit BOTH conventions for every percent
# token ("20%" -> 20 AND 0.2), doubling the surface area for a coincidental
# checkpoint hit (inflates E) and a free grounding hit (inflates G). 13.8% of
# eval questions contain a percent. Dual emission is now opt-in and used ONLY
# where GSM8K's own annotations are genuinely inconsistent: gold checkpoints.
# Restore the old behaviour with PERCENT_DUAL_EVERYWHERE = True.
PERCENT_DUAL_EVERYWHERE = False

# ------------------------------------------------------ execution fidelity (E)
# [AUDIT D6] Value-based E asks only "did the right NUMBER appear?". Measured:
# mean E = 0.397 vs mean E_struct = 0.106, and 627/1500 traces have E > 0 with
# E_struct == 0 -- the right number produced by the wrong operation on the wrong
# operands. But raw E_struct is a strict LOWER bound (a chained gold checkpoint
# the trace computes in separate steps cannot operand-match), so driving the
# cascade off it alone would over-fire NR.
#
# E_eff = max(E_struct, E_STRUCT_BLEND*E + (1-E_STRUCT_BLEND)*E_struct)
#   E_STRUCT_BLEND = 1.0 -> pure value-based E   (E_eff == E; current default)
#   E_STRUCT_BLEND = 0.0 -> pure structural E    (over-strict)
#   E_STRUCT_BLEND = 0.5 -> credit structure without punishing decomposition
#
# SHIPPED OFF BY DEFAULT, DELIBERATELY. Blending RESCALES the signal: at 0.5 the
# mean drops 0.397 -> 0.286 and the 0.30 boundary moves from the 45th to the 61st
# percentile of the distribution. Every E threshold below (NR_E_MAX, SM_E_MAX,
# CE_E_MIN, ABSTAIN_MARGIN) was calibrated against value-based E, so turning the
# blend on without recalibrating silently changes what every rule means.
#
# Enabling it is a TWO-STEP change, to be done on a dev slice and nowhere else:
#   1. set E_STRUCT_BLEND = 0.5
#   2. set NR_E_MAX = SM_E_MAX = CE_E_MIN = 0.17
#      (0.17 is the E_eff value at the same percentile 0.30 occupied under E,
#       measured on the 1,500-problem train run -- a starting point, not a
#       calibration; re-derive it on YOUR dev slice.)
# Then re-run fst_gate.py, because every label will have moved.
E_STRUCT_BLEND = 1.0

# ------------------------------------------------- diagnostic cascade
# [AUDIT D10-D14] The cascade is rebuilt. Order:
#   TR -> NR -> AL -> ST -> SM -> CE -> WP -> UNCLASSIFIED(abstain)
# Rationale for each new/changed rule is in diagnosis.py and FADE_AUDIT.md.
# All cascade E tests now read E_eff, not raw E.

NR_E_MAX = 0.30            # NR: E_eff < 0.30
NR_V_MAX = 0.40            # RESERVED / unused (kept so importers don't break)
# [AUDIT D8 -- CORRECTED] An earlier revision of this audit tried to define NR as
# "A <= 0.15", on the theory that A is a better 'did no verifiable work' test.
# That was WRONG and the data said so immediately: NR jumped 183 -> 688 (50.6% of
# all diagnoses), because A == 0 means "computed none of the GOLD checkpoints",
# and a trace can do plenty of valid arithmetic on the WRONG quantities. 616
# traces have A <= 0.15 while stating >= 1 real equation -- those are SM/WP/CE
# cases, not "never engaged". Using A for NR conflates 'did nothing' with 'did
# the wrong thing'.
#
# NR therefore keeps its original, correct test: NO stated arithmetic at all.
# A is used where it actually belongs -- the GOOD rule (GOOD_A_MIN) and, via
# A_struct, the redefined ST.
NR_USE_A = False           # True re-enables the A-based variant (not recommended)
NR_A_MAX = 0.15            # only consulted when NR_USE_A is True

# [AUDIT D14] ST fired 7 times in 1,500 because it was a LENGTH test against a
# model that writes 2.74x more lines than gold has steps. Redefined semantically:
# step omission = "you wrote plenty but never actually COMPUTED most of the
# required intermediates". Length is now only a floor, not the trigger.
ST_LEN_RATIO = 0.6         # ST: s < 0.6 * s_hat
ST_MIN_SHAT = 2            # only for problems with >= 2 gold checkpoints
ST_E_MAX = 0.60            # ST also requires E < 0.60
#
# HONEST OUTCOME: the semantic redefinition was TRIED AND REJECTED on the data.
# Defining ST as "A_struct < 0.34 with a length floor" made ST the LARGEST class
# (341 = 25.1% of diagnoses) and collapsed CE from 296 to 40 -- because A_struct
# averages 0.105 across the whole run, so "A_struct < 0.34" is true of almost
# everything and ST simply cannibalised CE.
#
# The real conclusion is that ST is genuinely near-empty for this model: it
# writes 2.74x MORE lines than gold has steps, so step OMISSION barely happens.
# That is a finding to report, not a rule to broaden until it fires. ST stays on
# its narrow, valid length definition and is expected to be rare (~7/1500).
#
# ST_USE_LEGACY = False enables the semantic variant for experimentation. Do NOT
# turn it on without re-tuning ST_ASTRUCT_MAX far below 0.34 on a dev slice.
ST_USE_LEGACY = False      # kept for the old length rule; see ST_MODE
ST_ASTRUCT_MAX = 0.34      # only consulted when ST_MODE == "astruct"

# [AUDIT D27] ST_MODE -- how step omission is detected.
#   "coverage" (DEFAULT) computational_steps / s_hat < ST_COVERAGE_MAX
#   "length"              s < ST_LEN_RATIO * s_hat   (the original rule)
#   "astruct"             A_struct < ST_ASTRUCT_MAX  (tried, rejected -- it
#                         cannibalised CE; see the note above)
#
# Why "coverage" is the right unit: `s` counts LINES and a chat model's lines are
# mostly prose, so s/s_hat = 2.74 says "verbose" while comp/s_hat = 0.67 says
# "under-computes". Only the second is about reasoning. Threshold swept on train:
#   thr    ST share   UNCLASSIFIED   majority-baseline
#   0.34     11.0%        13.2%          22.0%
#   0.50     12.6%        12.6%          21.9%   <- best conditioned
#   0.67     31.4%         6.9%          35.0%
ST_MODE = "coverage"
ST_COVERAGE_MAX = 0.5

# [AUDIT D12] AL -- algebraic non-execution. NEW TYPE. The model sets up symbolic
# variables and never resolves them to a number ("x + 3x/2 = 4x/2" -> "#### 4"
# when gold was 5250). Measured: 400/1500 traces (26.7%); accuracy 12.0% when
# algebraic vs 21.3% when not -- a 9.3-point cost. Detected from the trace with
# NO gold, which also makes it FST's most promising target class.
# >= this many UNRESOLVED relation lines (extraction.unresolved_relation_count).
# TUNED BY SWEEP on the train split, not guessed:
#   thr   AL share   UNCLASSIFIED   majority-baseline FST must beat
#    2      29.2%        13.9%           35.2%
#    3      18.4%        17.1%           23.1%   <-- best conditioned
#    4      10.8%        19.0%           24.8%
#    5       6.4%        20.4%           26.3%
# 3 gives the lowest majority baseline (the most balanced target set) while
# keeping AL a substantial, well-populated class. Re-sweep on your own dev slice
# if the traces change -- this was tuned on the OLD (pre-fix) generations.
AL_MIN_VAR_LINES = 3

# [AUDIT D13] TR -- truncated / unterminated generation. 154/1500 traces end
# mid-thought. These are GENERATION failures wearing a reasoning-failure costume;
# diagnosing them poisons FST's labels with noise unrelated to the question.
# They are routed OUT of the taxonomy and re-generated instead.
TR_ENABLED = True

SM_E_MAX = 0.30            # SM: E_eff < 0.30 (plus V, length, grounding, coherence)
CE_E_MIN = 0.30            # CE: E_eff >= 0.30
V_MIN = 0.40               # SM and CE require V > 0.40
V_MAX = 0.85               # CE requires V < 0.85; WP requires V >= 0.85
FULL_LEN_RATIO = 0.7       # SM / CE / WP require s >= 0.7 * s_hat

# [AUDIT D11] WP -- wrong plan. NEW TYPE. 243 traces have SOUND arithmetic and a
# wrong answer (mean V = 0.999) and were falling through to abstain because CE
# requires V < 0.85 and SM requires E < 0.30. They are NOT calculation errors:
# CE fires at mean V = 0.193. Handing them "verify each computation" is exactly
# the confidently-wrong cure the abstain design exists to prevent.
WP_ENABLED = True

# [AUDIT D10] The abstain band was discretisation noise, not uncertainty.
# E = hits/s_hat is a ratio of small integers: all 216 banded traces sat at
# exactly three values (1/3 = 142, 1/4 = 71, 2/7 = 3). A continuous +/-0.05 guard
# on a signal with ~6 distinct values below 0.5 swallows whole denominators.
# 0.02 leaves 3/216 abstaining. Restore with ABSTAIN_MARGIN = 0.05.
ABSTAIN_MARGIN = 0.02

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

# --------------------------------------------------- detailed run outputs
RESULTS_FILE = "results.jsonl"     # one record PER ATTEMPT (pass1 + retries)
SUMMARY_FILE = "run_summary.json"  # all aggregate metrics, machine-readable
REPORT_FILE = "report.md"          # the same, human-readable
CSV_FILE = "results.csv"           # per-problem final outcome, spreadsheet-ready
CONFIG_SNAPSHOT_FILE = "config_snapshot.json"  # every threshold used, for repro
PROMPT_SAMPLES_FILE = "prompt_samples.txt"     # [AUDIT D2] the gold-leak canary

# --------------------------------------------------- diagnosis confidence
# A typed diagnosis firing with confidence < DIAG_MIN_CONFIDENCE is downgraded to
# UNCLASSIFIED -> generic cure, rather than committing a confidently-wrong typed
# cure. Also keeps low-confidence labels OUT of FST's training data.
# DEFAULT 0.0 = OFF (emitted but never triggers a downgrade).
DIAG_MIN_CONFIDENCE = 0.0

# =============================================================================
# [AUDIT D2] PROMPT INTEGRITY
# =============================================================================
# The chat re-wrap bug (D1) sent the model an EXEMPLAR to solve for 1,367
# generations and ~8 GPU-hours without tripping anything: grounding collapsed
# 0.898 -> 0.128 and NR/SM recovered 0 of 351. Nothing asserted that the prompt
# actually contained the question we were asked to solve.
#
# PROMPT_ASSERT makes that a hard invariant on EVERY generation in EVERY arm.
# PROMPT_SAMPLE_EVERY dumps a verbatim prompt every N generations so the gold
# rule can be audited by grep after the fact.
PROMPT_ASSERT = True
PROMPT_ASSERT_PREFIX = 48   # chars of the live question that must survive
PROMPT_SAMPLE_EVERY = 250   # 0 = off

# =============================================================================
# [AUDIT D20] RETRY ECONOMICS
# =============================================================================
# MEDIUM_CORRECT consumed 149 of 1,367 retry generations (11%) and recovered 0 --
# impossible by construction, since `newly_correct` requires the id NOT to be in
# `solved` and a correct trace already is. Worse, retrying a trace the model got
# RIGHT made it WRONG 114/149 times (76.5%).
# Set True to restore polish retries (they can still add pool entries: 8/149).
RETRY_POLISH_MEDIUM_CORRECT = False

# [AUDIT SELF-REVIEW] Once polish retries are off, a CORRECT trace's diagnosis has
# no consumer -- nothing acts on it. It also distorts the headline: 60 of the 235
# remaining UNCLASSIFIED traces are CORRECT answers that simply cannot fire WP or
# AL (both require `not correct`), so they inflate the abstain rate for a
# non-reason. Measured over WRONG traces only, the abstain rate is 14.4%, not
# 17.3%. Set True to restore diagnosis of weak successes (e.g. if you re-enable
# RETRY_POLISH_MEDIUM_CORRECT, which needs the diagnosis to pick a cure).
DIAGNOSE_CORRECT = False

# =============================================================================
# [AUDIT D15] RETRIEVAL
# =============================================================================
# Strategy 2 (few_shot -- the default, and the one the 1,500-problem run used)
# built its prompt from the FIXED seeds and never called _dyn(), so the
# accumulated pool and the 3-stage retriever had ZERO runtime. Confirmed:
# prompt_tokens sd = 25 across 1,500 problems, correlating 0.961 with question
# length -- a constant exemplar block.
#
# FEWSHOT_DYNAMIC = True  -> strategy 2 retrieves from the pool (FADE as designed)
# FEWSHOT_DYNAMIC = False -> the original fixed-seed arm, kept as the CONTROL
FEWSHOT_DYNAMIC = True

# [AUDIT D18 / retrieval upgrade 1] w_complexity used to dominate w_relevance
# 0.6/0.4. Pool traces average 7.12 lines and gold seeds 5.1, so complexity
# re-rank systematically preferred VERBOSE MODEL TRACES over CONCISE CORRECT GOLD.
RETRIEVAL_W_RELEVANCE = 0.5
RETRIEVAL_W_COMPLEXITY = 0.2
# [AUDIT D32] Structural question similarity, added because it MEASURED BEST.
# Offline benchmark (eval_retrieval.py, precision@8, 143-exemplar pool, TF-IDF):
#     random 0.252 | cosine 0.271 | 3stage_lines 0.258 (the shipped code!)
#     3stage_fixed 0.281 | structural 0.284 | hybrid 0.285
# Set to 0.0 to disable. RE-RUN eval_retrieval.py --embed minilm before trusting
# these weights: they were tuned under TF-IDF, and MiniLM may reorder them.
RETRIEVAL_W_STRUCTURAL = 0.3
# [retrieval upgrade 3] MMR deduped on question cosine. Two dissimilar questions
# can still demonstrate the identical reasoning shape. Also dedupe on the
# exemplar's operator multiset + step count.
RETRIEVAL_DUP_THRESHOLD = 0.85
RETRIEVAL_DEDUPE_SHAPE = True
# [retrieval upgrade 2] For a TYPED retry the diagnosis is known: score candidates
# jointly instead of filtering by type and throwing away the similarity ordering.
RETRIEVAL_TYPED_SOFT = True
RETRIEVAL_W_TYPEMATCH = 0.5

# [retrieval upgrade 5] similarity.embed's TF-IDF fallback fits a FRESH
# vectorizer per call, so vectors are not comparable across calls and retrieval
# silently degrades to noise. Fail loudly instead.
REQUIRE_MINILM = True