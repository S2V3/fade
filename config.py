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


# [AUDIT D39] The RETRY path was question-blind.
#
# select_typed_positives(pool, ftype, seeds, k) took no `question` argument, so
# EVERY retry of a given failure type received the IDENTICAL 8 exemplars, and
# select_generic_positives took the last 8 of the pool for every problem alike.
# Measured on the 1,500-problem run, mean TF-IDF similarity between the retried
# question and its 8 exemplars:
#
#     typed selection (type-fit only)     0.0059
#     generic selection (last 8)          0.0046
#     similarity retrieval (top-8)        0.0778   <- what PASS-1 already got
#
# The retry therefore handed the model exemplars 13x LESS topically relevant than
# the prompt that had just failed, and the typed-vs-generic comparison was
# between two flavours of irrelevant. Result: typed 40.3% vs generic 39.3%,
# McNemar p = 0.37.
#
# With this True, both arms retrieve by question similarity through
# exemplar_selector.get_exemplars_3stage, and the arms differ by exactly the
# thing under test: whether ftype (and the typed instruction) is supplied.
# Set False to reproduce the question-blind behaviour.
RETRY_QUESTION_AWARE = True          # revert: False


# [AUDIT D40] Type-fit must RANK, because the eligibility filter barely filters.
#
# _typed_candidates() passes 100% of the pool for ST and CE, 87% for WP, 82% for
# AL. For those types "type-eligible" means "everything", so filtering-then-
# retrieving leaves the typed arm identical to generic apart from a one-line
# instruction -- which is exactly what the ablation measured (typed 40.3% vs
# generic 39.3%, p=0.37).
#
# The fix: over-sample a relevance shortlist, then re-rank it by a blend of
# relevance position and _type_fit_score, so type-fit discriminates even when the
# boolean filter does not.
RETRY_TYPEFIT_OVERSAMPLE = 4         # shortlist = 4 x budget, then re-rank
RETRY_W_TYPEFIT = 0.70               # 0 = pure relevance, 1 = pure type-fit; revert: 0.45

# [AUDIT D40] Procedural typed instructions.
# The originals were single clauses ("Verify each computation.") that named the
# failure without saying what to DO. 'procedural' gives a short, checkable
# procedure per type. Revert with 'short'.
# [AUDIT D56] "redirect" replaced "procedural" as the default. The procedural
# instructions demanded effort ("check each result", "show every step") and the
# model complied by computing LESS: CE halved its false equations and lost 12.5
# points, AL did the same, NR and SM moved the wrong way outright. The only cure
# that gained asks for a DECISION (WP, +7.1). Redirecting instructions name a
# different artefact to produce instead of asking for more care.
TYPED_INSTRUCTION_STYLE = "redirect"     # options: redirect | procedural | short


# [AUDIT D40] Should the CONTROL arm also retrieve by question?
#
# It is a separate switch from RETRY_QUESTION_AWARE because it decides what the
# typed-vs-generic comparison actually measures:
#
#   True  (default)  both arms retrieve by question; they differ ONLY by ftype,
#                    the type-fit re-rank and the typed instruction. A typed win
#                    is attributable to TYPING.
#   False            generic stays question-blind (last k of the pool). A typed
#                    win then confounds typing with retrieval, and a reviewer can
#                    fairly say the result only shows that retrieval works.
#
# The pre-fix generic run (typed 40.3% vs generic 39.3%, p=0.374) is already the
# question-blind comparison, so setting this False re-runs an arm you have.
GENERIC_QUESTION_AWARE = True        # revert: False


# [AUDIT D41] Show the cure, do not only state it.
# A 7B follows format better than instructions. With this True the typed arm
# replaces ONE of its retrieved exemplars with a hand-written demonstration of
# the procedure for that failure type, so budget parity with generic is exact
# (both still show `budget` exemplars).
TYPED_APPROACH_DEMO = True           # revert: False


# [AUDIT D42] Put the most relevant exemplar NEAREST the live question.
#
# get_exemplars_3stage returns most-relevant FIRST, so the best-matched exemplar
# ends up furthest from the question the model is about to answer. In-context
# learning weights later demonstrations more heavily, so this reverses the block:
# least relevant first, most relevant last, immediately before the question.
#
# Applies to BOTH arms -- it is a prompt-ordering improvement, not a typed
# treatment, and letting only typed have it would confound the ablation.
RETRY_RELEVANCE_LAST = True          # revert: False


# [AUDIT D43] Typed-only response prefill.
# The typed retry prompt ends mid-sentence ("Solution: We need to find:") so the
# model must complete it. Generic ends at "Solution:" with nothing after, exactly
# as before. Costs no exemplar slot and no extra tokens.
TYPED_PREFILL_ENABLED = True         # revert: False


# [AUDIT D44] Type-conditioned acceptance tests.
# The diagnosis picks the cure AND checks, gold-free, whether the cure took. A
# retry that fails its own type's test is resampled once and the better candidate
# is kept. Generic has no diagnosis and therefore no test.
# [AUDIT D59] TURNED OFF. The test identifies bad candidates correctly and then
# replaces them with WORSE ones, because the resample is drawn at temperature 0.7
# while the original was greedy. Measured on 1,068 retries:
#
#     type   resampled   recovery if resampled   if not
#     WP           110            19.1%           18.1%
#     CE            27            22.2%           31.2%
#     ST            50            10.0%           17.2%
#     AL            52            13.5%           21.4%
#     UNCL          29             6.9%           16.9%
#     TR             4            25.0%            7.7%
#     ALL          272            15.4%           17.6%
#
# Every type except TR (n=4) and WP (neutral) does worse after resampling. The
# acceptance test is a good detector and a bad selector: passing it is only weakly
# correlated with being correct, so trading a greedy candidate for a stochastic
# one that happens to pass loses more than it gains. It also cost 272 extra
# generations (+10% over generic).
#
# Kept in the code as a measured negative result: a gold-free, per-type acceptance
# criterion CAN tell good retries from bad (1.8x lift, see ACCEPT_TEST_TYPES) but
# CANNOT be used to select a replacement unless the replacement is drawn at least
# as well as the original.
TYPED_ACCEPT_TEST = False            # revert: True
# [AUDIT D45] Never keep a resample that computes LESS than the candidate it
# replaces. Without this the acceptance test is gamed by degenerate traces.
ACCEPT_REJECT_DEGENERATE = True      # revert: False
# Measured on the 1,068 retries already in hand -- correct rate when the test
# PASSES vs when it FAILS:
#     CE  38.3% vs  3.6%   (10.6x)      <- keep
#     AL  30.0% vs  9.3%   ( 3.2x)      <- keep
#     TR   8.9% vs  0.0%   (   inf)     <- keep
#     ST  21.1% vs 14.5%   ( 1.5x)      <- keep
#     WP  19.2% vs 15.7%   ( 1.2x)      <- keep, weak
#     SM  13.3% vs 14.1%   ( 0.9x)      <- DROP, no signal
#     NR  11.1% vs 14.5%   ( 0.8x)      <- DROP, inverted
# Only the types where the test actually predicts correctness are gated. Testing
# SM and NR would burn generations resampling on noise.
# [AUDIT D45] WP kept only because its test (the answer must equal a computed
# value) cannot be passed by computing less -- it needs at least one equation.
ACCEPT_TEST_TYPES = ("CE", "AL", "TR", "ST", "WP")
ACCEPT_RESAMPLE_TEMP = 0.7           # >0, or the resample reproduces the failure
# [AUDIT D50] TR means the model ran out of room. Resampling at the SAME ceiling
# reproduces the truncation: on the pre-fix run TR retries truncated again 19.6%
# of the time, the worst of any type. The cure for "ran out of room" is room.
# Applies only to the TR resample, which is 56 problems -- a rounding error in the
# generation budget, already accounted for in the 4% typed/generic difference.
ACCEPT_TR_TOKEN_BOOST = 1.6          # revert: 1.0
# Compute parity: generic resamples at the SAME RATE as typed, chosen at random,
# keeping the second sample. Without this typed gets more generations and a
# reviewer can attribute the win to compute. Leave True for the paper.
# [AUDIT D48] Turned OFF by default -- the arms are already compute-matched.
#
# Measured on the 1,068 retries: symbolic repair returns WITHOUT generating on
# 15.5% of them, while the acceptance test spends an extra generation on ~23% of
# the rest. Net budget is 1.040 generations per retry for typed against 1.000 for
# generic -- a 4% difference, well inside what a re-run of generic would cost.
#
# So a completed generic arm does NOT need re-running to stay a fair control.
# Set this True only if you are running generic fresh anyway.
GENERIC_MATCH_RESAMPLE = False       # revert: True
GENERIC_RESAMPLE_RATE = 0.0          # set from the typed run; 0.0 = auto (measured)


# [AUDIT D46] A NON-GENERATIVE cure, routed by the diagnosis.
#
# Prompting fixes STRUCTURAL failures and cannot fix COMPUTATIONAL ones. The
# re-run showed exactly that: WP +6.0, ST +8.2, TR +6.1 over generic, but CE
# -16.0. So CE stops being re-prompted and gets its arithmetic repaired instead.
#
# Fires only when the trace is SELF-INCONSISTENT -- the answer it reports is the
# stated right-hand side of an equation that is arithmetically false. No gold is
# needed to detect that, or to know what the model's own arithmetic implies.
#
# Measured on 1,068 wrong traces: fires on 183 (17.1%), of which 44 land exactly
# on gold (+2.9 pts) for ZERO generations. On the 738-problem re-run it fixes 31
# the generative cure MISSED, worth +4.2 pts. It also saves GPU: repair runs
# BEFORE the retry, so ~20 of every 94 CE retries need no generation at all.
TYPED_SYMBOLIC_REPAIR = True         # revert: False
# [AUDIT D61] Repair must be a FALLBACK, never a PRE-EMPTION.
#
# The first design returned the repaired trace immediately and skipped the
# generative retry. That looked like a saving (166 free retries) and was a loss:
# repair's precision is only 24%, so 139 of those problems got a repaired-but-
# still-wrong answer AND forfeited the generative attempt that recovers ~17% of
# failures.
#
# Measured on the completed runs, scoring the union of attempts exactly as the
# harness already does:
#     GENERIC, plain retry                39.5%   (161 recovered)
#     EDER, repair pre-empts the retry    40.9%   (182)
#     repair as a fallback, both kept     42.3%   (202)
#
# Pre-empting cost about 20 problems. With this True the generative retry always
# runs, and the repaired candidate is kept alongside it; the one with no false
# equation wins, which is decidable without gold.
SYMBOLIC_REPAIR_AS_FALLBACK = True   # revert: False (pre-empt)
# Types whose diagnosis says the failure is computational. WP, ST and NR are
# excluded: repair fired on 0 of them, because arithmetic was never the problem.
SYMBOLIC_REPAIR_TYPES = ("CE", "SM", "AL", "TR", "UNCLASSIFIED")


# [AUDIT D49] The hard type filter starves retrieval, and D40 already replaced it.
#
# select_typed_positives filtered the pool through _typed_candidates() and then
# retrieved inside whatever survived. Measured cost in topical relevance:
#
#     type  eligible  share   filtered   unfiltered   relevance lost
#     SM        94     28%     0.0533      0.0913        -42%  (unfiltered is +71%)
#     AL       273     82%     0.0709      0.0893        -21%
#     NR       176     53%     0.0687      0.0861        -20%
#     WP       291     87%     0.0928      0.0963         -4%
#     ST/CE    334    100%     same        same            0%
#
# SM is the second-largest failure class and the filter was taking 42% of its
# relevance. The filter is also a no-op for ST and CE, where it passes the whole
# pool -- so it was contributing nothing where it was harmless and doing damage
# where it bit.
#
# D40's continuous type-fit re-rank supplies the typing signal already, blended
# with relevance, over the WHOLE pool. With this False, retrieval sees everything
# and typing comes from the blend rather than from a boolean gate.
RETRY_TYPE_HARD_FILTER = False       # revert: True


# [AUDIT D51] Write the question's numbers into SM's prefill instead of asking
# for them. 70% of SM traces (177 of 253) use a value the question never gave;
# a regex cannot hallucinate one.
SM_INJECT_QUESTION_NUMBERS = True    # revert: False

# [AUDIT D52] When the cascade abstains, still apply the best-matched cure.
# 84% of UNCLASSIFIED failures carry a clear gold-free signature (75% a false
# equation, 9% an unresolved relation) yet all of them got the generic cure --
# the worst-performing group at ~8-9% recovery. The RECORDED diagnosis stays
# UNCLASSIFIED, so the abstain rate the paper reports is unchanged.
UNCLASSIFIED_FALLBACK_CURE = True    # revert: False


# [AUDIT D55] Typed must be generic PLUS typing, never generic REPLACED by typing.
#
# select_typed_positives re-ranked a relevance shortlist by a blend of relevance
# and _type_fit_score, which DISPLACED the best-matched exemplars. Measured mean
# question-exemplar similarity, typed against generic on the same questions:
#
#     WP -22%   SM -22%   CE -30%   ST -25%   AL -14%   NR -14%
#
# Typed was handing the model materially worse exemplars than the control on
# every single type, and paying for it with a type-fit score that for CE barely
# varies at all (sd/|mean| = 0.11, because 3.0*V is constant over a pool where
# every exemplar is already correct). CE is typed's worst type and this is why.
#
# RETRY_RELEVANCE_FLOOR guarantees the top-N most relevant exemplars survive the
# re-rank. Typed then contains generic's best N, and typing decides only the
# remaining slots. Generic can no longer beat typed on exemplar quality, which is
# the property that makes "typed is generic plus a treatment" actually true.
# The knob, measured (40 problems, mean over WP/CE/SM/ST):
#     FLOOR   typed rel   vs generic   exemplars that differ
#         0      0.0991        -4%              2.6 of 8   <- the shipped bug
#         2      0.1060        +3%              2.5
#         3      0.1088        +5%              2.3        <- default
#         5      0.1099        +6%              1.8
#         7      0.1105        +7%              1.0
# 3 keeps typed comfortably above generic on relevance while leaving the most
# room for typing to differ. Note that even at FLOOR=0 only 2.6 exemplars
# differed -- the exemplar channel was never the large lever; the instruction,
# the prefill and the symbolic repair are.
# [AUDIT D62] Re-tuned now that both arms share ONE shortlist (D55c). Measured
# over 60 problems -- type-selected exemplars against relevance versus generic:
#
#     FLOOR  W_TYPEFIT   type-selected   typed rel vs generic
#         3      0.45         1.4                +1%
#         2      0.45         1.6                +1%
#         3      0.70         2.4                -4%
#         2      0.70         2.8                -6%
#         1      0.70         3.1                -8%
#
# floor=3/W=0.45 left only 1.4 of 8 exemplars actually chosen by type -- typed and
# generic were near-identical on the exemplar channel, so the whole treatment
# rested on the instruction, prefill, demo and repair. floor=2/W=0.70 doubles the
# type-selected count while typed stays within a few percent of generic on
# relevance, which is the property D55 was protecting.
RETRY_RELEVANCE_FLOOR = 2            # of 8 slots; revert: 3


# [AUDIT D60] A retry that reproduces the failed answer is worthless -- resample it.
#
# Measured on the completed typed run: 175 of 1,068 retries (16%) returned the
# EXACT same answer as the attempt that failed. Their recovery rate is 0.0% by
# definition. Retries whose answer changed recover at 18-33% depending on type.
#
#     type   same answer   recovery | changed   recovery | same
#     WP         25%             24.7%               0.0%
#     ST         24%             18.4%               0.0%
#     SM         19%             18.0%               0.0%
#     CE         10%             32.8%               0.0%
#     ALL        16%
#
# This is a far stronger trigger than the D44 acceptance test, which resampled on
# a weak correlate and lost 2.2 points doing it. "The answer did not move" is not
# a correlate of failure -- it IS failure, observable without gold. Replacing a
# candidate known to be worth 0% cannot lose.
#
# Upper bound if resamples reach the changed-answer rate: +35 problems, roughly
# +3.3 points of recovery. Realistically less, since a resample that changes the
# answer is not guaranteed to change it usefully.
#
# NOTE ON FAIRNESS: this mechanism needs no diagnosis, so a generic arm could use
# it too. Applying it to typed alone makes the comparison a system claim rather
# than a typing claim. Either re-run generic with it, or report the pass-1 -> final
# improvement rather than the typed-vs-generic delta. Say which you did.
RETRY_RESAMPLE_IF_UNCHANGED = True   # revert: False
UNCHANGED_RESAMPLE_TEMP = 0.8        # must be >0 or the resample repeats verbatim
UNCHANGED_MAX_RESAMPLES = 1


# [AUDIT D63] Layered prompt: general examples first, then type-matched, then the
# question. This is the structure the exemplar block can express SAFELY.
#
# The obvious implementation -- writing "the previous attempt failed like this"
# between the exemplars and the question -- is exactly the [D1/D2] bug. Exemplars
# are parsed by generation._EX_BLOCK_RE into real dialogue turns; free text inside
# the block breaks the '\n\nQuestion:' lookahead, the live question gets swallowed
# into the last exemplar's answer, and the model solves an exemplar instead.
# Measured cost when that happened: grounding 0.89 -> 0.12, and 0 of 325 NR/SM
# failures recovered.
#
# So the layering is carried by ORDER, which costs nothing and cannot break the
# parse:
#     [ relevance-retrieved, ascending ] [ type-selected ] [ cure demonstration ]
#                                                          ^ nearest the question
# and the failure mode is NAMED in the system message, where the instruction
# already lives and where no parser is looking.
RETRY_LAYERED_ORDER = True           # revert: False (most-relevant-last)
RETRY_NAME_FAILURE_MODE = True       # prepend "a previous attempt ..." to the instruction
