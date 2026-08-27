"""
Selects the best seed exemplars from the first 300 GSM8K problems,
one per strategy (or per category for few-shot strategies), using
KNN via sentence-transformers + our quality evaluator scoring.

Design:
  • Each strategy declares how many exemplars it needs (STRATEGY_EXEMPLAR_NEEDS)
  • Zero-shot strategies need 0 exemplars
  • Few-shot strategies need N exemplars (one per category_type = 8 categories)
  • Self-consistency / sampling strategies use same pool as few-shot
  • Full pipeline needs all 8 categories

Selection method:
  1. Embed all 300 problems using sentence-transformers (all-MiniLM-L6-v2)
  2. Cluster by category_type (from categorizer)
  3. Within each category, score each problem with quality_evaluator
     (static heuristic score since we don't have traces yet)
  4. Pick top-k by quality score within each category
  5. Discard those problems from the generation pool

Returns:
  - manual_exemplars: Dict[strategy_id → List[exemplar_dict]]
  - remaining_problems: List of problem dicts NOT used as exemplars
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    import torch
    print("Torch OK")
except Exception as e:
    print(e)

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
    print(" sentence-transformers OK")

except Exception as e:
    print("[warn] SBERT unavailable:", e)
    SBERT_AVAILABLE = False
    print("[warn] sentence-transformers not found — falling back to TF-IDF similarity")

# ── Strategy exemplar requirements ────────────────────────────────────────────
# (strategy_id → n_per_category)
# n=0  → no exemplars at all (zero-shot strategies)
# n>0  → select exactly n exemplars FROM EACH of the 8 category types
#         so total = n × 8  (e.g. n=5 → 40 exemplars for few_shot)
#
# At prompt-build time, generation.py uses all stored exemplars for that
# strategy; KNN / dynamic strategies further filter to the most relevant ones.
STRATEGY_EXEMPLAR_NEEDS: Dict[int, int] = {
    0:  0,   # zero_shot            — no exemplars
    1:  0,   # zero_shot_cot        — no exemplars
    2:  5,   # few_shot             — 5 per category  (40 total)
    3:  3,   # few_shot_dynamic     — 3 per category  (24 total), KNN picks at runtime
    4:  0,   # self_consistency     — no exemplars
    5:  0,   # temp_low             — no exemplars
    6:  0,   # temp_high            — no exemplars
    7:  3,   # explicit_steps       — 3 per category  (24 total)
    8:  3,   # scratchpad           — 3 per category  (24 total)
    9:  3,   # math_first           — 3 per category  (24 total)
    10: 2,   # category_context     — 2 per category  (16 total)
    11: 2,   # worked_example       — 2 per category  (16 total; use best 1 at runtime)
    12: 1,   # reversed_qa          — 1 per category  (8 total; control condition)
    13: 5,   # cot_few_shot         — 5 per category  (40 total)
    14: 3,   # cot_self_consistency — 3 per category  (24 total)
    15: 5,   # full_pipeline        — 5 per category  (40 total)
    16: 5,   # fade_best            — 5 per category  (40 total), KNN picks 8 at runtime
}

CATEGORY_TYPES = [
    'percentage', 'monetary', 'time', 'rate',
    'counting', 'ratio', 'logic', 'arithmetic'
]


# ── Heuristic problem quality scorer (no trace needed) ────────────────────────

def _score_problem_richness(question: str) -> float:
    """
    Quick heuristic to score a problem's quality as a potential exemplar
    without needing a generated trace. Based on:
      - Length (too short = trivial, too long = complex)
      - Number count (more numbers = more computation steps)
      - Presence of multi-step indicators
    """
    import re
    q = question.lower()

    # Length score: ideal 50-200 chars
    l = len(question)
    if l < 40:
        len_score = 0.3
    elif l <= 200:
        len_score = 1.0
    else:
        len_score = max(0.4, 1.0 - (l - 200) / 500)

    # Number density: more numbers → more computation
    nums = re.findall(r'\b\d+(?:\.\d+)?\b', question)
    num_score = min(1.0, len(nums) / 5)

    # Multi-step indicators
    step_words = ['then', 'after', 'also', 'each', 'total', 'remaining',
                  'left', 'more', 'less', 'how many', 'how much', 'per']
    step_hits = sum(1 for w in step_words if w in q)
    step_score = min(1.0, step_hits / 3)

    return round(0.35 * len_score + 0.35 * num_score + 0.30 * step_score, 4)


# ── Embedding model (lazy load) ────────────────────────────────────────────────

_sbert_model = None

# [AUDIT / retrieval upgrade 4] get_knn_exemplars re-embedded the ENTIRE candidate
# pool on EVERY call (`_get_embeddings(all_texts + [query])`). At pool = 133 and
# n = 1500 that is ~200k redundant encodes per run. Pool records already persist
# an `embedding` under MiniLM; this cache covers the rest.
_EMB_CACHE: Dict[str, np.ndarray] = {}
_EMB_CACHE_MAX = 50000


def _get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Returns (N, D) embedding matrix, memoised per text.
    Uses sentence-transformers if available, else TF-IDF fallback.
    """
    global _sbert_model
    if SBERT_AVAILABLE:
        if _sbert_model is None:
            print("  Loading sentence-transformers (all-MiniLM-L6-v2)…")
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        missing = [t for t in dict.fromkeys(texts) if t not in _EMB_CACHE]
        if missing:
            vecs = _sbert_model.encode(missing, show_progress_bar=False,
                                       batch_size=64, normalize_embeddings=True)
            if len(_EMB_CACHE) > _EMB_CACHE_MAX:
                _EMB_CACHE.clear()
            for t, v in zip(missing, vecs):
                _EMB_CACHE[t] = v
        return np.asarray([_EMB_CACHE[t] for t in texts])
    else:
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(max_features=512, ngram_range=(1, 2))
        mat = vec.fit_transform(texts).toarray().astype(np.float32)
        return normalize(mat)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit vectors."""
    return float(np.dot(a, b))


# ── Main selector class ────────────────────────────────────────────────────────

class ExemplarSelector:
    """
    Select best manual exemplars from the first 300 GSM8K problems.

    Usage:
        selector = ExemplarSelector()
        manual_exemplars, remaining = selector.select(problems_300, categorizer)
    """

    def __init__(self, pool_size: int = 300):
        self.pool_size = pool_size

    def select(
        self,
        problems: List[Dict],          # List of {question, answer} from GSM8K
        categorizer,                    # QuestionCategorizer instance
        ground_truth_key: str = 'answer',
    ) -> Tuple[Dict[int, List[Dict]], List[Dict]]:
        """
        Args:
            problems        : First `pool_size` GSM8K problems
            categorizer     : QuestionCategorizer instance
            ground_truth_key: key in problem dict that holds ground truth

        Returns:
            manual_exemplars : {strategy_id → [exemplar_dict, …]}
            remaining_problems: problems NOT selected as exemplars
        """
        problems = problems[:self.pool_size]
        print(f"\n[ExemplarSelector] Scoring {len(problems)} problems…")

        # ── Step 1: Categorise + score every problem ──────────────────────────
        scored: List[Dict] = []
        for idx, prob in enumerate(problems):
            question = prob['question']
            cat_info = categorizer.categorize(question)
            richness = _score_problem_richness(question)
            scored.append({
                'original_index' : idx,
                'question'       : question,
                'ground_truth'   : prob[ground_truth_key],
                'richness_score' : richness,
                **cat_info,
            })

        # ── Step 2: Embed all questions ───────────────────────────────────────
        print("  Computing embeddings…")
        questions = [s['question'] for s in scored]
        embeddings = _get_embeddings(questions)   # shape (N, D)
        for i, s in enumerate(scored):
            s['_emb'] = embeddings[i]

        # ── Step 3: Group by category_type ───────────────────────────────────
        by_category: Dict[str, List[Dict]] = defaultdict(list)
        for s in scored:
            by_category[s['category_type']].append(s)

        # Within each category, sort by richness descending
        for cat in by_category:
            by_category[cat].sort(key=lambda x: x['richness_score'], reverse=True)

        # ── Step 4: Select exemplars per strategy ────────────────────────────
        # Rule: every strategy with n_per_cat > 0 gets exactly n_per_cat
        # exemplars FROM EACH of the 8 category types → total = n × 8.
        # Strategies are processed most-demanding-first so they get first pick
        # of the best problems in each category.
        selected_indices: set = set()
        manual_exemplars: Dict[int, List[Dict]] = {}

        strategy_order = sorted(
            range(16),
            key=lambda s: STRATEGY_EXEMPLAR_NEEDS[s],
            reverse=True,   # most demanding picks first
        )

        for sid in strategy_order:
            n_per_cat = STRATEGY_EXEMPLAR_NEEDS[sid]
            if n_per_cat == 0:
                manual_exemplars[sid] = []
                continue

            # n_per_cat exemplars from each category (top-richness, unused first)
            chosen = self._select_n_per_category(
                by_category, selected_indices, n_per_cat=n_per_cat
            )

            for ex in chosen:
                selected_indices.add(ex['original_index'])

            clean = [{k: v for k, v in ex.items() if k != '_emb'} for ex in chosen]
            manual_exemplars[sid] = clean

        # ── Step 5: Remaining problems ────────────────────────────────────────
        remaining = [
            {k: v for k, v in s.items() if k != '_emb'}
            for s in scored
            if s['original_index'] not in selected_indices
        ]

        print(f"  Selected {len(selected_indices)} exemplars across strategies")
        print(f"  Remaining for generation: {len(remaining)} problems\n")

        self._print_summary(manual_exemplars)
        return manual_exemplars, remaining

    # ── helpers ───────────────────────────────────────────────────────────────

    def _select_n_per_category(
        self,
        by_category: Dict[str, List[Dict]],
        used: set,
        n_per_cat: int,
    ) -> List[Dict]:
        """
        Pick exactly n_per_cat exemplars from each of the 8 category types,
        sorted by richness_score descending within each category.
        Skips already-used problems; falls back to reuse only if a category
        has fewer than n_per_cat unused problems left.
        """
        chosen = []
        for cat in CATEGORY_TYPES:
            all_in_cat  = by_category.get(cat, [])   # already sorted by richness
            unused      = [p for p in all_in_cat if p['original_index'] not in used]

            if len(unused) >= n_per_cat:
                picked = unused[:n_per_cat]
            else:
                # Use all unused first, then fill from already-used (reuse)
                reusable = [p for p in all_in_cat if p['original_index'] in used]
                picked   = unused + reusable[:n_per_cat - len(unused)]

            if not picked:
                print(f"  [warn] no problems found for category '{cat}' — skipping")

            chosen.extend(picked)
        return chosen

    # ── KNN lookup (used at generation time in main) ──────────────────────────

    def get_knn_exemplars(
        self,
        query_question: str,
        candidate_exemplars: List[Dict],
        k: int = 3,
        category_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        At generation time: retrieve K most similar exemplars from the
        accumulated pool using KNN cosine similarity.

        Args:
            query_question      : the problem being solved
            candidate_exemplars : list of exemplar dicts (must have 'question' key)
            k                   : number to retrieve
            category_type       : if given, boost same-category exemplars

        Returns:
            List of up to k exemplar dicts, most similar first
        """
        if not candidate_exemplars:
            return []

        all_texts     = [e['question'] for e in candidate_exemplars]
        all_embeddings = _get_embeddings(all_texts + [query_question])
        pool_embs     = all_embeddings[:-1]   # (N, D)
        query_emb     = all_embeddings[-1]    # (D,)

        sims = [_cosine_sim(query_emb, e) for e in pool_embs]

        # Category boost: +0.15 to same-category exemplars
        if category_type:
            for i, ex in enumerate(candidate_exemplars):
                if ex.get('category_type') == category_type:
                    sims[i] = min(1.0, sims[i] + 0.15)

        # Sort by similarity descending
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [candidate_exemplars[i] for i in ranked[:k]]

    def _print_summary(self, manual_exemplars: Dict[int, List[Dict]]):
        names = {
            0:'zero_shot',1:'zero_shot_cot',2:'few_shot',3:'few_shot_dynamic',
            4:'self_consistency',5:'temp_low',6:'temp_high',7:'explicit_steps',
            8:'scratchpad',9:'math_first',10:'category_context',
            11:'worked_example',12:'reversed_qa',13:'cot_few_shot',
            14:'cot_self_consistency',15:'full_pipeline'
        }
        print(f"\n  {'Strategy':<25} {'n/cat':>5} {'Total':>6}  Per-category counts")
        print(f"  {'-'*80}")
        for sid in range(16):
            exs       = manual_exemplars.get(sid, [])
            n_per_cat = STRATEGY_EXEMPLAR_NEEDS[sid]
            # Count how many were actually selected per category
            cat_counts = {cat: 0 for cat in CATEGORY_TYPES}
            for ex in exs:
                c = ex.get('category_type', '?')
                if c in cat_counts:
                    cat_counts[c] += 1
            if n_per_cat == 0:
                per_cat_str = 'N/A (zero-shot)'
            else:
                per_cat_str = '  '.join(
                    f"{cat[:4]}={cat_counts[cat]}" for cat in CATEGORY_TYPES
                )
            print(f"  {names[sid]:<25} {n_per_cat:>5} {len(exs):>6}  {per_cat_str}")

# =============================================================================
# THREE-STAGE RETRIEVAL (design doc Layer-1, section 7.3)
#
# get_knn_exemplars() above is STAGE 1 ONLY (cosine + category boost). It is
# deliberately kept as its own experimental arm: similarity-only retrieval IS
# the naive-accumulation baseline (SG-ICL lineage) that FADE must beat.
#
# This function adds the two missing stages:
#   Stage 2 -- COMPLEXITY RE-RANK. Fu et al. (2022): on math, *complex*
#              exemplars (more reasoning steps) outperform similar-but-simple
#              ones. Score = 0.4*relevance + 0.6*normalised step count.
#   Stage 3 -- MMR-STYLE DIVERSITY. Skip a candidate whose cosine to an
#              already-accepted exemplar exceeds 0.85: k near-duplicates teach
#              about as much as one, and waste context that could carry a
#              different reasoning pattern.
# =============================================================================

_COMPLEXITY_WORDS = {"simple": 2, "medium": 4, "complex": 6}


def exemplar_step_count(ex: Dict) -> int:
    """Best available estimate of an exemplar's REASONING DEPTH.

    [AUDIT D31] This used to prefer `signals['n_steps']` -- the PROSE LINE COUNT.
    The complexity re-rank carries 40-60% of the ranking score, so the single
    most load-bearing component of retrieval was ranking exemplars by how much
    text they contained. Measured on the pool: mean n_steps 5.62 vs mean
    n_checkpoints 2.58, and the two correlate at only **r = 0.279** -- so this is
    not a harmless proxy. Worse, D27 showed line count is ANTI-correlated with
    correctness (separation -0.324), which means Fu et al.'s "prefer complex
    exemplars" heuristic was being implemented as "prefer verbose ones".

    Order now: computational steps -> gold checkpoints -> never raw line count.
    """
    sig = ex.get("signals") or {}
    for key in ("n_comp_steps", "n_checkpoints"):
        if isinstance(sig.get(key), (int, float)) and sig[key]:
            return int(sig[key])
    c = ex.get("complexity")
    if isinstance(c, (int, float)) and c:
        return int(c)
    if isinstance(c, str) and c in _COMPLEXITY_WORDS:
        return _COMPLEXITY_WORDS[c]
    if isinstance(ex.get("estimated_steps"), (int, float)):
        return int(ex["estimated_steps"])
    trace = ex.get("trace") or ""
    return max(1, len([l for l in trace.splitlines()
                       if l.strip() and not l.strip().startswith("####")]))


# [AUDIT D32] Structural question similarity, measured to be the best-performing
# ranking component in the offline benchmark (eval_retrieval.py):
#     random 0.252 | cosine 0.271 | 3stage_lines 0.258 | 3stage_fixed 0.281
#     structural 0.284 | HYBRID 0.285   (precision@8, TF-IDF, 143-exemplar pool)
# Note 3stage_lines -- the shipped code before D31 -- was barely above random.
_Q_NUM = re.compile(r"\d+(?:\.\d+)?")
_Q_CUES = {
    "per": r"\bper\b|each|every", "pct": r"%|percent",
    "money": r"\$|dollar|cost|price|pay", "time": r"hour|minute|day|week|month|year",
    "cmp": r"more than|less than|twice|half|double",
    "tot": r"total|altogether|in all|combined|left|remaining",
}


def question_shape(q: str) -> tuple:
    """Gold-free structural fingerprint of a question: (#numbers, #words, cue bits)."""
    nums = _Q_NUM.findall(q or "")
    bits = tuple(1 if re.search(p, q or "", re.I) else 0 for p in _Q_CUES.values())
    return (len(nums), len((q or "").split()), bits)


def question_shape_sim(a: tuple, b: tuple) -> float:
    """Similarity of two question shapes, in [0,1]."""
    cue = sum(1 for x, y in zip(a[2], b[2]) if x == y) / max(len(a[2]), 1)
    dn = 1.0 - min(abs(a[0] - b[0]) / 6.0, 1.0)
    dw = 1.0 - min(abs(a[1] - b[1]) / 60.0, 1.0)
    return 0.5 * cue + 0.35 * dn + 0.15 * dw


def _reasoning_shape(ex: Dict) -> tuple:
    """[AUDIT / retrieval upgrade 3] A coarse fingerprint of HOW an exemplar
    reasons: (operator multiset, step-count band). MMR deduped on QUESTION cosine,
    but two questions can be textually dissimilar and still demonstrate the
    identical two-step multiply-then-add pattern -- so the diversity stage was
    spending context on redundant *reasoning* while believing it was diverse."""
    trace = ex.get("trace") or ""
    ops = frozenset(ch for ch in trace if ch in "+-*/")
    return (ops, min(exemplar_step_count(ex), 8))


def get_exemplars_3stage(
    self,
    query_question: str,
    candidate_exemplars: List[Dict],
    k: int = 3,
    category_type: Optional[str] = None,
    shortlist: int = 15,
    w_relevance: Optional[float] = None,
    w_complexity: Optional[float] = None,
    dup_threshold: Optional[float] = None,
    ftype: Optional[str] = None,
) -> List[Dict]:
    """Full Layer-1 retrieval: relevance -> complexity re-rank -> MMR diversity.

    [AUDIT] Three changes:
      * weights now come from config and DEFAULT THE OTHER WAY. They were
        w_relevance=0.4 / w_complexity=0.6, so complexity DOMINATED. Pool traces
        average 7.12 lines against gold seeds' 5.1, which meant the re-rank
        systematically preferred VERBOSE MODEL TRACES over CONCISE CORRECT GOLD.
      * diversity also dedupes on reasoning SHAPE, not only question cosine.
      * `ftype` enables soft type-matching: for a typed retry the diagnosis is
        known, so score type-match jointly with similarity instead of filtering
        by type first and discarding the similarity ordering inside it.
    """
    import config as _cfg
    if w_relevance is None:
        w_relevance = _cfg.RETRIEVAL_W_RELEVANCE
    if w_complexity is None:
        w_complexity = _cfg.RETRIEVAL_W_COMPLEXITY
    if dup_threshold is None:
        dup_threshold = _cfg.RETRIEVAL_DUP_THRESHOLD

    if not candidate_exemplars:
        return []
    if len(candidate_exemplars) <= k:
        return list(candidate_exemplars)

    from similarity import embed

    questions = [ex["question"] for ex in candidate_exemplars] + [query_question]
    embs = embed(questions)
    pool_embs, query_emb = embs[:-1], embs[-1]

    # ---- Stage 1: relevance (cosine + same-category boost) -> shortlist ----
    sims = [_cosine_sim(query_emb, e) for e in pool_embs]
    if category_type:
        for i, ex in enumerate(candidate_exemplars):
            if ex.get("category_type") == category_type:
                sims[i] = min(1.0, sims[i] + 0.15)
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    short = order[:max(shortlist, k)]

    # ---- Stage 2: complexity re-rank (Fu et al.: complex > similar-but-simple)
    #      plus optional soft TYPE match [retrieval upgrade 2]
    steps = [exemplar_step_count(candidate_exemplars[i]) for i in short]
    max_steps = max(steps) or 1

    type_bonus = {}
    if ftype and _cfg.RETRIEVAL_TYPED_SOFT:
        try:
            from diagnosis import FailureType
            import kaggle_run as _K
            matched = {id(r) for r in _K._typed_candidates(
                [candidate_exemplars[i] for i in short], FailureType(ftype))}
            type_bonus = {i: (1.0 if id(candidate_exemplars[i]) in matched else 0.0)
                          for i in short}
        except Exception:
            type_bonus = {}

    # [AUDIT D32] structural blend
    w_struct = getattr(_cfg, "RETRIEVAL_W_STRUCTURAL", 0.0)
    q_shape = question_shape(query_question) if w_struct else None
    shapes = ({i: question_shape(candidate_exemplars[i].get("question", ""))
               for i in short} if w_struct else {})

    def _score(i: int, st: int) -> float:
        base = w_relevance * sims[i] + w_complexity * (st / max_steps)
        if w_struct:
            base += w_struct * question_shape_sim(q_shape, shapes[i])
        if type_bonus:
            base += _cfg.RETRIEVAL_W_TYPEMATCH * type_bonus.get(i, 0.0)
        return base

    scored = sorted(zip(short, steps), key=lambda t: _score(t[0], t[1]), reverse=True)

    # ---- Stage 3: MMR-style diversity (drop near-duplicate exemplars) ----
    chosen: List[int] = []
    seen_shapes: set = set()
    for idx, _ in scored:
        if any(_cosine_sim(pool_embs[idx], pool_embs[j]) > dup_threshold
               for j in chosen):
            continue
        if _cfg.RETRIEVAL_DEDUPE_SHAPE:
            shape = _reasoning_shape(candidate_exemplars[idx])
            if shape in seen_shapes:
                continue
            seen_shapes.add(shape)
        chosen.append(idx)
        if len(chosen) >= k:
            break
    for idx, _ in scored:                     # top up if diversity was too strict
        if len(chosen) >= k:
            break
        if idx not in chosen:
            chosen.append(idx)
    return [candidate_exemplars[i] for i in chosen[:k]]


ExemplarSelector.get_exemplars_3stage = get_exemplars_3stage