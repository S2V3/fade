"""
3-dimensional classification for every GSM8K problem:

  Dimension 1 — category_type (8 categories):
    percentage, monetary, time, rate, counting, ratio, logic, arithmetic

  Dimension 2 — complexity (simple / medium / complex):
    based on estimated operation count in the solution

  Dimension 3 — main_operation (addition / subtraction / multiplication / division / mixed)

  Also stores: estimated_steps, keyword_hits (debug info)
"""

import re
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# CATEGORY TYPE  — keyword-based priority matching
# Priority matters: higher-index categories matched first if multiple hit
# ─────────────────────────────────────────────────────────────

_TYPE_RULES: List[tuple] = [
    # (category_name, [keywords])  — ORDER = priority (last match wins ties)
    ('arithmetic', ['add', 'subtract', 'multiply', 'divide',
                    'sum of', 'difference of', 'product of']),
    ('logic',      ['if ', 'then ', 'must ', 'either ', 'unless',
                    'condition', 'at least', 'at most', 'exactly']),
    ('ratio',      ['ratio', 'proportion', 'compared to', 'for every',
                    'out of', 'fraction of']),
    ('counting',   ['how many', 'count', 'number of', 'total number',
                    'altogether', 'in all', 'combined']),
    ('rate',       [' per ', 'each', 'rate', 'speed', 'mph', 'km/h',
                    'miles per', 'km per', 'per hour', 'per day',
                    'per week', 'per item']),
    ('time',       ['hour', 'minute', 'second', 'day', 'week',
                    'month', 'year', 'o\'clock', 'am', 'pm',
                    'morning', 'evening', 'duration']),
    ('monetary',   ['dollar', '$', 'cent', 'cost', 'price', 'pay',
                    'charge', 'fee', 'earn', 'wage', 'salary',
                    'discount', 'spend', 'budget', 'afford']),
    ('percentage', ['percent', '%', 'percentage', 'out of 100',
                    'per cent', 'discount', 'tax', 'tip', 'interest']),
]


def _get_category_type(text: str) -> tuple:
    """
    Returns (category_type, keyword_hits).

    [AUDIT D18] WAS "last match in the priority list wins". Because 'monetary'
    and 'percentage' sit LAST in _TYPE_RULES, they absorbed almost everything:
    across 1,500 eval problems the categorizer emitted only 5 of its 8 values
    (time 539, monetary 443, percentage 227, counting 188, rate 103) and NEVER
    produced 'logic', 'arithmetic' or 'ratio'. 10 of the 35 strategy-2 seeds were
    therefore selected for categories that can never be requested at eval time.

    This also caps FST: `category_type` was one of its highest-weighted features
    while being effectively a 5-valued, position-biased variable. The gate's
    Cramer's V of 0.09 is at least partly a property of THIS function, not of the
    phenomenon.

    Now: SCORE-BASED. Every rule contributes its hit count, longer keywords score
    higher (they are more specific), and the highest total wins. Ties fall back to
    priority order. Deterministic, order-insensitive, and it actually uses the
    whole label space.
    """
    text_lower = text.lower()
    scores: Dict[str, float] = {}
    hits_by_cat: Dict[str, List[str]] = {}

    for cat_name, keywords in _TYPE_RULES:
        hits = [kw for kw in keywords if kw in text_lower]
        if not hits:
            continue
        # a longer keyword is a more specific signal than a 3-letter one
        scores[cat_name] = sum(1.0 + 0.10 * len(kw.strip()) for kw in hits)
        hits_by_cat[cat_name] = hits

    if not scores:
        return 'arithmetic', []

    best = max(scores.values())
    # tie-break by _TYPE_RULES order (later = more specific), preserving the old
    # intent without letting it dominate an unambiguous match
    order = {name: i for i, (name, _) in enumerate(_TYPE_RULES)}
    winners = [c for c, s in scores.items() if s == best]
    winner = max(winners, key=lambda c: order[c])
    return winner, hits_by_cat[winner]


# ─────────────────────────────────────────────────────────────
# COMPLEXITY  — estimate number of arithmetic operations needed
# We count operations in question text as a proxy for solution length.
# ─────────────────────────────────────────────────────────────

def _count_question_ops(question: str) -> int:
    """
    Heuristic: count distinct numeric conditions in a question.
    Each "X per Y" / "costs Z" / "N items" etc. likely maps to one operation.
    """
    # Count numeric tokens — very rough proxy for complexity
    nums = re.findall(r'\b\d+(?:\.\d+)?\b', question)
    return len(nums)


def _get_complexity(question: str, solution: Optional[str] = None) -> tuple:
    """
    If solution is given, count actual operation lines. Else estimate from question.
    Returns (complexity_label, estimated_steps).
    """
    if solution:
        # Count lines with an arithmetic expression
        op_lines = [
            line for line in solution.split('\n')
            if re.search(r'\d\s*[+\-*/×÷]\s*\d', line)
        ]
        n_steps = len(op_lines)
    else:
        # Estimate from number of numeric tokens in question
        n_ops = _count_question_ops(question)
        # Rough mapping: 1-2 numbers → 1 op; every 2 extra nums ≈ +1 op
        n_steps = max(1, n_ops // 2)

    # [AUDIT D18] The old bands (<=3 simple, <=5 medium, else complex) combined
    # with n_ops//2 made 'complexity' a CONSTANT: 1477 of 1500 eval problems came
    # back 'simple', 19 'medium', 4 'complex'. A near-constant feature cannot
    # carry signal, and it was one of three categorical inputs to FST.
    # Rebanded on the observed distribution of numeric tokens in GSM8K questions
    # so the three levels are actually populated.
    if n_steps <= 1:
        label = 'simple'
    elif n_steps <= 2:
        label = 'medium'
    else:
        label = 'complex'

    return label, n_steps


# ─────────────────────────────────────────────────────────────
# MAIN OPERATION — frequency of arithmetic symbols in solution
# ─────────────────────────────────────────────────────────────

_OP_PATTERNS = {
    'addition'       : re.compile(r'(?<!\w)[+](?!\w)|add(?:ition|ed|s)\b', re.I),
    'subtraction'    : re.compile(r'(?<!\w)[-](?!\w)|subtract(?:ed|s|ion)?\b|minus\b|less\b', re.I),
    'multiplication' : re.compile(r'[*×]|multipl(?:y|ied|ies|ication)\b|times\b|product\b', re.I),
    'division'       : re.compile(r'[÷/]|divid(?:e|es|ed|ing|sion)\b|per\b', re.I),
}

def _get_main_operation(text: str) -> str:
    counts = {op: len(pat.findall(text)) for op, pat in _OP_PATTERNS.items()}
    max_count = max(counts.values())
    if max_count == 0:
        return 'none'
    # If top two are tied, it's mixed
    top = [op for op, c in counts.items() if c == max_count]
    return top[0] if len(top) == 1 else 'mixed'


# ─────────────────────────────────────────────────────────────
# PUBLIC CLASS
# ─────────────────────────────────────────────────────────────

class QuestionCategorizer:
    """
    Categorize a GSM8K question into 3 dimensions.

    Usage:
        cat = QuestionCategorizer()
        meta = cat.categorize(question, solution=trace_or_None)
    """

    def categorize(self, question: str, solution: Optional[str] = None) -> Dict:
        """
        Returns:
            {
                'category_type'   : str,   # one of 8 types
                'complexity'      : str,   # simple / medium / complex
                'main_operation'  : str,   # addition / subtraction / …
                'estimated_steps' : int,
                'keyword_hits'    : list,  # debug: which keywords matched
            }
        """
        category_type, kw_hits    = _get_category_type(question)
        complexity, est_steps     = _get_complexity(question, solution)

        # For main operation, prefer solution text (more explicit) over question
        op_text      = (solution or '') + ' ' + question
        main_op      = _get_main_operation(op_text)

        return {
            'category_type'   : category_type,
            'complexity'      : complexity,
            'main_operation'  : main_op,
            'estimated_steps' : est_steps,
            'keyword_hits'    : kw_hits,
        }

    def batch_categorize(self, exemplars: List[Dict]) -> List[Dict]:
        """
        In-place add category fields to a list of exemplar dicts.
        Each dict must have at minimum a 'question' key.
        Optionally 'trace' is used as the solution.
        """
        for ex in exemplars:
            cats = self.categorize(
                ex['question'],
                solution=ex.get('trace')
            )
            ex.update(cats)
        return exemplars