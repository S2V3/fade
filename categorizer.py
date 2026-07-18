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
    Last match in priority list wins (most specific).
    """
    text_lower = text.lower()
    matched_category = 'arithmetic'   # default
    matched_keywords: List[str] = []

    for cat_name, keywords in _TYPE_RULES:
        hits = [kw for kw in keywords if kw in text_lower]
        if hits:
            matched_category = cat_name
            matched_keywords = hits

    return matched_category, matched_keywords


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

    if n_steps <= 3:
        label = 'simple'
    elif n_steps <= 5:
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