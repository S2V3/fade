"""
preprocessing.py  --  the step FADE was missing: normalise text BEFORE the
instrument reads it.

Why this matters (and why it is not cosmetic): every signal in components.py is
built by REGEX + sympy over raw strings. A base model writes math in whatever
glyphs it feels like -- a unicode minus (U+2212 '\u2212'), a times sign (U+00D7
'\u00d7'), a division sign (U+00F7 '\u00f7'), a vulgar fraction ('\u00bd'), a
non-breaking space between a number and its unit. Every one of those silently
DEFEATS the equation regex and the value extractor, so a perfectly good step is
scored as a miss / a non-equation / incoherent. Normalising the trace before
scoring recovers those -- it raises the RECALL of E, V, A and coherence without
touching their meaning.

Two entry points:
  clean_question(q)   -- safe text tidy for the question shown in the prompt and
                         read by the categorizer / G-score (unicode + whitespace).
  normalize_math(t)   -- clean_question PLUS math-glyph folding and vulgar-fraction
                         expansion; use on the model's TRACE and on gold solutions
                         right before scoring.

Design rule: only *representation* is changed, never *value*. '\u2212' -> '-' is the
same number; '\u00d7' -> '*' is the same operator; '\u00bd' -> '1/2' is the same
quantity. Nothing here can change which answer a trace states.
"""

from __future__ import annotations

import re

# ---- glyphs that are unambiguously the ASCII math character -----------------
_MATH_MAP = {
    "\u2212": "-",   # minus sign        -> hyphen-minus
    "\u00d7": "*",   # multiplication x  -> star (extraction handles * and x)
    "\u2217": "*",   # asterisk operator
    "\u00f7": "/",   # division sign
    "\u2044": "/",   # fraction slash
    "\u2215": "/",   # division slash
    "\uff1d": "=",   # full-width equals
    "\uff0b": "+",   # full-width plus
}

# ---- whitespace variants that should all collapse to a normal space ---------
_SPACE_MAP = {
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ",
    "\u202f": " ", "\u2002": " ", "\u2003": " ", "\ufeff": "",
}

# ---- vulgar fractions -> "a/b" (helps the fraction/value extractor) ----------
_VULGAR = {
    "\u00bd": "1/2", "\u2153": "1/3", "\u2154": "2/3", "\u00bc": "1/4",
    "\u00be": "3/4", "\u2155": "1/5", "\u2156": "2/5", "\u2157": "3/5",
    "\u2158": "4/5", "\u2159": "1/6", "\u215a": "5/6", "\u215b": "1/8",
    "\u215c": "3/8", "\u215d": "5/8", "\u215e": "7/8",
}

# ---- smart quotes -> ascii (pure tidy; no effect on math) --------------------
_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2032": "'", "\u2033": '"',
}

_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")


def _apply(mapping: dict, s: str) -> str:
    for k, v in mapping.items():
        if k in s:
            s = s.replace(k, v)
    return s


def clean_question(text: str) -> str:
    """Safe tidy for question text: fold odd spaces and quotes, collapse runs of
    whitespace. Does NOT touch math glyphs (a question rarely computes)."""
    if not text:
        return text
    s = _apply(_SPACE_MAP, text)
    s = _apply(_QUOTES, s)
    s = _MULTISPACE.sub(" ", s)
    s = _MULTINEWLINE.sub("\n\n", s)
    return s.strip()


def normalize_math(text: str) -> str:
    """clean_question + math-glyph folding + vulgar-fraction expansion. Use on the
    model TRACE and on gold solutions immediately before scoring so the instrument
    sees canonical ASCII math."""
    if not text:
        return text
    s = clean_question(text)
    s = _apply(_VULGAR, s)
    s = _apply(_MATH_MAP, s)
    return s


def preprocess_problem(problem: dict) -> dict:
    """Return a shallow-cleaned copy of a GSM8K problem dict.
    Question is tidied; the annotated gold answer is math-normalised (idempotent
    on already-clean GSM8K, but robust if a variant slips in)."""
    out = dict(problem)
    if "question" in out:
        out["question"] = clean_question(out["question"])
    if "answer" in out:
        out["answer"] = normalize_math(out["answer"])
    return out