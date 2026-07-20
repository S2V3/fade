"""
loopguard.py  --  make a base model's degeneration loop harmless.

The problem: LLaMA-2-7B (a base model) frequently re-solves the same problem
several times in one generation. generation.py already stops cleanly when the
model emits '#### <number>' or a literal '\\nQuestion:' -- but a base model that
re-solves in PROSE ("Now let us solve again ...") or in repeated "Step 1/2/3"
blocks emits NEITHER anchor, so the whole loop runs to max_new_tokens and the
whole loop lands in the scored trace (inflating n_steps / duplicate equations,
cratering R, and misrouting the label).

This module adds two anchor-free defences:

  truncate_trace(text)        -- post-processing that keeps only the FIRST
                                 attempt, using (in order): a banned code fence,
                                 an explicit new-problem marker, the first
                                 '#### N', a "restart cue" after real content,
                                 and finally a repeated-substantial-line net that
                                 catches loops with no marker at all.

  RepeatLineStopping           -- a StoppingCriteria that halts GENERATION the
                                 moment a substantial line repeats, so you stop
                                 paying for the loop instead of just trimming it.

Both are conservative: they only ever cut at a *repeat* or an explicit
new-problem cue, so a legitimate single solution is never shortened.
"""

from __future__ import annotations

import re

# markers that mean "the model started a brand-new problem" (hard cut) ---------
STOP_MARKERS = ["\nQuestion:", "\nQ:", "\nExample", "\nProblem:"]

# a completed '#### <number>' (the intended answer terminator) -----------------
ANSWER_STOP_RE = re.compile(r"####\s*\$?-?\d[\d,]*(?:\.\d+)?")

# "restart cues" a base model uses when it loops WITHOUT emitting '####' --------
# (only meaningful once real content has been produced, hence the offset search)
_RESTART_RE = re.compile(
    r"\n\s*(?:question|q|problem|example"
    r"|now\s+(?:let|we|i)\b|again\b|let'?s\s+solve"
    r"|step\s*1\b|solution\s*:)",
    re.IGNORECASE,
)


def _norm_line(ln: str) -> str:
    """Normalised key for repeat detection: lowercase, alnum+space only."""
    k = re.sub(r"\s+", " ", ln.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", k)


def _cut_at_first_repeat(text: str, min_len: int = 12) -> str:
    """Cut at the first substantial line that has already appeared. A repeated
    non-trivial line is the signature of a re-solve loop."""
    seen: set[str] = set()
    kept: list[str] = []
    for ln in text.split("\n"):
        key = _norm_line(ln)
        if len(key) >= min_len:
            if key in seen:
                break                      # loop onset -> stop here
            seen.add(key)
        kept.append(ln)
    return "\n".join(kept)


def truncate_trace(text: str) -> str:
    """Keep only the model's FIRST attempt. Safe on a clean single solution."""
    # 0 -- never keep a code fence
    i = text.find("\\begin{code}")
    if i != -1:
        text = text[:i]

    # 1 -- explicit new-problem marker (the model started a fresh Question)
    cut = len(text)
    for mk in STOP_MARKERS:
        j = text.find(mk)
        if j != -1:
            cut = min(cut, j)
    text = text[:cut]

    # 2 -- the intended terminator: first '#### N' wins outright
    m = ANSWER_STOP_RE.search(text)
    if m:
        return text[:m.end()].strip()

    # 3 -- no '####': cut at the first restart cue that follows real content
    m2 = _RESTART_RE.search(text, 40)
    if m2:
        text = text[:m2.start()]

    # 4 -- final net: cut at the first repeated substantial line
    text = _cut_at_first_repeat(text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Generation-time stop: halt as soon as a substantial line repeats.
# Attach only for single-sequence generation (same as _StopOnAnswer).
# --------------------------------------------------------------------------- #
class RepeatLineStopping:
    """StoppingCriteria: stop when the generated region contains a repeated
    substantial line (a re-solve loop) OR a restart cue after real content.
    Decodes the whole generated region -- cheap at max_new_tokens<=512."""

    def __init__(self, tokenizer, prompt_len: int, min_len: int = 15,
                 check_every: int = 8):
        self.tok = tokenizer
        self.plen = prompt_len
        self.min_len = min_len
        self.check_every = check_every
        self._n = 0

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self._n += 1
        if self._n % self.check_every:          # throttle: check every k tokens
            return False
        new_ids = input_ids[0, self.plen:]
        if len(new_ids) < 12:
            return False
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        # restart cue after content?
        if _RESTART_RE.search(text, 40):
            return True
        # repeated substantial line?
        seen: set[str] = set()
        for ln in text.split("\n"):
            key = _norm_line(ln)
            if len(key) >= self.min_len:
                if key in seen:
                    return True
                seen.add(key)
        return False