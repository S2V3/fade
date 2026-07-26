"""
loopguard.py  --  make a base model's degeneration loop harmless, WITHOUT ever
cutting a legitimate single solution short.

Background: LLaMA-2-7B (a base model) often re-solves the same problem several
times in one generation. generation.py already stops cleanly when the model
emits '#### <number>' or a literal '\nQuestion:'. But a base model that
re-solves in prose or in repeated "Step 1/2/3" blocks emits NEITHER anchor, so
the whole loop runs to max_new_tokens and lands in the scored trace.

The ONLY reliable anchor-free signal that a loop has begun is a **verbatim
repeated substantial line**. An earlier draft of this file also treated phrases
like "now let's", "again", "solution:" as restart cues -- that was WRONG: those
are normal chain-of-thought connectives, so the guard cut correct traces off
before their answer. This version uses repeated-line detection only, so a clean
single solution is never shortened.

Two entry points:
  truncate_trace(text)   -- keep only the FIRST attempt: code-fence cut,
                            explicit new-problem marker, first '#### N', then a
                            repeated-substantial-line net. Nothing else.
  RepeatLineStopping     -- halt GENERATION the moment a substantial line
                            repeats (a real loop), so you stop paying for it.
"""

from __future__ import annotations

import re

# explicit "the model started a brand-new problem" markers (hard cut)
STOP_MARKERS = ["\nQuestion:", "\nQ:", "\nExample", "\nProblem:"]

# a completed '#### <number>' (the intended answer terminator)
ANSWER_STOP_RE = re.compile(r"####\s*\$?-?\d[\d,]*(?:\.\d+)?")


def _norm_line(ln: str) -> str:
    """Normalised key for repeat detection: lowercase, alnum+space only."""
    k = re.sub(r"\s+", " ", ln.strip().lower())
    return re.sub(r"[^a-z0-9 ]", "", k).strip()


def _cut_at_first_repeat(text: str, min_len: int = 12) -> str:
    """Cut at the first substantial line that has already appeared verbatim. A
    repeated non-trivial line is the signature of a re-solve loop; normal CoT
    does not repeat a whole line word-for-word."""
    seen = set()
    kept = []
    for ln in text.split("\n"):
        key = _norm_line(ln)
        if len(key) >= min_len:
            if key in seen:
                break                      # loop onset -> stop here
            seen.add(key)
        kept.append(ln)
    return "\n".join(kept)


def truncate_trace(text: str) -> str:
    """Keep only the model's FIRST attempt. Safe on a clean single solution: it
    only ever cuts at an explicit new-problem marker, the first answer, or a
    verbatim repeated line."""
    # 0 -- never keep a code fence
    i = text.find("\\begin{code}")
    if i != -1:
        text = text[:i]

    # 1 -- explicit new-problem marker (model started a fresh Question)
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

    # 3 -- no '####': cut only at a verbatim repeated substantial line.
    #      (No phrase/keyword cues -- those false-trigger on normal reasoning.)
    return _cut_at_first_repeat(text).strip()


# --------------------------------------------------------------------------- #
# Generation-time stop: halt as soon as a substantial line repeats verbatim.
# Attach only for single-sequence generation (same guard as _StopOnAnswer).
# Will NOT fire during a clean first attempt -- nothing is repeated yet.
# --------------------------------------------------------------------------- #
class RepeatLineStopping:
    def __init__(self, tokenizer, prompt_len, min_len=15, check_every=8):
        self.tok = tokenizer
        self.plen = prompt_len
        self.min_len = min_len
        self.check_every = check_every
        self._n = 0

    def __call__(self, input_ids, scores, **kwargs):
        self._n += 1
        if self._n % self.check_every:          # throttle: check every k tokens
            return False
        new_ids = input_ids[0, self.plen:]
        if len(new_ids) < 12:
            return False
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        seen = set()
        for ln in text.split("\n"):
            key = _norm_line(ln)
            if len(key) >= self.min_len:
                if key in seen:
                    return True             # a full line repeated -> loop
                seen.add(key)
        return False