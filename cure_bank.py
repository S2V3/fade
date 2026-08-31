"""
cure_bank.py -- turn a PREDICTED failure type into a preloaded prompt.

Phase 2 (EDER) selects a cure AFTER seeing the failed trace. Phase 3 (FST) must
select it BEFORE the first attempt, from the question alone. The exemplar
selection logic is identical; only the source of the type differs. This module is
therefore a thin wrapper over kaggle_run.select_typed_positives, kept separate so
the FST path has one obvious place to look.

TWO-SIDED CURES
---------------
Pass a NegativeBank and the cure gains a second half: alongside the k correct
demonstrations, a short description of how questions of this shape actually break,
drawn from real wrong traces of the predicted type.

The warning is returned in the INSTRUCTION slot, never in the exemplar list. That
is the whole safety design -- build_retry_prompt routes the instruction into the
system message via extra_system, so nothing wrong ever enters the demonstration
block the model's next-token objective pushes it to imitate. See negatives.py.

GOLD RULE: the pool handed in here must come from TraceStore.exemplars(), which
strips gold_solution/gold_answer. The negative bank applies the same sanitiser.
Nothing in this module can reach gold.
"""
from __future__ import annotations

from diagnosis import FailureType, TYPED_INSTRUCTION, TYPED_CURE


def cure_for(pred_type, pool, seeds, k=8, question=None, negatives=None, n_neg=0):
    """(exemplars, instruction) for a PREDICTED type.

    pred_type may be a FailureType, its string value, or None/UNCLASSIFIED --
    the last of which means the predictor abstained and we fall back to the
    generic cure, exactly as the cascade does.

    negatives : an optional NegativeBank. When given with n_neg > 0 and a
                `question`, a failure-mode warning is APPENDED TO THE
                INSTRUCTION -- not to the exemplar list.
    """
    from kaggle_run import select_typed_positives, select_generic_positives

    def _warn(ft_value, instruction):
        if not (negatives and n_neg > 0 and question):
            return instruction
        w = negatives.warning_for(question, ft_value, k=n_neg)
        if not w:
            return instruction
        return (instruction + "\n\n" + w) if instruction else w

    # [AUDIT D39] `question` is forwarded so retrieval is question-aware. Without
    # it every problem of a predicted type received the IDENTICAL 8 exemplars --
    # the same defect the deferred-retry path had.
    if pred_type is None or pred_type in ("UNCLASSIFIED", FailureType.UNCLASSIFIED):
        # abstain: generic positives, and no warning -- we do not know what to
        # warn about, and a wrong warning is worse than none.
        return select_generic_positives(pool, seeds, k, question=question), ""
    ft = pred_type if isinstance(pred_type, FailureType) else FailureType(pred_type)
    if ft is FailureType.TR:                      # a generation artefact has no cure
        return select_generic_positives(pool, seeds, k, question=question), ""
    return (select_typed_positives(pool, ft, seeds, k, question=question),
            _warn(ft.value, TYPED_INSTRUCTION.get(ft, "")))


def describe(pred_type):
    if pred_type in (None, "UNCLASSIFIED"):
        return "generic retrieval (predictor abstained)"
    ft = pred_type if isinstance(pred_type, FailureType) else FailureType(pred_type)
    return TYPED_CURE.get(ft, "generic")
