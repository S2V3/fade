"""
cure_bank.py -- turn a PREDICTED failure type into a preloaded prompt.

Phase 2 (EDER) selects a cure AFTER seeing the failed trace. Phase 3 (FST) must
select it BEFORE the first attempt, from the question alone. The exemplar
selection logic is identical; only the source of the type differs. This module is
therefore a thin wrapper over kaggle_run.select_typed_positives, kept separate so
the FST path has one obvious place to look.

GOLD RULE: the pool handed in here must come from TraceStore.exemplars(), which
strips gold_solution/gold_answer. Nothing in this module can reach gold.
"""
from __future__ import annotations

from diagnosis import FailureType, TYPED_INSTRUCTION, TYPED_CURE


def cure_for(pred_type, pool, seeds, k=8):
    """(exemplars, instruction) for a PREDICTED type.

    pred_type may be a FailureType, its string value, or None/UNCLASSIFIED --
    the last of which means the predictor abstained and we fall back to the
    generic cure, exactly as the cascade does.
    """
    from kaggle_run import select_typed_positives, select_generic_positives

    if pred_type is None or pred_type in ("UNCLASSIFIED", FailureType.UNCLASSIFIED):
        return select_generic_positives(pool, seeds, k), ""
    ft = pred_type if isinstance(pred_type, FailureType) else FailureType(pred_type)
    if ft is FailureType.TR:                      # a generation artefact has no cure
        return select_generic_positives(pool, seeds, k), ""
    return select_typed_positives(pool, ft, seeds, k), TYPED_INSTRUCTION.get(ft, "")


def describe(pred_type):
    if pred_type in (None, "UNCLASSIFIED"):
        return "generic retrieval (predictor abstained)"
    ft = pred_type if isinstance(pred_type, FailureType) else FailureType(pred_type)
    return TYPED_CURE.get(ft, "generic")
