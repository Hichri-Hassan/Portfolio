"""Evaluation and plotting helpers extracted from testtt.py.

This module simply exposes the rich evaluation utilities that already exist
inside `ActionableTradingPredictor` so that other parts of the codebase can
import them directly from `evaluate` without having to know about the original
class implementation.
"""

from typing import Any
from models import ActionableTradingPredictor

# A single hidden predictor instance to reuse the existing methods.  We disable
# logging and data exploration here because the evaluation helpers do not rely
# on those subsystems.
_PREDICTOR = ActionableTradingPredictor(enable_logging=False, enable_data_exploration=False)

evaluate_model_actionable = _PREDICTOR.evaluate_model_actionable
plot_confusion_matrix = _PREDICTOR.plot_confusion_matrix
print_detailed_classification_report = _PREDICTOR.print_detailed_classification_report
plot_feature_importance = _PREDICTOR.plot_feature_importance

__all__ = [
    "evaluate_model_actionable",
    "plot_confusion_matrix",
    "print_detailed_classification_report",
    "plot_feature_importance",
]