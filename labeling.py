"""Adaptive labeling module extracted from testtt.py.

This module provides `label_targets`, a thin wrapper around
`ActionableTradingPredictor.create_actionable_targets` so that the adaptive
labeling logic lives in its own file.
"""

from typing import Any
import pandas as pd

from models import ActionableTradingPredictor


def label_targets(
    df: pd.DataFrame,
    target_buy_pct: int = 20,
    target_sell_pct: int = 20,
    *args: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """Add BUY / HOLD / SELL labels to a feature-rich DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains the engineered features.
    target_buy_pct : int, default 20
        Desired proportion of BUY signals.
    target_sell_pct : int, default 20
        Desired proportion of SELL signals.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with a new `Target` column indicating the trading
        action.
    """
    predictor = ActionableTradingPredictor(enable_logging=False, enable_data_exploration=False)
    return predictor.create_actionable_targets(df, target_buy_pct, target_sell_pct, *args, **kwargs)