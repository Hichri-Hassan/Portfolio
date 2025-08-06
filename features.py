"""Feature engineering module extracted from testtt.py.
This module provides a single utility function `build_features` that wraps
`ActionableTradingPredictor.comprehensive_feature_engineering` so that the
rest of the codebase can depend on a smaller surface area.
"""

from typing import Any
import pandas as pd

# Re-use the comprehensive feature engineering implementation that already
# exists inside ActionableTradingPredictor.  We import the class from the new
# models.py façade which in turn re-exports the original implementation.
from models import ActionableTradingPredictor


def build_features(df: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Create the full feature set required by the trading models.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data with at least the columns expected by
        `ActionableTradingPredictor.comprehensive_feature_engineering`.
    *args, **kwargs : Any
        Forwarded to the underlying implementation allowing callers to stay
        fully compatible with the original interface.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with all engineered features.
    """
    predictor = ActionableTradingPredictor(enable_logging=False, enable_data_exploration=False)
    return predictor.comprehensive_feature_engineering(df, *args, **kwargs)