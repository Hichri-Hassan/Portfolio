"""Orchestration script for the trading prediction workflow.

This `main.py` file demonstrates a minimal end-to-end run that:
1. Loads (or synthesises) input data.
2. Builds the full feature set.
3. Generates BUY/HOLD/SELL labels with adaptive targets.
4. Trains and evaluates the actionable models.

The heavy-lifting implementations live in the sub-modules `features`,
`labeling`, `models`, and `evaluate`.

Run the script directly:

    python main.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

import features
import labeling
from models import ActionableTradingPredictor


DATA_FILE = "us_stocks_5years_with_fundamentals.csv"
RANDOM_STATE = 42


def _load_data() -> pd.DataFrame:
    """Load the market data set.

    If the expected CSV file is not available we fall back to a synthetic data
    generator so that the pipeline can still be executed for demonstration
    purposes.
    """
    if os.path.exists(DATA_FILE):
        print(f"📂 Loading data from '{DATA_FILE}' …")
        return pd.read_csv(DATA_FILE)

    # ---------------------------------------------------------------------
    # Synthetic fallback – relies on the helper test class that lives in the
    # original *testtt.py* module.  This keeps the example completely
    # self-contained.
    # ---------------------------------------------------------------------
    print("⚠️  Data file not found – generating synthetic sample data instead …")
    from testtt import TradingPredictorTests  # local import to avoid hard dep

    return TradingPredictorTests().create_sample_data(n_samples=1000, n_tickers=5)


def main() -> None:
    print("🚀 Actionable Trading Prediction – Orchestration Start")
    print("=" * 70)

    # 1. Data loading ------------------------------------------------------
    raw_df = _load_data()
    print(f"✅ Loaded data shape: {raw_df.shape}")

    # 2. Feature engineering ----------------------------------------------
    feats_df = features.build_features(raw_df)
    print(f"✅ Feature engineering complete – shape: {feats_df.shape}")

    # 3. Adaptive labeling -------------------------------------------------
    labeled_df = labeling.label_targets(feats_df, target_buy_pct=20, target_sell_pct=20)
    print("✅ Target labeling complete – distribution:")
    print(labeled_df["Target"].value_counts(normalize=True) * 100)

    # 4. Model training & evaluation --------------------------------------
    experiment_name = f"split_pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    predictor = ActionableTradingPredictor(
        random_state=RANDOM_STATE,
        experiment_name=experiment_name,
        enable_logging=True,
        enable_data_exploration=False,
    )

    results = predictor.train_and_evaluate_actionable(labeled_df)

    print("\n📊 Final leaderboard (top 10 rows):")
    print(results["results"].head(10))

    print("\n🏁 Workflow finished.")


if __name__ == "__main__":
    main()