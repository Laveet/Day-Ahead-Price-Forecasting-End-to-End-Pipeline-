"""
training.py

Trains the LightGBM day-ahead price model on a feature-engineered
dataframe.
"""

from __future__ import annotations
import lightgbm as lgb
import pandas as pd

from config import MODEL_FEATURES, LGBM_PARAMS


def train_price_model(feature_df: pd.DataFrame) -> lgb.LGBMRegressor:
    # Drop rows missing ANY required feature, not just price — otherwise
    # under-filled early-history rows (e.g. missing lag_168) silently
    # leak NaNs into training.
    train_df = feature_df.dropna(subset=["price_eur_mwh"] + MODEL_FEATURES)

    if train_df.empty:
        raise ValueError(
            "No training rows survived feature engineering. "
            "Check that master_data spans at least 7+ days (needed for lag_168)."
        )

    X = train_df[MODEL_FEATURES]
    y = train_df["price_eur_mwh"]

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)

    print("Model trained:", X.shape)
    return model
