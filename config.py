"""
config.py

Central configuration for the Day-Ahead Price Forecasting pipeline.
Change paths, zones, or hyperparameters here once — every other module
imports from this file so nothing gets out of sync.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
LAKEHOUSE_ROOT = Path("data/lakehouse")

HISTORICAL_SUBFOLDERS = ["day_ahead_prices", "total_load", "generation"]

FORECAST_ARCHIVE_DIR = LAKEHOUSE_ROOT / "forecasts"
EVALUATION_DIR = LAKEHOUSE_ROOT / "evaluation"

# ---------------------------------------------------------------------
# Market config
# ---------------------------------------------------------------------
DEFAULT_BIDDING_ZONE = "DE_LU"

# ---------------------------------------------------------------------
# Feature engineering config
# ---------------------------------------------------------------------
LAG_HOURS = [24, 48, 168]
LAG_COLUMNS = ["price_eur_mwh", "total_load_mw", "residual_load_mw"]
ROLLING_WINDOW = 24

# ---------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------
MODEL_FEATURES = [
    "total_load_mw",
    "solar_mw",
    "wind_offshore_mw",
    "wind_onshore_mw",

    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",

    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",

    "residual_load_mw",

    "price_eur_mwh_lag_24",
    "price_eur_mwh_lag_48",
    "price_eur_mwh_lag_168",

    "total_load_mw_lag_24",
    "total_load_mw_lag_48",
    "total_load_mw_lag_168",

    "residual_load_mw_lag_24",
    "residual_load_mw_lag_48",
    "residual_load_mw_lag_168",

    f"price_rolling_mean_{ROLLING_WINDOW}",
    f"price_rolling_std_{ROLLING_WINDOW}",
]

# ---------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    random_state=42,
    verbose=-1,
)
