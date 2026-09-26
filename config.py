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

# Delivery days, auctions and gate closure are defined in LOCAL market time,
# not UTC. A German delivery day is 00:00-24:00 Europe/Berlin
# (= 22:00-22:00 UTC in summer, 23:00-23:00 UTC in winter; 23 or 25 hours
# on DST-change days). Timestamps are still STORED in UTC; this timezone is
# only used to decide which UTC hours belong to which delivery day.
MARKET_TZ = "Europe/Berlin"

# ---------------------------------------------------------------------
# Feature engineering config
# ---------------------------------------------------------------------
# Lags per column are chosen by what is KNOWN at gate closure
# (12:00 CET on day D, bidding for delivery day D+1):
#   - prices for all of D were set in yesterday's auction -> lag 24 is known
#   - actual load / generation are only complete up to the end of D-1
#     -> lag 24 (= hours of D) is NOT known yet, the smallest usable lag is 48
LAG_SPEC = {
    "price_eur_mwh": [24, 48, 168],
    "total_load_mw": [48, 168],
    "residual_load_mw": [48, 168],
}
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

    "total_load_mw_lag_48",
    "total_load_mw_lag_168",

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
