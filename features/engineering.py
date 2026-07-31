"""
engineering.py

ONE feature engineering function used for BOTH training rows (real price)
and forecast rows (price_eur_mwh = NaN). Because lag/rolling features are
built with .shift() on a gapless hourly grid, a value at time t only ever
depends on strictly earlier timestamps -> no leakage, and forecast rows
automatically get correct lag values from real history.

Bug fixes baked in from debugging the live pipeline:
  1. Reindex to a complete hourly grid before any shift(). .shift(N)
     shifts by ROW POSITION, not real elapsed time — a single missing
     hour anywhere in history silently corrupts every later lag/rolling
     value (not just NaN, a WRONG value with no error raised).
  2. Rolling stats are frozen at the last known actual hour and ffilled
     across the forecast day, instead of sliding into hours that haven't
     been predicted yet (which produces NaN for hour 2 onward of any
     single-shot 24h-ahead forecast).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import LAG_HOURS, LAG_COLUMNS, ROLLING_WINDOW


def engineer_features(master_data: pd.DataFrame) -> pd.DataFrame:
    print("Starting feature engineering...")

    df = master_data.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -- 0. Reindex to a complete, gapless hourly grid --------------------
    # Guarantees shift(168) always means "168 hours ago", not "168 rows
    # ago". Genuinely missing hours become real NaN rows here instead of
    # silently corrupting every later shift.
    zone = df["bidding_zone"].dropna().iloc[0] if "bidding_zone" in df.columns else None
    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_index)
    df.index.name = "timestamp"
    df = df.reset_index()
    if zone is not None:
        df["bidding_zone"] = df["bidding_zone"].fillna(zone)

    # -- 1. Calendar features -----------------------------------------
    df["hour"] = df.timestamp.dt.hour
    df["dayofweek"] = df.timestamp.dt.dayofweek
    df["month"] = df.timestamp.dt.month
    df["dayofyear"] = df.timestamp.dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

    # -- 2. Residual load -----------------------------------------------
    # Uses only columns that ALSO exist as forecasts (total_load_mw,
    # solar_mw, wind_onshore_mw, wind_offshore_mw) so it's computable
    # identically for historical and forecast rows.
    df["residual_load_mw"] = (
        df["total_load_mw"] - df["solar_mw"] - df["wind_onshore_mw"] - df["wind_offshore_mw"]
    )

    # -- 3. Lag features --------------------------------------------------
    for col in LAG_COLUMNS:
        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # -- 4. Rolling price stats (leak-free, frozen for forecast rows) ----
    # shift(1) BEFORE rolling: the window ending at time t only contains
    # values strictly before t, so this is safe even when price at t
    # itself is NaN (a forecast row).
    df[f"price_rolling_mean_{ROLLING_WINDOW}"] = df["price_eur_mwh"].shift(1).rolling(ROLLING_WINDOW).mean()
    df[f"price_rolling_std_{ROLLING_WINDOW}"] = df["price_eur_mwh"].shift(1).rolling(ROLLING_WINDOW).std()

    # A sliding window breaks down inside the forecast day itself: hour 2's
    # window includes hour 1 of tomorrow, whose price is NaN (not
    # predicted yet) -> NaN, and it only gets worse from there. Real
    # day-ahead forecasting predicts the whole next day at once, before it
    # starts, so "recent price behaviour" should be one snapshot frozen at
    # the last known actual hour, applied to every hour of the forecast
    # day. ffill only ever copies a PAST value forward -> no leakage.
    df[f"price_rolling_mean_{ROLLING_WINDOW}"] = df[f"price_rolling_mean_{ROLLING_WINDOW}"].ffill()
    df[f"price_rolling_std_{ROLLING_WINDOW}"] = df[f"price_rolling_std_{ROLLING_WINDOW}"].ffill()

    return df
