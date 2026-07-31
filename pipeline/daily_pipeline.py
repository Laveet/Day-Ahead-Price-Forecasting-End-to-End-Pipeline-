"""
daily_pipeline.py

Orchestrates the full day-ahead workflow. Every run now closes the loop
automatically:

  0) Evaluate yesterday's forecast — if a forecast was archived for
     `day_x`, it's compared against the actuals that just arrived, and
     the comparison + metrics are saved. (Skipped silently on the very
     first run, when no forecast exists yet.)
  1) Download & append today's actuals
  2) Feature engineer + train on the updated history
  3) Download tomorrow's forecasts
  4) Build the unified feature matrix for tomorrow
  5) Predict tomorrow's 24 hourly prices AND archive the forecast, so
     step 0 has something to grade the next time the pipeline runs.
"""

from __future__ import annotations
import pandas as pd

from config import DEFAULT_BIDDING_ZONE, MODEL_FEATURES
from data.live_data import fetch_align_and_save_live_data
from data.forecast_data import fetch_next_day_forecast_inputs
from features.engineering import engineer_features
from models.training import train_price_model
from evaluation.forecast_tracker import save_forecast, compare_forecast_vs_actual


def run_day_ahead_pipeline(
    master_data: pd.DataFrame,
    day_x: str,
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
):
    """
    Parameters
    ----------
    master_data : your existing master_energy_df
    day_x : "today" as a string, e.g. "2026-07-30" — the day whose
        actuals just became available.

    Returns
    -------
    (master_updated, model, forecast_result)
    """
    print(f"Processing {day_x}")

    # ---- STEP 1: download & append today's actuals ----------------------
    actual_day = fetch_align_and_save_live_data(day_x, bidding_zone)

    # ---- STEP 0 (runs here, once actuals exist): grade yesterday's forecast
    compare_forecast_vs_actual(actual_day, day_x, bidding_zone)

    master_updated = pd.concat([master_data, actual_day], ignore_index=True)
    master_updated = (
        master_updated
        .drop_duplicates(subset=["timestamp", "bidding_zone"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---- STEP 2: feature engineer + train on updated history -----------
    features = engineer_features(master_updated)
    model = train_price_model(features)

    # ---- STEP 3: download tomorrow's forecasts --------------------------
    next_day = (pd.Timestamp(day_x) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_inputs = fetch_next_day_forecast_inputs(next_day, bidding_zone)
    print(forecast_inputs.tail(10))
    # ---- STEP 4: unified timeline -> one feature engineering call ------
    combined = pd.concat([master_updated, forecast_inputs], ignore_index=True)
    combined["bidding_zone"] = combined["bidding_zone"].ffill().bfill()
    combined_features = engineer_features(combined)
   
    future_features = combined_features[
        combined_features.timestamp.dt.date == pd.Timestamp(next_day).date()
    ]
    

    # if len(future_features) != 24:
    #     raise ValueError(
    #         f"Expected 24 forecast rows for {next_day}, got {len(future_features)}. "
    #         "Check that the ENTSO-E forecast query returned a full day."
    #     )

    # nan_mask = future_features[MODEL_FEATURES].isna()
    # if nan_mask.any().any():
    #     bad_cols = future_features[MODEL_FEATURES].columns[nan_mask.any()].tolist()
    #     raise ValueError(
    #         f"NaNs in forecast feature matrix for columns: {bad_cols}. "
    #         "Likely a forecast column-naming mismatch or insufficient history for lag_168."
    #     )

    # ---- STEP 5: predict + archive ----------------------------------------
    predictions = model.predict(future_features[MODEL_FEATURES])

    forecast_result = pd.DataFrame({
        "timestamp": future_features.timestamp.values,
        "forecast_price_eur_mwh": predictions,
    })

    save_forecast(forecast_result, next_day, bidding_zone)

    return master_updated, model, forecast_result
