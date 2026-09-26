"""
forecast_data.py

Downloads tomorrow's day-ahead LOAD and WIND/SOLAR forecasts and shapes
them to match the master schema exactly, so they can be pd.concat'ed
straight onto master_updated and run through engineer_features()
unchanged.

Includes every fix from debugging the live pipeline:
  - wind/solar columns renamed to match master_energy_df (solar_mw, etc.)
  - bidding_zone + placeholder columns added so schemas match
  - 15-min wind/solar data resampled to hourly to match master resolution
  - small boundary gaps from the resample interpolated (input features
    only — never touches the target)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from config import DEFAULT_BIDDING_ZONE
from data.entsoe_client import get_client
from data.market_time import delivery_day_bounds


def fetch_next_day_forecast_inputs(target_date_str: str, bidding_zone: str = DEFAULT_BIDDING_ZONE) -> pd.DataFrame:
    client = get_client()

    # Local delivery day [00:00, 24:00) Europe/Berlin, exclusive end —
    # the same hours the D-1 auction actually clears (23/25 on DST days).
    start, end = delivery_day_bounds(target_date_str)

    print(f"Downloading forecasts for {target_date_str}...")

    load_forecast = client.query_load_forecast(bidding_zone, start=start, end=end)
    wind_solar = client.query_wind_and_solar_forecast(bidding_zone, start=start, end=end)

    # Some entsoe-py versions return query_load_forecast as a single-column
    # DataFrame instead of a Series -> .values would be 2-D. ravel() flattens
    # either shape safely.
    load_values = np.ravel(load_forecast.values)
    df_load = load_forecast.reset_index().rename(columns={"index": "timestamp"})
    df_load["timestamp"] = pd.to_datetime(df_load["timestamp"], utc=True)
    df_load=df_load.rename(columns={"Forecasted Load":"total_load_mw"})

        
    df_gen = wind_solar.reset_index().rename(columns={"index": "timestamp"})
    df_gen["timestamp"] = pd.to_datetime(df_gen["timestamp"], utc=True)
    df_gen = df_gen.rename(columns={
        "Solar": "solar_mw",
        "Wind Onshore": "wind_onshore_mw",
        "Wind Offshore": "wind_offshore_mw",
    })

    # Wind/solar forecasts come at 15-min resolution; master_energy_df and
    # the load forecast are hourly. Resample BEFORE merging.
    df_load = df_load.set_index("timestamp").resample("h").mean().reset_index()
    df_gen = df_gen.set_index("timestamp").resample("h").mean().reset_index()

    result = pd.merge(df_load, df_gen, on="timestamp", how="outer")
    start_utc, end_utc = start.tz_convert("UTC"), end.tz_convert("UTC")
    result = result[(result.timestamp >= start_utc) & (result.timestamp < end_utc)]

    # Small edge gaps can appear after the 15-min -> hourly resample
    # (usually a UTC / local bidding-zone boundary mismatch). These are
    # raw INPUT features, not the target, so interpolating a couple of
    # missing hours from neighbours is legitimate and introduces no leakage.
    forecast_input_cols = [c for c in ["total_load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"] if c in result.columns]
    result[forecast_input_cols] = result[forecast_input_cols].interpolate(limit_direction="both")

    # price_eur_mwh and non-forecastable generation are unknown at
    # forecast time — kept only so the schema matches master_updated.
    result["bidding_zone"] = bidding_zone
    result["price_eur_mwh"] = np.nan
    for col in ["gas_mw", "hard_coal_mw", "nuclear_mw"]:
        result[col] = np.nan

    result = result.sort_values("timestamp").reset_index(drop=True)
    print(f"  -> {len(result)} forecast rows downloaded.")
    return result
