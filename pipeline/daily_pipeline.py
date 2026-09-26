"""
daily_pipeline.py

Orchestrates the day-ahead workflow with the information set a trader
actually has at gate closure.

Market timeline (EPEX / SDAC, DE-LU)
------------------------------------
Run on the MORNING of day D (before the 12:00 CET gate closure). The bids
submitted before 12:00 on D are for delivery day D+1.

    What is known on D before 12:00      How it is used
    --------------------------------      -----------------------------------
    prices for all of D                   price lag 24 for D+1, grading the
      (cleared in the D-1 auction)          forecast made yesterday for D
    actual load / generation up to D-1    training rows, load lags >= 48 h
    load + wind/solar FORECASTS for D+1   inputs for the D+1 feature rows

    NOT known yet: D's actual load and generation (D hasn't ended) and
    therefore anything derived from them, e.g. load lag 24 for D+1.

The old version took `day_x`'s full actuals (only available after day_x
ends) and then forecast day_x+1, whose auction had already closed at noon
on day_x. The forecast could not have been used for bidding.

Every run:
  0) refresh the gas price (TTF) and align it: delivery day T uses the
     last close dated <= T-2 (the D-1 settlement is not known at 12:00 on D)
  1) actual load + generation for D-1 (+ its prices) -> full history rows
  2) prices for D                                     -> price-only rows
  3) grade the forecast archived yesterday for D
  4) feature engineering + training on the updated history
  5) download D+1 load and wind/solar forecasts
  6) predict all hours of delivery day D+1 and archive the forecast
"""

from __future__ import annotations
import pandas as pd

from config import DEFAULT_BIDDING_ZONE, MODEL_FEATURES
from data.live_data import fetch_and_save_prices, fetch_align_and_save_live_data
from data.forecast_data import fetch_next_day_forecast_inputs
from data.market_time import hours_in_delivery_day, local_date
from features.engineering import engineer_features
from data.fuel_data import refresh_fuel_data, join_fuel
from models.training import train_price_model
from evaluation.forecast_tracker import save_forecast, compare_forecast_vs_actual


def _append(master: pd.DataFrame, chunk: pd.DataFrame | None) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return master
    out = pd.concat([master, chunk], ignore_index=True)
    return (
        out.drop_duplicates(subset=["timestamp", "bidding_zone"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _as_of_gate_closure(master: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """
    Cut history to what was known at 12:00 CET on `run_date` (D):
      - nothing after delivery day D
      - for day D itself: prices only (load/generation of D not known yet)
    For a normal live run this changes nothing. For a REPLAY of a past day
    (e.g. `python main.py 2026-09-25` while later data is already in the
    lakehouse) it prevents training on / feeding in information from the
    future.
    """
    D = pd.Timestamp(run_date).date()
    days = local_date(master["timestamp"])
    out = master[days <= D].copy()
    fundamentals = [c for c in out.columns if c not in ("timestamp", "bidding_zone", "price_eur_mwh")]
    out.loc[local_date(out["timestamp"]) == D, fundamentals] = float("nan")
    return out.reset_index(drop=True)


def run_day_ahead_pipeline(
    master_data: pd.DataFrame,
    run_date: str,
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
):
    """
    Parameters
    ----------
    master_data : master_energy_df from historical_loader
    run_date : day D, the day you are running on (before 12:00 CET),
        e.g. "2026-09-26". The forecast is produced for delivery day D+1.

    Returns
    -------
    (master_updated, model, forecast_result)
    """
    D = pd.Timestamp(run_date)
    d_prev = (D - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    d_today = D.strftime("%Y-%m-%d")
    d_target = (D + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Run date D={d_today} | actuals through {d_prev} | prices through {d_today} "
          f"| forecasting delivery day {d_target}")

    # Keep the full history aside: in a replay it already contains the
    # actual prices of D+1, which lets us grade the forecast immediately.
    known_later = master_data
    master_data = _as_of_gate_closure(master_data, d_today)

    # ---- STEP 0: refresh gas price (TTF) -----------------------------------
    # Fails loudly if Yahoo returns nothing or the data fails validation:
    # a model trained with gas must not silently run without it.
    try:
        fuel = refresh_fuel_data(verbose=True)
    except Exception as e:
        raise RuntimeError(
            f"Gas price (TTF) refresh failed: {e}\n"
            "The model uses ttf_eur_mwh as an input, so the forecast is stopped. "
            "Check your internet, try `pip install -U yfinance`, or run "
            "`python -m data.fuel_data` to see the full error."
        ) from e

    # ---- STEP 1: complete actuals for D-1 ---------------------------------
    actual_prev = fetch_align_and_save_live_data(d_prev, bidding_zone)
    master_updated = _append(master_data, actual_prev)

    # ---- STEP 2: prices for D (known since yesterday's auction) -----------
    # Appended AFTER the full rows, so these price-only rows never overwrite
    # a complete row. Load/generation stay NaN for D -> those rows are used
    # for price lags / rolling stats but dropped from training.
    prices_today = fetch_and_save_prices(d_today, bidding_zone)
    if prices_today is None or prices_today.empty:
        raise RuntimeError(
            f"No day-ahead prices for {d_today} yet. They are published after "
            f"the auction on {d_prev} (~12:45 CET); without them the D+1 "
            "forecast would have no price lag 24."
        )
    master_updated = _append(master_updated, prices_today)

    # ---- STEP 3: grade yesterday's forecast for D --------------------------
    compare_forecast_vs_actual(prices_today, d_today, bidding_zone)

    # ---- STEP 4: feature engineer + train ----------------------------------
    features = join_fuel(engineer_features(master_updated), fuel)
    model = train_price_model(features)

    # ---- STEP 5: forecasts for D+1 -----------------------------------------
    forecast_inputs = fetch_next_day_forecast_inputs(d_target, bidding_zone)

    # ---- STEP 6: unified timeline -> one feature-engineering call ----------
    combined = pd.concat([master_updated, forecast_inputs], ignore_index=True)
    combined["bidding_zone"] = combined["bidding_zone"].ffill().bfill()
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined_features = join_fuel(engineer_features(combined), fuel)

    future_features = combined_features[
        local_date(combined_features.timestamp) == pd.Timestamp(d_target).date()
    ]

    expected = hours_in_delivery_day(d_target)   # 23 / 24 / 25
    if len(future_features) != expected:
        raise ValueError(
            f"Expected {expected} forecast rows for delivery day {d_target}, "
            f"got {len(future_features)}. Check the ENTSO-E forecast download."
        )

    target_fuel = fuel[fuel["delivery_date"] == pd.Timestamp(d_target)]
    if target_fuel.empty or target_fuel["ttf_eur_mwh"].isna().any():
        raise RuntimeError(f"No gas price available for delivery day {d_target}.")
    age = int(target_fuel["ttf_age_days"].iloc[0])
    print(f"Gas price for {d_target}: {target_fuel.ttf_eur_mwh.iloc[0]:.2f} EUR/MWh "
          f"(close of {target_fuel.ttf_obs_date.iloc[0].date()}, {age} days before delivery)")
    if age > 5:
        print(f"  WARNING: gas price is {age} days old — Yahoo may be lagging.")

    nan_cols = future_features[MODEL_FEATURES].columns[
        future_features[MODEL_FEATURES].isna().any()
    ].tolist()
    if nan_cols:
        # LightGBM can predict through NaNs, so warn instead of failing. A
        # known case: on the 25-hour October DST day the last hour's lag-48
        # load falls on D itself, which is not known yet.
        print(f"  WARNING: NaNs in forecast features {nan_cols} — "
              "LightGBM will use its missing-value branches for those rows.")

    predictions = model.predict(future_features[MODEL_FEATURES])

    forecast_result = pd.DataFrame({
        "timestamp": future_features.timestamp.values,
        "forecast_price_eur_mwh": predictions,
    })

    save_forecast(forecast_result, d_target, bidding_zone)

    # Replay: actual prices for D+1 are already known -> grade right away.
    actual_target = known_later[local_date(known_later["timestamp"]) == pd.Timestamp(d_target).date()]
    actual_target = actual_target.dropna(subset=["price_eur_mwh"])
    if len(actual_target) == expected:
        print(f"Actual prices for {d_target} already known (replay) — grading now:")
        compare_forecast_vs_actual(actual_target, d_target, bidding_zone)

    return master_updated, model, forecast_result
