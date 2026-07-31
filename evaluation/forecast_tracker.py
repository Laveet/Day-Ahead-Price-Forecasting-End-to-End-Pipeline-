"""
forecast_tracker.py

Archives every forecast the pipeline produces, then compares the saved
forecast against actuals once they become available — so accuracy is
tracked run over run with no manual bookkeeping.

Storage layout
--------------
data/lakehouse/forecasts/zone=<zone>/year=<yyyy>/month=<mm>/
    forecast_target=<target_date>_generated=<run_timestamp>.parquet

data/lakehouse/evaluation/zone=<zone>/year=<yyyy>/month=<mm>/
    evaluation_<target_date>.parquet   (hour-by-hour comparison)
    metrics_<target_date>.csv          (summary metrics)

Typical flow
------------
Day 1 (running for July 30, forecasting July 31):
    save_forecast(forecast_result, "2026-07-31", "DE_LU")

Day 2 (running for July 31 — actuals for July 31 just arrived):
    compare_forecast_vs_actual(actual_day, "2026-07-31", "DE_LU")
    -> loads the forecast saved on Day 1 and grades it

The daily pipeline calls both of these automatically — see
pipeline/daily_pipeline.py.
"""

from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from config import FORECAST_ARCHIVE_DIR, EVALUATION_DIR, DEFAULT_BIDDING_ZONE


def save_forecast(
    forecast_result: pd.DataFrame,
    target_date: str,
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
) -> None:
    """
    Archives a forecast so it can be compared against actuals later.

    Parameters
    ----------
    forecast_result : DataFrame with ['timestamp', 'forecast_price_eur_mwh']
    target_date : the date being forecast, e.g. "2026-07-31"
    """
    target_ts = pd.Timestamp(target_date)
    generated_at = datetime.now(timezone.utc)

    out = forecast_result.copy()
    out["target_date"] = target_ts.date().isoformat()
    out["generated_at"] = generated_at.isoformat()
    out["bidding_zone"] = bidding_zone

    out_dir = (
        FORECAST_ARCHIVE_DIR
        / f"zone={bidding_zone}"
        / f"year={target_ts.year}"
        / f"month={target_ts.month:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"forecast_target={target_ts.date()}"
        f"_generated={generated_at.strftime('%Y%m%dT%H%M%S')}.parquet"
    )
    out.to_parquet(out_dir / filename, index=False)
    print(f"Saved forecast for {target_ts.date()} -> {out_dir / filename}")


def load_latest_forecast(
    target_date: str,
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
) -> pd.DataFrame | None:
    """
    Loads the most recently generated forecast for `target_date`.
    Returns None if no forecast was ever archived for that day (e.g. the
    pipeline wasn't run the day before, or this is the very first run).
    """
    target_ts = pd.Timestamp(target_date)
    search_dir = (
        FORECAST_ARCHIVE_DIR
        / f"zone={bidding_zone}"
        / f"year={target_ts.year}"
        / f"month={target_ts.month:02d}"
    )
    if not search_dir.exists():
        return None

    matches = sorted(search_dir.glob(f"forecast_target={target_ts.date()}_generated=*.parquet"))
    if not matches:
        return None

    # Filenames sort chronologically because generated_at is a zero-padded
    # timestamp string, so the last match is the most recent forecast run.
    return pd.read_parquet(matches[-1])


def compare_forecast_vs_actual(
    actual_df: pd.DataFrame,
    target_date: str,
    bidding_zone: str = DEFAULT_BIDDING_ZONE,
):
    """
    Compares the most recently archived forecast for `target_date` against
    real prices, once they're available, and saves the comparison.

    Parameters
    ----------
    actual_df : any dataframe containing actual ['timestamp', 'price_eur_mwh']
        rows for target_date — e.g. the `actual_day` chunk returned by
        fetch_align_and_save_live_data().
    target_date : the date to evaluate, e.g. "2026-07-31"

    Returns
    -------
    (comparison_df, metrics_dict), or None if there was nothing to compare
    (no archived forecast, or no overlapping timestamps).
    """
    forecast_df = load_latest_forecast(target_date, bidding_zone)
    if forecast_df is None:
        print(f"No archived forecast found for {target_date} — nothing to compare.")
        return None

    actual = actual_df.copy()
    actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True)
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], utc=True)

    comparison = pd.merge(
        forecast_df[["timestamp", "forecast_price_eur_mwh"]],
        actual[["timestamp", "price_eur_mwh"]],
        on="timestamp",
        how="inner",
    )

    if comparison.empty:
        print(f"Forecast and actuals for {target_date} share no matching timestamps.")
        return None

    comparison["error"] = comparison["price_eur_mwh"] - comparison["forecast_price_eur_mwh"]
    comparison["abs_error"] = comparison["error"].abs()
    comparison["pct_error"] = (
        comparison["error"] / comparison["price_eur_mwh"].replace(0, np.nan)
    ) * 100

    metrics = {
        "target_date": target_date,
        "bidding_zone": bidding_zone,
        "n_hours_compared": len(comparison),
        "mae": comparison["abs_error"].mean(),
        "rmse": float(np.sqrt((comparison["error"] ** 2).mean())),
        "mape_pct": comparison["pct_error"].abs().mean(),
        "bias": comparison["error"].mean(),  # positive => model under-predicted
    }

    _save_evaluation(comparison, metrics, target_date, bidding_zone)

    print(
        f"Forecast accuracy for {target_date}: "
        f"MAE={metrics['mae']:.2f} EUR/MWh, "
        f"RMSE={metrics['rmse']:.2f} EUR/MWh, "
        f"MAPE={metrics['mape_pct']:.1f}%, "
        f"bias={metrics['bias']:+.2f} EUR/MWh"
    )

    return comparison, metrics


def _save_evaluation(comparison: pd.DataFrame, metrics: dict, target_date: str, bidding_zone: str) -> None:
    target_ts = pd.Timestamp(target_date)
    out_dir = (
        EVALUATION_DIR
        / f"zone={bidding_zone}"
        / f"year={target_ts.year}"
        / f"month={target_ts.month:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison.to_parquet(out_dir / f"evaluation_{target_ts.date()}.parquet", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / f"metrics_{target_ts.date()}.csv", index=False)
