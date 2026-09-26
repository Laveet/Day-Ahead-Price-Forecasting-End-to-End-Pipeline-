"""
live_data.py

Downloads ACTUAL market data from ENTSO-E for one local delivery day,
resamples it to the hourly master resolution, saves lakehouse partitions
and returns the hourly chunk.

Split into two functions because prices and fundamentals become available
at different times (this is what makes the gate-closure timing correct):

  fetch_and_save_prices(D)        -> known since the D-1 auction (~12:45 CET on D-1)
  fetch_and_save_fundamentals(D)  -> actual load + generation, complete only
                                     after D has ended

Fixes vs. the previous version
------------------------------
1. 15-min -> hourly. Since 1 Oct 2025 the DE-LU day-ahead auction clears
   96 quarter-hour prices per day. They used to be saved raw, and the later
   inner merge with hourly load kept only the :00 quarter-hour — so the
   "hourly" target was really the first-quarter-hour price. Everything is
   now resampled to the hourly MEAN before it is saved or merged.
2. Missing last hour. The query end was `start + 1 day - 1 hour`, but
   ENTSO-E / entsoe-py treat `end` as exclusive, so hour 23 was never
   downloaded (-> 23 rows per day in every evaluation). Queries now use
   the exclusive local-day bounds from market_time.delivery_day_bounds().
3. Days are local (Europe/Berlin) delivery days, not UTC days.
"""

from __future__ import annotations
import pandas as pd

from config import LAKEHOUSE_ROOT, DEFAULT_BIDDING_ZONE
from data.entsoe_client import get_client
from data.market_time import delivery_day_bounds

GENERATION_MAPPING = {
    "Solar_Actual Aggregated": "solar_mw",
    "Wind Onshore_Actual Aggregated": "wind_onshore_mw",
    "Wind Offshore_Actual Aggregated": "wind_offshore_mw",
    "Fossil Gas_Actual Aggregated": "gas_mw",
    "Fossil Hard coal_Actual Aggregated": "hard_coal_mw",
    "Nuclear_Actual Aggregated": "nuclear_mw",
}


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _to_hourly(df: pd.DataFrame, bidding_zone: str) -> pd.DataFrame:
    """
    Resample any-resolution data (15-min or hourly) to hourly means on a
    UTC index. Hourly data passes through unchanged. The mean of four
    quarter-hour prices equals the price of an hourly baseload block over
    that hour, so it is the natural hourly target.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    numeric = df.select_dtypes(include="number").columns
    out = df.set_index("timestamp")[numeric].resample("h").mean()
    out = out.dropna(how="all").reset_index()
    out["bidding_zone"] = bidding_zone
    return out


def _partition_dir(dataset: str, bidding_zone: str, day: pd.Timestamp):
    d = (LAKEHOUSE_ROOT / dataset / f"zone={bidding_zone}"
         / f"year={day.year}" / f"month={day.month:02d}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_complete(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, name: str) -> None:
    expected = int((end - start) / pd.Timedelta(hours=1))
    if len(df) != expected:
        print(f"  WARNING: {name} has {len(df)} hourly rows, expected {expected} "
              f"for {start.date()} (data not fully published yet?)")


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------
def fetch_and_save_prices(target_date_str: str, bidding_zone: str = DEFAULT_BIDDING_ZONE) -> pd.DataFrame | None:
    """Hourly day-ahead prices for local delivery day `target_date_str`."""
    client = get_client()
    start, end = delivery_day_bounds(target_date_str)
    print(f"Fetching day-ahead prices for {bidding_zone}, delivery day {target_date_str}...")

    try:
        prices = client.query_day_ahead_prices(bidding_zone, start=start, end=end)
    except Exception as e:
        print(f"Error fetching prices for {target_date_str}: {e}")
        return None

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(prices.index, utc=True),
        "price_eur_mwh": pd.to_numeric(pd.Series(prices.values.ravel()), errors="coerce").values,
    })
    # keep only this delivery day (entsoe-py can return one extra boundary point)
    df = df[(df.timestamp >= start.tz_convert("UTC")) & (df.timestamp < end.tz_convert("UTC"))]
    df = _to_hourly(df, bidding_zone)
    _check_complete(df, start, end, "prices")

    out = _partition_dir("day_ahead_prices", bidding_zone, start)
    df.to_parquet(out / f"price_{start.strftime('%Y-%m-%d')}.parquet", index=False)
    return df


def fetch_and_save_fundamentals(target_date_str: str, bidding_zone: str = DEFAULT_BIDDING_ZONE) -> pd.DataFrame | None:
    """Hourly ACTUAL load + generation for local delivery day `target_date_str`."""
    client = get_client()
    start, end = delivery_day_bounds(target_date_str)
    start_utc, end_utc = start.tz_convert("UTC"), end.tz_convert("UTC")
    print(f"Fetching actual load + generation for {bidding_zone}, {target_date_str}...")

    try:
        load = client.query_load(bidding_zone, start=start, end=end)
        generation = client.query_generation(bidding_zone, start=start, end=end)
    except Exception as e:
        print(f"Error fetching load/generation for {target_date_str}: {e}")
        return None

    # --- load ---
    df_load = pd.DataFrame({
        "timestamp": pd.to_datetime(load.index, utc=True),
        "total_load_mw": pd.to_numeric(pd.Series(load.values.ravel()), errors="coerce").values,
    })
    df_load = df_load[(df_load.timestamp >= start_utc) & (df_load.timestamp < end_utc)]
    df_load = _to_hourly(df_load, bidding_zone)
    _check_complete(df_load, start, end, "load")

    # --- generation ---
    df_gen = generation.copy()
    if isinstance(df_gen.columns, pd.MultiIndex):
        df_gen.columns = ["_".join(col).strip() for col in df_gen.columns.values]
    df_gen.index = pd.to_datetime(df_gen.index, utc=True)
    df_gen.index.name = "timestamp"
    df_gen = df_gen.reset_index().rename(columns=GENERATION_MAPPING)
    df_gen = df_gen[(df_gen.timestamp >= start_utc) & (df_gen.timestamp < end_utc)]
    df_gen = _to_hourly(df_gen, bidding_zone)
    _check_complete(df_gen, start, end, "generation")

    load_dir = _partition_dir("total_load", bidding_zone, start)
    df_load.to_parquet(load_dir / f"load_{start.strftime('%Y-%m-%d')}.parquet", index=False)
    gen_dir = _partition_dir("generation", bidding_zone, start)
    df_gen.to_parquet(gen_dir / f"gen_{start.strftime('%Y-%m-%d')}.parquet", index=False)

    return pd.merge(df_load, df_gen, on=["timestamp", "bidding_zone"], how="inner")


def fetch_align_and_save_live_data(target_date_str: str, bidding_zone: str = DEFAULT_BIDDING_ZONE) -> pd.DataFrame | None:
    """
    Backwards-compatible wrapper: full hourly row (price + load + generation)
    for a day that has ENDED. Useful for backfilling history.
    """
    prices = fetch_and_save_prices(target_date_str, bidding_zone)
    fundamentals = fetch_and_save_fundamentals(target_date_str, bidding_zone)
    if prices is None or fundamentals is None:
        return None
    return pd.merge(prices, fundamentals, on=["timestamp", "bidding_zone"], how="inner")
