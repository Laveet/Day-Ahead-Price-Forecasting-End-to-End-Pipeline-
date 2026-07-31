"""
live_data.py

Downloads actual day-ahead prices, load, and generation for a given date,
aligns everything to the master schema, saves parquet partitions, and
returns the merged chunk. This is your existing, already-working logic —
unchanged, just moved into its own module.
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path

from config import LAKEHOUSE_ROOT, DEFAULT_BIDDING_ZONE
from data.entsoe_client import get_client


def fetch_align_and_save_live_data(target_date_str: str, bidding_zone: str = DEFAULT_BIDDING_ZONE) -> pd.DataFrame:
    client = get_client()
    start = pd.Timestamp(target_date_str, tz="UTC")
    end = start + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    print(f"Fetching live ENTSO-E data for zone {bidding_zone} on {target_date_str}...")

    try:
        prices = client.query_day_ahead_prices(bidding_zone, start=start, end=end)
        load = client.query_load(bidding_zone, start=start, end=end)
        generation = client.query_generation(bidding_zone, start=start, end=end, resolution="PT60M")

        df_price_live = pd.DataFrame({
            "timestamp": pd.to_datetime(prices.index, utc=True),
            "bidding_zone": bidding_zone,
            "price_eur_mwh": pd.to_numeric(prices.values.ravel(), errors="coerce"),
        }).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        df_load_live = pd.DataFrame({
            "timestamp": pd.to_datetime(load.index, utc=True),
            "bidding_zone": bidding_zone,
            "total_load_mw": pd.to_numeric(load.values.ravel(), errors="coerce"),
        }).sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        df_gen_live = generation.copy()
        if isinstance(df_gen_live.columns, pd.MultiIndex):
            df_gen_live.columns = ["_".join(col).strip() for col in df_gen_live.columns.values]

        df_gen_live = df_gen_live.reset_index()
        if "index" in df_gen_live.columns:
            df_gen_live.rename(columns={"index": "timestamp"}, inplace=True)
        elif "level_0" in df_gen_live.columns:
            df_gen_live.rename(columns={"level_0": "timestamp"}, inplace=True)

        df_gen_live["timestamp"] = pd.to_datetime(df_gen_live["timestamp"], utc=True)
        df_gen_live["bidding_zone"] = bidding_zone

        mapping_rules = {
            "Solar_Actual Aggregated": "solar_mw",
            "Wind Onshore_Actual Aggregated": "wind_onshore_mw",
            "Wind Offshore_Actual Aggregated": "wind_offshore_mw",
            "Fossil Gas_Actual Aggregated": "gas_mw",
            "Fossil Hard coal_Actual Aggregated": "hard_coal_mw",
            "Nuclear_Actual Aggregated": "nuclear_mw",
        }
        for old_col, new_col in mapping_rules.items():
            if old_col in df_gen_live.columns:
                df_gen_live.rename(columns={old_col: new_col}, inplace=True)

        df_gen_live = df_gen_live.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        # --- Save to lakehouse folders as Parquet ---
        year = start.year
        month = f"{start.month:02d}"

        price_dir = LAKEHOUSE_ROOT / "day_ahead_prices" / f"zone={bidding_zone}" / f"year={year}" / f"month={month}"
        price_dir.mkdir(parents=True, exist_ok=True)
        df_price_live.to_parquet(price_dir / f"price_{start.strftime('%Y-%m-%d')}.parquet", index=False)

        load_dir = LAKEHOUSE_ROOT / "total_load" / f"zone={bidding_zone}" / f"year={year}" / f"month={month}"
        load_dir.mkdir(parents=True, exist_ok=True)
        df_load_live.to_parquet(load_dir / f"load_{start.strftime('%Y-%m-%d')}.parquet", index=False)

        gen_dir = LAKEHOUSE_ROOT / "generation" / f"zone={bidding_zone}" / f"year={year}" / f"month={month}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        df_gen_live.to_parquet(gen_dir / f"gen_{start.strftime('%Y-%m-%d')}.parquet", index=False)

        print(f"Successfully saved aligned live partitions (Price, Load, Generation) for {target_date_str}.")

        # --- Merge into a live master chunk ---
        live_master_chunk = pd.merge(df_price_live, df_load_live, on=["timestamp", "bidding_zone"], how="inner")

        merge_gen_keys = ["timestamp"]
        if "bidding_zone" in df_gen_live.columns:
            merge_gen_keys.append("bidding_zone")

        live_master_chunk = pd.merge(live_master_chunk, df_gen_live, on=merge_gen_keys, how="inner")

        return live_master_chunk

    except Exception as e:
        print(f"Error during live ENTSO-E generation extraction: {e}")
        return None
