"""
historical_loader.py

Loads and cleans the historical lakehouse data into master_energy_df.
This is your existing, already-working logic — unchanged, just moved
into its own module.
"""

from __future__ import annotations
import pandas as pd

from config import LAKEHOUSE_ROOT, HISTORICAL_SUBFOLDERS


def _file_to_hourly(df: pd.DataFrame, mtime: float) -> pd.DataFrame:
    """
    Resample ONE file to hourly means and record how complete each hour is.

    The lakehouse mixes resolutions: hourly history, raw 15-min files from
    after the Oct-2025 switch, and hourly files written by the fixed
    live_data.py / backfill. Deduplicating raw rows across those would mix
    an hourly mean with raw quarter-hours of the same hour. Resampling per
    file first makes every file hourly, and `_coverage` lets the loader
    prefer a complete hour (4/4 quarter-hours, or 1/1 for hourly data) over
    a partial one when two files cover the same timestamp.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    if df.empty:
        return df

    step = df["timestamp"].diff().dropna()
    step = step.mode().iloc[0] if not step.empty else pd.Timedelta(hours=1)
    per_hour = max(1, int(pd.Timedelta(hours=1) / step)) if step <= pd.Timedelta(hours=1) else 1

    df["_hour"] = df["timestamp"].dt.floor("h")
    numeric = [c for c in df.select_dtypes(include="number").columns]
    agg = {c: "mean" for c in numeric}
    for c in df.columns:
        if c not in numeric and c not in ("timestamp", "_hour"):
            agg[c] = "first"
    out = df.groupby("_hour").agg(agg)
    out["_coverage"] = df.groupby("_hour").size() / per_hour
    out["_mtime"] = mtime
    return out.rename_axis("timestamp").reset_index()


def load_partitioned_parquet(subfolder_name: str) -> pd.DataFrame | None:
    """
    Reads every parquet file in a subfolder, converts each to hourly, and
    keeps ONE row per hour: the most complete one, newest file on ties
    (so a re-fetched / backfilled day replaces the old partial file).
    """
    folder_path = LAKEHOUSE_ROOT / subfolder_name
    if not folder_path.exists():
        print(f"Warning: Directory {folder_path} does not exist.")
        return None

    parquet_files = list(folder_path.glob("**/*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return None

    print(f"Found {len(parquet_files)} parquet files in '{subfolder_name}'. Reading and merging...")
    df_list = [_file_to_hourly(pd.read_parquet(f), f.stat().st_mtime) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)
    df = (
        df.sort_values(["timestamp", "_coverage", "_mtime"], kind="stable")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .drop(columns=["_coverage", "_mtime"])
        .reset_index(drop=True)
    )
    return df


def load_all_historical() -> dict[str, pd.DataFrame | None]:
    """Loads all three raw historical datasets into a dict."""
    raw_data = {}
    for sf in HISTORICAL_SUBFOLDERS:
        raw_data[sf] = load_partitioned_parquet(sf)

    for sf, df in raw_data.items():
        if df is not None:
            print(f"\n--- Dataset: {sf} ---")
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())

    return raw_data


def clean_and_merge_data(raw_data_dict: dict, missing_threshold: float = 0.90) -> pd.DataFrame:
    print("Starting data cleaning and harmonization...")

    # --- 1. Clean Day-Ahead Prices ---
    df_price = raw_data_dict.get("day_ahead_prices")
    if df_price is not None:
        df_price["timestamp"] = pd.to_datetime(df_price["timestamp"], utc=True)
        df_price = df_price.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        price_cols = ["timestamp", "price_eur_mwh"]
        if "bidding_zone" in df_price.columns:
            price_cols.append("bidding_zone")
        df_price = df_price[price_cols]

        # Since 1 Oct 2025 DE-LU day-ahead prices are 15-minute products
        # (96/day). Load and generation are resampled to hourly below, so
        # prices must be too — otherwise the merge keeps only the :00
        # quarter-hour and the target silently becomes "first-quarter-hour
        # price" instead of the hourly average. Hourly data (pre-Oct 2025)
        # passes through unchanged.
        df_price.set_index("timestamp", inplace=True)
        agg_price = {"price_eur_mwh": "mean"}
        if "bidding_zone" in df_price.columns:
            agg_price["bidding_zone"] = "first"
        df_price = df_price.resample("h").agg(agg_price).reset_index()
        df_price = df_price.dropna(subset=["price_eur_mwh"])

    # --- 2. Clean Total Load ---
    df_load = raw_data_dict.get("total_load")
    if df_load is not None:
        df_load["timestamp"] = pd.to_datetime(df_load["timestamp"], utc=True)

        # Different pipeline versions stored load under three names. Combine
        # ALL of them: the old code rebuilt total_load_mw from load_mw /
        # total_load only, which overwrote the column that every live file
        # since July 2026 uses -> all recent load silently became NaN.
        load_series = [df_load[c] for c in ["total_load_mw", "load_mw", "total_load"] if c in df_load.columns]
        combined_load = load_series[0]
        for s_ in load_series[1:]:
            combined_load = combined_load.combine_first(s_)
        df_load["total_load_mw"] = combined_load

        load_cols = ["timestamp", "total_load_mw"]
        if "bidding_zone" in df_load.columns:
            load_cols.append("bidding_zone")
        df_load = df_load[[c for c in load_cols if c in df_load.columns]]
        df_load = df_load.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        df_load.set_index("timestamp", inplace=True)
        numeric_cols_load = df_load.select_dtypes(include=["number"]).columns
        agg_dict = {col: "mean" for col in numeric_cols_load}
        if "bidding_zone" in df_load.columns:
            agg_dict["bidding_zone"] = "first"
        df_load = df_load.resample("h").agg(agg_dict).reset_index()

    # --- 3. Clean Generation ---
    df_gen = raw_data_dict.get("generation")
    if df_gen is not None:
        df_gen["timestamp"] = pd.to_datetime(df_gen["timestamp"], utc=True)

        mapping_rules = {
            "Solar_Actual Aggregated": "solar_mw",
            "Wind Onshore_Actual Aggregated": "wind_onshore_mw",
            "Wind Offshore_Actual Aggregated": "wind_offshore_mw",
            "Fossil Gas_Actual Aggregated": "gas_mw",
            "Fossil Hard coal_Actual Aggregated": "hard_coal_mw",
            "Nuclear_Actual Aggregated": "nuclear_mw",
        }

        for old_col, new_col in mapping_rules.items():
            if old_col in df_gen.columns and new_col in df_gen.columns:
                df_gen[new_col] = df_gen[new_col].combine_first(df_gen[old_col])
                df_gen.drop(columns=[old_col], inplace=True)
            elif old_col in df_gen.columns:
                df_gen.rename(columns={old_col: new_col}, inplace=True)

        df_gen = df_gen.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

        nan_fraction = df_gen.isna().mean()
        cols_to_drop = nan_fraction[nan_fraction > missing_threshold].index
        cols_to_drop = [c for c in cols_to_drop if c not in ["timestamp", "bidding_zone"]]
        df_gen.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        df_gen.set_index("timestamp", inplace=True)
        numeric_cols_gen = df_gen.select_dtypes(include=["number"]).columns
        agg_dict_gen = {col: "mean" for col in numeric_cols_gen}
        if "bidding_zone" in df_gen.columns:
            agg_dict_gen["bidding_zone"] = "first"
        df_gen = df_gen.resample("h").agg(agg_dict_gen).reset_index()

    # --- 4. Merge All Datasets on Timestamp ---
    print("Merging datasets...")
    master_df = df_price.copy()

    if df_load is not None:
        merge_keys = ["timestamp"]
        if "bidding_zone" in master_df.columns and "bidding_zone" in df_load.columns:
            merge_keys.append("bidding_zone")
        # LEFT join on the price spine: at gate closure the prices of day D
        # are known but D's actual load/generation are not. Keeping those
        # price rows lets price lags run without gaps; rows with missing
        # fundamentals are dropped later in training via dropna.
        master_df = pd.merge(master_df, df_load, on=merge_keys, how="left")

    if df_gen is not None:
        merge_keys = ["timestamp"]
        if "bidding_zone" in master_df.columns and "bidding_zone" in df_gen.columns:
            merge_keys.append("bidding_zone")
        master_df = pd.merge(master_df, df_gen, on=merge_keys, how="left")

    master_df = master_df.sort_values("timestamp").reset_index(drop=True)

    print(f"Master DataFrame successfully created with shape: {master_df.shape}")
    return master_df
