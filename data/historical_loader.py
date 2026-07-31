"""
historical_loader.py

Loads and cleans the historical lakehouse data into master_energy_df.
This is your existing, already-working logic — unchanged, just moved
into its own module.
"""

from __future__ import annotations
import pandas as pd

from config import LAKEHOUSE_ROOT, HISTORICAL_SUBFOLDERS


def load_partitioned_parquet(subfolder_name: str) -> pd.DataFrame | None:
    """Recursively reads and concatenates all parquet files in a subfolder."""
    folder_path = LAKEHOUSE_ROOT / subfolder_name
    if not folder_path.exists():
        print(f"Warning: Directory {folder_path} does not exist.")
        return None

    parquet_files = list(folder_path.glob("**/*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return None

    print(f"Found {len(parquet_files)} parquet files in '{subfolder_name}'. Reading and merging...")
    df_list = [pd.read_parquet(file) for file in parquet_files]
    return pd.concat(df_list, ignore_index=True)


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

    # --- 2. Clean Total Load ---
    df_load = raw_data_dict.get("total_load")
    if df_load is not None:
        df_load["timestamp"] = pd.to_datetime(df_load["timestamp"], utc=True)

        if "load_mw" in df_load.columns and "total_load" in df_load.columns:
            df_load["total_load_mw"] = df_load["load_mw"].combine_first(df_load["total_load"])
        elif "load_mw" in df_load.columns:
            df_load["total_load_mw"] = df_load["load_mw"]
        elif "total_load" in df_load.columns:
            df_load["total_load_mw"] = df_load["total_load"]

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
        master_df = pd.merge(master_df, df_load, on=merge_keys, how="inner")

    if df_gen is not None:
        merge_keys = ["timestamp"]
        if "bidding_zone" in master_df.columns and "bidding_zone" in df_gen.columns:
            merge_keys.append("bidding_zone")
        master_df = pd.merge(master_df, df_gen, on=merge_keys, how="inner")

    master_df = master_df.sort_values("timestamp").reset_index(drop=True)

    print(f"Master DataFrame successfully created with shape: {master_df.shape}")
    return master_df
