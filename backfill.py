"""
backfill.py

Finds every local delivery day whose history is incomplete (price, load or
wind/solar has fewer hours than the day should have — 23/24/25 on DST days)
and re-downloads it from ENTSO-E with the fixed fetch logic (hourly means,
exclusive end, Europe/Berlin days).

    python backfill.py                    # scan 2023-01-01 .. yesterday, fetch gaps
    python backfill.py --dry-run          # only list incomplete days
    python backfill.py --start 2026-07-01 --end 2026-09-25

Safe to re-run: re-fetched days overwrite that day's price_/load_/gen_
files, and the loader prefers the most complete row for every hour, so old
partial files cannot override the backfilled data. Days that are still
incomplete after the fetch (e.g. ENTSO-E has not published them) are
listed at the end.
"""

from __future__ import annotations
import argparse
import time
import pandas as pd

from config import MARKET_TZ, DEFAULT_BIDDING_ZONE
from data.historical_loader import load_all_historical, clean_and_merge_data
from data.live_data import fetch_align_and_save_live_data
from data.market_time import local_date, hours_in_delivery_day

CHECK_COLS = ["price_eur_mwh", "total_load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]


def find_incomplete_days(start: str, end: str) -> pd.DataFrame:
    master = clean_and_merge_data(load_all_historical())
    master["day"] = local_date(master["timestamp"])
    days = pd.date_range(start, end).date
    cols = [c for c in CHECK_COLS if c in master.columns]
    counts = master.groupby("day")[cols].count().reindex(days, fill_value=0)
    expected = pd.Series([hours_in_delivery_day(str(d)) for d in days], index=days)
    incomplete = counts[counts.lt(expected, axis=0).any(axis=1)].copy()
    incomplete.insert(0, "expected_hours", expected[incomplete.index])
    return incomplete


def main():
    yesterday = (pd.Timestamp.now(tz=MARKET_TZ) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=yesterday, help="last day to check (must have ended)")
    ap.add_argument("--zone", default=DEFAULT_BIDDING_ZONE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between days (API rate limit)")
    args = ap.parse_args()

    incomplete = find_incomplete_days(args.start, args.end)
    print(f"\n{len(incomplete)} incomplete day(s) between {args.start} and {args.end}:")
    print(incomplete.to_string() if len(incomplete) else "  none")
    if args.dry_run or incomplete.empty:
        return

    failed = []
    for i, day in enumerate(incomplete.index, 1):
        d = str(day)
        print(f"\n[{i}/{len(incomplete)}] {d}")
        for attempt in range(3):
            chunk = fetch_align_and_save_live_data(d, args.zone)
            if chunk is not None and not chunk.empty:
                print(f"  -> {len(chunk)} hourly rows saved")
                break
            time.sleep(5 * (attempt + 1))
        else:
            failed.append(d)
        time.sleep(args.pause)

    print("\nRe-checking history after backfill...")
    still = find_incomplete_days(args.start, args.end)
    print(f"{len(still)} day(s) still incomplete:")
    print(still.to_string() if len(still) else "  none — history is complete")
    if failed:
        print("Download failed for:", ", ".join(failed))


if __name__ == "__main__":
    main()
