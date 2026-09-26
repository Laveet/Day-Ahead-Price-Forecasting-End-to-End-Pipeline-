"""
market_time.py

One place that defines what a "delivery day" is, so every module queries
and filters the same hours.
"""

from __future__ import annotations
import pandas as pd

from config import MARKET_TZ


def delivery_day_bounds(date_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    [start, end) of a local delivery day, as tz-aware Europe/Berlin timestamps.

    `end` is EXCLUSIVE — entsoe-py treats the end of a query as exclusive,
    so passing start + 1 day returns every interval of the day. (The old
    `start + 1 day - 1 hour` silently dropped the last hour of every day.)

    Using local midnights also makes DST days come out right: 23 hours in
    March, 25 in October.
    """
    start = pd.Timestamp(date_str, tz=MARKET_TZ)
    end = start + pd.DateOffset(days=1)   # calendar day, not 24h -> DST-safe
    return start, end


def hours_in_delivery_day(date_str: str) -> int:
    start, end = delivery_day_bounds(date_str)
    return int((end - start) / pd.Timedelta(hours=1))


def local_date(ts: pd.Series) -> pd.Series:
    """Local delivery date of each UTC timestamp."""
    return pd.to_datetime(ts, utc=True).dt.tz_convert(MARKET_TZ).dt.date
