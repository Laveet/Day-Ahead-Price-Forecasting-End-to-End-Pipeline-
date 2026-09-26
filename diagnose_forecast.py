"""
diagnose_forecast.py

Shows which day-ahead forecasts ENTSO-E has published right now, for today
and tomorrow (Europe/Berlin), so you can see whether main.py can run yet.

    python diagnose_forecast.py
"""
import pandas as pd
from entsoe.exceptions import NoMatchingDataError

from config import MARKET_TZ, DEFAULT_BIDDING_ZONE
from data.entsoe_client import get_client
from data.market_time import delivery_day_bounds

client = get_client()
now = pd.Timestamp.now(tz=MARKET_TZ)
print(f"Now: {now:%Y-%m-%d %H:%M %Z}\n")

queries = {
    "load forecast": client.query_load_forecast,
    "wind/solar forecast": client.query_wind_and_solar_forecast,
}
for label, day in [("today", now), ("tomorrow", now + pd.Timedelta(days=1))]:
    d = day.strftime("%Y-%m-%d")
    start, end = delivery_day_bounds(d)
    for name, fn in queries.items():
        try:
            res = fn(DEFAULT_BIDDING_ZONE, start=start, end=end)
            cols = list(res.columns) if isinstance(res, pd.DataFrame) else ["value"]
            print(f"{label:9s} {d}  {name:20s} OK   {len(res):3d} rows, "
                  f"{res.index.min():%H:%M} -> {res.index.max():%H:%M}, columns={cols}")
        except NoMatchingDataError:
            print(f"{label:9s} {d}  {name:20s} NOT PUBLISHED")
        except Exception as e:
            print(f"{label:9s} {d}  {name:20s} ERROR {type(e).__name__}: {e}")
