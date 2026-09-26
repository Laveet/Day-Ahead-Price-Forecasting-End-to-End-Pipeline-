"""
main.py

Entry point — run once per day, on the MORNING of day D (before the
12:00 CET day-ahead gate closure):

    python main.py              # D = today (Europe/Berlin)
    python main.py 2026-09-26   # explicit D, e.g. for replaying a past day

The pipeline will:
  - add D-1's complete actuals and D's (already known) prices to history
  - grade the forecast previously made for D
  - retrain and forecast delivery day D+1, the day you are bidding for
"""

import sys
import pandas as pd

from config import MARKET_TZ
from data.historical_loader import load_all_historical, clean_and_merge_data
from pipeline.daily_pipeline import run_day_ahead_pipeline

if __name__ == "__main__":
    now_local = pd.Timestamp.now(tz=MARKET_TZ)
    run_date = sys.argv[1] if len(sys.argv) > 1 else now_local.strftime("%Y-%m-%d")

    if len(sys.argv) == 1 and now_local.hour >= 12:
        print("NOTE: it is past 12:00 CET — today's gate for tomorrow's delivery "
              "has closed, so this forecast is for evaluation only.")

    raw_data = load_all_historical()
    master_energy_df = clean_and_merge_data(raw_data)

    master_updated, model, forecast_result = run_day_ahead_pipeline(master_energy_df, run_date)

    print("\nForecast:")
    print(forecast_result)
