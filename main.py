"""
main.py

Entry point — run this once per day.

    python main.py 2026-07-30

`day_x` is "today": the date whose actuals have just become available.
The pipeline will:
  - evaluate the forecast previously made for `day_x` (if any)
  - retrain on the updated history
  - forecast and archive `day_x + 1`
"""

import sys
from data.historical_loader import load_all_historical, clean_and_merge_data
from pipeline.daily_pipeline import run_day_ahead_pipeline

if __name__ == "__main__":
    today = sys.argv[1] if len(sys.argv) > 1 else "today"
    print(today)

    raw_data = load_all_historical()
    master_energy_df = clean_and_merge_data(raw_data)

    master_updated, model, forecast_result = run_day_ahead_pipeline(master_energy_df, today)

    print("\nForecast:")
    print(forecast_result)
