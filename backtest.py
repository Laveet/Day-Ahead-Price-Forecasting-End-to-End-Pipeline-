"""
backtest.py

Walk-forward (rolling-origin) backtest of the day-ahead price model.

For every day D in the test period it simulates the real morning routine:
  - information cut at 12:00 CET on D (same rules as the live pipeline):
      * training rows: delivery days <= D-1 (actual load/generation of D is
        not known yet)
      * D+1 features: price lags from D and earlier, load lags >= 48 h,
        24h rolling price stats FROZEN at the end of D
  - forecast all hours of delivery day D+1
  - store forecast next to the actual price and the naive baselines

It compares several model variants on exactly the same days, so a
difference in error is caused by the features, not by the test period.

    python backtest.py                      # last 365 days, retrain weekly
    python backtest.py --days 90            # shorter test period
    python backtest.py --retrain daily      # closest to live, ~7x slower

IMPORTANT LIMITATION (state it when you report results)
-------------------------------------------------------
For D+1 the backtest uses ACTUAL load / wind / solar, because historical
day-ahead forecasts are not stored in the lakehouse. Live, the model only
has forecasts, so absolute errors here are optimistic. The comparison
between variants is still fair, because every variant gets the same inputs.

Outputs (data/lakehouse/backtest/run=<timestamp>/):
  predictions.parquet   one row per hour: actual, every model, every baseline
  summary.csv           MAE / RMSE / bias / rMAE per model
  by_month.csv          MAE per month and model
  by_hour.csv           MAE per local hour and model
  rolling_mae.png       30-day rolling MAE per model
"""

from __future__ import annotations
import argparse
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import LAKEHOUSE_ROOT, BASE_FEATURES, MODEL_FEATURES, LGBM_PARAMS, ROLLING_WINDOW
from data.historical_loader import load_all_historical, clean_and_merge_data
from data.market_time import local_date, hours_in_delivery_day
from features.engineering import engineer_features
from data.fuel_data import FUEL_ASOF_PATH, load_fuel_asof

ROLL_COLS = [f"price_rolling_mean_{ROLLING_WINDOW}", f"price_rolling_std_{ROLLING_WINDOW}"]

# Model variants to compare. Add a variant here to test a new feature.
VARIANTS = {
    "lgbm_base": BASE_FEATURES,            # without gas
    "lgbm_ttf": MODEL_FEATURES,            # current live model (with gas)
}
BASELINES = {
    "naive_week": "price_eur_mwh_lag_168",   # same hour last week
    "naive_day": "price_eur_mwh_lag_24",     # same hour on D (known at gate closure)
}
REFERENCE = "naive_week"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_feature_table() -> pd.DataFrame:
    master = clean_and_merge_data(load_all_historical())
    feats = engineer_features(master)
    feats["delivery_date"] = pd.to_datetime(local_date(feats["timestamp"])).astype("datetime64[ns]")

    if FUEL_ASOF_PATH.exists():
        fuel = load_fuel_asof()[["delivery_date", "ttf_eur_mwh"]]
        feats = feats.merge(fuel, on="delivery_date", how="left")
    else:
        print(f"NOTE: {FUEL_ASOF_PATH} not found -> run `python -m data.fuel_data` first; "
              "variants needing ttf_eur_mwh are skipped.")
    return feats


def complete_days(feats: pd.DataFrame) -> set:
    # a day is usable only if price AND the D+1 inputs (load, wind, solar) are complete
    need = ["price_eur_mwh", "total_load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]
    counts = feats.dropna(subset=need).groupby("delivery_date").size()
    expected = pd.Series([hours_in_delivery_day(str(d.date())) for d in counts.index], index=counts.index)
    return set(counts.index[counts == expected])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def diebold_mariano(err_a: pd.Series, err_b: pd.Series, days: pd.Series) -> tuple[float, float]:
    """
    DM test on absolute errors, computed on DAILY mean loss differentials
    (hourly errors within a day are strongly correlated; daily series is
    closer to the test's assumptions). Newey-West variance with lag 7 for
    remaining weekly autocorrelation. Negative statistic -> model A better.
    Returns (statistic, two-sided p-value).
    """
    d = (err_a.abs() - err_b.abs()).groupby(days).mean().to_numpy()
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    dc = d - d.mean()
    lag = min(7, n - 1)
    var = dc @ dc / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def summarise(pred: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    ref_mae = (pred["actual"] - pred[REFERENCE]).abs().mean()
    for m in models:
        e = pred["actual"] - pred[m]
        rows.append({
            "model": m,
            "MAE": e.abs().mean(),
            "RMSE": math.sqrt((e ** 2).mean()),
            "bias (actual-forecast)": e.mean(),
            f"rMAE vs {REFERENCE}": e.abs().mean() / ref_mae,
            "MAE negative-price hours": e[pred["actual"] < 0].abs().mean(),
        })
    return pd.DataFrame(rows).set_index("model").round(3)


# ---------------------------------------------------------------------------
# backtest loop
# ---------------------------------------------------------------------------
def run(days: int, retrain: str, end: str | None) -> None:
    t0 = time.time()
    feats = build_feature_table()
    variants = {k: v for k, v in VARIANTS.items() if all(c in feats.columns for c in v)}
    ok_days = complete_days(feats)

    last = pd.Timestamp(end) if end else max(ok_days)
    targets = [d for d in pd.date_range(last - pd.Timedelta(days=days - 1), last) if d in ok_days]
    retrain_every = 1 if retrain == "daily" else 7
    print(f"\nTest period: {targets[0].date()} -> {targets[-1].date()} ({len(targets)} delivery days), "
          f"retrain {retrain}, variants: {list(variants)}")

    models, results = {}, []
    for i, target in enumerate(targets):
        D = target - pd.Timedelta(days=1)                      # run day (gate closure)

        # ---- (re)train on delivery days <= D-1 ---------------------------
        if i % retrain_every == 0 or not models:
            train = feats[feats["delivery_date"] <= D - pd.Timedelta(days=1)]
            for name, cols in variants.items():
                t = train.dropna(subset=["price_eur_mwh"] + cols)
                models[name] = lgb.LGBMRegressor(**LGBM_PARAMS).fit(t[cols], t["price_eur_mwh"])

        # ---- features for D+1 as known at gate closure ---------------------
        test = feats[feats["delivery_date"] == target].copy()
        # rolling stats frozen at the end of D (value at the first hour of D+1
        # uses the 24 hours before it, i.e. D's prices) -> identical to live
        for c in ROLL_COLS:
            test[c] = test[c].iloc[0]

        out = pd.DataFrame({"timestamp": test["timestamp"].values,
                            "delivery_date": target,
                            "hour": test["hour"].values,
                            "actual": test["price_eur_mwh"].values})
        for name, cols in variants.items():
            out[name] = models[name].predict(test[cols])
        for name, col in BASELINES.items():
            out[name] = test[col].values
        results.append(out)

        if (i + 1) % 30 == 0 or i + 1 == len(targets):
            done = pd.concat(results)
            maes = {m: (done.actual - done[m]).abs().mean() for m in list(variants) + [REFERENCE]}
            print(f"  {i + 1:4d}/{len(targets)}  {target.date()}  running MAE: "
                  + "  ".join(f"{k}={v:.2f}" for k, v in maes.items())
                  + f"  ({time.time() - t0:.0f}s)")

    pred = pd.concat(results, ignore_index=True).dropna(subset=["actual"])
    all_models = list(variants) + list(BASELINES)

    # ---- report ----------------------------------------------------------
    summary = summarise(pred, all_models)
    print("\n=== Summary (EUR/MWh) ===")
    print(summary.to_string())

    names = list(variants)
    if len(names) >= 2:
        a, b = names[1], names[0]
        stat, p = diebold_mariano(pred.actual - pred[a], pred.actual - pred[b], pred.delivery_date)
        verdict = ("significantly better" if stat < 0 else "significantly worse") if p < 0.05 else "no significant difference"
        print(f"\nDiebold-Mariano {a} vs {b}: stat={stat:.2f}, p={p:.3f} -> {a} is {verdict} (5% level)")

    pred["month"] = pred.delivery_date.dt.to_period("M").astype(str)
    abs_err = pd.DataFrame({m: (pred.actual - pred[m]).abs() for m in all_models})
    by_month = abs_err.groupby(pred["month"]).mean().round(2)
    by_hour = abs_err.groupby(pred["hour"]).mean().round(2)
    print("\n=== MAE by month ===")
    print(by_month.to_string())

    out_dir = LAKEHOUSE_ROOT / "backtest" / f"run={datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    summary.to_csv(out_dir / "summary.csv")
    by_month.to_csv(out_dir / "by_month.csv")
    by_hour.to_csv(out_dir / "by_hour.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        daily = abs_err.groupby(pred["delivery_date"]).mean()
        ax = daily.rolling(30, min_periods=7).mean().plot(figsize=(12, 4), lw=1.2)
        ax.set_ylabel("MAE, 30-day rolling (EUR/MWh)"); ax.set_title("Walk-forward backtest")
        plt.tight_layout(); plt.savefig(out_dir / "rolling_mae.png", dpi=120); plt.close()
    except ImportError:
        pass
    print(f"\nSaved results to {out_dir}  (total {time.time() - t0:.0f}s)")
    print("Reminder: D+1 load/wind/solar are ACTUALS here -> absolute errors are optimistic; "
          "the comparison between variants is fair.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="number of delivery days to test")
    ap.add_argument("--retrain", choices=["weekly", "daily"], default="weekly")
    ap.add_argument("--end", default=None, help="last delivery day (default: latest complete day)")
    args = ap.parse_args()
    run(args.days, args.retrain, args.end)
