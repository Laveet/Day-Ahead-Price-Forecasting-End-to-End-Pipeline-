"""
fuel_data.py

Gas price (TTF) for the day-ahead model: download, validate, and align
point-in-time to delivery days.

    python -m data.fuel_data        # refresh everything (download -> validate -> align)

Used by:
  - fuel_prices.ipynb   (exploration / plots)
  - backtest.py         (reads the aligned table)
  - later: the daily pipeline, to refresh before each forecast

Point-in-time rule
------------------
Bids for delivery day T are submitted before 12:00 CET on T-1. The gas
settlement of T-1 is not known then, so delivery day T uses the last
closing price dated T-2 or earlier. This prevents leakage.

Files (data/lakehouse/fuel_prices/)
  raw/ttf_yahoo_daily.parquet   date, ttf_eur_mwh        (one row per trading day)
  fuel_daily_asof.parquet       delivery_date, ttf_eur_mwh, ttf_obs_date, ttf_age_days
"""

from __future__ import annotations
import pandas as pd

from config import LAKEHOUSE_ROOT, MARKET_TZ

FUEL_DIR = LAKEHOUSE_ROOT / "fuel_prices"
RAW_TTF_PATH = FUEL_DIR / "raw" / "ttf_yahoo_daily.parquet"
FUEL_ASOF_PATH = FUEL_DIR / "fuel_daily_asof.parquet"

TTF_TICKER = "TTF=F"            # ICE Endex Dutch TTF front-month future, EUR/MWh
START = "2022-12-01"            # one month before the price history starts
FIRST_DELIVERY_DAY = "2023-01-01"
AVAILABLE_LAG_DAYS = 2          # delivery day T uses values dated <= T-2


# ---------------------------------------------------------------------------
# 1. download
# ---------------------------------------------------------------------------
def download_ttf(start: str = START) -> pd.DataFrame:
    """Daily TTF front-month closing prices from Yahoo Finance; saved to RAW_TTF_PATH."""
    import yfinance as yf     # imported here so the rest of the project works without it

    end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(TTF_TICKER, start=start, end=end, interval="1d",
                      auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {TTF_TICKER}. "
                           "Check your internet, or run `pip install -U yfinance`.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):          # newer yfinance: one column per ticker
        close = close.iloc[:, 0]
    idx = pd.to_datetime(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    df = (pd.DataFrame({"date": idx.normalize().astype("datetime64[ns]"),
                        "ttf_eur_mwh": close.to_numpy(dtype=float)})
          .dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True))
    RAW_TTF_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_TTF_PATH, index=False)
    return df


def load_raw_ttf() -> pd.DataFrame:
    df = pd.read_parquet(RAW_TTF_PATH)
    df["date"] = df["date"].astype("datetime64[ns]")   # parquet may return [ms]
    return df


# ---------------------------------------------------------------------------
# 2. validate
# ---------------------------------------------------------------------------
def validate_ttf(df: pd.DataFrame, lo: float = 5, hi: float = 350, start_by: str = "2022-12-31",
                 max_age_days: int = 7, max_gap_days: int = 5, max_jump: float = 0.25
                 ) -> tuple[list[str], list[str]]:
    """
    Returns (errors, warnings).
    Errors mean the data must not be used: empty, starts too late, stale,
    duplicate dates, missing or implausible values.
    Warnings need a human look: long gaps (holidays are normal) and large
    day-to-day moves (contract roll or real market shock).
    """
    col, errors, warnings = "ttf_eur_mwh", [], []
    d = df.sort_values("date").reset_index(drop=True)
    if d.empty:
        return ["no rows"], []

    if d["date"].min() > pd.Timestamp(start_by):
        errors.append(f"starts {d.date.min().date()}, needs data by {start_by}")
    age = (pd.Timestamp.today().normalize() - d["date"].max()).days
    if age > max_age_days:
        errors.append(f"last value is {age} days old ({d.date.max().date()})")
    if d["date"].duplicated().any():
        errors.append(f"{d['date'].duplicated().sum()} duplicate dates")
    if d[col].isna().any():
        errors.append(f"{d[col].isna().sum()} missing values")
    bad = d[(d[col] < lo) | (d[col] > hi)]
    if len(bad):
        errors.append(f"{len(bad)} values outside [{lo}, {hi}], first on {bad.date.iloc[0].date()}")

    gaps = d["date"].diff().dt.days
    for i in d.index[gaps > max_gap_days]:
        warnings.append(f"gap of {int(gaps[i])} days ending {d.date[i].date()}")
    jumps = d[col].pct_change().abs()
    for i in d.index[jumps > max_jump]:
        warnings.append(f"{jumps[i]:.0%} move on {d.date[i].date()} (value {d[col][i]:.2f})")
    return errors, warnings


# ---------------------------------------------------------------------------
# 3. align point-in-time to delivery days
# ---------------------------------------------------------------------------
def build_fuel_asof(ttf: pd.DataFrame, first_day: str = FIRST_DELIVERY_DAY,
                    last_day: str | None = None) -> pd.DataFrame:
    """
    One row per delivery day T with the last TTF close dated <= T-2.
    Saved to FUEL_ASOF_PATH. Raises if any value is from after T-2 or missing.
    """
    if last_day is None:
        last_day = (pd.Timestamp.now(tz=MARKET_TZ).tz_localize(None).normalize()
                    + pd.Timedelta(days=1))
    days = pd.DataFrame({"delivery_date": pd.date_range(first_day, last_day, freq="D").astype("datetime64[ns]")})
    days["asof_date"] = days["delivery_date"] - pd.Timedelta(days=AVAILABLE_LAG_DAYS)

    obs = ttf.rename(columns={"date": "ttf_obs_date"}).copy()
    obs["ttf_obs_date"] = obs["ttf_obs_date"].astype("datetime64[ns]")
    aligned = pd.merge_asof(days.sort_values("asof_date"), obs.sort_values("ttf_obs_date"),
                            left_on="asof_date", right_on="ttf_obs_date", direction="backward")
    aligned["ttf_age_days"] = (aligned.delivery_date - aligned.ttf_obs_date).dt.days

    if not (aligned.ttf_obs_date <= aligned.asof_date).all():
        raise AssertionError("TTF value from after T-2 used (leakage)")
    if aligned["ttf_eur_mwh"].isna().any():
        raise AssertionError(f"{aligned.ttf_eur_mwh.isna().sum()} delivery days without a TTF value")

    aligned = aligned.drop(columns="asof_date")
    aligned.to_parquet(FUEL_ASOF_PATH, index=False)
    return aligned


def load_fuel_asof() -> pd.DataFrame:
    df = pd.read_parquet(FUEL_ASOF_PATH)
    df["delivery_date"] = df["delivery_date"].astype("datetime64[ns]")
    return df


def join_fuel(hourly: pd.DataFrame, fuel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ttf_eur_mwh to an hourly frame (UTC `timestamp`) by local delivery date."""
    from data.market_time import local_date
    fuel = load_fuel_asof() if fuel is None else fuel
    out = hourly.copy()
    out["delivery_date"] = pd.to_datetime(local_date(out["timestamp"])).astype("datetime64[ns]")
    merged = out.merge(fuel[["delivery_date", "ttf_eur_mwh"]], on="delivery_date", how="left")
    if len(merged) != len(out):
        raise AssertionError("fuel join changed the number of rows")
    return merged


# ---------------------------------------------------------------------------
# one call for scripts / the pipeline
# ---------------------------------------------------------------------------
def refresh_fuel_data(verbose: bool = True) -> pd.DataFrame:
    """Download -> validate (raise on errors) -> align. Returns the aligned table."""
    ttf = download_ttf()
    errors, warnings = validate_ttf(ttf)
    if verbose:
        print(f"TTF: {len(ttf)} trading days, {ttf.date.min().date()} -> {ttf.date.max().date()}")
        for w in warnings:
            print("  warning:", w)
    if errors:
        raise ValueError("TTF validation failed:\n  - " + "\n  - ".join(errors))
    aligned = build_fuel_asof(ttf)
    if verbose:
        print(f"Aligned {len(aligned)} delivery days -> {FUEL_ASOF_PATH} "
              f"(value age {aligned.ttf_age_days.min()}-{aligned.ttf_age_days.max()} days)")
    return aligned


if __name__ == "__main__":
    refresh_fuel_data()
