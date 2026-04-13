"""
Build features and labels from Datastream data.

Same logic as the CRSP pipeline but uses infocode instead of permno,
and computes returns from Datastream's total return index.

Usage:
    python scripts/build_features_labels_datastream.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"

DAILY_LAGS = list(range(1, 21))
MULTI_PERIOD_LAGS = list(range(40, 241, 20))
ALL_LAGS = DAILY_LAGS + MULTI_PERIOD_LAGS
FEATURE_COLS = [f"R{m}" for m in ALL_LAGS]


def compute_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute 31 lagged return features from daily returns."""
    df = returns[["infocode", "date", "ret"]].copy()
    df = df.sort_values(["infocode", "date"]).reset_index(drop=True)

    # Build cumulative price index per stock
    df["price_idx"] = df.groupby("infocode")["ret"].transform(
        lambda r: (1 + r).cumprod()
    )

    # Compute R_m = price_idx_t / price_idx_{t-m} - 1 for each lag
    for m in ALL_LAGS:
        lagged = df.groupby("infocode")["price_idx"].shift(m)
        df[f"R{m}"] = df["price_idx"] / lagged - 1

    # Drop rows with any NaN feature (insufficient lookback)
    df = df.dropna(subset=FEATURE_COLS)
    df = df[["infocode", "date"] + FEATURE_COLS].reset_index(drop=True)
    return df


def compute_labels(returns: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Compute classification and regression labels."""
    ret = returns[["infocode", "date", "ret"]].copy()
    ret = ret.sort_values(["infocode", "date"]).reset_index(drop=True)

    # Next-day return and date
    ret["next_day_ret"] = ret.groupby("infocode")["ret"].shift(-1)
    ret["next_day_date"] = ret.groupby("infocode")["date"].shift(-1)
    ret = ret.dropna(subset=["next_day_ret"])
    ret["next_day_date"] = ret["next_day_date"].astype("datetime64[ns]")

    # Restrict to eligible stock-days
    labels = eligible.merge(ret, on=["date", "infocode"], how="inner")

    # Next-day eligibility check
    eligible_next = eligible.rename(columns={"date": "next_day_date"})
    labels = labels.merge(eligible_next, on=["next_day_date", "infocode"], how="inner")

    # Cross-sectional median on next_day_date
    median_by_day = (
        labels.groupby("next_day_date")["next_day_ret"]
        .median()
        .rename("next_day_median")
    )
    labels = labels.merge(median_by_day, on="next_day_date", how="left")

    labels["u_excess"] = labels["next_day_ret"] - labels["next_day_median"]
    labels["y_binary"] = (labels["u_excess"] > 0).astype(int)

    labels = labels[
        ["date", "infocode", "next_day_date", "next_day_ret",
         "next_day_median", "u_excess", "y_binary"]
    ].sort_values(["date", "infocode"]).reset_index(drop=True)

    return labels


def main():
    print("Loading Datastream data...")
    returns = pd.read_parquet(DS_DIR / "ds_daily_returns.parquet")
    eligible = pd.read_parquet(DS_DIR / "ds_universe_daily.parquet")
    returns["date"] = pd.to_datetime(returns["date"])
    eligible["date"] = pd.to_datetime(eligible["date"])
    print(f"  Returns: {len(returns):,} rows, {returns['infocode'].nunique()} stocks")
    print(f"  Eligible: {len(eligible):,} stock-days")

    print("\nComputing features (31 lagged returns)...")
    features = compute_features(returns)
    print(f"  {len(features):,} rows, {features['infocode'].nunique()} stocks")
    features.to_parquet(DS_DIR / "ds_features.parquet", index=False)

    print("\nComputing labels...")
    labels = compute_labels(returns, eligible)
    print(f"  {len(labels):,} rows, {labels['infocode'].nunique()} stocks")
    print(f"  y_binary=1 rate: {labels['y_binary'].mean():.4f}")
    labels.to_parquet(DS_DIR / "ds_labels.parquet", index=False)

    print("\nDone.")
    for f in sorted(DS_DIR.glob("ds_features*")) + sorted(DS_DIR.glob("ds_labels*")):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
