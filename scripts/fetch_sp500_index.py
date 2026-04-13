"""
Fetch the official S&P 500 total return index from CRSP `dsi`.

This is the S&P 500 daily return series sourced from S&P Dow Jones Indices,
which is what most academic papers (including Krauss et al. 2017) use as
the market benchmark.

Output: data/raw/sp500_index_daily.parquet
"""

import wrds
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "sp500_index_daily.parquet"


def main():
    conn = wrds.Connection()

    print("Fetching S&P 500 daily return + level from crsp.dsi...")
    df = conn.raw_sql(
        "SELECT date, sprtrn, spindx "
        "FROM crsp.dsi "
        "WHERE date BETWEEN '1989-01-01' AND '2015-12-31' "
        "ORDER BY date"
    )
    df["date"] = pd.to_datetime(df["date"])
    df["sprtrn"] = pd.to_numeric(df["sprtrn"], errors="coerce")
    df["spindx"] = pd.to_numeric(df["spindx"], errors="coerce")
    df = df.dropna(subset=["sprtrn"]).reset_index(drop=True)

    print(f"  {len(df):,} daily observations")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Mean daily return: {df['sprtrn'].mean():.6f}")
    print(f"  Annualized: {df['sprtrn'].mean() * 252:.4f}")
    print(f"  Std: {df['sprtrn'].std():.4f}")

    df.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")

    conn.close()


if __name__ == "__main__":
    main()
