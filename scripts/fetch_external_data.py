"""
Fetch external data from WRDS needed for reproduction tables/figures.

Usage:
    python scripts/fetch_external_data.py

Fetches:
    1. GICS industry codes (for Tables 1, 6)    -> data/raw/gics_industry.parquet
    2. Fama-French daily factors (for Table 4)   -> data/raw/ff_factors_daily.parquet
    3. VIX daily (for Table 4, Figure 2)         -> data/raw/vix_daily.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from krauss.data.wrds_extract import (
    get_connection,
    fetch_gics_industry,
    fetch_ff_factors,
)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def main():
    print("=" * 60)
    print("FETCH EXTERNAL DATA FROM WRDS")
    print("=" * 60)

    conn = get_connection()
    print("Connected to WRDS.\n")

    # 1. GICS industry codes
    print("[1/3] Fetching GICS industry codes...")
    gics = fetch_gics_industry(conn)
    gics.to_parquet(RAW / "gics_industry.parquet", index=False)
    n_mapped = gics["permno"].nunique()
    print(f"      {len(gics)} rows, {n_mapped} unique PERMNOs")
    print(f"      Sectors: {sorted(gics['gsector'].dropna().unique())}")
    print(f"      Saved -> data/raw/gics_industry.parquet\n")

    # 2. Fama-French factors
    print("[2/3] Fetching Fama-French daily factors...")
    ff = fetch_ff_factors(conn, "1990-01-01", "2015-12-31")
    ff.to_parquet(RAW / "ff_factors_daily.parquet", index=False)
    print(f"      {len(ff)} daily obs, {ff['date'].min().date()} to {ff['date'].max().date()}")
    print(f"      Columns: {list(ff.columns)}")
    print(f"      Saved -> data/raw/ff_factors_daily.parquet\n")

    conn.close()

    # 3. VIX (not available on WRDS — download from FRED)
    print("[3/3] Fetching daily VIX from FRED...")
    vix_url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        "?id=VIXCLS&cosd=1990-01-01&coed=2015-12-31"
    )
    vix = pd.read_csv(vix_url)
    vix.columns = ["date", "vix"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix["vix"] = pd.to_numeric(vix["vix"], errors="coerce")
    vix = vix.dropna(subset=["vix"]).reset_index(drop=True)
    vix.to_parquet(RAW / "vix_daily.parquet", index=False)
    print(f"      {len(vix)} daily obs, {vix['date'].min().date()} to {vix['date'].max().date()}")
    print(f"      Saved -> data/raw/vix_daily.parquet\n")
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
