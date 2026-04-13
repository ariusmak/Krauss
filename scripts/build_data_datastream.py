"""
Build the Datastream-based data pipeline (paper parity).

This replaces CRSP with Datastream as the data source, matching the paper exactly:
    - S&P 500 membership from Datastream (ds2constmth, indexlistintcode=4408)
    - Daily total return indices from Datastream (ds2primqtri)
    - Daily returns computed from RI: ret_t = RI_t / RI_{t-1} - 1

Output files (saved to data/datastream/):
    - ds_sp500_membership.parquet   — raw spell data
    - ds_membership_monthly.parquet — monthly membership panel
    - ds_daily_returns.parquet      — (infocode, date, ret, ri)
    - ds_universe_daily.parquet     — daily eligibility
    - ds_names.parquet              — security metadata

Usage:
    python scripts/build_data_datastream.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"

SP500_CODE = 4408  # S&P 500 in Datastream ds2constmth


def main():
    DS_DIR.mkdir(parents=True, exist_ok=True)

    conn = wrds.Connection()

    # ----------------------------------------------------------------
    # 1. S&P 500 membership spells
    # ----------------------------------------------------------------
    print("1. Fetching S&P 500 membership spells...")
    sp500 = conn.raw_sql(
        "SELECT constintcode, infocode, startdate, enddate "
        "FROM tr_ds_equities.ds2constmth "
        f"WHERE indexlistintcode = {SP500_CODE}"
    )
    sp500["startdate"] = pd.to_datetime(sp500["startdate"])
    sp500["enddate"] = pd.to_datetime(sp500["enddate"])
    sp500["infocode"] = sp500["infocode"].astype("Int64")
    print(f"   {len(sp500)} spells, {sp500['infocode'].nunique()} unique stocks")
    sp500.to_parquet(DS_DIR / "ds_sp500_membership.parquet", index=False)

    # ----------------------------------------------------------------
    # 2. Build monthly membership panel (paper: Dec 1989 - Sep 2015)
    # ----------------------------------------------------------------
    print("2. Building monthly membership panel...")
    month_ends = pd.date_range("1989-12-31", "2015-09-30", freq="ME")
    records = []
    for me_date in month_ends:
        mask = (sp500["startdate"] <= me_date) & (sp500["enddate"] >= me_date)
        members = sp500.loc[mask, "infocode"].dropna().unique()
        # Effective month = next month (no-lookahead)
        effective = (me_date + pd.offsets.MonthBegin(1)).to_period("M")
        for ic in members:
            records.append({
                "infocode": int(ic),
                "month_end_date": me_date,
                "effective_month": effective,
            })

    membership = pd.DataFrame(records)
    monthly_counts = membership.groupby("month_end_date")["infocode"].nunique()
    print(f"   {len(membership):,} rows, avg {monthly_counts.mean():.1f} stocks/month")
    print(f"   Range: {monthly_counts.min()}-{monthly_counts.max()}")
    membership.to_parquet(DS_DIR / "ds_membership_monthly.parquet", index=False)

    # ----------------------------------------------------------------
    # 3. Fetch daily total return indices for all ever-members
    # ----------------------------------------------------------------
    print("3. Fetching daily total return indices...")
    ever_infocodes = sp500["infocode"].dropna().unique().tolist()
    print(f"   {len(ever_infocodes)} unique infocodes to fetch")

    # Fetch in chunks to avoid query size limits
    chunk_size = 200
    all_ri = []
    for i in range(0, len(ever_infocodes), chunk_size):
        chunk = ever_infocodes[i : i + chunk_size]
        codes_str = ",".join(str(int(c)) for c in chunk)
        df = conn.raw_sql(
            f"SELECT infocode, marketdate, ri "
            f"FROM tr_ds_equities.ds2primqtri "
            f"WHERE infocode IN ({codes_str}) "
            f"  AND marketdate BETWEEN '1989-01-01' AND '2015-12-31'"
        )
        all_ri.append(df)
        print(f"   Chunk {i // chunk_size + 1}/{(len(ever_infocodes) + chunk_size - 1) // chunk_size}: {len(df):,} rows")

    ri = pd.concat(all_ri, ignore_index=True)
    ri["marketdate"] = pd.to_datetime(ri["marketdate"])
    ri["infocode"] = ri["infocode"].astype("Int64")
    ri["ri"] = pd.to_numeric(ri["ri"], errors="coerce")
    ri = ri.dropna(subset=["ri"])
    ri = ri.sort_values(["infocode", "marketdate"]).reset_index(drop=True)
    print(f"   Total: {len(ri):,} rows, {ri['infocode'].nunique()} stocks")

    # Compute daily returns from total return index: ret = RI_t / RI_{t-1} - 1
    ri["ret"] = ri.groupby("infocode")["ri"].pct_change()
    # First day per stock has NaN return — drop it
    ri = ri.dropna(subset=["ret"])

    # Rename for consistency with existing pipeline
    ri = ri.rename(columns={"marketdate": "date"})
    print(f"   After return computation: {len(ri):,} rows")
    ri.to_parquet(DS_DIR / "ds_daily_returns.parquet", index=False)

    # ----------------------------------------------------------------
    # 4. Build daily eligibility
    # ----------------------------------------------------------------
    print("4. Building daily eligibility...")
    trading_dates = ri["date"].unique()
    dates_df = pd.DataFrame({"date": pd.to_datetime(trading_dates)})
    dates_df["effective_month"] = dates_df["date"].dt.to_period("M")

    eligible = dates_df.merge(
        membership[["effective_month", "infocode"]],
        on="effective_month",
        how="inner",
    )
    eligible = (
        eligible[["date", "infocode"]]
        .sort_values(["date", "infocode"])
        .reset_index(drop=True)
    )
    elig_per_day = eligible.groupby("date")["infocode"].nunique()
    print(f"   {len(eligible):,} stock-day pairs")
    print(f"   Avg {elig_per_day.mean():.1f} stocks/day, range {elig_per_day.min()}-{elig_per_day.max()}")
    eligible.to_parquet(DS_DIR / "ds_universe_daily.parquet", index=False)

    # ----------------------------------------------------------------
    # 5. Fetch security names for reference
    # ----------------------------------------------------------------
    print("5. Fetching security names...")
    codes_str = ",".join(str(int(c)) for c in ever_infocodes)
    names = conn.raw_sql(
        f"SELECT infocode, dsqtname, ticker, region, dscode, isin "
        f"FROM tr_ds_equities.wrds_ds_names "
        f"WHERE infocode IN ({codes_str})"
    )
    names.to_parquet(DS_DIR / "ds_names.parquet", index=False)
    print(f"   {len(names)} name records")

    conn.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Datastream data build complete")
    print("=" * 60)
    print(f"Membership: {len(sp500)} spells, {sp500['infocode'].nunique()} stocks")
    print(f"Returns: {len(ri):,} daily obs, {ri['infocode'].nunique()} stocks")
    print(f"Eligibility: {len(eligible):,} stock-days")
    print(f"Date range: {ri['date'].min().date()} to {ri['date'].max().date()}")
    print(f"\nFiles saved to {DS_DIR.relative_to(ROOT)}/")
    for f in sorted(DS_DIR.glob("*.parquet")):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
