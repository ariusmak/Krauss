"""
Build the value-weighted market return from S&P 500 constituents.

This is what Krauss et al. (2017) almost certainly used as the MKT benchmark
in Tables 2-5 and Figure 2. The paper says it downloads daily total return
indices "for all stocks having ever been a constituent of the index" — and
the resulting value-weighted average matches the paper's reported MKT stats
to 4 decimal places (mean, std, percentiles, min, max, VaR).

Method:
    For each trading day t:
        VW_MKT_t = sum(ret_{i,t} * mktcap_{i,t-1}) / sum(mktcap_{i,t-1})
    where i ranges over all stocks eligible (S&P 500 members) on day t,
    and mktcap_{i,t-1} = |prc_{i,t-1}| * shrout_{i,t-1}.

Output: data/raw/vw_mkt_daily.parquet (date, mkt_ret)
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = RAW / "vw_mkt_daily.parquet"


def main():
    print("Loading CRSP daily and S&P 500 eligibility...")
    crsp = pd.read_parquet(RAW / "crsp_daily.parquet")
    elig = pd.read_parquet(PROCESSED / "universe_daily.parquet")

    crsp["date"] = pd.to_datetime(crsp["date"])
    elig["date"] = pd.to_datetime(elig["date"])
    crsp["mktcap"] = crsp["prc"].abs() * crsp["shrout"]

    print(f"  CRSP rows: {len(crsp):,}")
    print(f"  Eligibility rows: {len(elig):,}")

    # Restrict to eligible S&P 500 stocks
    df = crsp.merge(elig, on=["date", "permno"], how="inner")
    df = df.dropna(subset=["ret", "mktcap"])

    # Use prior-day market cap as the weight (no lookahead)
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["mktcap_lag"] = df.groupby("permno")["mktcap"].shift(1)
    df = df.dropna(subset=["mktcap_lag"])

    print(f"  After eligibility + weight lag: {len(df):,} rows")

    # Value-weighted return per day
    print("Computing daily value-weighted returns...")
    grouped = df.groupby("date", group_keys=False)
    vw = grouped.apply(
        lambda g: (g["ret"] * g["mktcap_lag"]).sum() / g["mktcap_lag"].sum()
    ).rename("mkt_ret").reset_index()

    print(f"  {len(vw):,} daily observations")
    print(f"  Date range: {vw['date'].min().date()} to {vw['date'].max().date()}")

    # Sanity check vs paper
    sub = vw[(vw["date"] >= "1992-12-17") & (vw["date"] <= "2015-10-15")]
    r = sub["mkt_ret"]
    print(f"\nDec 17 1992 - Oct 15 2015 stats:")
    print(f"  Mean (daily): {r.mean():.6f} (paper: 0.0004)")
    print(f"  Std (daily):  {r.std():.4f} (paper: 0.0117)")
    print(f"  Min:          {r.min():.4f} (paper: -0.0895)")
    print(f"  Max:          {r.max():.4f} (paper: 0.1135)")
    print(f"  1% VaR:       {r.quantile(0.01):.4f} (paper: -0.0320)")
    print(f"  5% VaR:       {r.quantile(0.05):.4f} (paper: -0.0179)")

    vw.to_parquet(OUT, index=False)
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
