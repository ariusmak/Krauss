"""Explore Datastream S&P 500: membership + total return index."""
import wrds
import pandas as pd

conn = wrds.Connection()

# ds2constmth is a spell table: (indexlistintcode, constintcode, startdate, enddate, infocode)
# indexlistintcode=4408 is S&P 500
# infocode joins to ds2primqtri for total return index

# 1. Full S&P 500 membership summary
print("=== S&P 500 membership spells (ds2constmth, code 4408) ===")
sp500 = conn.raw_sql(
    "SELECT constintcode, infocode, startdate, enddate "
    "FROM tr_ds_equities.ds2constmth "
    "WHERE indexlistintcode = 4408"
)
sp500['startdate'] = pd.to_datetime(sp500['startdate'])
sp500['enddate'] = pd.to_datetime(sp500['enddate'])
print(f"Total spells: {len(sp500)}")
print(f"Unique constintcode: {sp500['constintcode'].nunique()}")
print(f"Unique infocode: {sp500['infocode'].nunique()}")
print(f"Date range: {sp500['startdate'].min()} to {sp500['enddate'].max()}")
print()

# 2. How many stocks per month-end (reconstruct monthly snapshots)
# Check Dec 1989 through Sep 2015 (paper's range)
print("=== Monthly constituent counts (paper range) ===")
month_ends = pd.date_range('1989-12-31', '2015-09-30', freq='ME')
counts = []
for me in month_ends:
    n = ((sp500['startdate'] <= me) & (sp500['enddate'] >= me)).sum()
    counts.append({'date': me, 'n_stocks': n})
counts_df = pd.DataFrame(counts)
print(f"Mean stocks/month: {counts_df['n_stocks'].mean():.1f}")
print(f"Min: {counts_df['n_stocks'].min()} ({counts_df.loc[counts_df['n_stocks'].idxmin(), 'date'].date()})")
print(f"Max: {counts_df['n_stocks'].max()} ({counts_df.loc[counts_df['n_stocks'].idxmax(), 'date'].date()})")
print(f"Paper says: 499.7 avg")
print()

# Show crisis period months
crisis_months = counts_df[counts_df['date'].between('2008-06-30', '2009-06-30')]
print("Crisis months:")
for _, row in crisis_months.iterrows():
    print(f"  {row['date'].date()}: {row['n_stocks']} stocks")
print()

# 3. Get all ever-member infocodes for pulling return data
ever_infocodes = sp500['infocode'].dropna().unique()
print(f"Ever-member infocodes: {len(ever_infocodes)}")
print()

# 4. Sample total return index for a known stock during crisis
# Get Apple (likely in S&P 500)
print("=== Sample: look up a few stocks ===")
sample = sp500.head(10)
codes_str = ','.join(str(int(c)) for c in sample['infocode'].dropna().unique()[:5])
names = conn.raw_sql(
    f"SELECT infocode, dsqtname, ticker, region "
    f"FROM tr_ds_equities.wrds_ds_names "
    f"WHERE infocode IN ({codes_str})"
)
print(names.to_string())
print()

# 5. Total return index for first stock during crisis
code = int(sample['infocode'].dropna().iloc[0])
print(f"=== Total return index for infocode={code} (Sep-Oct 2008) ===")
ri = conn.raw_sql(
    f"SELECT infocode, marketdate, ri "
    f"FROM tr_ds_equities.ds2primqtri "
    f"WHERE infocode = {code} "
    f"  AND marketdate BETWEEN '2008-09-01' AND '2008-10-31' "
    f"ORDER BY marketdate"
)
print(ri.to_string())
print()

# 6. Compare to CRSP: how many CRSP S&P 500 members vs Datastream?
print("=== CRSP vs Datastream comparison ===")
crsp = pd.read_parquet('data/raw/sp500_membership.parquet')
print(f"CRSP spells: {len(crsp)}, unique PERMNOs: {crsp['permno'].nunique()}")
print(f"Datastream spells: {len(sp500)}, unique infocodes: {sp500['infocode'].nunique()}")

conn.close()
