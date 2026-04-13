"""
WRDS connection and raw data extraction.

Prompts for WRDS username interactively. The wrds library caches
credentials in ~/.pgpass after first use.
"""

import wrds
import pandas as pd


def get_connection() -> wrds.Connection:
    """Open a WRDS connection (prompts for credentials if not cached)."""
    return wrds.Connection()


def fetch_sp500_membership(conn: wrds.Connection) -> pd.DataFrame:
    """
    Fetch full S&P 500 historical membership from crsp.dsp500list.

    Returns
    -------
    pd.DataFrame
        permno : int
        start  : datetime — date stock entered S&P 500
        ending : datetime — date stock left (NaT if still active)
    """
    query = """
        SELECT permno, start, ending
        FROM crsp.dsp500list
        ORDER BY permno, start
    """
    df = conn.raw_sql(query)
    df["start"] = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_daily_stock_data(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch CRSP daily stock file (dsf) for all securities in date range.

    Parameters
    ----------
    start_date, end_date : str
        Format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        permno, date, ret, prc, shrout, cfacpr, cfacshr
    """
    query = f"""
        SELECT permno, date, ret, prc, shrout, cfacpr, cfacshr
        FROM crsp.dsf
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, date
    """
    df = conn.raw_sql(query)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_delisting_returns(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch CRSP delisting returns for survivorship-bias correction.

    Parameters
    ----------
    start_date, end_date : str
        Format 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        permno, dlstdt, dlret, dlstcd
    """
    query = f"""
        SELECT permno, dlstdt, dlret, dlstcd
        FROM crsp.dsedelist
        WHERE dlstdt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY permno, dlstdt
    """
    df = conn.raw_sql(query)
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_gics_industry(conn: wrds.Connection) -> pd.DataFrame:
    """
    Fetch GICS sector codes via CRSP-Compustat link.

    Returns
    -------
    pd.DataFrame
        permno, gvkey, gsector, ggroup, gind, conm
    """
    query = """
        SELECT DISTINCT
            a.lpermno AS permno,
            a.gvkey,
            b.gsector,
            b.ggroup,
            b.gind,
            b.conm
        FROM crsp.ccmxpf_lnkhist AS a
        INNER JOIN comp.company AS b
            ON a.gvkey = b.gvkey
        WHERE a.linktype IN ('LU', 'LC')
          AND a.linkprim IN ('P', 'C')
          AND a.lpermno IS NOT NULL
        ORDER BY a.lpermno
    """
    df = conn.raw_sql(query)
    df["permno"] = df["permno"].astype(int)
    return df


def fetch_ff_factors(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch daily Fama-French factors from WRDS.

    Returns FF3 (mktrf, smb, hml, rf) + momentum (umd) + short-term
    reversal (st_rev) + FF5 (rmw, cma).

    Returns
    -------
    pd.DataFrame
        date, mktrf, smb, hml, rf, umd, st_rev, smb5, hml5, rmw, cma
    """
    # FF3 + RF
    ff3_query = f"""
        SELECT date, mktrf, smb, hml, rf
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    ff3 = conn.raw_sql(ff3_query)
    ff3["date"] = pd.to_datetime(ff3["date"])

    # Momentum (UMD)
    mom_query = f"""
        SELECT date, umd
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    try:
        mom = conn.raw_sql(mom_query)
        mom["date"] = pd.to_datetime(mom["date"])
        ff3 = ff3.merge(mom, on="date", how="left")
    except Exception:
        ff3["umd"] = pd.NA

    # FF5 factors
    ff5_query = f"""
        SELECT date,
               smb AS smb5, hml AS hml5, rmw AS rmw5, cma AS cma5
        FROM ff.fivefactors_daily
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    try:
        ff5 = conn.raw_sql(ff5_query)
        ff5["date"] = pd.to_datetime(ff5["date"])
        ff3 = ff3.merge(ff5, on="date", how="left")
    except Exception:
        for c in ["smb5", "hml5", "rmw5", "cma5"]:
            ff3[c] = pd.NA

    # Short-term reversal
    rev_query = f"""
        SELECT date, st_rev
        FROM ff.factors_daily
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
    """
    try:
        rev = conn.raw_sql(rev_query)
        rev["date"] = pd.to_datetime(rev["date"])
        ff3 = ff3.merge(rev, on="date", how="left")
    except Exception:
        ff3["st_rev"] = pd.NA

    return ff3


def fetch_vix(
    conn: wrds.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch daily VIX index from WRDS (CBOE volatility index).

    Returns
    -------
    pd.DataFrame
        date, vix
    """
    query = f"""
        SELECT caldt AS date, vix
        FROM cboe.cboe
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY caldt
    """
    try:
        df = conn.raw_sql(query)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        # Alternative table name
        query2 = f"""
            SELECT date, close AS vix
            FROM cboe_new.cboe_vix
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date
        """
        df = conn.raw_sql(query2)
        df["date"] = pd.to_datetime(df["date"])
        return df
