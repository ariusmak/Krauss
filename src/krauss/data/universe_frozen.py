"""
Frozen-per-period universe construction (paper parity variant).

The paper defines:
    "Let n_i denote the number of stocks in the S&P 500 at the end of the
    training period of study period i, having full price information available."

Key difference vs monthly-updated universe:
    - Monthly-updated: stocks leave the universe when removed from S&P 500.
      Stocks entering S&P 500 mid-trading-period are added.
    - Frozen: the stock set is FIXED at end of training. Stocks remain
      eligible for the entire trading period even if they leave the S&P 500.
      Stocks joining mid-period are NOT added.

This matters most during the crisis: stocks removed from the S&P 500
(e.g., due to distress) remain tradeable in the frozen universe. A model
that correctly predicts they'll underperform can short them through their
worst declines. In the monthly-updated universe, they exit and the
short opportunity is lost.

Stocks are only excluded from daily ranking when they have no return data
(actual delisting). The frozen set determines ELIGIBILITY, but actual
trading requires a return observation on that day.
"""

import pandas as pd
import numpy as np


def build_frozen_universe(
    sp,  # StudyPeriod namedtuple
    membership: pd.DataFrame,
    returns: pd.DataFrame,
    min_train_coverage: float = 0.90,
) -> set:
    """
    Build the frozen stock universe for one study period.

    Takes S&P 500 members at the end of the training period that have
    sufficient data coverage during the training window.

    Parameters
    ----------
    sp : StudyPeriod
        From study_periods.build_study_periods().
    membership : pd.DataFrame
        Monthly membership panel (from universe.build_membership_matrix()).
    returns : pd.DataFrame
        Full return panel (permno, date, ret).
    min_train_coverage : float
        Minimum fraction of training days a stock must have return data for.
        Stocks with less coverage are excluded (they don't have enough
        history for reliable feature computation).

    Returns
    -------
    set of int
        PERMNOs in the frozen universe for this study period.
    """
    # S&P 500 members at end of training period
    train_end = pd.to_datetime(sp.train_end)
    train_end_month = train_end.to_period("M")
    sp500_at_end = set(
        membership[membership["effective_month"] == train_end_month]["permno"]
    )

    # Check data coverage during the TRAINING window
    train_dates_set = set(pd.to_datetime(sp.train_dates))
    n_train = len(train_dates_set)

    train_ret = returns[
        returns["date"].isin(train_dates_set)
        & returns["permno"].isin(sp500_at_end)
    ]
    train_counts = train_ret.groupby("permno")["date"].nunique()

    threshold = int(n_train * min_train_coverage)
    has_sufficient_data = set(train_counts[train_counts >= threshold].index)

    frozen = sp500_at_end & has_sufficient_data
    return frozen


def build_frozen_daily_eligibility(
    frozen_permnos: set,
    returns: pd.DataFrame,
    trade_dates: list,
) -> pd.DataFrame:
    """
    Build daily eligibility for the frozen universe during the trade window.

    A stock is eligible on a trade day if:
        1. It's in the frozen set (determined at end of training), AND
        2. It has a return observation on that day (still trading, not delisted).

    Parameters
    ----------
    frozen_permnos : set
        PERMNOs in the frozen universe.
    returns : pd.DataFrame
        Full return panel (permno, date, ret).
    trade_dates : list
        Trading dates for this period.

    Returns
    -------
    pd.DataFrame
        date, permno — one row per eligible stock-day.
    """
    trade_dates_set = set(pd.to_datetime(trade_dates))

    # Get actual return observations for frozen stocks during trade window
    trade_ret = returns[
        returns["date"].isin(trade_dates_set)
        & returns["permno"].isin(frozen_permnos)
    ]

    eligible = trade_ret[["date", "permno"]].drop_duplicates()
    eligible = eligible.sort_values(["date", "permno"]).reset_index(drop=True)
    return eligible
