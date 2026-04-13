"""
Precompute and save all backtest results for Phase 1 reproduction.

Usage:
    python scripts/build_backtest_results.py

Reads:
    data/processed/predictions_phase1.parquet
    data/processed/daily_returns.parquet

Saves:
    data/processed/backtest/daily_returns_{model}_k{k}.parquet
        — daily portfolio return series (pre/post cost, long/short legs)
    data/processed/backtest/holdings_{model}_k{k}.parquet
        — daily stock-level holdings (date, permno, side, weight, return)
    data/processed/backtest/turnover_{model}_k{k}.parquet
        — daily turnover
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.backtest.ranking import rank_and_select
from krauss.backtest.portfolio import build_daily_portfolios, aggregate_portfolio_returns
from krauss.backtest.costs import compute_turnover, apply_transaction_costs

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
BT_DIR = PROCESSED / "backtest"

MODEL_COLS = ["p_dnn", "p_xgb", "p_rf", "p_ens1", "p_ens2", "p_ens3"]
MODEL_NAMES = {
    "p_dnn": "DNN", "p_xgb": "GBT", "p_rf": "RAF",
    "p_ens1": "ENS1", "p_ens2": "ENS2", "p_ens3": "ENS3",
}
K_VALUES = [10, 50, 100, 150, 200]
COST_BPS = 5


def main():
    BT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    pred = pd.read_parquet(PROCESSED / "predictions_phase1.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    print(f"  Predictions: {len(pred):,} rows")
    print(f"  Returns: {len(returns):,} rows\n")

    total = len(MODEL_COLS) * len(K_VALUES)
    with tqdm(total=total, desc="Backtests") as pbar:
        for model_col in MODEL_COLS:
            for k in K_VALUES:
                tag = f"{MODEL_NAMES[model_col]}_k{k}"

                # Check cache
                daily_path = BT_DIR / f"daily_{tag}.parquet"
                hold_path = BT_DIR / f"holdings_{tag}.parquet"
                turn_path = BT_DIR / f"turnover_{tag}.parquet"

                if daily_path.exists() and hold_path.exists() and turn_path.exists():
                    pbar.update(1)
                    continue

                # Run backtest
                sel = rank_and_select(pred, k=k, score_col=model_col)
                holdings = build_daily_portfolios(sel, returns, k=k)
                daily = aggregate_portfolio_returns(holdings)
                turnover = compute_turnover(holdings, k=k)
                daily = apply_transaction_costs(daily, turnover, COST_BPS)

                # Save daily returns
                daily.to_parquet(daily_path, index=False)

                # Save holdings
                holdings.to_parquet(hold_path, index=False)

                # Save turnover
                turnover.to_parquet(turn_path, index=False)

                pbar.update(1)

    # Summary
    n_files = len(list(BT_DIR.glob("*.parquet")))
    total_mb = sum(f.stat().st_size for f in BT_DIR.glob("*.parquet")) / 1e6
    print(f"\nDone. {n_files} files saved to data/processed/backtest/ ({total_mb:.1f} MB)")


if __name__ == "__main__":
    main()
