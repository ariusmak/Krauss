"""
Test frozen-universe vs monthly-updated universe on periods 0, 15, 20.

Uses H2O models (same framework as paper) with both universe approaches
to isolate the effect of universe construction on results.
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods
from krauss.data.features import FEATURE_COLS
from krauss.data.universe_frozen import build_frozen_universe, build_frozen_daily_eligibility
from krauss.backtest.ranking import rank_and_select
from krauss.backtest.portfolio import build_daily_portfolios, aggregate_portfolio_returns
from krauss.backtest.costs import compute_turnover, apply_transaction_costs

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def build_panel(features, labels, eligible_permnos, dates_set):
    """Build modeling panel for a set of dates and eligible stocks."""
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][
        ["date", "permno", "y_binary"]
    ].copy()

    panel = feat.merge(lab, on=["date", "permno"], how="inner")
    # Filter to eligible stocks (frozen set)
    panel = panel[panel["permno"].isin(eligible_permnos)]
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def run_test(test_periods):
    import h2o
    from krauss.models.h2o_dnn_phase1 import build_h2o_dnn, train_h2o_dnn, predict_h2o_dnn
    from krauss.models.h2o_gbt_phase1 import build_h2o_gbt, train_h2o_gbt, predict_h2o_gbt
    from krauss.models.h2o_rf_phase1 import build_h2o_rf, train_h2o_rf, predict_h2o_rf
    from krauss.models.ensembles_phase1 import ens1_predictions

    h2o.init(max_mem_size="8G", nthreads=1)

    print("Loading data...")
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    membership = pd.read_parquet(PROCESSED / "membership_monthly.parquet")

    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)

    paper_ens1_pre = {
        0: 0.0179,   # period 0 ~ early era
        15: 0.0291,  # period 15 ~ crisis
        20: 0.0014,  # period 20 ~ late era
    }

    all_frozen_preds = []

    for pid in test_periods:
        sp = periods[pid]
        print(f"\n{'='*60}")
        print(f"Period {pid}: trade {sp.trade_start.date()} to {sp.trade_end.date()}")
        print(f"{'='*60}")

        # Build frozen universe
        frozen = build_frozen_universe(sp, membership, returns)
        frozen_elig = build_frozen_daily_eligibility(
            frozen, returns, sp.trade_dates
        )

        print(f"Frozen universe: {len(frozen)} stocks")
        print(f"Avg stocks per trade day: {frozen_elig.groupby('date')['permno'].nunique().mean():.0f}")

        # Build panels using frozen universe
        train_dates = set(pd.to_datetime(sp.usable_train_dates))
        trade_dates = set(pd.to_datetime(sp.trade_dates))

        train_panel = build_panel(features, labels, frozen, train_dates)
        trade_panel = build_panel(features, labels, frozen, trade_dates)

        y_train = train_panel["y_binary"]
        print(f"Train: {len(train_panel):,} obs, {train_panel['permno'].nunique()} stocks")
        print(f"Trade: {len(trade_panel):,} obs, {trade_panel['permno'].nunique()} stocks")

        # Train H2O models
        t0 = time.time()

        print("  Training RF...", end=" ", flush=True)
        rf = build_h2o_rf()
        rf = train_h2o_rf(rf, train_panel, y_train)
        p_rf_trade = predict_h2o_rf(rf, trade_panel)
        p_rf_train = predict_h2o_rf(rf, train_panel)
        print(f"{time.time()-t0:.0f}s")

        t1 = time.time()
        print("  Training GBT...", end=" ", flush=True)
        gbt = build_h2o_gbt()
        gbt = train_h2o_gbt(gbt, train_panel, y_train)
        p_gbt_trade = predict_h2o_gbt(gbt, trade_panel)
        p_gbt_train = predict_h2o_gbt(gbt, train_panel)
        print(f"{time.time()-t1:.0f}s")

        t1 = time.time()
        print("  Training DNN...", end=" ", flush=True)
        dnn = build_h2o_dnn()
        dnn = train_h2o_dnn(dnn, train_panel, y_train)
        p_dnn_trade = predict_h2o_dnn(dnn, trade_panel)
        p_dnn_train = predict_h2o_dnn(dnn, train_panel)
        print(f"{time.time()-t1:.0f}s")

        # Build prediction df
        result = trade_panel[["date", "permno"]].copy()
        result["period_id"] = pid
        result["p_rf"] = p_rf_trade
        result["p_xgb"] = p_gbt_trade
        result["p_dnn"] = p_dnn_trade
        result["p_ens1"] = ens1_predictions(p_dnn_trade, p_gbt_trade, p_rf_trade)

        all_frozen_preds.append(result)

        # Backtest
        k = 10
        print(f"\n  k={k} backtest results:")
        print(f"  {'Model':6s} {'Pre-cost':>10s} {'Post-cost':>10s} {'Turnover':>10s}")
        print(f"  {'-'*40}")

        for m_name, col in [("DNN", "p_dnn"), ("GBT", "p_xgb"), ("RAF", "p_rf"), ("ENS1", "p_ens1")]:
            sel = rank_and_select(result, k=k, score_col=col)
            hold = build_daily_portfolios(sel, returns, k=k)
            daily = aggregate_portfolio_returns(hold)
            turn = compute_turnover(hold, k=k)
            daily = apply_transaction_costs(daily, turn, 5.0)

            pre = daily["port_ret"].mean()
            post = daily["port_ret_net"].mean()
            avg_turn = turn["turnover"].mean()
            print(f"  {m_name:6s} {pre:10.4f} {post:10.4f} {avg_turn:10.1f}")

        # Compare to monthly-updated H2O results
        try:
            h2o_monthly = pd.read_parquet(PROCESSED / "predictions_phase1_h2o.parquet")
            h2o_period = h2o_monthly[h2o_monthly["period_id"] == pid]
            if len(h2o_period) > 0 and "p_ens1" in h2o_period.columns:
                sel_m = rank_and_select(h2o_period, k=k, score_col="p_ens1")
                hold_m = build_daily_portfolios(sel_m, returns, k=k)
                daily_m = aggregate_portfolio_returns(hold_m)
                pre_m = daily_m["port_ret"].mean()
                print(f"\n  Monthly-updated ENS1 pre-cost: {pre_m:.4f}")
                print(f"  Frozen universe ENS1 pre-cost: {daily['port_ret'].mean():.4f}")
        except Exception:
            pass

    # Save
    all_preds = pd.concat(all_frozen_preds, ignore_index=True)
    all_preds.to_parquet(PROCESSED / "predictions_frozen_test.parquet", index=False)
    print(f"\nSaved frozen predictions -> predictions_frozen_test.parquet")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    run_test([15, 16])
