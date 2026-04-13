"""
Test H2O models on Datastream data for key periods.

Compares Datastream vs CRSP results to isolate data-source effects.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods
from krauss.models.ensembles_phase1 import (
    ens1_predictions, ens2_predictions, ens3_predictions,
)
from krauss.backtest.ranking import rank_and_select
from krauss.backtest.portfolio import build_daily_portfolios, aggregate_portfolio_returns
from krauss.backtest.costs import compute_turnover, apply_transaction_costs

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "data" / "datastream"

FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_panel(features, labels, eligible_codes, dates_set):
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][
        ["date", "infocode", "y_binary"]
    ].copy()
    panel = feat.merge(lab, on=["date", "infocode"], how="inner")
    panel = panel[panel["infocode"].isin(eligible_codes)]
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def main():
    import h2o
    from krauss.models.h2o_dnn_phase1 import build_h2o_dnn, train_h2o_dnn, predict_h2o_dnn
    from krauss.models.h2o_gbt_phase1 import build_h2o_gbt, train_h2o_gbt, predict_h2o_gbt
    from krauss.models.h2o_rf_phase1 import build_h2o_rf, train_h2o_rf, predict_h2o_rf

    h2o.init(max_mem_size="12G", nthreads=-1)

    print("Loading Datastream data (US calendar filtered)...")
    features = pd.read_parquet(DS_DIR / "ds_features_usonly.parquet")
    labels = pd.read_parquet(DS_DIR / "ds_labels_usonly.parquet")
    returns = pd.read_parquet(DS_DIR / "ds_daily_returns_usonly.parquet")
    membership = pd.read_parquet(DS_DIR / "ds_membership_monthly.parquet")
    eligible = pd.read_parquet(DS_DIR / "ds_universe_daily_usonly.parquet")

    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])
    returns["date"] = pd.to_datetime(returns["date"])
    eligible["date"] = pd.to_datetime(eligible["date"])

    # Build study periods from Datastream trading dates
    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)
    print(f"{len(periods)} study periods")

    # H2O models use FEATURE_COLS which reference "permno" internally
    # We need to adapt: the models just need the feature columns, not the ID
    # The predict functions take X_test[FEATURE_COLS] so this works directly

    test_periods = list(range(23))

    paper_pre = {
        "DNN": 0.0033, "GBT": 0.0037, "RAF": 0.0043, "ENS1": 0.0045,
    }

    all_results = []
    DS_MODELS = ROOT / "data" / "models_ds_h2o"

    for pid in test_periods:
        sp = periods[pid]

        # Check cache
        cache_dir = DS_MODELS / f"period_{pid:02d}"
        cache_path = cache_dir / "predictions.parquet"
        if cache_path.exists():
            print(f"  Period {pid}: loaded from cache")
            all_results.append(pd.read_parquet(cache_path))
            continue

        print(f"\n{'='*60}")
        print(f"Period {pid}: trade {sp.trade_start.date()} to {sp.trade_end.date()}")
        print(f"{'='*60}")

        # Monthly-updated eligibility (matching paper's month-end constituency lists)
        train_dates = set(pd.to_datetime(sp.usable_train_dates))
        trade_dates = set(pd.to_datetime(sp.trade_dates))

        # Get eligible infocodes per date from monthly membership
        train_elig_codes = set(eligible[eligible["date"].isin(train_dates)]["infocode"])
        trade_elig_codes = set(eligible[eligible["date"].isin(trade_dates)]["infocode"])

        train_panel = build_panel(features, labels, train_elig_codes, train_dates)
        trade_panel = build_panel(features, labels, trade_elig_codes, trade_dates)

        y_train = train_panel["y_binary"]
        print(f"Train: {len(train_panel):,} obs, {train_panel['infocode'].nunique()} stocks")
        print(f"Trade: {len(trade_panel):,} obs, {trade_panel['infocode'].nunique()} stocks")

        # Train models — rename infocode to permno temporarily for compatibility
        # (the model functions only look at FEATURE_COLS, not the ID column)

        t0 = time.time()
        print("  Training RF...", end=" ", flush=True)
        rf = build_h2o_rf()
        rf = train_h2o_rf(rf, train_panel, y_train)
        p_rf = predict_h2o_rf(rf, trade_panel)
        p_rf_tr = predict_h2o_rf(rf, train_panel)
        print(f"{time.time()-t0:.0f}s")
        h2o.remove_all()

        t1 = time.time()
        print("  Training GBT...", end=" ", flush=True)
        gbt = build_h2o_gbt()
        gbt = train_h2o_gbt(gbt, train_panel, y_train)
        p_gbt = predict_h2o_gbt(gbt, trade_panel)
        p_gbt_tr = predict_h2o_gbt(gbt, train_panel)
        print(f"{time.time()-t1:.0f}s")
        h2o.remove_all()

        t1 = time.time()
        print("  Training DNN...", end=" ", flush=True)
        dnn = build_h2o_dnn()
        dnn = train_h2o_dnn(dnn, train_panel, y_train)
        p_dnn = predict_h2o_dnn(dnn, trade_panel)
        p_dnn_tr = predict_h2o_dnn(dnn, train_panel)
        print(f"{time.time()-t1:.0f}s")
        h2o.remove_all()

        p_ens1 = ens1_predictions(p_dnn, p_gbt, p_rf)
        y_train_arr = y_train.values
        p_ens2 = ens2_predictions(
            p_dnn, p_gbt, p_rf,
            y_train_arr, p_dnn_tr, p_gbt_tr, p_rf_tr,
        )
        p_ens3 = ens3_predictions(
            p_dnn, p_gbt, p_rf,
            y_train_arr, p_dnn_tr, p_gbt_tr, p_rf_tr,
        )

        # Build result with infocode as permno (for backtest compatibility)
        result = trade_panel[["date", "infocode"]].copy()
        result = result.rename(columns={"infocode": "permno"})
        result["p_dnn"] = p_dnn
        result["p_xgb"] = p_gbt
        result["p_rf"] = p_rf
        result["p_ens1"] = p_ens1
        result["p_ens2"] = p_ens2
        result["p_ens3"] = p_ens3

        # Backtest — need returns with "permno" column
        ret_bt = returns[["date", "infocode", "ret"]].rename(
            columns={"infocode": "permno"}
        )

        k = 10
        print(f"\n  k={k} results:")
        print(f"  {'Model':6s} {'Pre-cost':>10s} {'Post-cost':>10s} {'Turnover':>10s} {'Paper':>10s} {'Ratio':>8s}")
        print(f"  {'-'*56}")

        for m_name, col in [("DNN", "p_dnn"), ("GBT", "p_xgb"),
                            ("RAF", "p_rf"), ("ENS1", "p_ens1")]:
            sel = rank_and_select(result, k=k, score_col=col)
            hold = build_daily_portfolios(sel, ret_bt, k=k)
            daily = aggregate_portfolio_returns(hold)
            turn = compute_turnover(hold, k=k)
            daily = apply_transaction_costs(daily, turn, 5.0)

            pre = daily["port_ret"].mean()
            post = daily["port_ret_net"].mean()
            avg_turn = turn["turnover"].mean()
            pp = paper_pre[m_name]
            print(f"  {m_name:6s} {pre:10.4f} {post:10.4f} {avg_turn:10.1f} {pp:10.4f} {pre/pp:8.1%}")

        # Prediction spreads
        print(f"\n  Prediction spreads:")
        for col in ["p_dnn", "p_xgb", "p_rf"]:
            std = result[col].std()
            cs = result.groupby("date")[col].std().mean()
            print(f"    {col}: std={std:.4f}  daily_cs={cs:.4f}")

        # Cache results
        result["period_id"] = pid
        cache_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
        all_results.append(result)

        # Save consolidated after each period
        consolidated = pd.concat(all_results, ignore_index=True)
        consolidated.to_parquet(DS_DIR / "predictions_ds_h2o.parquet", index=False)

    # Final consolidated save
    if all_results:
        predictions = pd.concat(all_results, ignore_index=True)
        out_path = DS_DIR / "predictions_ds_h2o.parquet"
        predictions.to_parquet(out_path, index=False)
        print(f"\n{'='*60}")
        print(f"DONE — {len(predictions):,} predictions across {predictions['period_id'].nunique()} periods")
        print(f"Saved -> {out_path}")
        for col in ["p_rf", "p_xgb", "p_dnn", "p_ens1"]:
            if col in predictions.columns:
                print(f"  {col} mean={predictions[col].mean():.4f} std={predictions[col].std():.4f}")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
