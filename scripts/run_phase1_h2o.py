"""
Run Phase 1 reproduction using H2O models (exact paper framework).

Parallel to run_phase1.py but uses H2O DNN, GBM, DRF instead of
PyTorch/XGBoost/sklearn. Results saved separately for comparison.

Usage:
    python scripts/run_phase1_h2o.py
    python scripts/run_phase1_h2o.py --model dnn       # DNN only
    python scripts/run_phase1_h2o.py --periods 0 15 20  # specific periods

Outputs:
    data/models_h2o/period_{id}/predictions.parquet
    data/processed/predictions_phase1_h2o.parquet
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods
from krauss.data.features import FEATURE_COLS
from krauss.models.ensembles_phase1 import (
    ens1_predictions, ens2_predictions, ens3_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models_h2o"


def load_data():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    eligible = pd.read_parquet(PROCESSED / "universe_daily.parquet")
    return features, labels, returns, eligible


def build_panel(features, labels, eligible, dates_set):
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][
        ["date", "permno", "y_binary"]
    ].copy()
    panel = feat.merge(lab, on=["date", "permno"], how="inner")
    panel = panel.merge(eligible, on=["date", "permno"], how="inner")
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def run_period(
    period_id, sp, features, labels, eligible,
    run_rf=True, run_gbt=True, run_dnn=True,
):
    import h2o
    from krauss.models.h2o_dnn_phase1 import build_h2o_dnn, train_h2o_dnn, predict_h2o_dnn
    from krauss.models.h2o_gbt_phase1 import build_h2o_gbt, train_h2o_gbt, predict_h2o_gbt
    from krauss.models.h2o_rf_phase1 import build_h2o_rf, train_h2o_rf, predict_h2o_rf
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    model_dir = MODELS / f"period_{period_id:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check cache
    pred_path = model_dir / "predictions.parquet"
    if pred_path.exists():
        print(f"  Period {period_id}: loaded from cache")
        return pd.read_parquet(pred_path)

    train_dates = set(pd.to_datetime(sp.usable_train_dates))
    trade_dates = set(pd.to_datetime(sp.trade_dates))

    train_panel = build_panel(features, labels, eligible, train_dates)
    trade_panel = build_panel(features, labels, eligible, trade_dates)

    y_train = train_panel["y_binary"]

    print(f"  Train: {len(train_panel):,} obs, "
          f"{train_panel['permno'].nunique()} stocks, "
          f"{len(train_dates)} days")
    print(f"  Trade: {len(trade_panel):,} obs, "
          f"{trade_panel['permno'].nunique()} stocks, "
          f"{len(trade_dates)} days")

    result = trade_panel[["date", "permno"]].copy()
    result["period_id"] = period_id

    p_rf_train = p_gbt_train = p_dnn_train = None

    if run_rf:
        t1 = time.time()
        rf = build_h2o_rf()
        rf = train_h2o_rf(rf, train_panel, y_train)
        result["p_rf"] = predict_h2o_rf(rf, trade_panel)
        p_rf_train = predict_h2o_rf(rf, train_panel)
        print(f"    RF: {time.time()-t1:.1f}s")

    if run_gbt:
        t1 = time.time()
        gbt = build_h2o_gbt()
        gbt = train_h2o_gbt(gbt, train_panel, y_train)
        result["p_xgb"] = predict_h2o_gbt(gbt, trade_panel)
        p_gbt_train = predict_h2o_gbt(gbt, train_panel)
        print(f"    GBT: {time.time()-t1:.1f}s")

    if run_dnn:
        t1 = time.time()
        dnn = build_h2o_dnn()
        dnn = train_h2o_dnn(dnn, train_panel, y_train)
        result["p_dnn"] = predict_h2o_dnn(dnn, trade_panel)
        p_dnn_train = predict_h2o_dnn(dnn, train_panel)
        print(f"    DNN: {time.time()-t1:.1f}s")

    # Ensembles
    if run_rf and run_gbt and run_dnn:
        result["p_ens1"] = ens1_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values
        )
        result["p_ens2"] = ens2_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values,
            y_train.values, p_dnn_train, p_gbt_train, p_rf_train,
        )
        result["p_ens3"] = ens3_predictions(
            result["p_dnn"].values, result["p_xgb"].values, result["p_rf"].values,
            y_train.values, p_dnn_train, p_gbt_train, p_rf_train,
        )
        # Save ensemble metadata
        ens_meta = {}
        for name, p_tr in [("rf", p_rf_train), ("gbt", p_gbt_train), ("dnn", p_dnn_train)]:
            auc = roc_auc_score(y_train.values, p_tr)
            gini = 2 * auc - 1
            ens_meta[name] = {"auc": round(auc, 6), "gini": round(gini, 6)}
        with open(model_dir / "ensemble_meta.json", "w") as f:
            json.dump(ens_meta, f, indent=2)

    # Save meta
    meta = {
        "period_id": period_id,
        "trade_start": str(sp.trade_start.date()),
        "trade_end": str(sp.trade_end.date()),
        "n_train_obs": len(train_panel),
        "n_trade_obs": len(trade_panel),
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    result.to_parquet(pred_path, index=False)
    print(f"  Period {period_id} total: {time.time()-t0:.1f}s\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                        choices=["rf", "gbt", "dnn", "all"])
    parser.add_argument("--periods", type=int, nargs="*", default=None)
    args = parser.parse_args()

    run_rf = args.model in ("rf", "all")
    run_gbt = args.model in ("gbt", "all")
    run_dnn = args.model in ("dnn", "all")

    print("=" * 60)
    print("PHASE 1 — H2O Model Training & Prediction")
    print("=" * 60)
    print(f"Models: RF={run_rf}, GBT={run_gbt}, DNN={run_dnn}\n")

    # Start H2O
    import h2o
    h2o.init(max_mem_size="8G", nthreads=1)
    print()

    # Load data
    print("Loading data...")
    features, labels, returns, eligible = load_data()
    print(f"  Features: {len(features):,}")
    print(f"  Labels:   {len(labels):,}")

    # Build study periods
    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)
    print(f"\n{len(periods)} study periods")

    if args.periods is not None:
        period_ids = args.periods
    else:
        period_ids = list(range(len(periods)))

    print(f"Running periods: {period_ids}\n")

    all_results = []

    for pid in period_ids:
        sp = periods[pid]
        print(f"--- Period {pid}: trade {sp.trade_start.date()} "
              f"to {sp.trade_end.date()} ---")
        result = run_period(
            pid, sp, features, labels, eligible,
            run_rf=run_rf, run_gbt=run_gbt, run_dnn=run_dnn,
        )
        all_results.append(result)

        # Save consolidated after each period
        all_so_far = pd.concat(all_results, ignore_index=True)
        all_so_far.to_parquet(PROCESSED / "predictions_phase1_h2o.parquet", index=False)

    predictions = pd.concat(all_results, ignore_index=True)
    out_path = PROCESSED / "predictions_phase1_h2o.parquet"
    predictions.to_parquet(out_path, index=False)

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nPredictions: {len(predictions):,} rows")
    print(f"Saved -> {out_path.relative_to(ROOT)}")

    for col in ["p_rf", "p_xgb", "p_dnn", "p_ens1"]:
        if col in predictions.columns:
            m = predictions[col].mean()
            s = predictions[col].std()
            print(f"  {col} mean: {m:.4f}  std: {s:.4f}")

    # Shutdown H2O
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
