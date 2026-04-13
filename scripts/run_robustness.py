"""
Run robustness/sensitivity analysis (Table 7) — alternative model configurations.

Usage:
    python scripts/run_robustness.py --alt 1        # Alternative 1 (low param)
    python scripts/run_robustness.py --alt 2        # Alternative 2 (high param)
    python scripts/run_robustness.py --alt 3        # Alternative 3 (shallow DNN)
    python scripts/run_robustness.py                # All three
    python scripts/run_robustness.py --alt 1 --periods 0 1 2

Configurations (from Table 7 of the paper):
    Baseline:      DNN 31-31-10-5-2,  GBT 100 trees, RAF 1000 trees
    Alternative 1: DNN 31-15-10-5-2,  GBT 50 trees,  RAF 500 trees
    Alternative 2: DNN 31-62-10-5-2,  GBT 200 trees, RAF 2000 trees
    Alternative 3: NN  31-31-2 (tanh, no dropout) — DNN only

Outputs:
    data/models_alt{N}/period_{id}/  — saved models per period
    data/processed/predictions_alt{N}.parquet — predictions
"""

import argparse
import json
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from krauss.data.study_periods import build_study_periods
from krauss.data.features import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


# ── Alternative configurations ──────────────────────────────────────────
ALT_CONFIGS = {
    1: {
        "name": "Alternative 1 (low parameterization)",
        "rf": {"n_estimators": 500},
        "xgb": {"n_estimators": 50},
        "dnn": {"h1_size": 15},  # 31-15-10-5-2
    },
    2: {
        "name": "Alternative 2 (high parameterization)",
        "rf": {"n_estimators": 2000},
        "xgb": {"n_estimators": 200},
        "dnn": {"h1_size": 62},  # 31-62-10-5-2
    },
    3: {
        "name": "Alternative 3 (shallow NN)",
        "rf": None,   # not applicable
        "xgb": None,   # not applicable
        "dnn": {"shallow": True},  # 31-31-2, tanh, no dropout
    },
}


def load_data():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    returns = pd.read_parquet(PROCESSED / "daily_returns.parquet")
    eligible = pd.read_parquet(PROCESSED / "universe_daily.parquet")
    return features, labels, returns, eligible


def build_panel(features, labels, eligible, dates_set):
    feat = features[features["date"].isin(dates_set)].copy()
    lab = labels[labels["date"].isin(dates_set)][["date", "permno", "y_binary"]].copy()
    panel = feat.merge(lab, on=["date", "permno"], how="inner")
    panel = panel.merge(eligible, on=["date", "permno"], how="inner")
    panel = panel.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return panel


def run_period(period_id, sp, features, labels, eligible, alt_id, models_dir):
    """Train alternative models for one period."""
    t0 = time.time()
    cfg = ALT_CONFIGS[alt_id]

    model_dir = models_dir / f"period_{period_id:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_dates = set(pd.to_datetime(sp.usable_train_dates))
    trade_dates = set(pd.to_datetime(sp.trade_dates))

    train_panel = build_panel(features, labels, eligible, train_dates)
    trade_panel = build_panel(features, labels, eligible, trade_dates)

    print(f"  Train: {len(train_panel):,} obs, "
          f"{train_panel['permno'].nunique()} stocks")
    print(f"  Trade: {len(trade_panel):,} obs, "
          f"{trade_panel['permno'].nunique()} stocks")

    # Save panels for subprocesses
    train_panel.to_parquet(model_dir / "_train_panel.parquet", index=False)
    trade_panel.to_parquet(model_dir / "_trade_panel.parquet", index=False)

    src_path = str(ROOT / "src")
    result = trade_panel[["date", "permno"]].copy()
    result["period_id"] = period_id

    def _run_subprocess(name, script):
        t1 = time.time()
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for line in proc.stdout:
            print(f"      {line}", end="", flush=True)
        proc.wait()
        elapsed = time.time() - t1
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            print(f"    {name} STDERR: {stderr}")
            raise RuntimeError(f"{name} failed (exit {proc.returncode})")
        print(f"    {name}: {elapsed:.1f}s -> {model_dir.name}/")

    # ── DNN ──
    if cfg["dnn"] is not None:
        if cfg["dnn"].get("shallow"):
            # Alternative 3: shallow 31-31-2 with tanh, no dropout
            _run_subprocess("DNN(shallow)", f"""
import sys; sys.path.insert(0, {src_path!r})
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from krauss.data.features import FEATURE_COLS

tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})

# Simple 31-31-2 network with tanh, no dropout
class ShallowNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(31, 31), nn.Tanh(),
            nn.Linear(31, 2),
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(1); np.random.seed(1)
model = ShallowNN()

X_all = torch.tensor(tp[FEATURE_COLS].values.astype(np.float32))
y_all = torch.tensor(tp['y_binary'].values.astype(np.int64))
n_val = max(int(len(X_all) * 0.1), 1)
idx = torch.randperm(len(X_all))
X_tr, y_tr = X_all[idx[n_val:]], y_all[idx[n_val:]]
X_val, y_val = X_all[idx[:n_val]], y_all[idx[:n_val]]

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024, shuffle=True)
opt = torch.optim.Adadelta(model.parameters())
crit = nn.CrossEntropyLoss()
best_val, best_state, no_imp = float('inf'), None, 0

for epoch in range(400):
    model.train()
    for xb, yb in loader:
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = crit(model(X_val), y_val).item()
    if vl < best_val:
        best_val = vl; best_state = {{k: v.cpu().clone() for k, v in model.state_dict().items()}}; no_imp = 0
    else:
        no_imp += 1
    if epoch % 50 == 0:
        print(f'  epoch {{epoch}}: val_loss={{vl:.6f}} best={{best_val:.6f}} no_imp={{no_imp}}', flush=True)
    if no_imp >= 20:
        break

if best_state: model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_trade = torch.softmax(model(torch.tensor(tdp[FEATURE_COLS].values.astype(np.float32))), dim=1)[:, 1].numpy()
    p_train = torch.softmax(model(torch.tensor(tp[FEATURE_COLS].values.astype(np.float32))), dim=1)[:, 1].numpy()
pd.DataFrame({{'p': p_trade}}).to_parquet({str(model_dir / '_dnn_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train}}).to_parquet({str(model_dir / '_dnn_train.parquet')!r}, index=False)
torch.save(model.state_dict(), {str(model_dir / 'dnn.pt')!r})
""")
        else:
            # Alternative 1 or 2: different H1 size
            h1 = cfg["dnn"]["h1_size"]
            _run_subprocess(f"DNN(h1={h1})", f"""
import sys; sys.path.insert(0, {src_path!r})
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from krauss.data.features import FEATURE_COLS
from krauss.models.dnn_phase1 import MaxoutLayer

tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})

class DNN_Alt(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dropout = nn.Dropout(0.1)
        self.h1 = MaxoutLayer(31, {h1}, 2); self.drop1 = nn.Dropout(0.5)
        self.h2 = MaxoutLayer({h1}, 10, 2); self.drop2 = nn.Dropout(0.5)
        self.h3 = MaxoutLayer(10, 5, 2); self.drop3 = nn.Dropout(0.5)
        self.output = nn.Linear(5, 2)
    def forward(self, x):
        x = self.input_dropout(x)
        x = self.drop1(self.h1(x))
        x = self.drop2(self.h2(x))
        x = self.drop3(self.h3(x))
        return self.output(x)

torch.manual_seed(1); np.random.seed(1)
model = DNN_Alt()

X_all = torch.tensor(tp[FEATURE_COLS].values.astype(np.float32))
y_all = torch.tensor(tp['y_binary'].values.astype(np.int64))
n_val = max(int(len(X_all) * 0.1), 1)
idx = torch.randperm(len(X_all))
X_tr, y_tr = X_all[idx[n_val:]], y_all[idx[n_val:]]
X_val, y_val = X_all[idx[:n_val]], y_all[idx[:n_val]]

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024, shuffle=True)
opt = torch.optim.Adadelta(model.parameters())
crit = nn.CrossEntropyLoss()
best_val, best_state, no_imp = float('inf'), None, 0

for epoch in range(400):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss = crit(model(xb), yb)
        l1 = sum(p.abs().sum() for p in model.parameters())
        loss = loss + 1e-5 * l1
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = crit(model(X_val), y_val).item()
    if vl < best_val:
        best_val = vl; best_state = {{k: v.cpu().clone() for k, v in model.state_dict().items()}}; no_imp = 0
    else:
        no_imp += 1
    if epoch % 50 == 0:
        print(f'  epoch {{epoch}}: val_loss={{vl:.6f}} best={{best_val:.6f}} no_imp={{no_imp}}', flush=True)
    if no_imp >= 20:
        break

if best_state: model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_trade = torch.softmax(model(torch.tensor(tdp[FEATURE_COLS].values.astype(np.float32))), dim=1)[:, 1].numpy()
    p_train = torch.softmax(model(torch.tensor(tp[FEATURE_COLS].values.astype(np.float32))), dim=1)[:, 1].numpy()
pd.DataFrame({{'p': p_trade}}).to_parquet({str(model_dir / '_dnn_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': p_train}}).to_parquet({str(model_dir / '_dnn_train.parquet')!r}, index=False)
torch.save(model.state_dict(), {str(model_dir / 'dnn.pt')!r})
""")
        result["p_dnn"] = pd.read_parquet(model_dir / "_dnn_trade.parquet")["p"].values

    # ── XGB ──
    if cfg["xgb"] is not None:
        n_trees = cfg["xgb"]["n_estimators"]
        _run_subprocess(f"XGB(n={n_trees})", f"""
import sys; sys.path.insert(0, {src_path!r})
import pandas as pd
from krauss.models.xgb_phase1 import train_xgb, predict_xgb
import xgboost as xgb
import numpy as np

tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})
from krauss.data.features import FEATURE_COLS

m = xgb.XGBClassifier(
    n_estimators={n_trees}, max_depth=3, learning_rate=0.1,
    colsample_bynode=15/31, objective='binary:logistic',
    eval_metric='logloss', random_state=1, n_jobs=-1,
)
m.fit(tp[FEATURE_COLS], tp['y_binary'])
pd.DataFrame({{'p': m.predict_proba(tdp[FEATURE_COLS])[:, 1]}}).to_parquet({str(model_dir / '_xgb_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': m.predict_proba(tp[FEATURE_COLS])[:, 1]}}).to_parquet({str(model_dir / '_xgb_train.parquet')!r}, index=False)
m.save_model({str(model_dir / 'xgb.json')!r})
""")
        result["p_xgb"] = pd.read_parquet(model_dir / "_xgb_trade.parquet")["p"].values

    # ── RF ──
    if cfg["rf"] is not None:
        n_trees = cfg["rf"]["n_estimators"]
        _run_subprocess(f"RF(n={n_trees})", f"""
import sys; sys.path.insert(0, {src_path!r})
import joblib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from krauss.data.features import FEATURE_COLS

tp = pd.read_parquet({str(model_dir / '_train_panel.parquet')!r})
tdp = pd.read_parquet({str(model_dir / '_trade_panel.parquet')!r})

rf = RandomForestClassifier(
    n_estimators={n_trees}, max_depth=20,
    max_features=int(np.floor(np.sqrt(31))),
    random_state=1, n_jobs=-1,
)
rf.fit(tp[FEATURE_COLS], tp['y_binary'])
pd.DataFrame({{'p': rf.predict_proba(tdp[FEATURE_COLS])[:, 1]}}).to_parquet({str(model_dir / '_rf_trade.parquet')!r}, index=False)
pd.DataFrame({{'p': rf.predict_proba(tp[FEATURE_COLS])[:, 1]}}).to_parquet({str(model_dir / '_rf_train.parquet')!r}, index=False)
joblib.dump(rf, {str(model_dir / 'rf.pkl')!r})
""")
        result["p_rf"] = pd.read_parquet(model_dir / "_rf_trade.parquet")["p"].values

    # Clean temp files
    for f in model_dir.glob("_*.parquet"):
        f.unlink()

    # Save predictions
    result.to_parquet(model_dir / "predictions.parquet", index=False)
    print(f"    Predictions -> {model_dir.name}/predictions.parquet")
    print(f"  Period {period_id} total: {time.time()-t0:.1f}s\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt", type=int, nargs="*", default=[1, 2, 3],
                        choices=[1, 2, 3])
    parser.add_argument("--periods", type=int, nargs="*", default=None)
    args = parser.parse_args()

    # Load data once
    print("Loading data...")
    features, labels, returns, eligible = load_data()
    trading_dates = np.sort(returns["date"].unique())
    periods = build_study_periods(trading_dates)
    print(f"  {len(periods)} study periods\n")

    if args.periods is not None:
        period_ids = args.periods
    else:
        period_ids = list(range(len(periods)))

    for alt_id in args.alt:
        cfg = ALT_CONFIGS[alt_id]
        models_dir = ROOT / "data" / f"models_alt{alt_id}"
        models_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"ROBUSTNESS — {cfg['name']}")
        print(f"  RF:  {cfg['rf']}")
        print(f"  XGB: {cfg['xgb']}")
        print(f"  DNN: {cfg['dnn']}")
        print(f"  Output: {models_dir.relative_to(ROOT)}/")
        print("=" * 60)

        from tqdm import tqdm
        all_results = []

        # Resume support
        for pid in period_ids:
            pred_path = models_dir / f"period_{pid:02d}" / "predictions.parquet"
            if pred_path.exists():
                all_results.append(pd.read_parquet(pred_path))
                print(f"  Period {pid}: loaded from cache")

        completed = {r["period_id"].iloc[0] for r in all_results}
        remaining = [pid for pid in period_ids if pid not in completed]
        print(f"  {len(completed)} cached, {len(remaining)} remaining\n")

        for pid in tqdm(remaining, desc=f"Alt{alt_id}", unit="period"):
            sp = periods[pid]
            print(f"\n--- Period {pid}: trade {sp.trade_start.date()} "
                  f"to {sp.trade_end.date()} ---")
            result = run_period(pid, sp, features, labels, eligible,
                                alt_id, models_dir)
            all_results.append(result)

            # Save progress
            all_so_far = pd.concat(all_results, ignore_index=True)
            all_so_far.to_parquet(PROCESSED / f"predictions_alt{alt_id}.parquet",
                                  index=False)

        predictions = pd.concat(all_results, ignore_index=True)
        out_path = PROCESSED / f"predictions_alt{alt_id}.parquet"
        predictions.to_parquet(out_path, index=False)

        print(f"\nAlt{alt_id} done: {len(predictions):,} rows -> {out_path.name}")
        for col in [c for c in predictions.columns if c.startswith("p_")]:
            print(f"  {col} mean: {predictions[col].mean():.4f}")
        print()


if __name__ == "__main__":
    main()
