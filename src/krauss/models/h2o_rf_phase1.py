"""
H2O Random Forest for Phase 1 — exact paper configuration.

This uses H2O's DRF (the same framework as the original paper),
with all parameters matched to Section 4.3.3.

Parameters:
    B_RAF = 1000 trees
    J_RAF = 20 (max depth)
    m_RAF = floor(sqrt(31)) = 5 (features per split)
    Seed: 1
"""

import numpy as np
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)
SEED = 1


def build_h2o_rf() -> H2ORandomForestEstimator:
    """Create H2O DRF with exact paper parameters."""
    n_features = len(FEATURE_COLS)  # 31
    return H2ORandomForestEstimator(
        ntrees=1000,
        max_depth=20,
        mtries=int(np.floor(np.sqrt(n_features))),  # 5
        # H2O DRF defaults the paper relies on
        sample_rate=0.6320,
        nbins=20,
        seed=SEED,
        distribution="bernoulli",
    )


def train_h2o_rf(
    model: H2ORandomForestEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> H2ORandomForestEstimator:
    """Train the H2O DRF on training data."""
    train_df = X_train[FEATURE_COLS].copy()
    train_df["y"] = y_train.values.astype(str)

    h2o_train = h2o.H2OFrame(train_df)
    h2o_train["y"] = h2o_train["y"].asfactor()

    model.train(x=FEATURE_COLS, y="y", training_frame=h2o_train)
    return model


def predict_h2o_rf(
    model: H2ORandomForestEstimator,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate P(y=1) predictions."""
    h2o_test = h2o.H2OFrame(X_test[FEATURE_COLS])
    preds = model.predict(h2o_test)
    return preds["p1"].as_data_frame().values.ravel()
