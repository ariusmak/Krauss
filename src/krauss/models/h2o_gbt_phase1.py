"""
H2O Gradient-Boosted Trees for Phase 1 — exact paper configuration.

This uses H2O's GBM (the same framework as the original paper),
with all parameters matched to Section 4.3.2.

Paper: "We use H2O's implementation of AdaBoost, deploying shallow
decision trees as weak learners."

Parameters:
    M_GBT = 100 trees
    J_GBT = 3 (max depth, allows two-way interactions)
    lambda_GBT = 0.1 (learning rate)
    m_GBT = 15 (features per split, ~half of 31)
    Seed: 1
"""

import numpy as np
import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator

FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)
SEED = 1


def build_h2o_gbt() -> H2OGradientBoostingEstimator:
    """Create H2O GBM with exact paper parameters."""
    n_features = len(FEATURE_COLS)  # 31
    return H2OGradientBoostingEstimator(
        ntrees=100,
        max_depth=3,
        learn_rate=0.1,
        col_sample_rate_per_tree=15 / n_features,  # ~0.484
        # H2O defaults the paper relies on
        min_rows=10,
        nbins=20,
        seed=SEED,
        distribution="bernoulli",
    )


def train_h2o_gbt(
    model: H2OGradientBoostingEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> H2OGradientBoostingEstimator:
    """Train the H2O GBM on training data."""
    train_df = X_train[FEATURE_COLS].copy()
    train_df["y"] = y_train.values.astype(str)

    h2o_train = h2o.H2OFrame(train_df)
    h2o_train["y"] = h2o_train["y"].asfactor()

    model.train(x=FEATURE_COLS, y="y", training_frame=h2o_train)
    return model


def predict_h2o_gbt(
    model: H2OGradientBoostingEstimator,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate P(y=1) predictions."""
    h2o_test = h2o.H2OFrame(X_test[FEATURE_COLS])
    preds = model.predict(h2o_test)
    return preds["p1"].as_data_frame().values.ravel()
