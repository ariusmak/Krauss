"""
H2O Deep Neural Network for Phase 1 — exact paper configuration.

This uses the same H2O framework as the original paper, with all
parameters matched to the paper's Section 4.3.1 and H2O defaults.

Architecture: 31-31-10-5-2
Activation: Maxout (with dropout)
Output: 2-class softmax
Optimizer: ADADELTA (rho=0.99, eps=1e-8, rate=0.005, rate_annealing=1e-6)
Dropout: hidden=0.5, input=0.1
L1: 1e-5
Epochs: 400 max, early stopping
Seed: 1, single thread (reproducible=True)
"""

import numpy as np
import pandas as pd
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)
SEED = 1


def build_h2o_dnn() -> H2ODeepLearningEstimator:
    """Create H2O DNN with exact paper parameters."""
    return H2ODeepLearningEstimator(
        # Architecture: 31 inputs -> 31-10-5 hidden -> 2 outputs
        hidden=[31, 10, 5],
        activation="MaxoutWithDropout",
        # Regularization
        hidden_dropout_ratios=[0.5, 0.5, 0.5],
        input_dropout_ratio=0.1,
        l1=1e-5,
        # Training
        epochs=400,
        # ADADELTA with H2O defaults
        adaptive_rate=True,
        rho=0.99,
        epsilon=1e-8,
        # Early stopping: avg of last 5 scores, stop after 5 non-improvements
        overwrite_with_best_model=True,
        stopping_rounds=5,
        stopping_metric="logloss",
        stopping_tolerance=0.0,
        # Reproducibility
        seed=SEED,
        reproducible=True,  # single thread, deterministic
        # Classification
        distribution="bernoulli",
        # Defaults the paper relies on (explicit for clarity)
        score_training_samples=10000,
        mini_batch_size=1,
        rate=0.005,
        rate_annealing=1e-6,
    )


def train_h2o_dnn(
    model: H2ODeepLearningEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> H2ODeepLearningEstimator:
    """Train the H2O DNN on training data."""
    train_df = X_train[FEATURE_COLS].copy()
    train_df["y"] = y_train.values.astype(str)  # H2O needs string for classification

    h2o_train = h2o.H2OFrame(train_df)
    h2o_train["y"] = h2o_train["y"].asfactor()

    model.train(x=FEATURE_COLS, y="y", training_frame=h2o_train)
    return model


def predict_h2o_dnn(
    model: H2ODeepLearningEstimator,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate P(y=1) predictions."""
    h2o_test = h2o.H2OFrame(X_test[FEATURE_COLS])
    preds = model.predict(h2o_test)
    # H2O returns columns: predict, p0, p1
    return preds["p1"].as_data_frame().values.ravel()
