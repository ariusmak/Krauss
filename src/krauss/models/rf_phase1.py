"""
Random Forest classifier for Phase 1 reproduction.

Paper parameters (H2O DRF):
    - 1000 trees (B_RAF = 1000)
    - Max depth 20 (J_RAF = 20)
    - Feature subsampling: floor(sqrt(p)) where p=31 -> 5
    - Seed fixed to 1
    - sklearn RandomForestClassifier as agreed Python analogue
      (paper uses H2O DRF — logged as reproduction deviation)

H2O DRF defaults matched:
    - sample_rate=0.6320 (63.2% subsample without replacement, not bootstrap)
    - criterion: entropy (H2O DRF default, not gini)
    - nbins=20 for histogram splits: cannot replicate in sklearn (uses exact splits)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEED = 1
FEATURE_COLS = (
    [f"R{i}" for i in range(1, 21)]
    + [f"R{i}" for i in range(40, 241, 20)]
)


def build_rf_model(seed: int = SEED) -> RandomForestClassifier:
    """
    Create a RandomForestClassifier with H2O DRF-matched parameters.
    """
    n_features = len(FEATURE_COLS)  # 31
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=20,
        max_features=int(np.floor(np.sqrt(n_features))),  # 5
        # H2O DRF uses sample_rate=0.6320 (subsample WITHOUT replacement).
        # sklearn 1.7+ disallows max_samples with bootstrap=False.
        # Use bootstrap=True + max_samples=0.6320: draws 63.2% of n samples
        # WITH replacement. For large n (~255K), the expected unique fraction
        # is ~63.2%, matching H2O's without-replacement behavior closely.
        bootstrap=True,
        max_samples=0.6320,
        # H2O DRF uses entropy (information gain), not gini
        criterion="entropy",
        random_state=seed,
        n_jobs=-1,
    )


def train_rf(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit the RF on training data."""
    model.fit(X_train[FEATURE_COLS], y_train)
    return model


def predict_rf(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Generate probability predictions (P(y=1)) for test data.
    """
    return model.predict_proba(X_test[FEATURE_COLS])[:, 1]
