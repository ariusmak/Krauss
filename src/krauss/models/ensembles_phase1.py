"""
Phase 1 ensemble methods.

ENS1: Equal-weight average of DNN, GBT, RAF probabilities.
ENS2: Training-period Gini/AUC-weighted average.
ENS3: Training-period rank-weighted average.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def ens1_predictions(
    p_dnn: np.ndarray,
    p_gbt: np.ndarray,
    p_raf: np.ndarray,
) -> np.ndarray:
    """
    ENS1: Simple equal-weighted average.

    P_ENS = (P_DNN + P_GBT + P_RAF) / 3
    """
    return (p_dnn + p_gbt + p_raf) / 3.0


def _compute_auc(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Compute AUC (area under ROC curve)."""
    return roc_auc_score(y_true, p_hat)


def _compute_gini(y_true: np.ndarray, p_hat: np.ndarray) -> float:
    """Compute Gini coefficient = 2 * AUC - 1."""
    return 2.0 * _compute_auc(y_true, p_hat) - 1.0


def ens2_predictions(
    p_dnn: np.ndarray,
    p_gbt: np.ndarray,
    p_raf: np.ndarray,
    y_train: np.ndarray,
    p_dnn_train: np.ndarray,
    p_gbt_train: np.ndarray,
    p_raf_train: np.ndarray,
) -> np.ndarray:
    """
    ENS2: Gini/AUC-based weighted average.

    Weights proportional to each model's training-period Gini coefficient.
    """
    gini_dnn = max(_compute_gini(y_train, p_dnn_train), 0)
    gini_gbt = max(_compute_gini(y_train, p_gbt_train), 0)
    gini_raf = max(_compute_gini(y_train, p_raf_train), 0)

    total = gini_dnn + gini_gbt + gini_raf
    if total == 0:
        return ens1_predictions(p_dnn, p_gbt, p_raf)

    w_dnn = gini_dnn / total
    w_gbt = gini_gbt / total
    w_raf = gini_raf / total

    return w_dnn * p_dnn + w_gbt * p_gbt + w_raf * p_raf


def ens3_predictions(
    p_dnn: np.ndarray,
    p_gbt: np.ndarray,
    p_raf: np.ndarray,
    y_train: np.ndarray,
    p_dnn_train: np.ndarray,
    p_gbt_train: np.ndarray,
    p_raf_train: np.ndarray,
) -> np.ndarray:
    """
    ENS3: Rank-based weighted average (paper Equation 7).

    w_i = (1/R_i) / (1/R_DNN + 1/R_GBT + 1/R_RAF)

    where R_i is the Gini-based performance rank of base learner i.
    Rank 1 = best (highest Gini), rank 3 = worst (lowest Gini),
    following Aiolfi & Timmermann (2006).
    """
    ginis = {
        "dnn": _compute_gini(y_train, p_dnn_train),
        "gbt": _compute_gini(y_train, p_gbt_train),
        "raf": _compute_gini(y_train, p_raf_train),
    }
    # Rank 1 = best (highest Gini), rank 3 = worst (lowest Gini)
    sorted_models = sorted(ginis, key=lambda m: ginis[m], reverse=True)
    ranks = {m: r + 1 for r, m in enumerate(sorted_models)}

    # Paper Eq. 7: w_i = (1/R_i) / sum(1/R_j)
    inv_ranks = {m: 1.0 / ranks[m] for m in ranks}
    total_inv = sum(inv_ranks.values())

    w_dnn = inv_ranks["dnn"] / total_inv
    w_gbt = inv_ranks["gbt"] / total_inv
    w_raf = inv_ranks["raf"] / total_inv

    return w_dnn * p_dnn + w_gbt * p_gbt + w_raf * p_raf
