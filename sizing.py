# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
"""
═══════════════════════════════════════════════════════════════
FILE 4: sizing.py — Conviction Score + Position Sizing
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from config import PARAMS

def compute_conviction(row, weights=None):
    """
    Compute conviction score from tensor features at a single bar.

    Parameters
    ----------
    row : pd.Series — one bar from the crossover dataframe
    weights : dict or None — component weights

    Returns
    -------
    conviction : float in [0, 1]
    components : dict of individual scores
    """
    if weights is None:
        weights = PARAMS["conviction_weights"]

    stress_val = row.get("stress_B", 0.0)
    age_val = row.get("T_regime_age", 0)
    alpha_conv = row.get("T_alpha_conv", 0.0)
    kl_slope = row.get("T_kl_slope_3s", 0.0)

    components = {
        "stress": np.clip((0.02 - stress_val) / 0.02, 0, 1),
        "age": np.clip(age_val / 20.0, 0, 1),
        "alpha": np.clip(alpha_conv * 2, 0, 1),
        "kl_slope": np.clip(kl_slope * 500, 0, 1),
    }

    conviction = sum(weights.get(k, 0) * v for k, v in components.items())
    return np.clip(conviction, 0, 1), components


def size_position(conviction, tiers=None):
    """
    Map conviction to contract count.

    Parameters
    ----------
    conviction : float in [0, 1]
    tiers : list of dicts with threshold/contracts

    Returns
    -------
    n_contracts : int
    """
    if tiers is None:
        tiers = PARAMS["contract_tiers"]

    for tier in tiers:
        if conviction >= tier["threshold"]:
            return tier["contracts"]
    return 1


def compute_conviction_column(xo):
    """
    Add conviction and contract columns to crossover dataframe.

    Parameters
    ----------
    xo : pd.DataFrame — gated crossover bars

    Returns
    -------
    xo : pd.DataFrame with conviction, n_contracts columns
    """
    xo = xo.copy()
    convictions = []
    contracts = []

    for idx in xo.index:
        conv, _ = compute_conviction(xo.loc[idx])
        convictions.append(conv)
        contracts.append(size_position(conv))

    xo["conviction"] = convictions
    xo["n_contracts"] = contracts

    return xo
    
