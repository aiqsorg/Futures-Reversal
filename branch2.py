# region imports
from AlgorithmImports import *
# endregion

# Your New Python File

import numpy as np
from config import PARAMS
"""
═══════════════════════════════════════════════════════════════
FILE 1: branch2.py — Stress EWMA (Session-Reset)
═══════════════════════════════════════════════════════════════
Computes fast/slow EWMA of KL values with session resets.
In live: fed by tensor slot 2 KL (most recent bar with backward info).
"""
import numpy as np
from config import PARAMS


def compute_stress(kl_array, session_dates, fast_span=None, slow_span=None):
    """
    Session-reset EWMA stress indicator.

    Parameters
    ----------
    kl_array : np.ndarray [N] — per-bar KL values
    session_dates : np.ndarray [N] — session date per bar
    fast_span, slow_span : int — EWMA spans

    Returns
    -------
    stress : np.ndarray [N] — fast EWMA - slow EWMA
    """
    if fast_span is None: fast_span = PARAMS["ewma_fast"]
    if slow_span is None: slow_span = PARAMS["ewma_slow"]

    n = len(kl_array)
    kf = np.zeros(n)
    ks = np.zeros(n)
    af = 2.0 / (fast_span + 1)
    asl = 2.0 / (slow_span + 1)

    prev_sess = None
    fv = sv = 0.0

    for i in range(n):
        se = session_dates[i]
        kv = kl_array[i]
        if np.isnan(kv):
            kv = fv  # carry forward

        if se != prev_sess:
            # Session reset
            fv = sv = kv
            prev_sess = se
        else:
            fv = af * kv + (1 - af) * fv
            sv = asl * kv + (1 - asl) * sv

        kf[i] = fv
        ks[i] = sv

    return kf - ks


def attach_stress(df, hmm_results, mode="B"):
    """
    Compute and attach stress columns to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    hmm_results : dict from hmm_model.run_hmm_all_sessions()
    mode : "A" (full backward KL) or "B" (tensor slot 2 KL)

    Returns
    -------
    df : pd.DataFrame with stress columns
    """
    df = df.copy()
    sd_arr = df["SessionDate"].values

    # Mode A stress (from full backward — for comparison)
    df["stress_A"] = compute_stress(hmm_results["kl_a"], sd_arr)

    # Mode B stress (from tensor slot 2 — live feasible)
    kl_b_slot2 = hmm_results["kl_b_slots"][:, 2]
    df["stress_B"] = compute_stress(kl_b_slot2, sd_arr)

    return df
