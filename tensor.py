# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
"""
tensor.py — Tensor Builder + Feature Extraction
==================================================
Builds the [T, 5, 11] tensor per session from HMM outputs.
Extracts all features: continuous, argmax, patterns, disagreement,
micro, interactions.

Uses Mode B (progressive backward) gamma/KL from hmm_model.py.
The tensor is the SINGLE SOURCE OF TRUTH for all decision features.

Tensor layout per slot [11 columns]:
  [0:2]  gamma P(vol), P(quiet)
  [2:4]  alpha P(vol), P(quiet)
  [4:6]  proj  P(vol), P(quiet)     — alpha @ transition matrix
  [6]    KL(alpha || gamma)
  [7:11] micro features              — frozen at bar creation

Slots: [t-3, t-2, t-1, t_current, t+1_projection]
"""
import numpy as np
from config import PARAMS, MICRO_FEATURES


# ══════════════════════════════════════════════════════════════
# TENSOR BUILDER
# ══════════════════════════════════════════════════════════════

def build_tensor_session(alpha, kl_slots, gamma_slots, micro, active, transmat):
    """
    Build [T, 5, 11] tensor for one session using Mode B outputs.

    Parameters
    ----------
    alpha : [T, 2] — forward posteriors
    kl_slots : [T, 4] — KL per tensor slot from progressive backward
    gamma_slots : [T, 4, 2] — gamma per tensor slot from progressive backward
    micro : [T, 4] — micro features per bar
    active : [T] bool — activity mask
    transmat : [2, 2] — HMM transition matrix

    Returns
    -------
    tensor : [T, 5, 11] — full tensor history
    """
    T = len(alpha)
    N_S = PARAMS["tensor_slots"]
    N_C = PARAMS["tensor_cols"]
    A = transmat

    tensor = np.zeros((T, N_S, N_C))

    for t in range(T):
        # ── Slots 0-3: historical + current ──
        for slot in range(4):
            abs_bar = t - (3 - slot)

            if abs_bar < 0:
                # Before session — uniform
                tensor[t, slot, 0:2] = [0.5, 0.5]  # gamma
                tensor[t, slot, 2:4] = [0.5, 0.5]  # alpha
                tensor[t, slot, 4:6] = [0.5, 0.5]  # proj
                tensor[t, slot, 6] = 0.0            # KL
                tensor[t, slot, 7:11] = 0.0         # micro
                continue

            # Gamma from progressive backward
            tensor[t, slot, 0:2] = gamma_slots[t, slot]

            # Alpha (frozen at bar creation)
            tensor[t, slot, 2:4] = alpha[abs_bar]

            # Projection
            tensor[t, slot, 4:6] = alpha[abs_bar] @ A

            # KL from progressive backward
            tensor[t, slot, 6] = kl_slots[t, slot]

            # Micro (frozen at bar creation)
            tensor[t, slot, 7:11] = micro[abs_bar]

        # ── Slot 4: projection ──
        alpha_next = alpha[t] @ A
        proj_next = alpha_next @ A
        tensor[t, 4, 0:2] = alpha_next    # gamma = alpha for projection
        tensor[t, 4, 2:4] = alpha_next
        tensor[t, 4, 4:6] = proj_next
        tensor[t, 4, 6] = 0.0
        tensor[t, 4, 7:11] = 0.0

    return tensor


def build_tensors_all_sessions(df, hmm_results, sessions, model):
    """
    Build tensors for all sessions.

    Parameters
    ----------
    df : pd.DataFrame — with micro feature columns
    hmm_results : dict — from hmm_model.run_hmm_all_sessions()
    sessions : list of session dates
    model : fitted GaussianHMM

    Returns
    -------
    all_tensors : [N, 5, 11] — tensor for every bar
    """
    N = len(df)
    all_tensors = np.zeros((N, PARAMS["tensor_slots"], PARAMS["tensor_cols"]))

    offset = 0
    for sess in sessions:
        sm = df["SessionDate"] == sess
        T_s = sm.sum()

        alpha_s = hmm_results["alpha"][offset:offset + T_s]
        kl_s = hmm_results["kl_b_slots"][offset:offset + T_s]
        gamma_s = hmm_results["gamma_b_slots"][offset:offset + T_s]
        micro_s = df.loc[sm, MICRO_FEATURES].values
        active_s = df.loc[sm, "Active"].values

        tens = build_tensor_session(
            alpha_s, kl_s, gamma_s, micro_s, active_s, model.transmat_
        )
        all_tensors[offset:offset + T_s] = tens
        offset += T_s

    return all_tensors


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_features(tensor_history, alpha_all, active):
    """
    Extract ALL features from tensor. Returns dict of [T] arrays.
    Every feature name is prefixed with T_ when attached to dataframe.

    Categories: KL, alpha_continuous, alpha_argmax, gamma,
    disagreement, projection, micro, regime_persistence, interactions.
    """
    T = len(tensor_history)
    f = {}

    # ── Shortcuts ──
    gamma_q  = tensor_history[:, :4, 1]     # [T, 4] gamma P(quiet)
    alpha_q  = tensor_history[:, :4, 3]     # [T, 4] alpha P(quiet)
    proj_q   = tensor_history[:, :, 3]      # [T, 5] proj P(quiet) — includes slot 4
    kl_slots = tensor_history[:, :4, 6]     # [T, 4] KL
    micro_sl = tensor_history[:, :4, 7:11]  # [T, 4, 4] micro

    alpha_am = (alpha_q > 0.5).astype(int)
    gamma_am = (gamma_q > 0.5).astype(int)
    proj_am  = (proj_q > 0.5).astype(int)

    # ════════════════════════════════════════
    # A. KL FEATURES
    # ════════════════════════════════════════

    f["kl_slope_3s"] = _slope_3(kl_slots[:, :3])
    f["kl_slope_4s"] = _slope_4(kl_slots)
    f["kl_mean_3s"]  = np.nanmean(kl_slots[:, :3], axis=1)
    f["kl_max_3s"]   = np.nanmax(kl_slots[:, :3], axis=1)
    f["kl_min_3s"]   = np.nanmin(kl_slots[:, :3], axis=1)
    f["kl_range_3s"] = f["kl_max_3s"] - f["kl_min_3s"]
    f["kl_slot2"]    = kl_slots[:, 2]
    f["kl_accel"]    = (kl_slots[:, 2] - kl_slots[:, 1]) - (kl_slots[:, 1] - kl_slots[:, 0])

    # ════════════════════════════════════════
    # B. ALPHA CONTINUOUS
    # ════════════════════════════════════════

    f["alpha_std"]   = alpha_q.std(axis=1)
    f["alpha_mean"]  = alpha_q.mean(axis=1)
    f["alpha_conv"]  = np.abs(alpha_all[:, 1] - 0.5)
    f["alpha_slope"] = _slope_4(alpha_q)
    f["alpha_range"] = alpha_q.max(axis=1) - alpha_q.min(axis=1)

    # ════════════════════════════════════════
    # C. ALPHA ARGMAX
    # ════════════════════════════════════════

    for s in range(4):
        f[f"am_alpha_s{s}"] = alpha_am[:, s].astype(float)

    f["am_alpha_sum"]            = alpha_am.sum(axis=1).astype(float)
    f["am_alpha_unan_quiet"]     = (f["am_alpha_sum"] == 4).astype(float)
    f["am_alpha_unan_vol"]       = (f["am_alpha_sum"] == 0).astype(float)
    f["am_alpha_unanimous"]      = ((f["am_alpha_sum"] == 4) | (f["am_alpha_sum"] == 0)).astype(float)
    f["am_alpha_majority_quiet"] = (f["am_alpha_sum"] >= 3).astype(float)
    f["am_alpha_majority_vol"]   = (f["am_alpha_sum"] <= 1).astype(float)

    f["am_alpha_flips"] = _count_flips(alpha_am)

    f["am_alpha_last_flip"]  = _last_flip_pos(alpha_am)
    f["am_alpha_first_flip"] = _first_flip_pos(alpha_am)

    f["am_alpha_pattern"] = (alpha_am[:, 0]*8 + alpha_am[:, 1]*4 +
                              alpha_am[:, 2]*2 + alpha_am[:, 3]*1).astype(float)

    f["am_pattern_stable_quiet"]  = (f["am_alpha_pattern"] == 15).astype(float)
    f["am_pattern_stable_vol"]    = (f["am_alpha_pattern"] == 0).astype(float)
    f["am_pattern_entering_quiet"] = np.isin(f["am_alpha_pattern"], [7, 3, 1]).astype(float)
    f["am_pattern_entering_vol"]   = np.isin(f["am_alpha_pattern"], [8, 12, 14]).astype(float)
    f["am_pattern_mixed"]          = np.isin(f["am_alpha_pattern"], [5, 6, 9, 10]).astype(float)

    f["am_alpha_current_quiet"]  = alpha_am[:, 3].astype(float)
    f["am_alpha_transition_now"] = (alpha_am[:, 3] != alpha_am[:, 2]).astype(float)
    f["am_alpha_recent_trans"]   = (f["am_alpha_flips"] > 0).astype(float)

    # ════════════════════════════════════════
    # D. GAMMA CONTINUOUS + ARGMAX
    # ════════════════════════════════════════

    f["gamma_std"]   = gamma_q.std(axis=1)
    f["gamma_mean"]  = gamma_q.mean(axis=1)
    f["gamma_range"] = gamma_q.max(axis=1) - gamma_q.min(axis=1)
    f["gamma_slope"] = _slope_4(gamma_q)

    for s in range(4):
        f[f"am_gamma_s{s}"] = gamma_am[:, s].astype(float)

    f["am_gamma_sum"]        = gamma_am.sum(axis=1).astype(float)
    f["am_gamma_unan_quiet"] = (f["am_gamma_sum"] == 4).astype(float)
    f["am_gamma_unan_vol"]   = (f["am_gamma_sum"] == 0).astype(float)
    f["am_gamma_flips"]      = _count_flips(gamma_am)
    f["am_gamma_pattern"]    = (gamma_am[:, 0]*8 + gamma_am[:, 1]*4 +
                                 gamma_am[:, 2]*2 + gamma_am[:, 3]*1).astype(float)

    # ════════════════════════════════════════
    # E. DISAGREEMENT (alpha vs gamma per slot)
    # ════════════════════════════════════════

    disagree = (alpha_am != gamma_am).astype(float)
    for s in range(4):
        f[f"disagree_s{s}"] = disagree[:, s]

    f["disagree_count"]   = disagree.sum(axis=1)
    f["disagree_any"]     = (f["disagree_count"] > 0).astype(float)
    f["disagree_current"] = disagree[:, 3]
    f["disagree_trend"]   = (disagree[:, 2] + disagree[:, 3]) - (disagree[:, 0] + disagree[:, 1])
    f["disagree_pattern"] = (disagree[:, 0]*8 + disagree[:, 1]*4 +
                              disagree[:, 2]*2 + disagree[:, 3]*1)

    for s in range(4):
        f[f"ag_direction_s{s}"] = (alpha_am[:, s] - gamma_am[:, s]).astype(float)
    f["ag_direction_net"] = sum(f[f"ag_direction_s{s}"] for s in range(4))

    # ════════════════════════════════════════
    # F. PROJECTION
    # ════════════════════════════════════════

    f["proj_quiet_val"]      = tensor_history[:, 4, 3]
    f["proj_conv"]           = np.abs(tensor_history[:, 4, 3] - 0.5)
    f["am_proj_quiet"]       = proj_am[:, 4].astype(float)
    f["proj_agrees_alpha"]   = (proj_am[:, 4] == alpha_am[:, 3]).astype(float)
    f["proj_agrees_gamma"]   = (proj_am[:, 4] == gamma_am[:, 3]).astype(float)
    f["proj_error"]          = np.abs(tensor_history[:, 4, 3] - tensor_history[:, 3, 3])

    slot23_trend = alpha_am[:, 3] - alpha_am[:, 2]
    proj_trend = proj_am[:, 4].astype(int) - alpha_am[:, 3]
    f["proj_continues_trend"] = np.where(
        slot23_trend == 0,
        (proj_am[:, 4] == alpha_am[:, 3]).astype(float),
        (np.sign(proj_trend) == np.sign(slot23_trend)).astype(float)
    )

    f["am_5slot_pattern"] = (alpha_am[:, 0]*16 + alpha_am[:, 1]*8 +
                              alpha_am[:, 2]*4 + alpha_am[:, 3]*2 +
                              proj_am[:, 4]*1).astype(float)

    # ════════════════════════════════════════
    # G. MICRO FEATURES
    # ════════════════════════════════════════

    micro_names = ["gk", "range", "vol", "cpos"]
    for m in range(4):
        f[f"micro_{micro_names[m]}_mean"]  = micro_sl[:, :, m].mean(axis=1)
        f[f"micro_{micro_names[m]}_std"]   = micro_sl[:, :, m].std(axis=1)
        f[f"micro_{micro_names[m]}_last"]  = micro_sl[:, 3, m]
        f[f"micro_{micro_names[m]}_slope"] = _slope_4(micro_sl[:, :, m])

    # ════════════════════════════════════════
    # H. REGIME PERSISTENCE (session-aware — filled separately)
    # ════════════════════════════════════════

    f["regime_age"] = np.zeros(T)  # filled by attach_session_features()

    f["bars_since_flip"] = np.full(T, 4.0)
    for i in range(T):
        for j in range(3, 0, -1):
            if alpha_am[i, j] != alpha_am[i, j - 1]:
                f["bars_since_flip"][i] = 3 - j
                break

    f["current_streak"] = np.ones(T)
    for i in range(T):
        curr = alpha_am[i, 3]
        for j in range(2, -1, -1):
            if alpha_am[i, j] == curr:
                f["current_streak"][i] += 1
            else:
                break

    # ════════════════════════════════════════
    # I. INTERACTIONS
    # ════════════════════════════════════════

    f["kl_x_alpha_std"]  = f["kl_mean_3s"] * f["alpha_std"]
    f["kl_x_flips"]      = f["kl_mean_3s"] * f["am_alpha_flips"]
    f["kl_x_disagree"]   = f["kl_mean_3s"] * f["disagree_count"]
    f["conv_x_streak"]   = f["alpha_conv"] * f["current_streak"]
    f["conv_x_unanimous"] = f["alpha_conv"] * f["am_alpha_unanimous"]

    kl_p75 = np.nanpercentile(f["kl_mean_3s"], 75) if np.any(~np.isnan(f["kl_mean_3s"])) else 0
    f["kl_high_unstable"] = ((f["kl_mean_3s"] > kl_p75) & (f["alpha_std"] > 0.1)).astype(float)
    f["kl_high_stable"]   = ((f["kl_mean_3s"] > kl_p75) & (f["alpha_std"] < 0.05)).astype(float)
    kl_p25 = np.nanpercentile(f["kl_mean_3s"], 25) if np.any(~np.isnan(f["kl_mean_3s"])) else 0
    f["kl_low_stable"]    = ((f["kl_mean_3s"] < kl_p25) & (f["alpha_std"] < 0.05)).astype(float)

    f["disagree_at_transition"] = f["disagree_current"] * f["am_alpha_transition_now"]
    f["vol_calming_quiet"]      = ((f["micro_gk_slope"] < 0) &
                                    (f["am_alpha_current_quiet"] == 1)).astype(float)

    return f


def attach_features_to_df(df, features, sessions):
    """
    Attach tensor features to dataframe with T_ prefix.
    Also fills session-aware features (regime_age).
    """
    df = df.copy()
    for k, v in features.items():
        df[f"T_{k}"] = v

    # Fill regime_age (needs session awareness)
    df["T_regime_age"] = 0
    for sess in sessions:
        idx = df.index[df["SessionDate"] == sess]
        aq = (df.loc[idx, "Alpha_P1"].values > 0.5).astype(int)
        age, prev = 0, -1
        for i in range(len(idx)):
            if aq[i] == prev: age += 1
            else: age = 0; prev = aq[i]
            df.loc[idx[i], "T_regime_age"] = age

    # age × conviction interaction
    df["T_age_x_conv"] = df["T_regime_age"] * df["T_alpha_conv"]

    return df


# ══════════════════════════════════════════════════════════════
# SLOPE UTILITIES
# ══════════════════════════════════════════════════════════════

def _slope_3(arr_2d):
    """3-point linear slope per row. arr_2d: [T, 3]."""
    T = len(arr_2d)
    slope = np.full(T, 0.0)
    for i in range(T):
        w = arr_2d[i]
        if np.all(~np.isnan(w)) and np.all(w != 0) and np.std(w) > 1e-8:
            slope[i] = np.polyfit(np.arange(3), w, 1)[0]
    return slope


def _slope_4(arr_2d):
    """4-point linear slope per row. arr_2d: [T, 4]."""
    T = len(arr_2d)
    slope = np.full(T, 0.0)
    for i in range(T):
        w = arr_2d[i]
        if np.all(~np.isnan(w)) and np.std(w) > 1e-8:
            slope[i] = np.polyfit(np.arange(4), w, 1)[0]
    return slope


def _count_flips(am):
    """Count state flips in [T, 4] argmax array."""
    T = len(am)
    flips = np.zeros(T)
    for i in range(T):
        for j in range(1, 4):
            if am[i, j] != am[i, j - 1]:
                flips[i] += 1
    return flips


def _last_flip_pos(am):
    """Position of last flip (0=no flip, 1-3=between slots)."""
    T = len(am)
    pos = np.zeros(T)
    for i in range(T):
        for j in range(3, 0, -1):
            if am[i, j] != am[i, j - 1]:
                pos[i] = j
                break
    return pos


def _first_flip_pos(am):
    """Position of first flip."""
    T = len(am)
    pos = np.zeros(T)
    for i in range(T):
        for j in range(1, 4):
            if am[i, j] != am[i, j - 1]:
                pos[i] = j
                break
    return pos
