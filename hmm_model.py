# region imports
from AlgorithmImports import *
# endregion
"""
hmm_model.py — HMM Regime Detection
=======================================
Fits 2-state Gaussian HMM on configurable emissions.
Provides both Mode A (full backward, backtest ceiling) and
Mode B (progressive backward, live-feasible).

Pure functions. No global state.

Mode A: Full session forward-backward. Gamma at bar t uses bars t+1..T.
        NOT available in live. Used for comparison/ceiling.

Mode B: Progressive backward. At bar t, run backward on 0..t.
        Gamma at bar t-3 has 3 bars of backward info.
        Gamma at bar t (frontier) ≈ alpha (beta=1).
        This is what you'd run in live.
"""
import numpy as np
from hmmlearn import hmm
from config import PARAMS


# ══════════════════════════════════════════════════════════════
# HMM FITTING
# ══════════════════════════════════════════════════════════════

def fit_hmm(df, emission_cols, train_sessions, n_states=None, n_iter=None):
    """
    Fit Gaussian HMM on specified emissions and training sessions.

    Parameters
    ----------
    df : pd.DataFrame — must contain emission_cols and SessionDate
    emission_cols : list of str — column names for emissions
    train_sessions : list — session dates for training
    n_states : int — number of HMM states (default from config)
    n_iter : int — EM iterations (default from config)

    Returns
    -------
    model : fitted hmmlearn.GaussianHMM
    """
    if n_states is None: n_states = PARAMS["hmm_states"]
    if n_iter is None: n_iter = PARAMS["hmm_iter"]

    train_mask = df["SessionDate"].isin(train_sessions)
    X_train = df.loc[train_mask, emission_cols].values
    lengths = [int((df["SessionDate"] == s).sum()) for s in train_sessions]

    model = hmm.GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=n_iter, random_state=PARAMS["hmm_random_state"],
    )
    model.fit(X_train, lengths=lengths)

    # Enforce label consistency: State 0 = high vol, State 1 = quiet
    # Convention: higher GK_norm mean = higher vol state
    # For single-emission models, use first emission's variance
    if n_states == 2:
        if len(emission_cols) >= 2:
            # Use second emission (GK) if available
            swap = model.means_[0, 1] < model.means_[1, 1]
        else:
            # Use variance of first emission
            swap = model.covars_[0, 0, 0] < model.covars_[1, 0, 0]

        if swap:
            p = [1, 0]
            model.means_ = model.means_[p]
            model.covars_ = model.covars_[p]
            model.transmat_ = model.transmat_[np.ix_(p, p)]
            model.startprob_ = model.startprob_[p]

    diag = np.diag(model.transmat_)
    print(f"  HMM fitted on {len(train_sessions)} sessions, "
          f"{len(emission_cols)} emissions")
    print(f"    Transition diag: [{diag[0]:.3f}, {diag[1]:.3f}]")

    return model


# ══════════════════════════════════════════════════════════════
# FORWARD PASS (same for Mode A and Mode B)
# ══════════════════════════════════════════════════════════════

def run_forward(model, X, active):
    """
    Forward algorithm with activity masking.

    Parameters
    ----------
    model : fitted GaussianHMM
    X : np.ndarray [T, n_emissions]
    active : np.ndarray [T] bool

    Returns
    -------
    alpha : np.ndarray [T, n_states]
    B : np.ndarray [T, n_states] — emission probabilities (cached for backward)
    """
    T = len(X)
    N = model.n_components
    B = np.exp(model._compute_log_likelihood(X))

    alpha = np.zeros((T, N))
    alpha[0] = model.startprob_ * (B[0] if active[0] else 1.0)
    s = alpha[0].sum()
    if s > 0: alpha[0] /= s

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ model.transmat_) * (B[t] if active[t] else 1.0)
        s = alpha[t].sum()
        if s > 0: alpha[t] /= s

    return alpha, B


# ══════════════════════════════════════════════════════════════
# MODE A: FULL BACKWARD (backtest ceiling)
# ══════════════════════════════════════════════════════════════

def run_full_backward(model, B, active):
    """
    Full backward pass over entire session.
    NOT live-feasible — uses future data within session.

    Parameters
    ----------
    model : fitted GaussianHMM
    B : np.ndarray [T, n_states] — emission probs from forward pass
    active : np.ndarray [T] bool

    Returns
    -------
    beta : np.ndarray [T, n_states]
    """
    T = len(B)
    N = model.n_components
    beta = np.zeros((T, N))
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        bn = B[t + 1] if active[t + 1] else 1.0
        beta[t] = (model.transmat_ * bn * beta[t + 1]).sum(axis=1)
        s = beta[t].sum()
        if s > 0: beta[t] /= s

    return beta


def compute_gamma_kl(alpha, beta, active):
    """
    Compute smoothed posteriors (gamma) and KL divergence.

    Parameters
    ----------
    alpha : np.ndarray [T, n_states]
    beta : np.ndarray [T, n_states]
    active : np.ndarray [T] bool

    Returns
    -------
    gamma : np.ndarray [T, n_states]
    kl : np.ndarray [T] — KL(alpha || gamma), NaN for inactive bars
    """
    fb = alpha * beta
    fb_sum = fb.sum(axis=1, keepdims=True)
    fb_sum[fb_sum == 0] = 1.0
    gamma = fb / fb_sum

    kl = _kl_div(alpha, gamma)
    kl[~active] = np.nan

    return gamma, kl


def mode_a_session(model, X, active):
    """
    Full Mode A (backtest ceiling) for one session.

    Returns
    -------
    alpha : [T, 2], gamma : [T, 2], kl : [T]
    """
    alpha, B = run_forward(model, X, active)
    beta = run_full_backward(model, B, active)
    gamma, kl = compute_gamma_kl(alpha, beta, active)
    return alpha, gamma, kl


# ══════════════════════════════════════════════════════════════
# MODE B: PROGRESSIVE BACKWARD (live-feasible)
# ══════════════════════════════════════════════════════════════

def mode_b_session(model, X, active):
    """
    Progressive backward for one session. At each bar t, runs
    backward on bars 0..t to get gamma/KL for the tensor slots.

    Returns
    -------
    alpha : [T, 2] — forward posteriors (same as Mode A)
    kl_slots : [T, 4] — KL for tensor slots [t-3, t-2, t-1, t]
        Slot 3 (frontier) will have KL ≈ 0 since beta[t]=1
    gamma_slots : [T, 4, 2] — gamma for tensor slots
    """
    T = len(X)
    N = model.n_components
    alpha, B = run_forward(model, X, active)
    A = model.transmat_

    kl_slots = np.full((T, 4), np.nan)
    gamma_slots = np.zeros((T, 4, N))

    for t in range(T):
        # Backward pass on bars 0..t
        start = max(0, t - 3)
        length = t - start + 1

        b_partial = np.zeros((length, N))
        b_partial[-1] = 1.0  # beta[t] = 1 always

        for tb in range(length - 2, -1, -1):
            abs_idx = start + tb + 1
            bn = B[abs_idx] if active[abs_idx] else 1.0
            b_partial[tb] = (A * bn * b_partial[tb + 1]).sum(axis=1)
            s = b_partial[tb].sum()
            if s > 0: b_partial[tb] /= s

        # Fill slots
        for slot in range(4):
            abs_bar = t - (3 - slot)  # slot 0 = t-3, slot 3 = t
            if abs_bar < 0 or abs_bar < start:
                gamma_slots[t, slot] = [0.5, 0.5]
                kl_slots[t, slot] = 0.0
                continue

            local_idx = abs_bar - start
            fb = alpha[abs_bar] * b_partial[local_idx]
            s = fb.sum()
            if s > 0: fb /= s

            gamma_slots[t, slot] = fb
            if active[abs_bar]:
                kl_slots[t, slot] = _kl_div_single(alpha[abs_bar], fb)
            else:
                kl_slots[t, slot] = 0.0

    return alpha, kl_slots, gamma_slots


# ══════════════════════════════════════════════════════════════
# RUN ON ALL SESSIONS
# ══════════════════════════════════════════════════════════════

def run_hmm_all_sessions(df, model, emission_cols, sessions, mode="progressive"):
    """
    Run HMM forward + backward on all sessions.

    Parameters
    ----------
    df : pd.DataFrame
    model : fitted GaussianHMM
    emission_cols : list of str
    sessions : list of session dates
    mode : "progressive" (Mode B) or "full" (Mode A)

    Returns
    -------
    results : dict with keys:
        alpha : [N, 2] — forward posteriors for all bars
        kl_a : [N] — Mode A KL (full backward), always computed for comparison
        kl_b_slots : [N, 4] — Mode B KL per tensor slot (if mode includes progressive)
        gamma_b_slots : [N, 4, 2] — Mode B gamma per tensor slot
    """
    N = len(df)
    n_states = model.n_components

    alpha_all = np.zeros((N, n_states))
    kl_a_all = np.full(N, np.nan)

    # Always compute Mode B (it's cheap and needed for tensor)
    kl_b_slots = np.full((N, 4), np.nan)
    gamma_b_slots = np.zeros((N, 4, n_states))

    offset = 0
    for si, sess in enumerate(sessions):
        sm = df["SessionDate"] == sess
        sd = df[sm]
        X = sd[emission_cols].values
        act = sd["Active"].values
        T_s = len(X)

        # Mode A (always, for comparison)
        alpha_s, gamma_a_s, kl_a_s = mode_a_session(model, X, act)
        alpha_all[offset:offset + T_s] = alpha_s
        kl_a_all[offset:offset + T_s] = kl_a_s

        # Mode B (progressive backward)
        _, kl_b_s, gamma_b_s = mode_b_session(model, X, act)
        kl_b_slots[offset:offset + T_s] = kl_b_s
        gamma_b_slots[offset:offset + T_s] = gamma_b_s

        offset += T_s

        if (si + 1) % 200 == 0:
            print(f"    ...{si+1}/{len(sessions)} sessions")

    return {
        "alpha": alpha_all,
        "kl_a": kl_a_all,
        "kl_b_slots": kl_b_slots,
        "gamma_b_slots": gamma_b_slots,
    }


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def _kl_div(p, q, eps=1e-10):
    """KL(p || q) per row. Returns [T] array."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q), axis=-1)


def _kl_div_single(p, q, eps=1e-10):
    """KL(p || q) for single pair of distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q))


def get_train_sessions(sessions, train_frac=None):
    """Split sessions into train/val by fraction."""
    if train_frac is None:
        train_frac = PARAMS["hmm_train_frac"]
    n_train = int(len(sessions) * train_frac)
    return sessions[:n_train], sessions[n_train:]
