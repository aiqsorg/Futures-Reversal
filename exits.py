# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
"""
exits.py — Dynamic Exit Strategies
=====================================
Called by backtest.simulate_trades(). Each exit function takes the
same signature and returns a trade_record dict with full path logging.

Exit strategies:
  fixed      — hold N bars, exit at close
  atr        — TP/SL based on entry-bar ATR
  tensor     — exit on regime flip, stress spike, or micro change
  mae_stop   — exit if drawdown exceeds historical winner MAE
  trailing   — trailing stop after MFE threshold reached
  combined   — ATR envelope + tensor monitoring
  adaptive   — early exit for quick movers, extended hold for runners

All exits log bar-by-bar tensor state during the hold for post-hoc analysis.
"""
import numpy as np
from config import PARAMS


# ══════════════════════════════════════════════════════════════
# CORE: SIMULATE ONE TRADE (called by backtest.py)
# ══════════════════════════════════════════════════════════════

def simulate_trade(df, entry_loc, direction, cost, exit_cfg,
                   all_tensors=None, entry_row=None):
    """
    Simulate one trade from entry to exit with full diagnostics.

    Parameters
    ----------
    df : pd.DataFrame — full bar data
    entry_loc : int — integer location of entry bar in df
    direction : int — +1 long, -1 short
    cost : float — round-trip cost in points
    exit_cfg : dict — from config.EXIT_CONFIGS
    all_tensors : np.ndarray [N, 5, 11] or None
    entry_row : pd.Series — the entry bar's data (for pre-computed features)

    Returns
    -------
    record : dict — full trade record with path, MAE/MFE, tensor states, context
    """
    entry_close = df.iloc[entry_loc]["Close"]
    entry_sess = df.iloc[entry_loc]["SessionDate"]
    entry_atr = df.iloc[entry_loc].get("ATR", np.nan)
    entry_time = df.index[entry_loc]

    exit_type = exit_cfg["type"]
    max_hold = exit_cfg.get("max_bars", exit_cfg.get("bars", PARAMS["hold_bars"]))

    # ── Collect bar-by-bar path ──
    bars = []
    exit_bar = None
    exit_reason = "max_hold"

    # State tracking for adaptive/trailing exits
    peak_pnl = 0.0
    trail_active = False

    for b in range(1, max_hold + 1):
        if entry_loc + b >= len(df):
            exit_reason = "data_end"; break
        if df.iloc[entry_loc + b]["SessionDate"] != entry_sess:
            exit_reason = "session_end"; break

        bar_data = df.iloc[entry_loc + b]
        bc = bar_data["Close"]
        bh = bar_data["High"]
        bl = bar_data["Low"]

        unrealized = (bc - entry_close) * direction
        bar_best = ((bh - entry_close) if direction == 1 else (entry_close - bl)) * abs(direction)
        bar_worst = ((bl - entry_close) if direction == 1 else (entry_close - bh)) * abs(direction)

        peak_pnl = max(peak_pnl, unrealized, bar_best)

        # ── Tensor state during hold (if available) ──
        tensor_state = None
        hold_alpha_regime = None
        hold_stress = None
        hold_micro_gk = None

        if all_tensors is not None and entry_loc + b < len(all_tensors):
            tensor_state = all_tensors[entry_loc + b]  # [5, 11]
            hold_alpha_regime = 1 if tensor_state[3, 3] > 0.5 else 0  # current slot alpha
            hold_micro_gk = tensor_state[3, 7]  # current slot micro GK

        if "stress_B" in bar_data.index:
            hold_stress = bar_data["stress_B"]

        bar_record = {
            "bar": b, "close": bc, "high": bh, "low": bl,
            "unrealized": unrealized, "best": bar_best, "worst": bar_worst,
            "peak_pnl": peak_pnl,
            "alpha_regime": hold_alpha_regime,
            "stress": hold_stress,
            "micro_gk": hold_micro_gk,
        }
        bars.append(bar_record)

        # ── EXIT LOGIC ──
        exited = _check_exit(
            exit_type, exit_cfg, b, unrealized, peak_pnl,
            entry_atr, bar_data, entry_loc, df, all_tensors,
            direction, trail_active
        )

        if exited:
            exit_reason = exited
            break

        # Update trailing state
        if exit_type in ("trailing", "adaptive", "combined"):
            trail_thresh = exit_cfg.get("trail_activation", 0.5)
            avg_mfe = exit_cfg.get("avg_mfe", 10.0)  # needs to be set from historical data
            if peak_pnl > trail_thresh * avg_mfe:
                trail_active = True

    if len(bars) == 0:
        return None

    # ── Compute trade metrics ──
    unr = [b["unrealized"] for b in bars]
    bst = [b["best"] for b in bars]
    wst = [b["worst"] for b in bars]

    final = unr[-1]
    mfe = max(max(unr), max(bst))
    mae = min(min(unr), min(wst))
    eff = final / mfe if mfe > 0 else 0.0
    mfe_bar = np.argmax([max(u, b) for u, b in zip(unr, bst)]) + 1

    # ── Entry context ──
    pre_start = max(0, entry_loc - 5)
    pre_bars = df.iloc[pre_start:entry_loc]
    pre_momentum = 0.0
    pre_vol = 0.0
    if len(pre_bars) >= 2:
        pre_momentum = (pre_bars["Close"].iloc[-1] - pre_bars["Close"].iloc[0]) / \
                       pre_bars["Close"].iloc[0] * direction
        pre_vol = pre_bars["GK"].mean() if "GK" in pre_bars.columns else 0.0

    # ── Entry tensor snapshot ──
    entry_tensor_features = {}
    if entry_row is not None:
        for col in entry_row.index:
            if col.startswith("T_") or col in ("stress_A", "stress_B"):
                val = entry_row[col]
                if not isinstance(val, (str, bool)):
                    entry_tensor_features[col] = val

    # ── Session position ──
    sess_bars = df[df["SessionDate"] == entry_sess]
    bar_in_session = entry_loc - df.index.get_loc(sess_bars.index[0])
    session_pct = bar_in_session / len(sess_bars) if len(sess_bars) > 0 else 0.5

    # ── Regime during hold summary ──
    hold_regimes = [b["alpha_regime"] for b in bars if b["alpha_regime"] is not None]
    regime_flipped_during = False
    if len(hold_regimes) >= 2:
        regime_flipped_during = any(hold_regimes[i] != hold_regimes[0]
                                     for i in range(1, len(hold_regimes)))

    hold_stresses = [b["stress"] for b in bars if b["stress"] is not None]
    max_stress_during = max(hold_stresses) if hold_stresses else np.nan

    record = {
        # Identity
        "entry_time": entry_time,
        "entry_close": entry_close,
        "entry_atr": entry_atr,
        "session_date": entry_sess,
        "session_pct": session_pct,

        # Outcome
        "final_pnl": final,
        "net_pnl": final - cost,
        "mfe": mfe,
        "mae": mae,
        "entry_eff": eff,
        "mfe_bar": mfe_bar,
        "bars_held": len(bars),
        "exit_reason": exit_reason,
        "winner": final > cost,

        # Path
        "path": unr,
        "path_best": bst,
        "path_worst": wst,

        # Context
        "pre_momentum": pre_momentum,
        "pre_vol": pre_vol,
        "direction": direction,

        # Tensor at entry
        "entry_features": entry_tensor_features,

        # Dynamics during hold
        "regime_flipped_during": regime_flipped_during,
        "max_stress_during": max_stress_during,
        "hold_regimes": hold_regimes,
        "hold_stresses": hold_stresses,

        # Sizing (filled by caller)
        "conviction": entry_row.get("conviction", np.nan) if entry_row is not None else np.nan,
        "n_contracts": entry_row.get("n_contracts", 1) if entry_row is not None else 1,
    }

    return record


# ══════════════════════════════════════════════════════════════
# EXIT CHECKS
# ══════════════════════════════════════════════════════════════

def _check_exit(exit_type, cfg, bar_num, unrealized, peak_pnl,
                entry_atr, bar_data, entry_loc, df, all_tensors,
                direction, trail_active):
    """
    Check if exit condition is met. Returns exit_reason string or None.
    """
    # ── Fixed ──
    if exit_type == "fixed":
        if bar_num >= cfg["bars"]:
            return "fixed_hold"

    # ── ATR-based ──
    elif exit_type == "atr":
        if not np.isnan(entry_atr) and entry_atr > 0:
            if unrealized >= entry_atr * cfg["tp_mult"]:
                return "atr_tp"
            if unrealized <= -entry_atr * cfg["sl_mult"]:
                return "atr_sl"

    # ── Tensor-monitored ──
    elif exit_type == "tensor":
        # Regime flip
        if cfg.get("exit_on_regime_flip", True):
            entry_regime = 1 if df.iloc[entry_loc].get("Alpha_P1", 0.5) > 0.5 else 0
            curr_regime = 1 if bar_data.get("Alpha_P1", 0.5) > 0.5 else 0
            if curr_regime != entry_regime:
                return "regime_flip"

        # Stress spike
        if cfg.get("exit_on_stress_spike", True):
            curr_stress = bar_data.get("stress_B", 0)
            if curr_stress > cfg.get("stress_exit_threshold", 0.05):
                return "stress_spike"

        # Micro GK spike (optional)
        if cfg.get("exit_on_micro_spike", False):
            # If micro GK at current bar is > 2x the entry bar's micro GK
            entry_gk = df.iloc[entry_loc].get("micro_gk", 0)
            curr_gk = bar_data.get("micro_gk", 0)
            if entry_gk > 0 and curr_gk > entry_gk * 2.0:
                return "micro_spike"

    # ── MAE-informed stop ──
    elif exit_type == "mae":
        max_mae = cfg.get("max_mae_pts", None)
        if max_mae is not None and unrealized <= -abs(max_mae):
            return "mae_stop"

    # ── Trailing stop ──
    elif exit_type == "trailing":
        if trail_active:
            trail_pct = cfg.get("trail_pct", 0.4)
            trail_level = peak_pnl * (1 - trail_pct)
            if unrealized <= trail_level and peak_pnl > 0:
                return "trailing_stop"

    # ── Adaptive (combines time-based + trailing + MAE) ──
    elif exit_type == "adaptive":
        # Quick profit take: if > 1.5x avg profit in first 2 bars, take it
        quick_thresh = cfg.get("quick_profit_mult", 1.5)
        avg_profit = cfg.get("avg_profit", 5.0)
        if bar_num <= 2 and unrealized > quick_thresh * avg_profit:
            return "quick_profit"

        # MAE stop
        max_mae = cfg.get("max_mae_pts", None)
        if max_mae is not None and unrealized <= -abs(max_mae):
            return "mae_stop"

        # Trailing after threshold
        if trail_active:
            trail_pct = cfg.get("trail_pct", 0.4)
            trail_level = peak_pnl * (1 - trail_pct)
            if unrealized <= trail_level and peak_pnl > 0:
                return "trailing_stop"

    # ── Combined (ATR + tensor) ──
    elif exit_type == "combined":
        # ATR envelope
        if not np.isnan(entry_atr) and entry_atr > 0:
            if unrealized >= entry_atr * cfg.get("atr_tp_mult", 1.5):
                return "atr_tp"
            if unrealized <= -entry_atr * cfg.get("atr_sl_mult", 2.5):
                return "atr_sl"

        # Regime flip
        if cfg.get("exit_on_regime_flip", True):
            entry_regime = 1 if df.iloc[entry_loc].get("Alpha_P1", 0.5) > 0.5 else 0
            curr_regime = 1 if bar_data.get("Alpha_P1", 0.5) > 0.5 else 0
            if curr_regime != entry_regime:
                return "regime_flip"

        # Stress spike
        if cfg.get("exit_on_stress_spike", True):
            curr_stress = bar_data.get("stress_B", 0)
            if curr_stress > cfg.get("stress_exit_threshold", 0.05):
                return "stress_spike"

    return None  # no exit triggered


# ══════════════════════════════════════════════════════════════
# HELPER: compute historical MAE/MFE thresholds from past trades
# ══════════════════════════════════════════════════════════════

def compute_exit_thresholds(historical_trades, percentile=75):
    """
    Compute data-driven exit thresholds from historical trade data.
    Use this to set mae_stop, trailing activation, quick profit levels.

    Parameters
    ----------
    historical_trades : pd.DataFrame — from prior backtest run
    percentile : int — which percentile of winner MAE to use as stop

    Returns
    -------
    thresholds : dict
    """
    if len(historical_trades) < 20:
        return {}

    winners = historical_trades[historical_trades["winner"]]
    losers = historical_trades[~historical_trades["winner"]]

    winner_mae = winners["mae"].values if len(winners) > 0 else np.array([0])
    all_mfe = historical_trades["mfe"].values

    thresholds = {
        "max_mae_pts": abs(np.percentile(winner_mae, 100 - percentile)),
        "avg_mfe": np.mean(all_mfe),
        "avg_profit": winners["final_pnl"].mean() if len(winners) > 0 else 1.0,
        "median_mfe": np.median(all_mfe),
    }

    print(f"  Exit thresholds computed from {len(historical_trades)} trades:")
    print(f"    MAE stop (P{percentile} winner MAE): {thresholds['max_mae_pts']:.2f} pts")
    print(f"    Avg MFE: {thresholds['avg_mfe']:.2f} pts")
    print(f"    Avg winner profit: {thresholds['avg_profit']:.2f} pts")

    return thresholds


def update_exit_config(exit_cfg, thresholds):
    """
    Update an exit config dict with computed thresholds.
    """
    cfg = exit_cfg.copy()
    if "max_mae_pts" in thresholds:
        cfg["max_mae_pts"] = thresholds["max_mae_pts"]
    if "avg_mfe" in thresholds:
        cfg["avg_mfe"] = thresholds["avg_mfe"]
    if "avg_profit" in thresholds:
        cfg["avg_profit"] = thresholds["avg_profit"]
    return cfg
