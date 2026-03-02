"""
config.py — Central Configuration
====================================
Single source of truth for all parameters, instrument configs,
gate definitions, exit strategies, and sweep configurations.

Every other module imports from here. Change behavior by editing
this file, not by rewriting pipeline code.
"""
from AlgorithmImports import *

# ══════════════════════════════════════════════════════════════
# 1. GLOBAL PARAMETERS
# ══════════════════════════════════════════════════════════════

PARAMS = {
    # Data
    "lookback_days": 1500,
    "rth_start": "09:30",
    "rth_end": "15:55",
    "bar_size": "5min",
    "warmup_bars": 6,
    "gk_floor": 0.00005,
    "hmm_ewma_span": 390,

    # HMM
    "hmm_states": 2,
    "hmm_iter": 50,
    "hmm_train_frac": 0.40,
    "hmm_random_state": 42,

    # Branch 2 (stress EWMA)
    "ewma_fast": 3,
    "ewma_slow": 15,
    "stress_threshold": 0.02,

    # Entry gates
    "kl_slope_threshold": 0.0,
    "alpha_std_threshold": 0.05,
    "no_trade_after_hour": 15,

    # MA
    "ma_fast": 8,
    "ma_slow": 20,

    # Tensor
    "tensor_slots": 5,
    "tensor_cols": 11,
    # Layout per slot: [gamma(2), alpha(2), proj(2), kl(1), micro(4)]
    "tensor_gamma_idx": (0, 2),
    "tensor_alpha_idx": (2, 4),
    "tensor_proj_idx": (4, 6),
    "tensor_kl_idx": 6,
    "tensor_micro_idx": (7, 11),

    # Exit (defaults)
    "hold_bars": 7,
    "max_hold_bars": 12,

    # Sizing
    "conviction_weights": {
        "stress": 0.30,
        "age": 0.25,
        "alpha": 0.20,
        "kl_slope": 0.25,
        # NQ-calibrated adaptive (from diagnostics: MFE=32.5, winner MAE=-16.4)
    "adaptive_nq": {
        "type": "adaptive",
        "max_bars": 10,
        "quick_profit_mult": 1.5,
        "avg_profit": 32.0,
        "max_mae_pts": 35.0,      # P75 winner MAE on NQ
        "trail_activation": 0.5,
        "trail_pct": 0.4,
        "avg_mfe": 32.5,
    },
    # ES-calibrated adaptive (from diagnostics: MFE=7.2, winner MAE=-3.5)
    "adaptive_es": {
        "type": "adaptive",
        "max_bars": 10,
        "quick_profit_mult": 1.5,
        "avg_profit": 7.0,
        "max_mae_pts": 8.0,
        "trail_activation": 0.5,
        "trail_pct": 0.4,
        "avg_mfe": 7.2,
    },
    # YM-calibrated adaptive (from diagnostics: MFE=60, winner MAE=-57)
    "adaptive_ym": {
        "type": "adaptive",
        "max_bars": 12,
        "quick_profit_mult": 1.0,
        "avg_profit": 73.0,
        "max_mae_pts": 80.0,       # YM winners survive deep MAE
        "trail_activation": 0.4,
        "trail_pct": 0.35,
        "avg_mfe": 60.0,
    },
},
    "contract_tiers": [
        {"threshold": 0.7, "contracts": 3},
        {"threshold": 0.5, "contracts": 2},
        {"threshold": 0.0, "contracts": 1},
    ],

    # Validation
    "warmup_sessions": 60,
    "bootstrap_n": 10000,
}


# ══════════════════════════════════════════════════════════════
# 2. INSTRUMENT CONFIGURATIONS
# ══════════════════════════════════════════════════════════════

INSTRUMENTS = {
    "NQ": {
        "future": Futures.Indices.NASDAQ100EMini,
        "cross": "bull_x",
        "direction": 1,
        "gate": "stress",
        "cost_pts": 1.10,
        "point_value": 20.0,
        "rth_start": "09:30",
        "rth_end": "15:55",
        "notes": "Momentum long. Highest retail/systematic participation.",
    },
    "ES": {
        "future": Futures.Indices.SP500EMini,
        "cross": "bear_x",
        "direction": 1,
        "gate": "kl+alpha+stress",
        "cost_pts": 0.58,
        "point_value": 50.0,
        "rth_start": "09:30",
        "rth_end": "15:55",
        "notes": "Contrarian long. Regime-dependent.",
    },
    "YM": {
        "future": Futures.Indices.Dow30EMini,
        "cross": "bull_x",
        "direction": -1,
        "gate": "kl+alpha+stress",
        "cost_pts": 2.00,
        "point_value": 5.0,
        "rth_start": "09:30",
        "rth_end": "15:55",
        "notes": "Mean-reversion short. Degenerate HMM [0.393, 0.803].",
    },
    "RTY": {
        "future": Futures.Indices.Russell2000EMini,
        "cross": "bear_x",
        "direction": 1,
        "gate": "kl+alpha+stress",
        "cost_pts": 0.40,
        "point_value": 50.0,
        "rth_start": "09:30",
        "rth_end": "15:55",
        "notes": "Contrarian long. Best quarterly consistency.",
    },
    # Dead instruments — kept for hypothesis testing
    "CL": {
        "future": Futures.Energies.CrudeOilWTI,
        "cross": "bear_x",
        "direction": -1,
        "gate": "kl+alpha+stress",
        "cost_pts": 0.04,
        "point_value": 1000.0,
        "rth_start": "09:00",
        "rth_end": "14:25",
        "notes": "DEAD. Commercial hedger dominated. No edge expected.",
    },
    "GC": {
        "future": Futures.Metals.Gold,
        "cross": "bull_x",
        "direction": 1,
        "gate": "kl+alpha+stress",
        "cost_pts": 0.40,
        "point_value": 100.0,
        "rth_start": "09:00",
        "rth_end": "14:25",
        "notes": "DEAD. Macro fund dominated. No edge expected.",
    },
}


# ══════════════════════════════════════════════════════════════
# 3. HMM EMISSION CONFIGURATIONS
# ══════════════════════════════════════════════════════════════
# Each config specifies which normalized columns to feed the HMM.
# data_pipeline.py builds ALL possible columns; HMM uses the subset.

EMISSION_CONFIGS = {
    "lr+gk": {
        "columns": ["LR_norm", "GK_norm"],
        "notes": "Default. Log returns + Garman-Klass vol.",
    },
    "gk_only": {
        "columns": ["GK_norm"],
        "notes": "Volatility only. Tests if returns add noise.",
    },
    "lr_only": {
        "columns": ["LR_norm"],
        "notes": "Returns only. Tests if GK is redundant.",
    },
    "lr+rvol": {
        "columns": ["LR_norm", "RVol_norm"],
        "notes": "Returns + realized vol (std of returns).",
    },
    "lr+gk+spread": {
        "columns": ["LR_norm", "GK_norm", "Spread_norm"],
        "notes": "3-emission with spread. Requires quote data.",
    },
}


# ══════════════════════════════════════════════════════════════
# 4. GATE CONFIGURATIONS
# ══════════════════════════════════════════════════════════════
# Each gate is a function: takes crossover df, returns boolean mask.
# Use T_ prefix for tensor-derived features, stress_B for Mode B stress.

def _gate_stress(xo):
    return xo["stress_B"] < PARAMS["stress_threshold"]

def _gate_kl_alpha_stress(xo):
    return (
        (xo["T_kl_slope_3s"] > PARAMS["kl_slope_threshold"]) &
        (xo["T_alpha_std"] < PARAMS["alpha_std_threshold"]) &
        (xo["stress_B"] < PARAMS["stress_threshold"])
    )

def _gate_kl_stress(xo):
    return (
        (xo["T_kl_slope_3s"] > PARAMS["kl_slope_threshold"]) &
        (xo["stress_B"] < PARAMS["stress_threshold"])
    )

def _gate_micro_stress(xo):
    """Micro GK variability + stress. From tensor feature scan."""
    return (
        (xo["T_micro_gk_std"] > xo["T_micro_gk_std"].median()) &
        (xo["stress_B"] < PARAMS["stress_threshold"])
    )

def _gate_argmax_trans(xo):
    """Argmax transition at current bar + stress. NQ's strongest single."""
    return (
        (xo["T_am_alpha_transition_now"] == 1) &
        (xo["stress_B"] < PARAMS["stress_threshold"])
    )

def _gate_argmax_majvol_kl(xo):
    """Majority volatile + kl_slope. YM's strongest combo."""
    return (
        (xo["T_am_alpha_majority_vol"] == 1) &
        (xo["T_kl_slope_3s"] > 0)
    )

def _gate_none(xo):
    """Ungated — crossover + EOD filter only."""
    import pandas as pd
    return pd.Series(True, index=xo.index)

GATE_CONFIGS = {
    "none":              _gate_none,
    "stress":            _gate_stress,
    "kl+alpha+stress":   _gate_kl_alpha_stress,
    "kl+stress":         _gate_kl_stress,
    "micro+stress":      _gate_micro_stress,
    "am_trans+stress":   _gate_argmax_trans,
    "am_majvol+kl":      _gate_argmax_majvol_kl,
}


# ══════════════════════════════════════════════════════════════
# 5. EXIT CONFIGURATIONS
# ══════════════════════════════════════════════════════════════

EXIT_CONFIGS = {
    "fixed_5": {
        "type": "fixed",
        "bars": 5,
    },
    "fixed_7": {
        "type": "fixed",
        "bars": 7,
    },
    "fixed_10": {
        "type": "fixed",
        "bars": 10,
    },
    "atr_15_20": {
        "type": "atr",
        "tp_mult": 1.5,
        "sl_mult": 2.0,
        "max_bars": 10,
        "atr_period": 14,
    },
    "atr_20_25": {
        "type": "atr",
        "tp_mult": 2.0,
        "sl_mult": 2.5,
        "max_bars": 10,
        "atr_period": 14,
    },
    "atr_10_15": {
        "type": "atr",
        "tp_mult": 1.0,
        "sl_mult": 1.5,
        "max_bars": 8,
        "atr_period": 14,
    },
    "tensor_monitored": {
        "type": "tensor",
        "max_bars": 10,
        "exit_on_regime_flip": True,
        "exit_on_stress_spike": True,
        "stress_exit_threshold": 0.05,
        "exit_on_micro_spike": False,
    },
    "mae_informed": {
        "type": "mae",
        "max_bars": 10,
        "mae_percentile": 75,
        # max_mae_pts computed from historical winner distribution
    },
    "trailing_mfe": {
        "type": "trailing",
        "max_bars": 10,
        "trail_activation": 0.5,   # activate after 50% of avg MFE
        "trail_pct": 0.4,          # give back 40% of peak
    },
    "combined_atr_tensor": {
        "type": "combined",
        "atr_tp_mult": 1.5,
        "atr_sl_mult": 2.5,
        "max_bars": 10,
        "atr_period": 14,
        "exit_on_regime_flip": True,
        "exit_on_stress_spike": True,
        "stress_exit_threshold": 0.05,
    },
}


# ══════════════════════════════════════════════════════════════
# 6. DIRECTION CONFIGS (for sweep)
# ══════════════════════════════════════════════════════════════
# Override instrument defaults to test all directions

DIRECTION_CONFIGS = {
    "long_bull":  {"cross": "bull_x", "direction":  1},
    "short_bull": {"cross": "bull_x", "direction": -1},
    "long_bear":  {"cross": "bear_x", "direction":  1},
    "short_bear": {"cross": "bear_x", "direction": -1},
}


# ══════════════════════════════════════════════════════════════
# 7. SWEEP CONFIGURATION
# ══════════════════════════════════════════════════════════════
# Define what the combinatoric sweep should test.
# Set to None to use instrument defaults.

SWEEP_CONFIG = {
    "instruments": ["NQ", "ES", "YM", "RTY"],
    "emissions": ["lr+gk", "gk_only"],          # None = default only
    "directions": None,                           # None = instrument default
    "gates": ["none", "stress", "kl+alpha+stress"],
    "exits": ["fixed_7", "atr_15_20", "tensor_monitored"],
}

# Full sweep (uncomment for intensive run):
# SWEEP_CONFIG = {
#     "instruments": ["NQ", "ES", "YM", "RTY", "CL", "GC"],
#     "emissions": list(EMISSION_CONFIGS.keys()),
#     "directions": list(DIRECTION_CONFIGS.keys()),
#     "gates": list(GATE_CONFIGS.keys()),
#     "exits": list(EXIT_CONFIGS.keys()),
# }


# ══════════════════════════════════════════════════════════════
# 8. MICRO FEATURE DEFINITIONS
# ══════════════════════════════════════════════════════════════
# Columns stored in tensor slots 7:11.
# Order matters — must match tensor builder.

MICRO_FEATURES = [
    "micro_gk",       # Garman-Klass vol of the bar
    "micro_range",     # (High - Low) / Close, normalized
    "micro_vol",       # Volume / EWMA(Volume), normalized
    "micro_cpos",      # (Close - Low) / (High - Low), close position
]

# With quote data available, these can be added:
MICRO_FEATURES_QUOTE = [
    "micro_spread",    # Ask - Bid
    "micro_imbalance", # |BidSize - AskSize| / (BidSize + AskSize)
]


# ══════════════════════════════════════════════════════════════
# 9. FEATURE NAMES (for diagnostics)
# ══════════════════════════════════════════════════════════════
# Canonical list of all tensor-derived features.
# diagnostics.py uses this for correlation scans.

TENSOR_FEATURE_NAMES = {
    # KL
    "kl_slope_3s", "kl_slope_4s", "kl_mean_3s", "kl_max_3s",
    "kl_min_3s", "kl_range_3s", "kl_slot2", "kl_accel",
    # Alpha continuous
    "alpha_std", "alpha_mean", "alpha_conv", "alpha_slope", "alpha_range",
    # Alpha argmax
    "am_alpha_s0", "am_alpha_s1", "am_alpha_s2", "am_alpha_s3",
    "am_alpha_sum", "am_alpha_unan_quiet", "am_alpha_unan_vol",
    "am_alpha_unanimous", "am_alpha_majority_quiet", "am_alpha_majority_vol",
    "am_alpha_flips", "am_alpha_last_flip", "am_alpha_first_flip",
    "am_alpha_pattern", "am_pattern_stable_quiet", "am_pattern_stable_vol",
    "am_pattern_entering_quiet", "am_pattern_entering_vol", "am_pattern_mixed",
    "am_alpha_current_quiet", "am_alpha_transition_now", "am_alpha_recent_trans",
    # Gamma
    "gamma_std", "gamma_mean", "gamma_range", "gamma_slope",
    "am_gamma_s0", "am_gamma_s1", "am_gamma_s2", "am_gamma_s3",
    "am_gamma_sum", "am_gamma_unan_quiet", "am_gamma_unan_vol",
    "am_gamma_flips", "am_gamma_pattern",
    # Disagreement
    "disagree_s0", "disagree_s1", "disagree_s2", "disagree_s3",
    "disagree_count", "disagree_any", "disagree_current",
    "disagree_trend", "disagree_pattern",
    "ag_direction_s0", "ag_direction_s1", "ag_direction_s2", "ag_direction_s3",
    "ag_direction_net",
    # Projection
    "proj_quiet_val", "proj_conv", "am_proj_quiet",
    "proj_agrees_alpha", "proj_agrees_gamma",
    "proj_continues_trend", "proj_error", "am_5slot_pattern",
    # Micro
    "micro_gk_mean", "micro_gk_std", "micro_gk_last", "micro_gk_slope",
    "micro_range_mean", "micro_range_std", "micro_range_last", "micro_range_slope",
    "micro_vol_mean", "micro_vol_std", "micro_vol_last", "micro_vol_slope",
    "micro_cpos_mean", "micro_cpos_std", "micro_cpos_last", "micro_cpos_slope",
    # Regime persistence
    "regime_age", "bars_since_flip", "current_streak",
    # Interactions
    "kl_x_alpha_std", "kl_x_flips", "kl_x_disagree",
    "conv_x_streak", "conv_x_unanimous",
    "kl_high_unstable", "kl_high_stable", "kl_low_stable",
    "disagree_at_transition", "vol_calming_quiet",
}
