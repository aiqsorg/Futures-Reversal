# region imports
from AlgorithmImports import *
# endregion

# Your New Python File
import numpy as np
import pandas as pd
from config import PARAMS, GATE_CONFIGS

"""
═══════════════════════════════════════════════════════════════
FILE 3: decision.py — Gate Application + Entry Generation
═══════════════════════════════════════════════════════════════
Applies gate configs to crossover bars, generates entry signals.
"""
import pandas as pd
from config import GATE_CONFIGS


def get_crossover_bars(df, instrument_cfg, sessions=None):
    """
    Extract crossover bars with forward returns and EOD filter.

    Parameters
    ----------
    df : pd.DataFrame — with MA_bull_x, MA_bear_x, fwd_7, tensor features
    instrument_cfg : dict — from config.INSTRUMENTS
    sessions : list — session dates to include (default: skip warmup)

    Returns
    -------
    xo : pd.DataFrame — crossover bars with ret_pts column
    """
    if sessions is None:
        all_sess = sorted(df["SessionDate"].unique())
        warmup = PARAMS["warmup_sessions"]
        sessions = all_sess[warmup:]

    cross_col = "MA_bear_x" if instrument_cfg["cross"] == "bear_x" else "MA_bull_x"
    direction = instrument_cfg["direction"]
    hold = PARAMS["hold_bars"]
    fwd_col = f"fwd_{hold}"

    mask = (
        df["SessionDate"].isin(sessions) &
        df["Active"] &
        df[fwd_col].notna() &
        (df.index.hour < PARAMS["no_trade_after_hour"])
    )
    va = df[mask].copy()
    xo = va[va[cross_col] == True].copy()
    xo["ret_pts"] = xo[fwd_col] * xo["Close"] * direction

    return xo


def apply_gate(xo, gate_name):
    """
    Apply a named gate config to crossover bars.

    Parameters
    ----------
    xo : pd.DataFrame — crossover bars
    gate_name : str — key in config.GATE_CONFIGS

    Returns
    -------
    mask : pd.Series of bool
    """
    gate_fn = GATE_CONFIGS[gate_name]
    return gate_fn(xo)


def generate_entries(df, instrument_cfg, gate_name=None):
    """
    Full entry generation: crossovers + EOD filter + gates.

    Parameters
    ----------
    df : pd.DataFrame — fully featured
    instrument_cfg : dict
    gate_name : str or None — if None, uses instrument default

    Returns
    -------
    entries : pd.DataFrame — gated crossover bars
    xo_all : pd.DataFrame — all crossover bars (for comparison)
    """
    if gate_name is None:
        gate_name = instrument_cfg["gate"]

    xo = get_crossover_bars(df, instrument_cfg)
    gate_mask = apply_gate(xo, gate_name)
    entries = xo[gate_mask].copy()

    print(f"  Entries: {len(entries)}/{len(xo)} crossovers "
          f"({gate_name}, {len(entries)/max(len(xo),1)*100:.0f}% pass rate)")

    return entries, xo
