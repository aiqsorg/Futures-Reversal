# region imports
from AlgorithmImports import *
# endregion

"""
═══════════════════════════════════════════════════════════════
FILE 2: ma_signal.py — MA Crossover Detection
═══════════════════════════════════════════════════════════════
Detects bullish and bearish 8/20 SMA crossovers per session.
"""
import numpy as np
import pandas as pd
from config import PARAMS

def compute_crossovers(df, fast=None, slow=None):
    """
    Detect bull and bear MA crossovers across all sessions.
    

    Parameters
    ----------
    df : pd.DataFrame — must have Close, SessionDate
    fast, slow : int — MA periods

    Returns
    -------
    df : pd.DataFrame with MA_bull_x, MA_bear_x, MA_signal columns
    """
    if fast is None: fast = PARAMS["ma_fast"]
    if slow is None: slow = PARAMS["ma_slow"]

    df = df.copy()
    df["MA_bull_x"] = False
    df["MA_bear_x"] = False
    df["MA_signal"] = 0

    sessions = sorted(df["SessionDate"].unique())
    n_bull = n_bear = 0

    for sess in sessions:
        sm = df["SessionDate"] == sess
        sc = df.loc[sm, "Close"]
        if len(sc) < slow + 1:
            continue

        fma = sc.rolling(fast, min_periods=fast).mean()
        sma = sc.rolling(slow, min_periods=slow).mean()

        sig = pd.Series(0, index=sc.index)
        v = fma.notna() & sma.notna()
        sig[v] = np.where(fma[v] > sma[v], 1, -1)

        ps = sig.shift(1)
        xo = (sig != ps) & (sig != 0) & (ps != 0)
        xo.iloc[0] = False

        df.loc[sm, "MA_signal"] = sig
        df.loc[sm, "MA_bull_x"] = xo & (sig == 1)
        df.loc[sm, "MA_bear_x"] = xo & (sig == -1)

        n_bull += (xo & (sig == 1)).sum()
        n_bear += (xo & (sig == -1)).sum()

    print(f"  MA crossovers ({fast}/{slow}): {n_bull} bull, {n_bear} bear "
          f"({(n_bull+n_bear)/len(sessions):.1f}/session)")

    return df
