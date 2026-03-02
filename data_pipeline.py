"""
data_pipeline.py — Multi-Instrument Data Infrastructure
=========================================================
Pulls raw futures data for any instrument, builds 5-min RTH
dataframe with all base features, micro features, and multiple
emission normalizations for HMM testing.

Pure functions. No global state. Config comes from config.py.
"""
from AlgorithmImports import *
import numpy as np
import pandas as pd
from config import PARAMS, MICRO_FEATURES


# ══════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def compute_gk(df):
    """Garman-Klass volatility estimator. Works on any df with OHLC."""
    return np.sqrt(np.maximum(
        0.5 * (np.log(df["High"] / df["Low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"])) ** 2,
        0.0
    ))


def pull_data(qb, instrument_cfg, lookback_days=None):
    """
    Pull raw 1-min data for any instrument.

    Parameters
    ----------
    qb : QuantBook
    instrument_cfg : dict — from config.INSTRUMENTS[ticker]
    lookback_days : int or None — override config default

    Returns
    -------
    raw_df : pd.DataFrame — 1-min OHLCV bars, datetime indexed
    has_quotes : bool — whether quote data is available
    quote_df : pd.DataFrame or None — 1-min quote bars if available
    """
    if lookback_days is None:
        lookback_days = PARAMS["lookback_days"]

    sym = qb.add_future(instrument_cfg["future"], Resolution.MINUTE)
    sym.set_filter(0, 90)
    sym.data_normalization_mode = DataNormalizationMode.RAW

    print(f"  Pulling trade data ({lookback_days} days)...")
    trade_hist = qb.history(sym.symbol, timedelta(days=lookback_days), Resolution.MINUTE)

    raw_df = trade_hist[["open", "high", "low", "close", "volume"]].copy()
    raw_df.index = raw_df.index.get_level_values("time")
    raw_df.columns = ["Open", "High", "Low", "Close", "Volume"]
    print(f"    Trade bars: {len(raw_df)}")

    # Try quote data (may not be available for all instruments/periods)
    has_quotes = False
    quote_df = None
    try:
        print(f"  Pulling quote data...")
        quote_hist = qb.history(
            QuoteBar, sym.symbol, timedelta(days=lookback_days), Resolution.MINUTE
        )
        if len(quote_hist) > 0:
            quote_df = quote_hist[["bidclose", "askclose", "bidsize", "asksize"]].copy()
            quote_df.index = quote_df.index.get_level_values("time")
            quote_df.columns = ["BidClose", "AskClose", "BidSize", "AskSize"]
            has_quotes = True
            print(f"    Quote bars: {len(quote_df)}")
    except Exception as e:
        print(f"    Quote data not available: {e}")

    return raw_df, has_quotes, quote_df


def build_5min_bars(raw_df, rth_start=None, rth_end=None,
                    warmup_bars=None, quote_df=None):
    """
    Build 5-min RTH dataframe with all base features.

    Parameters
    ----------
    raw_df : pd.DataFrame — 1-min OHLCV from pull_data()
    rth_start, rth_end : str — RTH window (from instrument config)
    warmup_bars : int — bars to skip at session start
    quote_df : pd.DataFrame or None — 1-min quote bars

    Returns
    -------
    df : pd.DataFrame — 5-min bars with all base features
    """
    if rth_start is None: rth_start = PARAMS["rth_start"]
    if rth_end is None: rth_end = PARAMS["rth_end"]
    if warmup_bars is None: warmup_bars = PARAMS["warmup_bars"]

    # ── Resample to 5-min ──
    rth_1m = raw_df.between_time(rth_start, rth_end)
    df = rth_1m.resample("5min").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    # ── Quote data if available ──
    if quote_df is not None:
        qdf_rth = quote_df.between_time(rth_start, rth_end)
        qdf_5m = qdf_rth.resample("5min").agg({
            "BidClose": "last", "AskClose": "last",
            "BidSize": "last", "AskSize": "last",
        }).dropna()
        df = df.join(qdf_5m, how="left")

    # ── Session indexing ──
    df["SessionDate"] = df.index.date
    df["_bn"] = df.groupby("SessionDate").cumcount()
    df = df[df["_bn"] >= warmup_bars].drop(columns=["_bn"])
    df["BarIndex"] = df.groupby("SessionDate").cumcount()

    # ── Core features ──
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["GK"] = compute_gk(df)

    # Activity mask
    df["Active"] = (df["High"] != df["Low"]) & (df["GK"] > PARAMS["gk_floor"])

    # Drop first bar per session (no valid return)
    first_bars = df.groupby("SessionDate").head(1).index
    df.loc[first_bars, "LogReturn"] = np.nan
    df.dropna(subset=["LogReturn", "GK"], inplace=True)

    # ── EWMA normalizations for HMM emissions ──
    span = PARAMS["hmm_ewma_span"]

    # LR_norm (always built)
    df["LR_norm"] = (
        (df["LogReturn"] - df["LogReturn"].ewm(span=span).mean())
        / df["LogReturn"].ewm(span=span).std()
    )

    # GK_norm (always built)
    df["GK_norm"] = (
        (df["GK"] - df["GK"].ewm(span=span).mean())
        / df["GK"].ewm(span=span).std()
    )

    # RVol_norm (realized vol = rolling std of returns)
    df["RVol"] = df["LogReturn"].rolling(20, min_periods=10).std()
    df["RVol_norm"] = (
        (df["RVol"] - df["RVol"].ewm(span=span).mean())
        / df["RVol"].ewm(span=span).std()
    )

    # Spread_norm (if quote data available)
    if "AskClose" in df.columns and "BidClose" in df.columns:
        df["Spread"] = df["AskClose"] - df["BidClose"]
        df["Spread_norm"] = (
            (df["Spread"] - df["Spread"].ewm(span=span).mean())
            / df["Spread"].ewm(span=span).std()
        )

    df.dropna(subset=["LR_norm", "GK_norm"], inplace=True)

    return df


def build_micro_features(df):
    """
    Add micro features for tensor columns 7-10.
    Works with or without quote data (graceful fallback).

    Parameters
    ----------
    df : pd.DataFrame — from build_5min_bars()

    Returns
    -------
    df : pd.DataFrame — with micro feature columns added
    """
    df = df.copy()

    # GK vol of the bar
    df["micro_gk"] = df["GK"]

    # Normalized bar range
    bar_range = df["High"] - df["Low"]
    df["micro_range"] = bar_range / df["Close"]

    # Normalized volume
    vol_ewma = df["Volume"].ewm(span=50).mean()
    df["micro_vol"] = df["Volume"] / vol_ewma.replace(0, 1)

    # Close position within bar
    df["micro_cpos"] = np.where(
        bar_range > 0,
        (df["Close"] - df["Low"]) / bar_range,
        0.5
    )

    # Fill any NaN
    for col in MICRO_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = 0.0

    return df


def build_atr(df, period=14):
    """
    Compute session-aware ATR for exit strategies.

    Parameters
    ----------
    df : pd.DataFrame — must have High, Low, Close, SessionDate

    Returns
    -------
    df : pd.DataFrame — with ATR column added
    """
    df = df.copy()
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            np.abs(df["High"] - df["Close"].shift(1)),
            np.abs(df["Low"] - df["Close"].shift(1))
        )
    )

    # Session-aware: compute ATR within each session
    df["ATR"] = np.nan
    for sess in df["SessionDate"].unique():
        idx = df.index[df["SessionDate"] == sess]
        tr_sess = df.loc[idx, "TR"]
        atr_sess = tr_sess.ewm(span=period, min_periods=min(period, len(tr_sess))).mean()
        df.loc[idx, "ATR"] = atr_sess

    df.drop(columns=["TR"], inplace=True)
    return df


def compute_forward_returns(df, hold_bars=None):
    """
    Compute forward log returns for backtest evaluation.

    Parameters
    ----------
    df : pd.DataFrame — must have LogReturn, SessionDate
    hold_bars : int — holding period in bars

    Returns
    -------
    df : pd.DataFrame — with fwd_{hold_bars} column
    """
    if hold_bars is None:
        hold_bars = PARAMS["hold_bars"]

    df = df.copy()
    col = f"fwd_{hold_bars}"
    df[col] = np.nan

    for sess in df["SessionDate"].unique():
        idx = df.index[df["SessionDate"] == sess]
        lr = df.loc[idx, "LogReturn"].values
        for i in range(len(idx) - hold_bars):
            df.loc[idx[i], col] = lr[i + 1:i + 1 + hold_bars].sum()

    return df


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════

def run_pipeline(qb, instrument_cfg, lookback_days=None):
    """
    Full data pipeline for one instrument.

    Parameters
    ----------
    qb : QuantBook
    instrument_cfg : dict — from config.INSTRUMENTS[ticker]
    lookback_days : int or None

    Returns
    -------
    df : pd.DataFrame — ready for HMM and downstream
    sessions : list — sorted session dates
    has_quotes : bool
    """
    rth_s = instrument_cfg.get("rth_start", PARAMS["rth_start"])
    rth_e = instrument_cfg.get("rth_end", PARAMS["rth_end"])

    raw_df, has_quotes, quote_df = pull_data(qb, instrument_cfg, lookback_days)

    df = build_5min_bars(raw_df, rth_start=rth_s, rth_end=rth_e,
                         quote_df=quote_df)
    df = build_micro_features(df)
    df = build_atr(df)
    df = compute_forward_returns(df)

    sessions = sorted(df["SessionDate"].unique())

    print(f"  Pipeline complete:")
    print(f"    {len(df)} bars, {len(sessions)} sessions")
    print(f"    Range: {sessions[0]} to {sessions[-1]}")
    print(f"    Active: {df['Active'].mean()*100:.1f}%")
    print(f"    Quotes: {'yes' if has_quotes else 'no'}")

    return df, sessions, has_quotes
