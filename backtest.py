"""
backtest.py — Orchestrator
=============================
Runs the full pipeline for one or many instrument × config combos.
Returns structured results with full trade logs.

Usage in notebook:
    from backtest import run_single, run_sweep
    results = run_single(qb, "NQ")
    sweep = run_sweep(qb, config.SWEEP_CONFIG)
"""
from AlgorithmImports import *
import numpy as np
import pandas as pd

from config import PARAMS, INSTRUMENTS, EMISSION_CONFIGS, GATE_CONFIGS, \
    EXIT_CONFIGS, DIRECTION_CONFIGS, SWEEP_CONFIG
import data_pipeline as dp
import hmm_model as hm
import tensor as tn
import branch2 as b2
# NOTE: ma_signal, decision, sizing are in the combined file.
# In QC, these would be separate files. For now, import from the combined module.
# Adjust imports based on your actual file structure:
# import ma_signal as ma
# import decision as dec
# import sizing as sz


# ══════════════════════════════════════════════════════════════
# SINGLE INSTRUMENT RUN
# ══════════════════════════════════════════════════════════════

def run_single(qb, ticker, gate_name=None, emission_name="lr+gk",
               exit_name="fixed_7", direction_override=None,
               lookback_days=None, verbose=True):
    """
    Full pipeline for one instrument × one config.

    Parameters
    ----------
    qb : QuantBook
    ticker : str — key in config.INSTRUMENTS
    gate_name : str — key in config.GATE_CONFIGS (None = instrument default)
    emission_name : str — key in config.EMISSION_CONFIGS
    exit_name : str — key in config.EXIT_CONFIGS
    direction_override : dict or None — override cross/direction
    lookback_days : int or None
    verbose : bool

    Returns
    -------
    results : dict with keys:
        df : full dataframe
        sessions : list
        tensors : [N, 5, 11]
        xo : all crossover bars
        entries : gated entries
        trades : trade records with paths
        model : fitted HMM
        config : dict of config used
    """
    cfg = INSTRUMENTS[ticker].copy()
    em_cfg = EMISSION_CONFIGS[emission_name]

    if direction_override:
        cfg["cross"] = direction_override["cross"]
        cfg["direction"] = direction_override["direction"]
    if gate_name is None:
        gate_name = cfg["gate"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {ticker} | {emission_name} | {gate_name} | {exit_name}")
        print(f"  {cfg['cross']} {'long' if cfg['direction']==1 else 'short'}")
        print(f"{'='*60}")

    # ── 1. Data ──
    df, sessions, has_quotes = dp.run_pipeline(qb, cfg, lookback_days)

    # ── 2. HMM ──
    emission_cols = em_cfg["columns"]
    # Check all emission columns exist
    missing = [c for c in emission_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing emission columns {missing}, falling back to lr+gk")
        emission_cols = ["LR_norm", "GK_norm"]

    train_sess, val_sess = hm.get_train_sessions(sessions)
    model = hm.fit_hmm(df, emission_cols, train_sess)

    if verbose:
        print(f"  Running HMM on {len(sessions)} sessions...")
    hmm_results = hm.run_hmm_all_sessions(df, model, emission_cols, sessions)

    # Attach alpha to df
    df["Alpha_P1"] = hmm_results["alpha"][:, 1]

    # ── 3. Tensor ──
    if verbose:
        print(f"  Building tensors...")
    all_tensors = tn.build_tensors_all_sessions(df, hmm_results, sessions, model)
    features = tn.extract_features(all_tensors, hmm_results["alpha"], df["Active"].values)
    df = tn.attach_features_to_df(df, features, sessions)

    # ── 4. Branch 2 (stress) ──
    df = b2.attach_stress(df, hmm_results)

    # ── 5. MA crossovers ──
    # Import from combined module or separate file
    from ma_signal import compute_crossovers
    df = compute_crossovers(df)

    # ── 6. Forward returns ──
    df = dp.compute_forward_returns(df)

    # ── 7. Decision (entries) ──
    from decision import generate_entries, get_crossover_bars
    entries, xo = generate_entries(df, cfg, gate_name)

    # ── 8. Sizing ──
    from sizing import compute_conviction_column
    if len(entries) > 0:
        entries = compute_conviction_column(entries)

    # ── 9. Trade simulation ──
    trades = simulate_trades(df, entries, cfg, EXIT_CONFIGS[exit_name], all_tensors)

    # ── 10. Summary ──
    if verbose and len(trades) > 0:
        cost = cfg["cost_pts"]
        pv = cfg["point_value"]
        net = trades["final_pnl"].values - cost
        w, l = net[net > 0], net[net <= 0]
        pf = w.sum() / abs(l.sum()) if len(l) > 0 and abs(l.sum()) > 0 else 0
        print(f"\n  RESULTS: n={len(trades)}, net={net.mean():+.3f} pts, "
              f"hit={(net>0).mean():.3f}, PF={pf:.2f}")
        print(f"  Total $: ${net.sum() * pv:+,.0f}")

    return {
        "df": df, "sessions": sessions, "tensors": all_tensors,
        "xo": xo, "entries": entries, "trades": trades,
        "model": model, "hmm_results": hmm_results,
        "config": {
            "ticker": ticker, "instrument": cfg, "gate": gate_name,
            "emission": emission_name, "exit": exit_name,
        },
    }


# ══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_trades(df, entries, instrument_cfg, exit_cfg, all_tensors):
    """
    Simulate trades with full path logging using exits.py.

    Parameters
    ----------
    df : pd.DataFrame
    entries : pd.DataFrame — gated entry bars
    instrument_cfg : dict
    exit_cfg : dict — from config.EXIT_CONFIGS
    all_tensors : [N, 5, 11]

    Returns
    -------
    trades : pd.DataFrame — one row per trade with full diagnostics
    """
    if len(entries) == 0:
        return pd.DataFrame()

    from exits import simulate_trade

    direction = instrument_cfg["direction"]
    cost = instrument_cfg["cost_pts"]

    records = []
    for entry_idx in entries.index:
        entry_loc = df.index.get_loc(entry_idx)
        entry_row = entries.loc[entry_idx]

        record = simulate_trade(
            df=df,
            entry_loc=entry_loc,
            direction=direction,
            cost=cost,
            exit_cfg=exit_cfg,
            all_tensors=all_tensors,
            entry_row=entry_row,
        )

        if record is not None:
            records.append(record)

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════

def run_sweep(qb, sweep_cfg=None, verbose=False):
    """
    Run combinatoric sweep across instruments × configs.

    Parameters
    ----------
    qb : QuantBook
    sweep_cfg : dict or None — from config.SWEEP_CONFIG
    verbose : bool

    Returns
    -------
    results_table : pd.DataFrame — one row per config with summary stats
    all_results : dict — full results for each config
    """
    if sweep_cfg is None:
        sweep_cfg = SWEEP_CONFIG

    instruments = sweep_cfg["instruments"]
    emissions = sweep_cfg.get("emissions") or ["lr+gk"]
    gates = sweep_cfg.get("gates") or [None]  # None = instrument default
    exits = sweep_cfg.get("exits") or ["fixed_7"]
    directions = sweep_cfg.get("directions")

    rows = []
    all_results = {}
    total = len(instruments) * len(emissions) * len(gates) * len(exits)
    if directions:
        total *= len(directions)

    print(f"SWEEP: {total} configurations")
    i = 0

    for ticker in instruments:
        cfg = INSTRUMENTS[ticker]
        dir_list = [None] if not directions else list(directions)

        for dir_name in dir_list:
            dir_override = DIRECTION_CONFIGS[dir_name] if dir_name else None

            for em_name in emissions:
                for gate_name in gates:
                    g = gate_name or cfg["gate"]
                    for exit_name in exits:
                        i += 1
                        key = f"{ticker}_{em_name}_{g}_{exit_name}"
                        if dir_name:
                            key += f"_{dir_name}"

                        print(f"\n[{i}/{total}] {key}")

                        try:
                            res = run_single(
                                qb, ticker,
                                gate_name=g,
                                emission_name=em_name,
                                exit_name=exit_name,
                                direction_override=dir_override,
                                verbose=verbose,
                            )
                            all_results[key] = res

                            # Summary row
                            trades = res["trades"]
                            cost = cfg["cost_pts"]
                            pv = cfg["point_value"]
                            if len(trades) > 0:
                                net = trades["final_pnl"].values - cost
                                w = net[net > 0]
                                l = net[net <= 0]
                                pf = w.sum()/abs(l.sum()) if len(l)>0 and abs(l.sum())>0 else 0
                                rows.append({
                                    "key": key, "ticker": ticker,
                                    "emission": em_name, "gate": g,
                                    "exit": exit_name,
                                    "direction": dir_name or "default",
                                    "n_trades": len(trades),
                                    "net_mean": net.mean(),
                                    "hit_rate": (net > 0).mean(),
                                    "pf": pf,
                                    "total_pts": net.sum(),
                                    "total_dollar": net.sum() * pv,
                                    "mfe_mean": trades["mfe"].mean(),
                                    "mae_mean": trades["mae"].mean(),
                                    "capture": trades["final_pnl"].mean() / trades["mfe"].mean() if trades["mfe"].mean() > 0 else 0,
                                })
                            else:
                                rows.append({
                                    "key": key, "ticker": ticker,
                                    "emission": em_name, "gate": g,
                                    "exit": exit_name,
                                    "direction": dir_name or "default",
                                    "n_trades": 0,
                                })

                        except Exception as e:
                            print(f"  FAILED: {e}")
                            rows.append({"key": key, "ticker": ticker, "error": str(e)})

    results_table = pd.DataFrame(rows)
    print(f"\nSweep complete: {len(results_table)} configs")

    return results_table, all_results
