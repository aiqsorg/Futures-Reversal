"""
diagnostics.py — Trade Quality + Feature Analysis + Bootstrap
================================================================
Analyzes results from backtest.run_single().

Functions:
  trade_report()      — MAE/MFE, capture ratio, winner/loser profiles
  trade_paths()       — bar-by-bar mean trade path, winner vs loser
  entry_context()     — pre-entry momentum, vol, session position
  feature_scan()      — correlate all tensor features with returns
  pattern_analysis()  — argmax pattern breakdown at crossover bars
  gate_combinatorics() — exhaustive single/2-way/3-way gate testing
  bootstrap()         — confidence intervals on net mean, hit, PF
  cross_asset_test()  — does edge correlate with retail flow?
  full_diagnostic()   — runs everything, prints comprehensive report

Usage:
    from diagnostics import full_diagnostic
    results = run_single(qb, "NQ")
    full_diagnostic(results)
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import PARAMS, TENSOR_FEATURE_NAMES, GATE_CONFIGS


# ══════════════════════════════════════════════════════════════
# 1. TRADE QUALITY REPORT
# ══════════════════════════════════════════════════════════════

def trade_report(results):
    """
    MAE/MFE analysis, capture ratio, winner/loser separation.

    Parameters
    ----------
    results : dict from backtest.run_single()
    """
    trades = results["trades"]
    cfg = results["config"]
    cost = cfg["instrument"]["cost_pts"]
    pv = cfg["instrument"]["point_value"]

    if len(trades) == 0:
        print("  No trades."); return

    t = trades
    w = t[t["winner"]]
    l = t[~t["winner"]]

    ticker = cfg["ticker"]
    print(f"\n{'='*65}")
    print(f"  TRADE REPORT: {ticker} | {cfg['gate']} | {cfg['exit']}")
    print(f"  {cfg['instrument']['cross']} {'long' if cfg['instrument']['direction']==1 else 'short'}")
    print(f"{'='*65}")

    # ── Summary ──
    net = t["final_pnl"].values - cost
    wn, ln = net[net > 0], net[net <= 0]
    pf = wn.sum() / abs(ln.sum()) if len(ln) > 0 and abs(ln.sum()) > 0 else 0

    print(f"\n  Overview:")
    print(f"    Trades: {len(t)}")
    print(f"    Net/trade: {net.mean():+.3f} pts (${net.mean()*pv:+.1f})")
    print(f"    Hit rate: {(net>0).mean():.3f}")
    print(f"    PF: {pf:.2f}")
    print(f"    Total: {net.sum():+.1f} pts (${net.sum()*pv:+,.0f})")
    print(f"    Avg bars held: {t['bars_held'].mean():.1f}")

    # ── Exit reasons ──
    print(f"\n  Exit reasons:")
    for reason, count in t["exit_reason"].value_counts().items():
        sub = t[t["exit_reason"] == reason]
        sub_net = sub["final_pnl"].values - cost
        print(f"    {reason:<20s} n={count:>4} net={sub_net.mean():+.3f}")

    # ── MAE / MFE ──
    print(f"\n  MAE / MFE Analysis:")
    print(f"    {'Metric':<20s} {'All':>10} {'Winners':>10} {'Losers':>10}")
    for m in ["final_pnl", "mfe", "mae", "entry_eff"]:
        a = t[m].mean()
        wv = w[m].mean() if len(w) > 0 else np.nan
        lv = l[m].mean() if len(l) > 0 else np.nan
        print(f"    {m:<20s} {a:>+10.3f} {wv:>+10.3f} {lv:>+10.3f}")

    # ── MAE distribution ──
    print(f"\n  MAE distribution (how far trades go against you):")
    for pct in [10, 25, 50, 75, 90]:
        print(f"    P{pct}: {np.percentile(t['mae'], pct):+.3f} pts")

    # ── Separation quality ──
    if len(w) >= 5 and len(l) >= 5:
        mae_sep = abs(w["mae"].mean() - l["mae"].mean())
        quality = "GOOD — entries have timing edge" if mae_sep > 1.0 else \
                  "MODERATE" if mae_sep > 0.5 else \
                  "WEAK — similar drawdown profile, entries are noisy"
        print(f"\n  MAE separation: {mae_sep:.3f} pts ({quality})")

    # ── Capture ratio ──
    capture = t["final_pnl"].mean() / t["mfe"].mean() if t["mfe"].mean() > 0 else 0
    cap_quality = "Good" if capture > 0.3 else "Moderate" if capture > 0.15 else \
                  "POOR — leaving most of the move on the table"
    print(f"  Capture ratio: {capture:.3f} ({cap_quality})")
    print(f"    Mean MFE:       {t['mfe'].mean():.3f} pts (${t['mfe'].mean()*pv:.1f})")
    print(f"    Mean final P&L: {t['final_pnl'].mean():.3f} pts")
    print(f"    Gap (wasted):   {t['mfe'].mean() - t['final_pnl'].mean():.3f} pts "
          f"(${(t['mfe'].mean() - t['final_pnl'].mean())*pv:.1f})")

    # ── MFE timing: where does the peak occur? ──
    print(f"\n  MFE timing (when does peak profit occur?):")
    paths = t["path"].values
    max_bar_per_trade = []
    for path in paths:
        if len(path) > 0:
            max_bar_per_trade.append(np.argmax(path) + 1)
    if len(max_bar_per_trade) > 0:
        mb = np.array(max_bar_per_trade)
        print(f"    Mean peak bar: {mb.mean():.1f}")
        for b in range(1, min(PARAMS["hold_bars"] + 1, 11)):
            pct = (mb == b).mean() * 100
            if pct > 3:
                print(f"    Bar {b}: {pct:.0f}% of trades peak here")

    # ── Structural vs Lucky ──
    if len(w) >= 5:
        # Structural: won with low MAE (didn't survive deep drawdown)
        mae_thresh = w["mae"].quantile(0.5)  # median winner MAE
        structural = w[w["mae"] >= mae_thresh]  # less adverse = above median (less negative)
        lucky = w[w["mae"] < mae_thresh]        # deep drawdown before winning

        print(f"\n  Winner quality:")
        print(f"    'Clean' winners (MAE above median): n={len(structural)}, "
              f"final={structural['final_pnl'].mean():.3f}, MAE={structural['mae'].mean():+.3f}")
        print(f"    'Lucky' winners (deep MAE):         n={len(lucky)}, "
              f"final={lucky['final_pnl'].mean():.3f}, MAE={lucky['mae'].mean():+.3f}")


# ══════════════════════════════════════════════════════════════
# 2. TRADE PATHS
# ══════════════════════════════════════════════════════════════

def trade_paths(results):
    """Bar-by-bar mean trade path, winner vs loser split."""
    trades = results["trades"]
    if len(trades) == 0: return

    paths = trades["path"].values
    max_len = max(len(p) for p in paths)
    pm = np.full((len(paths), max_len), np.nan)
    for i, p in enumerate(paths):
        pm[i, :len(p)] = p

    win_mask = trades["winner"].values

    print(f"\n  Mean Trade Path:")
    print(f"  {'Bar':<5} {'All':>8} {'Winners':>8} {'Losers':>8} {'Gap':>8} {'%>0':>6}")

    for b in range(min(max_len, 12)):
        col = pm[:, b]
        v_all = col[~np.isnan(col)]
        v_win = col[win_mask & ~np.isnan(col)]
        v_lose = col[~win_mask & ~np.isnan(col)]

        if len(v_all) < 5: continue
        gap = v_win.mean() - v_lose.mean() if len(v_win) > 0 and len(v_lose) > 0 else 0
        print(f"  B{b+1:<4} {v_all.mean():>+8.3f} "
              f"{v_win.mean() if len(v_win)>0 else 0:>+8.3f} "
              f"{v_lose.mean() if len(v_lose)>0 else 0:>+8.3f} "
              f"{gap:>+8.3f} {(v_all>0).mean()*100:>5.1f}%")

    # When do winners and losers diverge?
    if max_len >= 3:
        bar1_all = pm[:, 0]; bar1_all = bar1_all[~np.isnan(bar1_all)]
        bar1_win = pm[win_mask, 0]; bar1_win = bar1_win[~np.isnan(bar1_win)]
        bar1_lose = pm[~win_mask, 0]; bar1_lose = bar1_lose[~np.isnan(bar1_lose)]

        if len(bar1_win) > 5 and len(bar1_lose) > 5:
            t_stat, p_val = stats.ttest_ind(bar1_win, bar1_lose, equal_var=False)
            print(f"\n  Bar 1 divergence test: t={t_stat:.2f}, p={p_val:.4f}")
            if p_val < 0.05:
                print(f"    Winners and losers are distinguishable by bar 1 → signal has real timing")
            else:
                print(f"    Not distinguishable at bar 1 → signal doesn't time the immediate move")


# ══════════════════════════════════════════════════════════════
# 3. ENTRY CONTEXT
# ══════════════════════════════════════════════════════════════

def entry_context(results):
    """Pre-entry momentum, volatility, bar quality for winners vs losers."""
    trades = results["trades"]
    if len(trades) < 10: return

    w = trades[trades["winner"]]
    l = trades[~trades["winner"]]
    if len(w) < 5 or len(l) < 5: return

    print(f"\n  Entry Context (winner vs loser):")
    print(f"    {'Metric':<25s} {'Winners':>10} {'Losers':>10} {'Diff':>10}")

    for col in ["pre_momentum", "entry_atr", "conviction"]:
        if col in trades.columns:
            wv = w[col].dropna().mean()
            lv = l[col].dropna().mean()
            print(f"    {col:<25s} {wv:>+10.5f} {lv:>+10.5f} {wv-lv:>+10.5f}")


# ══════════════════════════════════════════════════════════════
# 4. FEATURE SCAN
# ══════════════════════════════════════════════════════════════

def feature_scan(results, top_n=30):
    """
    Correlate all tensor features with returns at crossover bars.

    Parameters
    ----------
    results : dict from run_single()
    top_n : int — how many features to print

    Returns
    -------
    feat_df : pd.DataFrame — sorted by significance
    """
    xo = results["xo"]
    cfg = results["config"]
    cost = cfg["instrument"]["cost_pts"]
    direction = cfg["instrument"]["direction"]

    if "ret_pts" not in xo.columns:
        hold = PARAMS["hold_bars"]
        fwd_col = f"fwd_{hold}"
        xo = xo.copy()
        xo["ret_pts"] = xo[fwd_col] * xo["Close"] * direction

    ret = xo["ret_pts"].values
    t_cols = sorted([c for c in xo.columns if c.startswith("T_")] +
                    ["stress_A", "stress_B"])

    results_list = []
    for col in t_cols:
        if col not in xo.columns: continue
        vals = xo[col].values
        ok = ~np.isnan(vals) & ~np.isnan(ret)
        if ok.sum() < 20: continue
        r, p = stats.pearsonr(vals[ok], ret[ok])
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        results_list.append({"feature": col, "r": r, "p": p, "sig": sig,
                             "nunique": len(np.unique(vals[ok]))})

    feat_df = pd.DataFrame(results_list).sort_values("p")

    print(f"\n  Feature Scan ({cfg['ticker']}, n={len(xo)} crossovers):")
    print(f"  {'Feature':<40s} {'r':>7} {'p':>10} {'sig':>4}")
    for _, row in feat_df.head(top_n).iterrows():
        if row["p"] < 0.2:
            print(f"  {row['feature']:<40s} {row['r']:>+7.4f} {row['p']:>10.4f} {row['sig']:>4}")

    n_sig = (feat_df["p"] < 0.05).sum()
    print(f"\n  {n_sig} features significant at p<0.05 out of {len(feat_df)}")

    return feat_df


# ══════════════════════════════════════════════════════════════
# 5. ARGMAX PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════

def pattern_analysis(results):
    """Argmax pattern breakdown at crossover bars."""
    xo = results["xo"]
    cost = results["config"]["instrument"]["cost_pts"]

    if "T_am_alpha_pattern" not in xo.columns:
        print("  No argmax pattern column."); return

    print(f"\n  Argmax Pattern Breakdown:")
    print(f"  {'Pattern':<12s} {'Bits':<6s} {'N':>5} {'Net':>8} {'Hit':>6}")

    for pat in sorted(xo["T_am_alpha_pattern"].unique()):
        sub = xo[xo["T_am_alpha_pattern"] == pat]
        if len(sub) < 5: continue
        net = sub["ret_pts"].values - cost
        bits = f"{int(pat):04b}"
        regime = "".join(["Q" if b == "1" else "V" for b in bits])
        print(f"  {regime:<12s} {bits:<6s} {len(sub):>5} {net.mean():>+8.3f} "
              f"{(net>0).mean():>6.3f}")


# ══════════════════════════════════════════════════════════════
# 6. GATE COMBINATORICS
# ══════════════════════════════════════════════════════════════

def gate_combinatorics(results, max_singles=20, max_combos=15):
    """
    Exhaustive gate testing on crossover bars.
    Tests all single gates, then 2-way combos of top singles.
    """
    xo = results["xo"]
    cost = results["config"]["instrument"]["cost_pts"]

    if len(xo) < 20:
        print("  Too few crossovers."); return

    # Build all single gates
    singles = _build_single_gates(xo)
    base_net = (xo["ret_pts"].values - cost).mean()

    print(f"\n  Gate Combinatorics ({len(xo)} crossovers, baseline net={base_net:+.3f}):")
    print(f"\n  {'Gate':<35s} {'N':>5} {'Net':>8} {'Hit':>6} {'PF':>6} {'vs base':>8}")

    single_results = {}
    for gname, gmask in singles.items():
        sub = xo[gmask]
        if len(sub) < 8: continue
        pts = sub["ret_pts"].values
        net = pts - cost
        w, l = net[net > 0], net[net <= 0]
        pf = w.sum() / abs(l.sum()) if len(l) > 0 and abs(l.sum()) > 0 else 0
        delta = net.mean() - base_net
        single_results[gname] = {"n": len(sub), "net": net.mean(), "hit": (net>0).mean(),
                                  "pf": pf, "delta": delta, "mask": gmask}
        if delta > 0.3 or pf > 1.1:
            print(f"  {gname:<35s} {len(sub):>5} {net.mean():>+8.3f} "
                  f"{(net>0).mean():>6.3f} {pf:>6.2f} {delta:>+8.3f}")

    # Top singles for combos
    ranked = sorted(single_results.items(), key=lambda x: -x[1]["net"])
    top = [k for k, v in ranked if v["net"] > base_net and v["n"] >= 15][:max_singles]

    if len(top) >= 2:
        print(f"\n  2-WAY COMBOS (top {max_combos}):")
        print(f"  {'Gate':<55s} {'N':>5} {'Net':>8} {'Hit':>6} {'PF':>6}")

        combos = []
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                mask = single_results[top[i]]["mask"] & single_results[top[j]]["mask"]
                sub = xo[mask]
                if len(sub) < 8: continue
                pts = sub["ret_pts"].values
                net = pts - cost
                w, l = net[net > 0], net[net <= 0]
                pf = w.sum() / abs(l.sum()) if len(l) > 0 and abs(l.sum()) > 0 else 0
                combos.append({"gate": f"{top[i]} + {top[j]}", "n": len(sub),
                                "net": net.mean(), "hit": (net>0).mean(), "pf": pf})

        combos.sort(key=lambda x: -x["net"])
        for c in combos[:max_combos]:
            if c["net"] > base_net:
                print(f"  {c['gate']:<55s} {c['n']:>5} {c['net']:>+8.3f} "
                      f"{c['hit']:>6.3f} {c['pf']:>6.2f}")


def _build_single_gates(xo):
    """Build dict of all single gate masks for combinatorics."""
    gates = {}

    # KL
    if "T_kl_slope_3s" in xo: gates["kl_slope_3s>0"] = xo["T_kl_slope_3s"] > 0
    if "T_kl_accel" in xo: gates["kl_accel>0"] = xo["T_kl_accel"] > 0
    if "T_kl_mean_3s" in xo:
        gates["kl_mean>med"] = xo["T_kl_mean_3s"] > xo["T_kl_mean_3s"].median()

    # Alpha continuous
    if "T_alpha_std" in xo:
        gates["alpha_std<0.05"] = xo["T_alpha_std"] < 0.05
        gates["alpha_std<0.10"] = xo["T_alpha_std"] < 0.10
    if "T_alpha_conv" in xo:
        gates["alpha_conv>0.3"] = xo["T_alpha_conv"] > 0.3
    if "T_alpha_slope" in xo:
        gates["alpha_slope>0"] = xo["T_alpha_slope"] > 0

    # Alpha argmax
    if "T_am_alpha_unan_quiet" in xo: gates["am_unan_quiet"] = xo["T_am_alpha_unan_quiet"] == 1
    if "T_am_alpha_majority_vol" in xo: gates["am_maj_vol"] = xo["T_am_alpha_majority_vol"] == 1
    if "T_am_alpha_flips" in xo:
        gates["am_flips==0"] = xo["T_am_alpha_flips"] == 0
        gates["am_flips>=2"] = xo["T_am_alpha_flips"] >= 2
    if "T_am_alpha_transition_now" in xo: gates["am_trans_now"] = xo["T_am_alpha_transition_now"] == 1
    if "T_am_alpha_current_quiet" in xo:
        gates["am_current_quiet"] = xo["T_am_alpha_current_quiet"] == 1
        gates["am_current_vol"] = xo["T_am_alpha_current_quiet"] == 0

    # Argmax patterns
    if "T_am_pattern_entering_vol" in xo: gates["am_pat_enter_v"] = xo["T_am_pattern_entering_vol"] == 1
    if "T_am_pattern_entering_quiet" in xo: gates["am_pat_enter_q"] = xo["T_am_pattern_entering_quiet"] == 1
    if "T_am_pattern_stable_quiet" in xo: gates["am_pat_stable_q"] = xo["T_am_pattern_stable_quiet"] == 1

    # Gamma
    if "T_gamma_std" in xo: gates["gamma_std<0.05"] = xo["T_gamma_std"] < 0.05
    if "T_am_gamma_unan_quiet" in xo: gates["gam_unan_quiet"] = xo["T_am_gamma_unan_quiet"] == 1

    # Disagreement
    if "T_disagree_any" in xo:
        gates["no_disagree"] = xo["T_disagree_any"] == 0
    if "T_ag_direction_net" in xo:
        gates["ag_dir_net>0"] = xo["T_ag_direction_net"] > 0
        gates["ag_dir_net<0"] = xo["T_ag_direction_net"] < 0

    # Projection
    if "T_proj_agrees_alpha" in xo: gates["proj_agrees"] = xo["T_proj_agrees_alpha"] == 1

    # Regime
    if "T_regime_age" in xo:
        gates["age>=5"] = xo["T_regime_age"] >= 5
        gates["age>=10"] = xo["T_regime_age"] >= 10
    if "T_current_streak" in xo: gates["streak>=3"] = xo["T_current_streak"] >= 3

    # Stress
    if "stress_A" in xo: gates["stress_A<0.02"] = xo["stress_A"] < 0.02
    if "stress_B" in xo: gates["stress_B<0.02"] = xo["stress_B"] < 0.02

    # Micro
    if "T_micro_gk_slope" in xo: gates["gk_slope<0"] = xo["T_micro_gk_slope"] < 0
    if "T_vol_calming_quiet" in xo: gates["vol_calm_quiet"] = xo["T_vol_calming_quiet"] == 1
    if "T_kl_high_stable" in xo: gates["kl_high_stable"] = xo["T_kl_high_stable"] == 1

    return gates


# ══════════════════════════════════════════════════════════════
# 7. BOOTSTRAP
# ══════════════════════════════════════════════════════════════

def bootstrap(results, n_boot=None):
    """Bootstrap confidence intervals on key metrics."""
    if n_boot is None: n_boot = PARAMS["bootstrap_n"]

    trades = results["trades"]
    cost = results["config"]["instrument"]["cost_pts"]
    if len(trades) < 15:
        print("  Too few trades for bootstrap."); return

    pts = trades["final_pnl"].values
    net = pts - cost

    b_means = np.zeros(n_boot)
    b_hits = np.zeros(n_boot)
    b_pfs = np.zeros(n_boot)

    for i in range(n_boot):
        s = np.random.choice(net, size=len(net), replace=True)
        b_means[i] = s.mean()
        b_hits[i] = (s > 0).mean()
        w, l = s[s > 0], s[s <= 0]
        b_pfs[i] = w.sum() / abs(l.sum()) if len(l) > 0 and abs(l.sum()) > 0 else 0

    print(f"\n  Bootstrap ({len(net)} trades, {n_boot} resamples):")
    print(f"    Net mean 90% CI:  [{np.percentile(b_means, 5):+.3f}, {np.percentile(b_means, 95):+.3f}]")
    print(f"    Hit rate 90% CI:  [{np.percentile(b_hits, 5):.3f}, {np.percentile(b_hits, 95):.3f}]")
    print(f"    PF 90% CI:        [{np.percentile(b_pfs, 5):.2f}, {np.percentile(b_pfs, 95):.2f}]")
    print(f"    P(net > 0):       {(b_means > 0).mean() * 100:.1f}%")
    print(f"    P(PF > 1):        {(b_pfs > 1).mean() * 100:.1f}%")


# ══════════════════════════════════════════════════════════════
# 8. CROSS-ASSET HYPOTHESIS TEST
# ══════════════════════════════════════════════════════════════

def cross_asset_test(all_results):
    """
    Test: does edge correlate with retail flow presence?
    Expected ranking: NQ > ES > RTY > YM/GC/CL.

    Parameters
    ----------
    all_results : dict of {ticker: results_dict}
    """
    print(f"\n{'='*65}")
    print(f"  CROSS-ASSET HYPOTHESIS TEST")
    print(f"  Thesis: edge exists where retail/systematic flow dominates")
    print(f"  Expected: NQ > ES > RTY >> YM > CL/GC")
    print(f"{'='*65}")

    rows = []
    for ticker, res in all_results.items():
        trades = res["trades"]
        cost = res["config"]["instrument"]["cost_pts"]
        if len(trades) == 0:
            rows.append({"ticker": ticker, "n": 0, "net": np.nan})
            continue
        net = trades["final_pnl"].values - cost
        w, l = net[net > 0], net[net <= 0]
        pf = w.sum() / abs(l.sum()) if len(l) > 0 and abs(l.sum()) > 0 else 0
        rows.append({
            "ticker": ticker, "n": len(trades), "net": net.mean(),
            "hit": (net > 0).mean(), "pf": pf, "total": net.sum(),
        })

    df = pd.DataFrame(rows).sort_values("net", ascending=False)
    print(f"\n  {'Ticker':<8s} {'N':>5} {'Net/tr':>8} {'Hit':>6} {'PF':>6} {'Total':>10}")
    for _, r in df.iterrows():
        if np.isnan(r.get("net", np.nan)):
            print(f"  {r['ticker']:<8s} {'no trades':>5}")
        else:
            print(f"  {r['ticker']:<8s} {r['n']:>5} {r['net']:>+8.3f} "
                  f"{r['hit']:>6.3f} {r['pf']:>6.2f} {r['total']:>+10.1f}")

    # Check if equity index futures outperform commodity futures
    equity = df[df["ticker"].isin(["NQ", "ES", "RTY", "YM"])]["net"].dropna()
    commodity = df[df["ticker"].isin(["CL", "GC"])]["net"].dropna()

    if len(equity) > 0 and len(commodity) > 0:
        print(f"\n  Equity index avg net: {equity.mean():+.3f}")
        print(f"  Commodity avg net:    {commodity.mean():+.3f}")
        if equity.mean() > commodity.mean():
            print(f"  ✓ Retail flow thesis SUPPORTED")
        else:
            print(f"  ✗ Retail flow thesis NOT SUPPORTED")


# ══════════════════════════════════════════════════════════════
# 9. FULL DIAGNOSTIC (runs everything)
# ══════════════════════════════════════════════════════════════

def trade_detail(results, n_best=5, n_worst=5):
    """
    Print detailed diagnostics for individual trades.
    Shows best/worst trades with full context, then clusters.
    """
    trades = results["trades"]
    cfg = results["config"]
    cost = cfg["instrument"]["cost_pts"]
    pv = cfg["instrument"]["point_value"]

    if len(trades) < 10: return

    trades_sorted = trades.sort_values("net_pnl", ascending=False)

    # ── Best trades ──
    print(f"\n  TOP {n_best} TRADES:")
    print(f"  {'#':<3} {'Time':<20s} {'Net':>8} {'MFE':>8} {'MAE':>8} "
          f"{'Bars':>5} {'Exit':<15s} {'Pattern':<8s} {'Stress':>7}")
    for i, (_, t) in enumerate(trades_sorted.head(n_best).iterrows()):
        ef = t.get("entry_features", {})
        pat = ef.get("T_am_alpha_pattern", np.nan)
        pat_str = f"{int(pat):04b}" if not np.isnan(pat) else "????"
        stress = ef.get("stress_B", np.nan)
        print(f"  {i+1:<3} {str(t['entry_time'])[:19]:<20s} {t['net_pnl']:>+8.1f} "
              f"{t['mfe']:>+8.1f} {t['mae']:>+8.1f} {t['bars_held']:>5} "
              f"{t['exit_reason']:<15s} {pat_str:<8s} {stress:>+7.4f}")

    # ── Worst trades ──
    print(f"\n  WORST {n_worst} TRADES:")
    print(f"  {'#':<3} {'Time':<20s} {'Net':>8} {'MFE':>8} {'MAE':>8} "
          f"{'Bars':>5} {'Exit':<15s} {'Pattern':<8s} {'Stress':>7}")
    for i, (_, t) in enumerate(trades_sorted.tail(n_worst).iterrows()):
        ef = t.get("entry_features", {})
        pat = ef.get("T_am_alpha_pattern", np.nan)
        pat_str = f"{int(pat):04b}" if not np.isnan(pat) else "????"
        stress = ef.get("stress_B", np.nan)
        print(f"  {i+1:<3} {str(t['entry_time'])[:19]:<20s} {t['net_pnl']:>+8.1f} "
              f"{t['mfe']:>+8.1f} {t['mae']:>+8.1f} {t['bars_held']:>5} "
              f"{t['exit_reason']:<15s} {pat_str:<8s} {stress:>+7.4f}")

    # ── Clustering: what do winning trades have in common? ──
    print(f"\n  WINNER vs LOSER CLUSTERING:")
    w = trades[trades["winner"]]
    l = trades[~trades["winner"]]

    # Extract features from entry_features dict
    feature_keys = set()
    for _, t in trades.iterrows():
        ef = t.get("entry_features", {})
        if isinstance(ef, dict):
            feature_keys.update(ef.keys())

    interesting = []
    for key in sorted(feature_keys):
        w_vals = [t["entry_features"].get(key, np.nan) for _, t in w.iterrows()
                  if isinstance(t.get("entry_features"), dict)]
        l_vals = [t["entry_features"].get(key, np.nan) for _, t in l.iterrows()
                  if isinstance(t.get("entry_features"), dict)]
        w_vals = [v for v in w_vals if not np.isnan(v)]
        l_vals = [v for v in l_vals if not np.isnan(v)]
        if len(w_vals) < 10 or len(l_vals) < 10: continue
        w_mean = np.mean(w_vals)
        l_mean = np.mean(l_vals)
        pooled_std = np.sqrt((np.var(w_vals) + np.var(l_vals)) / 2)
        if pooled_std > 0:
            effect_size = (w_mean - l_mean) / pooled_std
        else:
            effect_size = 0
        t_stat, p_val = stats.ttest_ind(w_vals, l_vals, equal_var=False)
        if abs(effect_size) > 0.1 or p_val < 0.1:
            interesting.append({
                "feature": key, "w_mean": w_mean, "l_mean": l_mean,
                "effect": effect_size, "p": p_val,
            })

    if interesting:
        interesting.sort(key=lambda x: x["p"])
        print(f"  {'Feature':<40s} {'W mean':>10} {'L mean':>10} {'Effect':>8} {'p':>8}")
        for row in interesting[:20]:
            sig = "***" if row["p"] < .001 else "**" if row["p"] < .01 else "*" if row["p"] < .05 else ""
            print(f"  {row['feature']:<40s} {row['w_mean']:>+10.4f} {row['l_mean']:>+10.4f} "
                  f"{row['effect']:>+8.3f} {row['p']:>7.4f} {sig}")

    # ── Hold dynamics: does regime flip during hold predict outcome? ──
    if "regime_flipped_during" in trades.columns:
        flipped = trades[trades["regime_flipped_during"] == True]
        stable = trades[trades["regime_flipped_during"] == False]
        if len(flipped) >= 5 and len(stable) >= 5:
            f_net = (flipped["final_pnl"].values - cost).mean()
            s_net = (stable["final_pnl"].values - cost).mean()
            print(f"\n  Hold dynamics:")
            print(f"    Regime flipped during hold: n={len(flipped)}, net={f_net:+.3f}")
            print(f"    Regime stable during hold:  n={len(stable)}, net={s_net:+.3f}")

    if "max_stress_during" in trades.columns:
        med_stress = trades["max_stress_during"].median()
        high_s = trades[trades["max_stress_during"] > med_stress]
        low_s = trades[trades["max_stress_during"] <= med_stress]
        if len(high_s) >= 5 and len(low_s) >= 5:
            h_net = (high_s["final_pnl"].values - cost).mean()
            l_net = (low_s["final_pnl"].values - cost).mean()
            print(f"    High stress during hold:    n={len(high_s)}, net={h_net:+.3f}")
            print(f"    Low stress during hold:     n={len(low_s)}, net={l_net:+.3f}")

    # ── MFE bar clustering ──
    if "mfe_bar" in trades.columns:
        print(f"\n  MFE bar clustering (net by when peak occurs):")
        print(f"  {'Peak bar':<10s} {'N':>5} {'Net':>8} {'Hit':>6}")
        for b in sorted(trades["mfe_bar"].unique()):
            sub = trades[trades["mfe_bar"] == b]
            if len(sub) >= 5:
                sub_net = sub["final_pnl"].values - cost
                print(f"  Bar {int(b):<6} {len(sub):>5} {sub_net.mean():>+8.3f} "
                      f"{(sub_net>0).mean():>6.3f}")


def reversal_analysis(results):
    """
    Check if reversing the strategy improves results.
    A very negative net implies the opposite direction works.
    """
    trades = results["trades"]
    cfg = results["config"]
    cost = cfg["instrument"]["cost_pts"]

    if len(trades) < 10: return

    # Original direction
    net_orig = trades["final_pnl"].values - cost
    # Reversed: negate the P&L (what if you went the other way?)
    net_rev = -trades["final_pnl"].values - cost

    orig_mean = net_orig.mean()
    rev_mean = net_rev.mean()

    print(f"\n  Reversal Analysis:")
    print(f"    Original ({cfg['instrument']['cross']} "
          f"{'long' if cfg['instrument']['direction']==1 else 'short'}): "
          f"net={orig_mean:+.3f}, hit={(net_orig>0).mean():.3f}")
    print(f"    Reversed: net={rev_mean:+.3f}, hit={(net_rev>0).mean():.3f}")

    if rev_mean > orig_mean and rev_mean > 0:
        improvement = rev_mean - orig_mean
        print(f"    ⚠ REVERSAL IS BETTER by {improvement:+.3f} pts/trade")
        print(f"    Consider flipping direction for this instrument.")
    elif orig_mean <= 0 and rev_mean <= 0:
        print(f"    Neither direction works well.")
    else:
        print(f"    Original direction is correct.")


def full_diagnostic(results):
    """Run complete diagnostic suite on one instrument's results."""
    trade_report(results)
    trade_paths(results)
    entry_context(results)
    trade_detail(results)
    reversal_analysis(results)
    bootstrap(results)
    feat_df = feature_scan(results)
    pattern_analysis(results)
    gate_combinatorics(results)
    return feat_df
