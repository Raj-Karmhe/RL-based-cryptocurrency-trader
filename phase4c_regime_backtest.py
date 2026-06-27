"""
phase4c_regime_backtest.py — HMM-Filtered Regime Backtest
==========================================================
Evaluates bear and crab specialist agents ONLY on the days the HMM
classifies as bear or crab respectively.

Bull days are REMOVED from the data entirely — the portfolio only
steps through non-bull candles, as if bull periods never happened.

USAGE
-----
  python phase4c_regime_backtest.py --dataset val
  python phase4c_regime_backtest.py --dataset test
  python phase4c_regime_backtest.py --dataset test --coin BTC/USDT

OUTPUT
------
  results/regime_filtered_backtest_<COIN>.html   — interactive Plotly chart
  results/regime_filtered_metrics.json           — per-coin metrics
"""

import json, os, sys, io, warnings, pickle

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_train import compute_turbulence
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# ──────────────────────────────────────────────────────────────────────────────
# HMM: CLASSIFY EVERY ROW
# ──────────────────────────────────────────────────────────────────────────────

def load_hmm():
    path = os.path.join(config.MODELS_DIR, "hmm_regime_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"HMM model not found at {path}. Run phase5b_train_hmm.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["state_map"]


def classify_all(df: pd.DataFrame, hmm_model, state_map) -> pd.Series:
    """
    Classifies every row into 'bull', 'bear', or 'crab' using the same
    24-step smoothed return+vol feature as the EnsembleMetaAgent.
    """
    regimes, ret_buf, vol_buf = [], [], []
    for i in range(len(df)):
        ret_buf.append(df.iloc[i]["1d_log_return"])
        vol_buf.append(df.iloc[i]["1d_hvol_20"])
        if len(ret_buf) > 24: ret_buf.pop(0); vol_buf.pop(0)
        state = hmm_model.predict(np.array([[np.mean(ret_buf), np.mean(vol_buf)]]))[0]
        regimes.append(state_map[state])
    return pd.Series(regimes, index=df.index, name="regime")


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ON A SINGLE FILTERED DataFrame (all rows passed to env)
# ──────────────────────────────────────────────────────────────────────────────

def backtest_on_df(model, df, feature_cols, turb_threshold, allow_short=False):
    """
    Runs a clean step-by-step backtest on df.
    df should already be filtered to only the rows you want (e.g. bear days).
    Returns a dict of portfolio_values, positions, dates, prices, metrics.
    """
    env = CryptoTradingEnv(
        df, feature_cols,
        initial_balance  = config.INITIAL_BALANCE,
        turb_threshold   = turb_threshold,
        allow_short      = allow_short,
    )
    obs, _ = env.reset()

    pv_list   = [env.portfolio_value]
    pos_list  = [0.0]
    dates     = list(df.index[env.seq_len:])
    prices    = list(df["Close"].values[env.seq_len:])

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        pv_list.append(env.portfolio_value)
        pos_list.append(float(env.position))

    pv   = np.array(pv_list)
    rets = np.diff(np.log(pv + 1e-8))
    total_ret    = float((pv[-1] / pv[0]) - 1)
    running_max  = np.maximum.accumulate(pv)
    max_dd       = float(np.max((running_max - pv) / (running_max + 1e-8)))
    sharpe       = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(8760))
    down         = rets[rets < 0]
    sortino      = float(rets.mean() / (down.std() + 1e-8) * np.sqrt(8760)) if len(down) > 1 else 0.0
    pos_arr      = np.array(pos_list[1:])
    win_rate     = float(np.mean(pos_arr > 0.05)) if len(pos_arr) > 0 else 0.0

    return {
        "portfolio_values": pv.tolist(),
        "positions":        pos_list,
        "dates":            dates,
        "prices":           prices,
        "metrics": {
            "total_return":    total_ret,
            "max_drawdown":    max_dd,
            "sharpe_ratio":    sharpe,
            "sortino_ratio":   sortino,
            "final_portfolio": float(pv[-1]),
            "n_steps":         len(df) - env.seq_len,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY CHART
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {"bear": "#e74c3c", "crab": "#f39c12", "bull": "#2ecc71"}

def save_plot(bear_res, crab_res, coin, dataset):
    sym = coin.replace("/", "_")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False,
        subplot_titles=[
            f"Bear Agent  (bear days only) — {coin}",
            f"Crab Agent  (crab days only) — {coin}",
            "Position over time",
        ],
        row_heights=[0.4, 0.4, 0.2],
        vertical_spacing=0.06,
    )

    for row, (res, label, color) in enumerate([
        (bear_res, "Bear", COLORS["bear"]),
        (crab_res, "Crab", COLORS["crab"]),
    ], start=1):
        if res is None:
            continue
        dates = res["dates"]
        pv    = res["portfolio_values"][1:]
        bh    = (np.array(res["prices"]) / res["prices"][0]) * config.INITIAL_BALANCE
        m     = res["metrics"]

        n = min(len(dates), len(pv), len(bh))
        fig.add_trace(go.Scatter(
            x=dates[:n], y=pv[:n],
            name=f"{label} Agent",
            line=dict(color=color, width=2),
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=dates[:n], y=bh[:n],
            name=f"B&H ({label} days)",
            line=dict(color="#95a5a6", width=1.2, dash="dash"),
        ), row=row, col=1)

        ann = (f"Return: {m['total_return']*100:+.1f}%  "
               f"Sharpe: {m['sharpe_ratio']:+.2f}  "
               f"MaxDD: {m['max_drawdown']*100:.1f}%  "
               f"Steps: {m['n_steps']}")
        fig.add_annotation(
            text=ann, xref="paper", yref=f"y{row if row > 1 else ''}",
            x=0.01, y=0.97, xanchor="left", showarrow=False,
            font=dict(size=11, color=color), row=row, col=1,
        )

        pos = res["positions"][1:]
        pos_color = [color if p > 0.05 else "#555" for p in pos]
        fig.add_trace(go.Bar(
            x=dates[:n], y=pos[:n],
            name=f"{label} pos",
            marker_color=pos_color, opacity=0.6,
        ), row=3, col=1)

    fig.update_layout(
        title=f"HMM Regime-Filtered Backtest: {coin} ({dataset.upper()}) — bull days removed",
        template="plotly_dark",
        height=900,
        hovermode="x unified",
    )

    out = os.path.join(config.RESULTS_DIR, f"regime_filtered_backtest_{sym}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"  [Plot] Saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(dataset="val", coin_override=None):
    print("\n" + "=" * 70)
    print(f"  HMM REGIME-FILTERED BACKTEST  ({dataset.upper()} SET)")
    print("  Bear model on bear days | Crab model on crab days | Bull days REMOVED")
    print("=" * 70)

    hmm_model, state_map = load_hmm()
    print(f"  [HMM]  State map: {state_map}")

    # ── Resolve model paths ───────────────────────────────────────────────
    def _resolve(regime):
        best  = os.path.join(config.MODELS_DIR, f"best_{regime}", "best_model.zip")
        final = os.path.join(config.MODELS_DIR, f"{config.MODEL_NAME}_{regime}.zip")
        if os.path.exists(best):  return best
        if os.path.exists(final): return final
        raise FileNotFoundError(f"No model for '{regime}'. Retrain first.")

    bear_path = _resolve("bear")
    crab_path = _resolve("crab")
    print(f"  [Bear] {bear_path}")
    print(f"  [Crab] {crab_path}")

    # ── Auto-detect features + action space from model ────────────────────
    _probe = PPO.load(bear_path[:-4])
    _n_feat = (_probe.observation_space.shape[0] - 2) // config.SEQ_LEN
    _allow_short = float(_probe.action_space.low[0]) < 0.0
    del _probe

    registry = getattr(config, "FEATURE_REGISTRY", {})
    feat_file = registry.get(_n_feat, config.GOLDEN_FEATURES_PATH)
    with open(feat_file) as fp:
        golden_features = json.load(fp)
    print(f"  [Features] {_n_feat} features → {os.path.basename(feat_file)}")

    coins = [coin_override] if coin_override else config.MULTI_COINS
    all_metrics = {}

    for coin in coins:
        sym = coin.replace("/", "_")
        print(f"\n{'─'*70}")
        print(f"  [Coin] {coin}")
        print(f"{'─'*70}")

        # ── Load data ─────────────────────────────────────────────────────
        train_path = os.path.join(config.DATA_DIR, f"{sym}_train_features.csv")
        full_train = pd.read_csv(train_path, index_col=0, parse_dates=True)

        if dataset == "val":
            split    = int(len(full_train) * 0.7)
            eval_df  = full_train.iloc[split:].copy()
            train_df = full_train.iloc[:split]
        elif dataset == "full-train":
            # Use the ENTIRE training CSV (2021-2023) — contains all 3 regimes.
            # Note: this is in-sample for the models, but it's the only dataset
            # that contains actual bear/crab periods for a meaningful filtered test.
            eval_df  = full_train.copy()
            train_df = full_train
            print("  [Note] Using full training history (in-sample, but contains bear/crab periods).")
        else:
            test_path = os.path.join(config.DATA_DIR, f"{sym}_test_features.csv")
            if not os.path.exists(test_path):
                print(f"  [Skip] {test_path} not found.")
                continue
            eval_df  = pd.read_csv(test_path, index_col=0, parse_dates=True)
            train_df = full_train

        train_turb     = compute_turbulence(train_df["Close"])
        turb_threshold = float(np.nanpercentile(train_turb[train_turb > 0], config.TURBULENCE_PERCENTILE))
        eval_df["Turbulence"] = compute_turbulence(eval_df["Close"]).values

        # ── Classify all rows ─────────────────────────────────────────────
        print("  [HMM]  Classifying regimes...")
        regimes = classify_all(eval_df, hmm_model, state_map)
        eval_df["regime"] = regimes

        counts = regimes.value_counts()
        total  = len(regimes)
        for r in ["bull", "bear", "crab"]:
            n = counts.get(r, 0)
            print(f"         {r.capitalize():5s}: {n:5d} steps ({n/total*100:.1f}%)")

        # ── Split into bear-only and crab-only DataFrames ─────────────────
        bear_df = eval_df[eval_df["regime"] == "bear"].copy()
        crab_df = eval_df[eval_df["regime"] == "crab"].copy()

        print(f"  [Filter] Removed {counts.get('bull',0)} bull steps.")
        print(f"           Bear subset: {len(bear_df)} rows | "
              f"Crab subset: {len(crab_df)} rows")

        bear_res = crab_res = None

        # ── Bear model on bear days ───────────────────────────────────────
        if len(bear_df) > config.SEQ_LEN + 10:
            print(f"\n  -- Bear agent on bear days --")
            bear_df["Turbulence"] = compute_turbulence(bear_df["Close"]).values
            dummy_env = DummyVecEnv([lambda: CryptoTradingEnv(
                bear_df, golden_features,
                initial_balance=config.INITIAL_BALANCE,
                turb_threshold=turb_threshold,
                allow_short=_allow_short,
            )])
            model_bear = PPO.load(
                bear_path[:-4], env=dummy_env,
                custom_objects={
                    "observation_space": dummy_env.observation_space,
                    "action_space":      dummy_env.action_space,
                },
            )
            bear_res = backtest_on_df(model_bear, bear_df, golden_features, turb_threshold, _allow_short)
            m = bear_res["metrics"]
            bh_ret = float((np.array(bear_res["prices"])[-1] / bear_res["prices"][0]) - 1)
            print(f"  Return: {m['total_return']*100:+.2f}%  "
                  f"(B&H: {bh_ret*100:+.2f}%)  "
                  f"Sharpe: {m['sharpe_ratio']:+.4f}  "
                  f"MaxDD: {m['max_drawdown']*100:.2f}%")
            dummy_env.close()
        else:
            print(f"  [Skip] Not enough bear rows ({len(bear_df)}) for a meaningful backtest.")

        # ── Crab model on crab days ───────────────────────────────────────
        if len(crab_df) > config.SEQ_LEN + 10:
            print(f"\n  -- Crab agent on crab days --")
            crab_df["Turbulence"] = compute_turbulence(crab_df["Close"]).values
            dummy_env = DummyVecEnv([lambda: CryptoTradingEnv(
                crab_df, golden_features,
                initial_balance=config.INITIAL_BALANCE,
                turb_threshold=turb_threshold,
                allow_short=_allow_short,
            )])
            model_crab = PPO.load(
                crab_path[:-4], env=dummy_env,
                custom_objects={
                    "observation_space": dummy_env.observation_space,
                    "action_space":      dummy_env.action_space,
                },
            )
            crab_res = backtest_on_df(model_crab, crab_df, golden_features, turb_threshold, _allow_short)
            m = crab_res["metrics"]
            bh_ret = float((np.array(crab_res["prices"])[-1] / crab_res["prices"][0]) - 1)
            print(f"  Return: {m['total_return']*100:+.2f}%  "
                  f"(B&H: {bh_ret*100:+.2f}%)  "
                  f"Sharpe: {m['sharpe_ratio']:+.4f}  "
                  f"MaxDD: {m['max_drawdown']*100:.2f}%")
            dummy_env.close()
        else:
            print(f"  [Skip] Not enough crab rows ({len(crab_df)}) for a meaningful backtest.")

        # ── Save chart ────────────────────────────────────────────────────
        if bear_res or crab_res:
            save_plot(bear_res, crab_res, coin, dataset)

        all_metrics[coin] = {
            "bear": bear_res["metrics"] if bear_res else None,
            "crab": crab_res["metrics"] if crab_res else None,
            "regime_counts": counts.to_dict(),
        }

    out_json = os.path.join(config.RESULTS_DIR, "regime_filtered_metrics.json")
    with open(out_json, "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    print(f"\n  [Save] Metrics → {out_json}")
    print("\n  REGIME-FILTERED BACKTEST COMPLETE ✓")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4c: HMM Regime-Filtered Backtest")
    parser.add_argument("--dataset", default="val", choices=["val", "test", "full-train"])
    parser.add_argument("--coin",    default=None,
                        help="Single coin to test (e.g. BTC/USDT). Defaults to all.")
    args = parser.parse_args()
    run(dataset=args.dataset, coin_override=args.coin)
