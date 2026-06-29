"""
phase3_train.py — CLSTM-PPO Training + Validation
==================================================
Phase 3, Component 3 — Training Loop.

WHAT THIS SCRIPT DOES
---------------------
1.  Loads the golden features from Phase 2.
2.  Builds the training and validation Gymnasium environments.
3.  Creates the PPO agent with our CLSTMFeatureExtractor plugged in.
4.  Trains for `TOTAL_TIMESTEPS` with:
        - A custom callback that logs portfolio value, Sharpe, and drawdown
          every N episodes so you can monitor learning progress.
        - Episode truncation logic: because the training dataset spans 4 years
          of continuous data, the LSTM hidden states are reset at the start of
          each PPO rollout.  We use episode length = N_STEPS so the agent
          experiences a wide variety of market regimes without maintaining
          unrealistically long "memory" across years.
5.  Evaluates on the VALIDATION set (data the agent has never seen during
    training, but used to monitor generalisation).
6.  Saves the trained model to disk.

WHY PPO?
--------
PPO (Proximal Policy Optimization) is chosen because:
    • It uses a clipped surrogate objective that prevents the policy from
      updating too aggressively — important for financial data where the
      signal is noisy and a bad update can cause catastrophic forgetting.
    • It is compatible with continuous action spaces (unlike DQN).
    • It has shown strong performance in finance RL papers.
    • The SB3 implementation is production-quality and well-tested.
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

warnings.filterwarnings("ignore")

# Force line-buffered stdout so logs appear immediately in background tasks
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_model import CLSTMFeatureExtractor

# ──────────────────────────────────────────────────────────────────────────────
# TRAINING CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

class TrainingMonitorCallback(BaseCallback):
    """
    Logs training progress to the console and records metrics for plotting.

    Reports every `log_freq` episodes:
    • Average episode reward (last 10 episodes)
    • Final portfolio value of the last episode
    • Approximate Sharpe ratio over recent episodes
    """

    def __init__(self, log_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq          = log_freq
        self._ep_reward        = 0.0
        self._ep_length        = 0
        self.episode_rewards   = []
        self.episode_lengths   = []
        self.portfolio_values  = []
        self.timesteps_log     = []

    def _on_step(self) -> bool:
        # Accumulate per-step reward
        self._ep_reward += self.locals["rewards"][0]
        self._ep_length += 1

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)
            self.timesteps_log.append(self.num_timesteps)

            # Extract portfolio value from info dict
            infos = self.locals.get("infos", [{}])
            if infos and "portfolio_value" in infos[0]:
                self.portfolio_values.append(infos[0]["portfolio_value"])

            n = len(self.episode_rewards)
            if self.verbose and n % self.log_freq == 0:
                avg_r  = np.mean(self.episode_rewards[-10:])
                last_p = self.portfolio_values[-1] if self.portfolio_values else 0
                avg_p  = np.mean(self.portfolio_values[-10:]) if self.portfolio_values else 0
                print(
                    f"  Ep {n:5d} | Steps {self.num_timesteps:8,} | "
                    f"Avg Reward: {avg_r:8.5f} | "
                    f"Last Portfolio: ${last_p:12,.0f} | "
                    f"Avg Portfolio: ${avg_p:12,.0f}"
                )

            # Reset for next episode
            self._ep_reward = 0.0
            self._ep_length = 0

        return True  # return True to continue training

    def save_reward_plot(self, path: str):
        """Plots the training reward curve and saves to disk."""
        if not self.episode_rewards:
            return
        smoothed = pd.Series(self.episode_rewards).rolling(20, min_periods=1).mean()
        fig, ax  = plt.subplots(figsize=(12, 4))
        ax.plot(self.timesteps_log, self.episode_rewards, alpha=0.3,
                color="#95a5a6", label="Episode Reward")
        ax.plot(self.timesteps_log, smoothed, color="#2980b9",
                linewidth=2, label="Smoothed (20 ep)")
        ax.set_title("Training Reward Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  [Plot] Training reward curve saved to {path}")


# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def make_env(df: pd.DataFrame, feature_cols: list, turb_threshold: float = None):
    """Returns a thunk (zero-argument callable) that constructs the env."""
    def _init():
        return CryptoTradingEnv(
            df              = df,
            feature_cols    = feature_cols,
            turb_threshold  = turb_threshold,
        )
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE TURBULENCE THRESHOLD
# ──────────────────────────────────────────────────────────────────────────────

def compute_turbulence(close: pd.Series, lookback: int = 252) -> pd.Series:
    """
    Single-asset turbulence index.

    Measures how far the current log-return deviates from the recent
    historical distribution (a Mahalanobis-like z-score squared).

    A high turbulence score signals a market crash / extreme move.
    We use the 90th percentile of TRAINING turbulence as the threshold.
    Positions are force-closed when turbulence exceeds this threshold.
    """
    log_ret = np.log(close / close.shift(1)).fillna(0)
    turb    = pd.Series(np.zeros(len(close)), index=close.index)

    for i in range(lookback, len(close)):
        hist   = log_ret.iloc[i - lookback : i].values
        mu     = hist.mean()
        sigma  = hist.std() + 1e-8
        y_t    = log_ret.iloc[i]
        turb.iloc[i] = ((y_t - mu) / sigma) ** 2

    return turb


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def run_training(regime: str = "all"):
    """
    Sets up the environment, builds the CLSTM-PPO agent, and trains it.
    regime: 'all', 'bull', 'bear', or 'crab'. Filters training data by date.
    """
    print("\n" + "=" * 70)
    print(f"  PHASE 3 — CLSTM-PPO TRAINING ({regime.upper()} REGIME)")
    print("=" * 70)

    # Automatically load Optuna optimized hyperparameters if they exist for this regime
    optuna_file = os.path.join(config.RESULTS_DIR, f"optuna_best_params_{regime}.json")
    if os.path.exists(optuna_file):
        print(f"\n[Optuna] Loading optimized hyperparameters for {regime} regime...")
        with open(optuna_file, "r") as f:
            best_params = json.load(f)
            # Override config defaults dynamically
            for k, v in best_params.items():
                config_key = k.upper()
                if hasattr(config, config_key):
                    setattr(config, config_key, v)
                    print(f"  -> config.{config_key} = {v}")
    else:
        print(f"\n[Optuna] No optimized hyperparameters found at {optuna_file}. Using defaults from config.py.")

    print("\n[Step 1] Loading data …")
    
    train_dfs = []
    val_dfs   = []

    symbols_to_process = config.MULTI_COINS if config.MULTI_COIN_MODE else [config.SYMBOL]
    
    for symbol in symbols_to_process:
        symbol_file = symbol.replace("/", "_")
        t_path = os.path.join(config.DATA_DIR, f"{symbol_file}_train_features.csv")
        v_path = os.path.join(config.DATA_DIR, f"{symbol_file}_val_features.csv")
        
        # Fallback to single coin mode paths if the explicit coin paths don't exist
        if not os.path.exists(t_path) and symbol == config.SYMBOL:
            t_path = config.TRAIN_FEAT_PATH
            v_path = config.VAL_FEAT_PATH

        if not os.path.exists(t_path):
            raise FileNotFoundError(f"{t_path} not found. Run Phases 1 & 2 first.")

        t_df = pd.read_csv(t_path, index_col=0, parse_dates=True)
        v_df = pd.read_csv(v_path, index_col=0, parse_dates=True)
        
        # ── Regime Filtering (Only filter training data) ───────────────
        if regime in config.REGIME_RANGES:
            start_date, end_date = config.REGIME_RANGES[regime]
            # Handle open-ended ranges properly
            if start_date and end_date:
                t_df = t_df.loc[start_date:end_date]
            elif start_date:
                t_df = t_df.loc[start_date:]
            elif end_date:
                t_df = t_df.loc[:end_date]
            
        train_dfs.append(t_df)
        val_dfs.append(v_df)
        
        print(f"  {symbol} ({regime}) -> Train: {len(t_df):,} rows | Val: {len(v_df):,} rows")

    with open(config.GOLDEN_FEATURES_PATH) as fp:
        golden_features = json.load(fp)

    print(f"  Golden features ({len(golden_features)}): {golden_features}")

    # ── Turbulence ─────────────────────────────────────────────────────────
    print("\n[Step 2] Computing turbulence index (using first coin for threshold) …")
    
    # We base the global turbulence threshold on the primary coin (BTC usually)
    train_turb = compute_turbulence(train_dfs[0]["Close"])
    
    turb_threshold = float(
        np.nanpercentile(train_turb[train_turb > 0],
                         config.TURBULENCE_PERCENTILE)
    )
    config.TURBULENCE_THRESHOLD = turb_threshold
    print(f"  Global Turbulence threshold ({config.TURBULENCE_PERCENTILE}th pct): {turb_threshold:.4f}")

    # Attach turbulence to DataFrames for the environment
    for i in range(len(train_dfs)):
        t_turb = compute_turbulence(train_dfs[i]["Close"])
        v_turb = compute_turbulence(val_dfs[i]["Close"])
        train_dfs[i] = train_dfs[i].copy()
        val_dfs[i]   = val_dfs[i].copy()
        train_dfs[i]["Turbulence"] = t_turb.values
        val_dfs[i]["Turbulence"]   = v_turb.values

    # ── GPU / Device detection ─────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  ✅  GPU detected: {gpu_name}  ({vram_gb:.1f} GB VRAM)")
        print(f"      Training will run on CUDA.")
    else:
        device = "cpu"
        print("\n  ⚠️   No CUDA GPU found — training on CPU.")
        print("      Install CUDA-enabled PyTorch to use your GPU.")

    # ── Build Environments ─────────────────────────────────────────────────
    print("\n[Step 3] Building environments …")
    n_envs = config.N_ENVS
    
    env_fns = []
    for i in range(n_envs):
        df_idx = i % len(train_dfs)
        env_fns.append(make_env(train_dfs[df_idx], golden_features, turb_threshold))

    if n_envs > 1:
        train_env = DummyVecEnv(env_fns)
        print(f"  Train env : DummyVecEnv  × {n_envs} workers")
    else:
        train_env = DummyVecEnv(env_fns)
        print("  Train env : DummyVecEnv × 1")
        
    val_env = DummyVecEnv([make_env(val_dfs[0], golden_features, turb_threshold)])
    print("  Val   env : DummyVecEnv × 1 (evaluating on primary coin)")

    # ── Build or load model ────────────────────────────────────────────────
    
    # Append regime to model name so we don't overwrite
    base_model_path = config.MODEL_PATH
    if regime != "all":
        base_model_path = f"{base_model_path}_{regime}"
    
    model_zip = base_model_path + ".zip"
    
    if os.path.exists(model_zip) and not config.FORCE_RETRAIN:
        print(f"\n[Step 4] Loading pre-trained model from {model_zip} …")
        model = PPO.load(base_model_path, env=train_env)
        print("  Model loaded. Skipping training.")
        callback = TrainingMonitorCallback(verbose=0)
    else:
        print(f"\n[Step 4] Building CLSTM-PPO model ({regime} regime) …")
        n_feat = len(golden_features)

        model = PPO(
            policy     = "MlpPolicy",
            env        = train_env,
            device     = device,           # ← GPU if available, else CPU
            # ── PPO Hyperparameters ──────────────────────────────────
            learning_rate = config.LEARNING_RATE,
            n_steps       = config.N_STEPS,
            batch_size    = config.BATCH_SIZE,
            n_epochs      = config.N_EPOCHS,
            gamma         = config.GAMMA,
            gae_lambda    = config.GAE_LAMBDA,
            clip_range    = config.CLIP_RANGE,
            ent_coef      = config.ENT_COEF,
            vf_coef       = config.VF_COEF,
            max_grad_norm = config.MAX_GRAD_NORM,
            seed          = config.SEED,
            verbose       = 0,
            # ── Custom CLSTM feature extractor ──────────────────────
            policy_kwargs = dict(
                features_extractor_class  = CLSTMFeatureExtractor,
                features_extractor_kwargs = dict(
                    seq_len           = config.SEQ_LEN,
                    n_market_features = n_feat,
                    hidden_size       = config.LSTM_HIDDEN_SIZE,
                    mlp_hidden        = config.MLP_HIDDEN_SIZE,
                    dropout           = config.DROPOUT_RATE,
                ),
                # Actor and Critic hidden layers AFTER the CLSTM extractor
                net_arch = dict(pi=[256, 128], vf=[256, 128]),
                # Adam optimizer settings
                optimizer_kwargs = dict(eps=1e-8, betas=(0.9, 0.999)),
            ),
        )

        print(f"\n  Model architecture:")
        print(f"    CLSTM: {n_feat} features × {config.SEQ_LEN} steps "
              f"→ {config.LSTM_HIDDEN_SIZE} hidden → {config.MLP_HIDDEN_SIZE} out")
        print(f"    Actor / Critic: [256, 128] MLP heads")
        print(f"    Total PPO steps: {config.TOTAL_TIMESTEPS:,}")

        # ── Train ──────────────────────────────────────────────────────────
        print("\n[Step 5] Training (this may take a while) …")
        # Save best model callback setup
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=os.path.join(config.MODELS_DIR, f"best_{regime}"),
            log_path=config.RESULTS_DIR,
            eval_freq=max(10_000, config.TOTAL_TIMESTEPS // 20),
            deterministic=True,
            render=False,
        )
        callback = TrainingMonitorCallback(log_freq=10, verbose=1)
        model.learn(
            total_timesteps = config.TOTAL_TIMESTEPS,
            callback        = [callback, eval_callback],
            progress_bar    = True,   # tqdm progress bar
        )

        # ── Save model ─────────────────────────────────────────────────────
        model.save(base_model_path)
        print(f"\n  [Save] Model saved to {model_zip}")

        # Save training reward curve
        reward_plot = os.path.join(config.RESULTS_DIR, "training_reward_curve.png")
        callback.save_reward_plot(reward_plot)



    print("\n  PHASE 3 TRAINING COMPLETE ✓")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST HELPER (used here and in Phase 4)
# ──────────────────────────────────────────────────────────────────────────────

def run_backtest(model: PPO,
                 df_scaled: pd.DataFrame,
                 feature_cols: list,
                 turb_threshold: float = None,
                 initial_balance: float = config.INITIAL_BALANCE) -> dict:
    """
    Runs the trained PPO agent deterministically through a dataset.

    Returns a dictionary with:
    • portfolio_values  : list of portfolio value at each step
    • positions         : list of position allocation
    • actions           : list of raw action values
    • dates             : list of corresponding timestamps
    • trade_log         : list of annotated trade dicts
    • metrics           : dict of performance metrics
    """
    env = CryptoTradingEnv(
        df              = df_scaled,
        feature_cols    = feature_cols,
        initial_balance = initial_balance,
        turb_threshold  = turb_threshold,
        transaction_fee = config.TRANSACTION_FEE,
        slippage        = config.SLIPPAGE,
        is_eval         = True
    )
    obs, _ = env.reset()

    portfolio_values = [initial_balance]
    positions        = [0.0]
    actions_taken    = []
    
    # Initialize prices with the starting price
    prices           = [env.df.iloc[env.current_step]['Close']]

    # Dates start from SEQ_LEN (warm-up period)
    start_idx = config.SEQ_LEN
    dates     = list(df_scaled.index)[start_idx:]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position"])
        actions_taken.append(float(action[0]))
        prices.append(info["current_price"])

        if terminated or truncated:
            break

    # Align lengths
    min_len          = min(len(portfolio_values), len(dates))
    portfolio_values = portfolio_values[:min_len]
    dates            = dates[:min_len]
    positions        = positions[:min_len]
    prices           = prices[:min_len]

    # ── Metrics ────────────────────────────────────────────────────────────
    pv   = np.array(portfolio_values)
    rets = np.diff(pv) / (pv[:-1] + 1e-8)
    rets = rets[np.isfinite(rets)]

    total_return = float((pv[-1] / pv[0]) - 1)

    ann_factor  = np.sqrt(8760)   # Hourly crypto market (365*24)
    sharpe      = float((np.mean(rets) / (np.std(rets) + 1e-8)) * ann_factor) \
                  if len(rets) > 1 else 0.0

    peak        = np.maximum.accumulate(pv)
    drawdown    = (peak - pv) / (peak + 1e-8)
    max_dd      = float(np.max(drawdown))

    win_rate    = float(np.sum(rets > 0) / len(rets)) if len(rets) > 0 else 0.0
    loss_rate   = float(np.sum(rets < 0) / len(rets)) if len(rets) > 0 else 0.0

    win_rets = rets[rets > 0]
    loss_rets = rets[rets < 0]
    avg_win  = float(np.mean(win_rets)) if len(win_rets) > 0 else 0.0
    avg_loss = float(np.mean(loss_rets)) if len(loss_rets) > 0 else 0.0
    max_win  = float(np.max(win_rets)) if len(win_rets) > 0 else 0.0
    max_loss = float(np.min(loss_rets)) if len(loss_rets) > 0 else 0.0

    sortino     = float((np.mean(rets) / (np.std(loss_rets) + 1e-8)) * ann_factor) \
                  if len(loss_rets) > 0 else float("inf")

    return {
        "portfolio_values": portfolio_values,
        "positions":        positions,
        "actions":          actions_taken,
        "prices":           prices,
        "dates":            dates,
        "trade_log":        env.trade_log,
        "total_trades":     env.total_trades,
        "total_fees":       env.total_fees_paid,
        "total_slippage":   env.total_slippage,
        "sl_triggers":      env.risk_manager.sl_count,
        "tp_triggers":      env.risk_manager.tp_count,
        "turb_exits":       env.turb_exits,
        "metrics": {
            "total_return":    total_return,
            "sharpe_ratio":    sharpe,
            "max_drawdown":    max_dd,
            "win_rate":        win_rate,
            "loss_rate":       loss_rate,
            "avg_win":         avg_win,
            "avg_loss":        avg_loss,
            "max_win":         max_win,
            "max_loss":        max_loss,
            "sortino_ratio":   sortino,
            "final_portfolio": float(pv[-1]),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLSTM-PPO Agent")
    parser.add_argument("--regime", type=str, default="all", choices=["all", "bull", "bear", "crab"],
                        help="Filter training data by market regime")
    args = parser.parse_args()
    
    model = run_training(regime=args.regime)
    print(f"\nPhase 3 standalone test PASSED ({args.regime})")
