"""
phase6_optuna_tune.py — Hyperparameter Optimization with Optuna
================================================================
Phase 6 — Automated hyperparameter search for the CLSTM-PPO pipeline.

WHAT THIS DOES
--------------
Uses Optuna's Tree-Structured Parzen Estimator (TPE) sampler to find the
optimal combination of PPO hyperparameters by training a short trial run and
evaluating the agent's Sharpe ratio on the validation set.

PARAMETERS SEARCHED
--------------------
  PPO Core:
    • learning_rate       [1e-5, 1e-3] (log-uniform)
    • n_steps             [512, 4096]  (step 512)
    • batch_size          [32, 512]    (powers of 2)
    • n_epochs            [3, 15]
    • gamma               [0.90, 0.999]
    • gae_lambda          [0.90, 0.999]
    • clip_range          [0.1, 0.4]
    • ent_coef            [1e-6, 0.01] (log-uniform)
    • vf_coef             [0.3, 0.9]

  Neural Architecture:
    • lstm_hidden_size    [32, 256]    (powers of 2)
    • mlp_hidden_size     [128, 512]   (powers of 2)
    • dropout_rate        [0.0, 0.3]

  Reward:
    • sortino_weight      [0.3, 0.9]  (only relevant for Strategy 3)

SEEDED BASELINE
---------------
Trial #0 is always your current config.py defaults so Optuna starts with
a known reference point and only improves from there.

USAGE
-----
  # Run 30 trials on the crab regime (recommended for speed):
  python phase6_optuna_tune.py --regime crab --n_trials 30

  # Run on all regimes:
  python phase6_optuna_tune.py --regime all --n_trials 50

  # Quick test with fewer timesteps:
  python phase6_optuna_tune.py --regime bull --n_trials 10 --timesteps 100000

OUTPUT
------
  • results/optuna_study_<regime>.pkl    — resumable Optuna study object
  • results/optuna_best_params_<regime>.json  — best hyperparameters found
  • results/optuna_history_<regime>.png  — optimization history chart
"""

import json
import os
import sys
import io
import warnings

# Force UTF-8 encoding for stdout/stderr to prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_model import CLSTMFeatureExtractor
from phase3_train import compute_turbulence, run_backtest, make_env


# ──────────────────────────────────────────────────────────────────────────────
# WINDOWS NOTIFICATION CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

def _notify(study: optuna.Study, trial: optuna.Trial) -> None:
    """
    Optuna callback: prints a clean summary line to the terminal after every trial.
    """
    value  = trial.value
    n      = trial.number
    total  = len(study.trials)
    best   = study.best_value if study.best_value is not None else float("-inf")
    failed = (value is None or value == float("-inf"))

    if failed:
        status = "FAILED"
        sharpe_str = "N/A"
    else:
        sharpe_str = f"{value:+.4f}"
        status = "*** NEW BEST ***" if value >= best else "done"

    print(
        f"  [Trial {n:3d}/{total:3d}] {status:<16} | "
        f"Sharpe: {sharpe_str:<10} | Best so far: {best:+.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# REGIME DATE RANGES (must match phase3_train.py)
# ──────────────────────────────────────────────────────────────────────────────
REGIME_RANGES = config.REGIME_RANGES


# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMS PRESET (used as Trial #0)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_PARAMS = {
    "learning_rate":    config.LEARNING_RATE,
    "n_steps":          config.N_STEPS,
    "batch_size":       config.BATCH_SIZE,
    "n_epochs":         config.N_EPOCHS,
    "gamma":            config.GAMMA,
    "gae_lambda":       config.GAE_LAMBDA,
    "clip_range":       config.CLIP_RANGE,
    "ent_coef":         config.ENT_COEF,
    "vf_coef":          config.VF_COEF,
    "lstm_hidden_size": config.LSTM_HIDDEN_SIZE,
    "mlp_hidden_size":  config.MLP_HIDDEN_SIZE,
    "dropout_rate":     config.DROPOUT_RATE,
    "sortino_weight":   config.SORTINO_WEIGHT,
}


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA HELPER
# ──────────────────────────────────────────────────────────────────────────────
def load_regime_data(regime: str, golden_features: list):
    """Loads and slices training data for the given regime."""
    primary_coin = config.MULTI_COINS[0]
    sym_file = primary_coin.replace('/', '_')
    train_path = os.path.join(config.DATA_DIR, f"{sym_file}_train_features.csv")

    df = pd.read_csv(train_path, index_col=0, parse_dates=True)

    if regime != "all" and regime in REGIME_RANGES:
        start, end = REGIME_RANGES[regime]
        mask = (df.index >= start) & (df.index <= end)
        df_regime = df[mask].copy()
        if len(df_regime) < 1000:
            print(f"  [Warning] Regime '{regime}' has only {len(df_regime)} rows. Using full dataset.")
            df_regime = df.copy()
    else:
        df_regime = df.copy()

    split = int(len(df_regime) * 0.7)
    train_df = df_regime.iloc[:split].copy()
    val_df   = df_regime.iloc[split:].copy()

    train_turb = compute_turbulence(train_df['Close'])
    turb_threshold = float(np.nanpercentile(train_turb[train_turb > 0], config.TURBULENCE_PERCENTILE))

    train_df['Turbulence'] = compute_turbulence(train_df['Close']).values
    val_df['Turbulence']   = compute_turbulence(val_df['Close']).values

    return train_df, val_df, turb_threshold


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def make_objective(regime: str, golden_features: list, trial_timesteps: int):
    """
    Returns an Optuna objective function for the given regime.
    The objective trains a short PPO trial and returns the validation Sharpe ratio.
    """
    train_df, val_df, turb_threshold = load_regime_data(regime, golden_features)

    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ─────────────────────────────────────────
        learning_rate    = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps          = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size       = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        n_epochs         = trial.suggest_int("n_epochs", 3, 15)
        gamma            = trial.suggest_float("gamma", 0.90, 0.999)
        gae_lambda       = trial.suggest_float("gae_lambda", 0.90, 0.999)
        clip_range       = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef         = trial.suggest_float("ent_coef", 1e-6, 0.01, log=True)
        vf_coef          = trial.suggest_float("vf_coef", 0.3, 0.9)
        lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256])
        mlp_hidden_size  = trial.suggest_categorical("mlp_hidden_size", [128, 256, 512])
        dropout_rate     = trial.suggest_float("dropout_rate", 0.0, 0.3)
        sortino_weight   = trial.suggest_float("sortino_weight", 0.3, 0.9)

        # Ensure batch_size <= n_steps
        if batch_size > n_steps:
            return float("-inf")

        # Temporarily override config for this trial
        config.LEARNING_RATE    = learning_rate
        config.N_STEPS          = n_steps
        config.BATCH_SIZE       = batch_size
        config.N_EPOCHS         = n_epochs
        config.GAMMA            = gamma
        config.GAE_LAMBDA       = gae_lambda
        config.CLIP_RANGE       = clip_range
        config.ENT_COEF         = ent_coef
        config.VF_COEF          = vf_coef
        config.LSTM_HIDDEN_SIZE = lstm_hidden_size
        config.MLP_HIDDEN_SIZE  = mlp_hidden_size
        config.DROPOUT_RATE     = dropout_rate
        config.SORTINO_WEIGHT   = sortino_weight

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            n_feat = len(golden_features)

            train_env = DummyVecEnv([make_env(train_df, golden_features, turb_threshold)])
            val_env   = DummyVecEnv([make_env(val_df,   golden_features, turb_threshold)])

            model = PPO(
                policy        = "MlpPolicy",
                env           = train_env,
                device        = device,
                learning_rate = learning_rate,
                n_steps       = n_steps,
                batch_size    = batch_size,
                n_epochs      = n_epochs,
                gamma         = gamma,
                gae_lambda    = gae_lambda,
                clip_range    = clip_range,
                ent_coef      = ent_coef,
                vf_coef       = vf_coef,
                max_grad_norm = config.MAX_GRAD_NORM,
                seed          = config.SEED,
                verbose       = 0,
                policy_kwargs = dict(
                    features_extractor_class  = CLSTMFeatureExtractor,
                    features_extractor_kwargs = dict(
                        seq_len           = config.SEQ_LEN,
                        n_market_features = n_feat,
                        hidden_size       = lstm_hidden_size,
                        mlp_hidden        = mlp_hidden_size,
                        dropout           = dropout_rate,
                    ),
                    net_arch = dict(pi=[256, 128], vf=[256, 128]),
                    optimizer_kwargs = dict(eps=1e-8, betas=(0.9, 0.999)),
                ),
            )

            model.learn(total_timesteps=trial_timesteps, progress_bar=False)

            # Evaluate on validation set — use Sharpe ratio as the objective
            results = run_backtest(model, val_df, golden_features, turb_threshold=turb_threshold)
            sharpe  = results["metrics"]["sharpe_ratio"]

            print(f"  Trial {trial.number:3d} | LR={learning_rate:.2e} | "
                  f"gamma={gamma:.3f} | LSTM={lstm_hidden_size} | "
                  f"sortino_w={sortino_weight:.2f} | Sharpe={sharpe:+.4f}")

            train_env.close()
            val_env.close()
            return float(sharpe)

        except Exception as e:
            print(f"  Trial {trial.number} FAILED: {e}")
            return float("-inf")

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────
def save_optimization_plots(study: optuna.Study, regime: str):
    """Saves optimization history and parameter importance plots."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 1. Optimization history
        values = [t.value for t in study.trials if t.value is not None and t.value > float("-inf")]
        trials = [t.number for t in study.trials if t.value is not None and t.value > float("-inf")]
        best_so_far = np.maximum.accumulate(values)

        axes[0].scatter(trials, values, alpha=0.5, s=20, color="#3498db", label="Trial Sharpe")
        axes[0].plot(trials, best_so_far, color="#e74c3c", lw=2, label="Best so far")
        axes[0].set_title(f"Optuna Optimization History ({regime})", fontweight="bold")
        axes[0].set_xlabel("Trial")
        axes[0].set_ylabel("Validation Sharpe Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Best param bar chart
        best = study.best_params
        keys   = list(best.keys())
        # Normalize values for display
        vals   = [best[k] for k in keys]
        axes[1].barh(keys, [abs(v) if isinstance(v, float) else v for v in vals],
                     color="#2ecc71", alpha=0.8)
        axes[1].set_title("Best Hyperparameters Found", fontweight="bold")
        axes[1].set_xlabel("Value (absolute)")

        plt.tight_layout()
        path = os.path.join(config.RESULTS_DIR, f"optuna_history_{regime}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [Plot] Optimization history saved to {path}")
    except Exception as e:
        print(f"  [Warning] Could not save plots: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def run_optuna(regime: str = "crab", n_trials: int = 30, trial_timesteps: int = 150_000, notify: bool = True):
    print("\n" + "=" * 70)
    print(f"  PHASE 6 — OPTUNA HYPERPARAMETER OPTIMIZATION ({regime.upper()} REGIME)")
    print("=" * 70)

    with open(config.GOLDEN_FEATURES_PATH) as fp:
        golden_features = json.load(fp)
    print(f"  [Features] {len(golden_features)} golden features loaded.")
    print(f"  [Search]   {n_trials} trials × {trial_timesteps:,} timesteps each")
    print(f"  [Device]   {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Use SQLite backend for parallel/multi-GPU execution
    db_path = os.path.join(config.RESULTS_DIR, f"optuna_study_{regime}.db")
    storage_name = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=f"clstm_ppo_{regime}",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=42),
    )

    if len(study.trials) == 0:
        print(f"  [Seed]     Enqueueing baseline trial from config.py defaults...")
        study.enqueue_trial(BASELINE_PARAMS)
    else:
        print(f"  [Resume]   Loaded existing study with {len(study.trials)} trials.")

    objective = make_objective(regime, golden_features, trial_timesteps)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_notify] if notify else [],
        catch=(Exception,),
    )

    # Print results
    print("\n" + "=" * 70)
    print(f"  BEST TRIAL: #{study.best_trial.number}")
    print(f"  BEST VALIDATION SHARPE: {study.best_value:+.4f}")
    print("=" * 70)
    print("\n  Best Hyperparameters:")
    for k, v in study.best_params.items():
        current = getattr(config, k.upper(), "N/A")
        print(f"    {k:<22} {v!r:<20}  (was: {current!r})")

    # Save best params as JSON
    best_path = os.path.join(config.RESULTS_DIR, f"optuna_best_params_{regime}.json")
    with open(best_path, "w") as fp:
        json.dump(study.best_params, fp, indent=2)
    print(f"\n  [Save]   Best params saved to {best_path}")
    print("         --> Copy these into config.py before running phase3_train.py!")

    save_optimization_plots(study, regime)
    print("\n  PHASE 6 COMPLETE")
    return study.best_params


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6: Optuna Hyperparameter Tuning")
    parser.add_argument("--regime",     type=str, default="crab",
                        choices=["all", "bull", "bear", "crab"],
                        help="Which regime's training data to optimize on (default: crab — fastest).")
    parser.add_argument("--n_trials",   type=int, default=30,
                        help="Number of Optuna trials to run (default: 30).")
    parser.add_argument("--timesteps",  type=int, default=150_000,
                        help="Training timesteps per trial (default: 150,000). Lower = faster but noisier.")
    parser.add_argument("--no-notify",  action="store_true",
                        help="Disable Windows toast + audio notifications after each trial.")
    args = parser.parse_args()

    run_optuna(regime=args.regime, n_trials=args.n_trials, trial_timesteps=args.timesteps, notify=not args.no_notify)
