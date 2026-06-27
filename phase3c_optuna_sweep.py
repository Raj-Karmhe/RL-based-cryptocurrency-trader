import optuna
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_model import CLSTMFeatureExtractor
from phase3_train import compute_turbulence, run_backtest, make_env

warnings.filterwarnings("ignore")

def objective(trial):
    # ── Sample hyperparameters ────────────────────────────────────────────────
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma         = trial.suggest_float("gamma", 0.90, 0.999)
    gae_lambda    = trial.suggest_float("gae_lambda", 0.90, 0.99)
    ent_coef      = trial.suggest_float("ent_coef", 1e-8, 0.05, log=True)
    vf_coef       = trial.suggest_float("vf_coef", 0.1, 0.9)
    n_steps       = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    lstm_hidden   = trial.suggest_categorical("lstm_hidden", [32, 64, 128])
    mlp_hidden    = trial.suggest_categorical("mlp_hidden", [64, 128, 256])

    # ── Load Data ────────────────────────────────────────────────────────────
    sym_file = config.SYMBOL.replace("/", "_")
    train_path = os.path.join(config.DATA_DIR, f"{sym_file}_train_features.csv")
    val_path   = os.path.join(config.DATA_DIR, f"{sym_file}_val_features.csv")
    
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    val_df   = pd.read_csv(val_path, index_col=0, parse_dates=True)
    
    with open(config.GOLDEN_FEATURES_PATH) as fp:
        golden_features = json.load(fp)
        
    turb_threshold = float(np.nanpercentile(
        compute_turbulence(train_df['Close'])[compute_turbulence(train_df['Close']) > 0],
        config.TURBULENCE_PERCENTILE
    ))
    
    # ── Build Environments ───────────────────────────────────────────────────
    train_env = DummyVecEnv([make_env(train_df, golden_features, turb_threshold)])
    
    n_feat = len(golden_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        device        = device,
        learning_rate = learning_rate,
        n_steps       = n_steps,
        batch_size    = batch_size,
        gamma         = gamma,
        gae_lambda    = gae_lambda,
        ent_coef      = ent_coef,
        vf_coef       = vf_coef,
        verbose       = 0,
        policy_kwargs = dict(
            features_extractor_class  = CLSTMFeatureExtractor,
            features_extractor_kwargs = dict(
                seq_len           = config.SEQ_LEN,
                n_market_features = n_feat,
                hidden_size       = lstm_hidden,
                mlp_hidden        = mlp_hidden,
                dropout           = config.DROPOUT_RATE,
            ),
            net_arch = dict(pi=[128, 64], vf=[128, 64]),
        ),
    )

    # Train for a small number of steps
    try:
        model.learn(total_timesteps=50_000, progress_bar=True)
    except Exception as e:
        # If the hyperparameters are completely unstable and cause a crash, prune it.
        raise optuna.TrialPruned()

    # ── Backtest on Validation Set ───────────────────────────────────────────
    val_turb = compute_turbulence(val_df['Close'])
    val_df['Turbulence'] = val_turb.values
    
    results = run_backtest(model, val_df, golden_features, turb_threshold)
    sharpe = results["metrics"]["sharpe_ratio"]
    
    return sharpe

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optuna trials")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  PHASE 3c — OPTUNA HYPERPARAMETER SWEEP")
    print("=" * 70)

    study = optuna.create_study(direction="maximize", study_name="CLSTM-PPO-Sweep")
    study.optimize(objective, n_trials=args.n_trials)

    print("\n  [Sweep Complete]")
    print("  Best trial:")
    best = study.best_trial
    print(f"    Value (Sharpe Ratio): {best.value}")
    print("    Params: ")
    for key, value in best.params.items():
        print(f"      {key}: {value}")
        
    out_path = os.path.join(config.RESULTS_DIR, "best_hyperparameters.json")
    with open(out_path, "w") as f:
        json.dump(best.params, f, indent=4)
    print(f"  Saved best parameters to {out_path}")
