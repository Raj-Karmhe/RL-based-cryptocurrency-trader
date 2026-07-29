"""
phase3_train.py - PPO Model Training and Validation

This script trains a PPO reinforcement learning agent on historical cryptocurrency features.
It sets up vectorized Gymnasium environments, attaches the custom recurrent feature extractor,
defines PPO hyperparameters, monitors training progress using a callback, and saves the trained model.
"""

import os
import sys
import json
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive plotting
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings("ignore")

# Force immediate log flushing
sys.stdout.reconfigure(line_buffering=True)

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase3_model import MarketRecurrentExtractor

class TradingMonitorCallback(BaseCallback):
    """
    Custom callback to track and log episodic performance metrics (portfolio value, trade count)
    during PPO training.
    """
    def __init__(self, log_frequency: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_frequency
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.timestep_records = []
        
        # Per-environment accumulators (initialized on first step)
        self._per_env_reward = None
        self._per_env_length = None
        
    def _on_step(self) -> bool:
        # Handle vectorized environments - track per-env episode stats
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals.get("infos", [{}] * len(rewards))
        n_envs = len(rewards)
        
        # Lazy initialization of per-env accumulators
        if self._per_env_reward is None:
            self._per_env_reward = [0.0] * n_envs
            self._per_env_length = [0] * n_envs
        
        for env_idx in range(n_envs):
            self._per_env_reward[env_idx] += rewards[env_idx]
            self._per_env_length[env_idx] += 1
            
            if dones[env_idx]:
                self.episode_rewards.append(self._per_env_reward[env_idx])
                self.episode_lengths.append(self._per_env_length[env_idx])
                self.timestep_records.append(self.num_timesteps)
                
                if infos[env_idx] and "portfolio_value" in infos[env_idx]:
                    self.portfolio_values.append(infos[env_idx]["portfolio_value"])
                else:
                    self.portfolio_values.append(config.INITIAL_BALANCE)
                    
                episode_num = len(self.episode_rewards)
                if self.verbose and episode_num % self.log_freq == 0:
                    recent_avg_reward = np.mean(self.episode_rewards[-10:])
                    last_portfolio = self.portfolio_values[-1]
                    recent_avg_portfolio = np.mean(self.portfolio_values[-10:])
                    
                    print(
                        f"  Episode {episode_num:5d} | Steps {self.num_timesteps:8,} | "
                        f"Avg Reward: {recent_avg_reward:8.5f} | "
                        f"Final Portfolio: ${last_portfolio:12,.2f} | "
                        f"Avg Portfolio: ${recent_avg_portfolio:12,.2f}"
                    )
                    
                # Reset accumulators for this specific environment
                self._per_env_reward[env_idx] = 0.0
                self._per_env_length[env_idx] = 0
            
        return True
        
    def save_reward_curve(self, save_path: str):
        """
        Saves a plot of the training rewards.
        """
        if not self.episode_rewards:
            return
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.timestep_records, self.episode_rewards, alpha=0.3, color="gray", label="Raw Rewards")
        
        # Add rolling average line
        smoothed = pd.Series(self.episode_rewards).rolling(15, min_periods=1).mean()
        ax.plot(self.timestep_records, smoothed.values, color="blue", linewidth=2, label="Rolling Mean (15 ep)")
        
        ax.set_title("PPO Agent Training Reward Curve")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved training reward curve to {save_path}")

def run_deterministic_eval(model, df_val, golden_cols):
    """
    Evaluates the model on validation data under deterministic conditions.
    """
    print("\nRunning deterministic evaluation on Validation Set...")
    val_env = CryptoTradingEnv(df_val, golden_cols, is_eval=True)
    obs, _ = val_env.reset()
    
    done = False
    truncated = False
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = val_env.step(action)
        
    final_portfolio = info["portfolio_value"]
    net_return = (final_portfolio - config.INITIAL_BALANCE) / config.INITIAL_BALANCE
    total_trades = info["trades"]
    
    print(f"  Validation Final Portfolio Value: ${final_portfolio:12,.2f}")
    print(f"  Validation Total Trades         : {total_trades}")
    print(f"  Validation Cumulative Return     : {net_return * 100:.2f}%")
    
    return final_portfolio, net_return

def execute_training(is_test_run: bool = False):
    """
    Orchestrates the environment setup, model instantiation, training, and evaluation.
    """
    print("=" * 60)
    print("Starting PPO Reinforcement Learning Agent Training")
    print("=" * 60)
    
    # 1. Load data
    if not os.path.exists(config.TRAIN_FEAT_PATH) or not os.path.exists(config.VAL_FEAT_PATH):
        raise FileNotFoundError("Feature files not found. Execute phase1_feature_engineering.py first.")
        
    train_df = pd.read_csv(config.TRAIN_FEAT_PATH, index_col="Date", parse_dates=True)
    val_df = pd.read_csv(config.VAL_FEAT_PATH, index_col="Date", parse_dates=True)
    
    # 2. Load golden feature columns
    if not os.path.exists(config.GOLDEN_FEATURES_PATH):
        raise FileNotFoundError("Golden features list not found. Execute phase2_feature_selection.py first.")
        
    with open(config.GOLDEN_FEATURES_PATH, "r") as f:
        golden_cols = json.load(f)
        
    print(f"Loaded {len(golden_cols)} Golden Features for State Space.")
    
    # 3. Build vectorized training environment with N_ENVS parallel envs
    def make_train_env():
        return CryptoTradingEnv(train_df, golden_cols, is_eval=False)
        
    train_env = DummyVecEnv([make_train_env for _ in range(config.N_ENVS)])
    print(f"Created {config.N_ENVS} parallel training environments.")
    
    # 4. Define custom neural network policy settings
    policy_kwargs = dict(
        features_extractor_class=MarketRecurrentExtractor,
        features_extractor_kwargs=dict(
            sequence_length=config.SEQ_LEN,
            num_features=len(golden_cols),
            gru_hidden=config.LSTM_HIDDEN_SIZE,
            mlp_hidden=config.MLP_HIDDEN_SIZE,
            dropout_prob=config.DROPOUT_RATE
        ),
        net_arch=dict(pi=[64], vf=[64])  # Actor/critic heads following feature extraction
    )
    
    # 5. Instantiate PPO
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        ent_coef=config.ENT_COEF,
        vf_coef=config.VF_COEF,
        max_grad_norm=config.MAX_GRAD_NORM,
        target_kl=config.TARGET_KL,
        seed=config.SEED,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Set timesteps (shorten if this is a verification run)
    timesteps = 2048 if is_test_run else config.TOTAL_TIMESTEPS
    print(f"Training agent for {timesteps:,} timesteps...")
    
    # 6. Train model with custom monitor callback
    monitor_callback = TradingMonitorCallback(log_frequency=10 if is_test_run else 50)
    
    model.learn(total_timesteps=timesteps, callback=monitor_callback)
    print("\nTraining completed.")
    
    # Save training reward plots
    reward_curve_path = os.path.join(config.RESULTS_DIR, "training_reward_curve.png")
    monitor_callback.save_reward_curve(reward_curve_path)
    
    # Save trained model checkpoint
    model.save(config.MODEL_PATH)
    print(f"Saved trained PPO model checkpoint to {config.MODEL_PATH}")
    
    # 7. Evaluate on validation set
    run_deterministic_eval(model, val_df, golden_cols)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="Perform a quick verification training loop")
    args = parser.parse_args()
    
    execute_training(is_test_run=args.test_run)
