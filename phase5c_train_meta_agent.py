import os
import sys
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv
from phase5c_meta_environment import MetaTradingEnv
from phase3_train import compute_turbulence

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: RESOLVE MODEL PATHS
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_model(regime):
    # Same resolution logic as phase5
    path_zip = os.path.join(config.MODELS_DIR, f"{config.MODEL_NAME}_{regime}.zip")
    if os.path.exists(path_zip): return path_zip
    legacy_zip = os.path.join(config.MODELS_DIR, f"clstm_ppo_strategy_0_optimized_no_exit_only_long_{regime}.zip")
    if os.path.exists(legacy_zip): return legacy_zip
    raise FileNotFoundError(f"Could not find model for regime: {regime}")

class MetaTrainingMonitorCallback(BaseCallback):
    def __init__(self, log_freq: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_reward = 0.0
        self.episode_rewards = []
        self.portfolio_values = []

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        
        # Log every N steps
        if self.verbose and self.num_timesteps % self.log_freq == 0:
            last_p = 0
            infos = self.locals.get("infos", [{}])
            if infos and "portfolio_value" in infos[0]:
                last_p = infos[0]["portfolio_value"]
            
            print(f"  Meta Steps {self.num_timesteps:8,} | Last Portfolio: ${last_p:12,.0f}")
            sys.stdout.flush()

        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self._ep_reward = 0.0
        return True

def train_meta_agent():
    print("\n" + "=" * 70)
    print("  PHASE 5C — TRAINING RL META-AGENT")
    print("=" * 70)
    
    # 1. Resolve and Load Expert Models
    print("[Step 1] Resolving expert models...")
    model_bull_path = _resolve_model("bull")
    model_bear_path = _resolve_model("bear")
    model_crab_path = _resolve_model("crab")
    print(f"  [Models] Bull : {model_bull_path}")
    print(f"  [Models] Bear : {model_bear_path}")
    print(f"  [Models] Crab : {model_crab_path}")
    
    # We need a dummy env to load the models successfully
    print("[Step 2] Loading features and creating Base Environment...")
    t_path = os.path.join(config.DATA_DIR, f"{config.SYMBOL_FILE}_train_features.csv")
    if not os.path.exists(t_path):
        t_path = config.TRAIN_FEAT_PATH
        
    t_df = pd.read_csv(t_path, index_col=0, parse_dates=True)
    
    # Determine feature space from the Bull model
    probe = PPO.load(model_bull_path[:-4])
    obs_dim = probe.observation_space.shape[0]
    n_feat = (obs_dim - 2) // config.SEQ_LEN
    registry = getattr(config, "FEATURE_REGISTRY", {})
    features_path = registry.get(n_feat, config.GOLDEN_FEATURES_PATH)
    with open(features_path) as fp:
        golden_features = json.load(fp)
    del probe
    
    turb = compute_turbulence(t_df['Close'])
    turb_threshold = float(np.nanpercentile(turb[turb > 0], config.TURBULENCE_PERCENTILE))
    
    # Create the base environment for training
    base_env = CryptoTradingEnv(
        df=t_df,
        feature_cols=golden_features,
        turb_threshold=turb_threshold
    )
    
    # Wrap in DummyVecEnv so SB3 can load the models properly
    dummy_base_env = DummyVecEnv([lambda: base_env])
    
    print("[Step 3] Loading frozen experts into memory...")
    model_bull = PPO.load(model_bull_path[:-4], env=dummy_base_env)
    model_bear = PPO.load(model_bear_path[:-4], env=dummy_base_env)
    model_crab = PPO.load(model_crab_path[:-4], env=dummy_base_env)
    
    print("[Step 4] Building Meta-Environment...")
    meta_env_fn = lambda: MetaTradingEnv(base_env, model_bull, model_bear, model_crab)
    meta_vec_env = DummyVecEnv([meta_env_fn])
    meta_vec_env = VecNormalize(meta_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    print("[Step 5] Initializing Meta-Agent PPO...")
    meta_agent = PPO(
        policy="MlpPolicy",
        env=meta_vec_env,
        learning_rate=config.META_AGENT_LR,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
        verbose=0
    )
    
    print(f"[Step 6] Training Meta-Agent for {config.META_AGENT_TIMESTEPS} timesteps...")
    callback = MetaTrainingMonitorCallback(log_freq=2048, verbose=1)
    meta_agent.learn(total_timesteps=config.META_AGENT_TIMESTEPS, callback=callback)
    
    meta_agent.save(config.META_AGENT_PATH)
    meta_vec_env.save(os.path.join(config.RESULTS_DIR, "meta_vec_normalize.pkl"))
    print(f"\n[Save] Meta-Agent saved to {config.META_AGENT_PATH}.zip")
    print(f"[Save] Meta-Agent VecNormalize stats saved to {os.path.join(config.RESULTS_DIR, 'meta_vec_normalize.pkl')}")
    print("  PHASE 5C TRAINING COMPLETE ✓")

if __name__ == "__main__":
    train_meta_agent()
