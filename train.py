"""
train.py

Train PPO Agent
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

import pandas as pd
from split_data import split_data
from environment import TradingEnvironment


# ----------------------------------------------------
# Build Dataset
# ----------------------------------------------------

df = pd.read_csv(
    "processed_data_with_regime.csv"
)

train_df, test_df = split_data(df)


# ----------------------------------------------------
# Create Environment
# ----------------------------------------------------

def make_env():

    env = TradingEnvironment(train_df)

    env = Monitor(env)

    return env


env = DummyVecEnv([make_env])

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)

# ----------------------------------------------------
# Save Directory
# ----------------------------------------------------

os.makedirs("models", exist_ok=True)

os.makedirs("logs", exist_ok=True)

# ----------------------------------------------------
# Checkpoints
# ----------------------------------------------------

checkpoint_callback = CheckpointCallback(

    save_freq=10000,

    save_path="models/",

    name_prefix="ppo_crypto"

)

# ----------------------------------------------------
# PPO Model
# ----------------------------------------------------

model = PPO(

    policy="MlpPolicy",

    env=env,

    learning_rate=3e-4,

    n_steps=2048,

    batch_size=64,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    verbose=1,

    tensorboard_log="./logs/"

)

# ----------------------------------------------------
# Train
# ----------------------------------------------------

model.learn(

    total_timesteps=100000,

    callback=checkpoint_callback

)

# ----------------------------------------------------
# Save
# ----------------------------------------------------

model.save("models/final_model")

env.save("models/vec_normalize.pkl")

print("\nTraining Finished Successfully!")