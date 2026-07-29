"""
evaluate.py

Evaluate Trained PPO Agent
"""

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pandas as pd
from split_data import split_data
from environment import TradingEnvironment


# -----------------------------------------------------
# Load Processed Dataset
# -----------------------------------------------------

df = pd.read_csv("processed_data_with_regime.csv")

train_df, test_df = split_data(df)


# -----------------------------------------------------
# Create Test Environment
# -----------------------------------------------------

def make_env():

    from stable_baselines3.common.monitor import Monitor

    return Monitor(TradingEnvironment(test_df))


env = DummyVecEnv([make_env])

env = VecNormalize.load(
    "models/vec_normalize.pkl",
    env
)

# VERY IMPORTANT
# Do NOT update statistics while evaluating

env.training = False

env.norm_reward = False


# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = PPO.load(
    "models/final_model",
    env=env
)


# -----------------------------------------------------
# Reset
# -----------------------------------------------------

obs = env.reset()

done = [False]


# -----------------------------------------------------
# Store Results
# -----------------------------------------------------

portfolio_values = []

actions = []

step = 0


# -----------------------------------------------------
# Evaluation Loop
# -----------------------------------------------------

while not done[0]:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    portfolio = float(info[0]["portfolio_value"])

    portfolio_values.append(portfolio)

    actions.append(info[0]["executed_action"])

    step += 1

    if step % 500 == 0:

        print(f"Step : {step}")

# -----------------------------------------------------
# Results
# -----------------------------------------------------

if not portfolio_values:
    print("No steps completed. Test set may be too small.")
else:
    print("\nEvaluation Finished")

    print(f"Total Steps : {step}")

    print(f"Initial Portfolio : 100000")

    print(f"Final Portfolio : {portfolio_values[-1]:.2f}")

    total_return = (

        portfolio_values[-1] - 100000

    ) / 100000 * 100

    print(f"Return : {total_return:.2f}%")

    print(f"Maximum Portfolio : {max(portfolio_values):.2f}")

    print(f"Minimum Portfolio : {min(portfolio_values):.2f}")

    print(f"Total Trades : {len(actions)}")