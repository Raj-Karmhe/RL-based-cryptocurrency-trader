"""
backtest.py

Backtest the trained PPO model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize

import pandas as pd
from split_data import split_data
from environment import TradingEnvironment


# ----------------------------------------
# Load Data
# ----------------------------------------

df = pd.read_csv("processed_data_with_regime.csv")

train_df, test_df = split_data(df)


# ----------------------------------------
# Environment
# ----------------------------------------

def make_env():
    return TradingEnvironment(test_df)


env = DummyVecEnv([make_env])

env = VecNormalize.load(
    "models/vec_normalize.pkl",
    env
)

env.training = False
env.norm_reward = False


# ----------------------------------------
# Load PPO
# ----------------------------------------

model = PPO.load(
    "models/final_model",
    env=env
)


# ----------------------------------------
# Backtest
# ----------------------------------------

obs = env.reset()

done = False

portfolio_history = []

reward_history = []

action_history = []

while not done:

    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = env.step(action)

    portfolio = float(info[0]["portfolio_value"])

    portfolio_history.append(portfolio)

    reward_history.append(float(reward[0]))

    action_history.append(info[0]["executed_action"])


# ----------------------------------------
# Metrics
# ----------------------------------------

portfolio = np.array(portfolio_history)

returns = np.diff(portfolio) / portfolio[:-1]

initial_portfolio = 100000

total_return = (
    portfolio[-1] - initial_portfolio
) / initial_portfolio

if np.std(returns) > 1e-8:
    sharpe = (
        np.mean(returns)
        /
        np.std(returns)
    ) * np.sqrt(252)
else:
    sharpe = 0.0

running_max = np.maximum.accumulate(portfolio)

drawdown = (
    portfolio - running_max
) / running_max

max_drawdown = np.min(drawdown)


buy_count = np.sum(np.array(action_history) > 0)

sell_count = np.sum(np.array(action_history) < 0)

hold_count = np.sum(np.array(action_history) == 0)


# ----------------------------------------
# Print
# ----------------------------------------

print("=" * 60)

print("BACKTEST RESULTS")

print("=" * 60)

print(f"Final Portfolio : {portfolio[-1]:.2f}")

print(f"Return           : {total_return*100:.2f}%")

print(f"Sharpe Ratio     : {sharpe:.3f}")

print(f"Max Drawdown     : {max_drawdown*100:.2f}%")

print()

print(f"Buy Actions      : {buy_count}")

print(f"Sell Actions     : {sell_count}")

print(f"Hold Actions     : {hold_count}")

print()

print(f"Highest Portfolio : {portfolio.max():.2f}")

print(f"Lowest Portfolio  : {portfolio.min():.2f}")

print("=" * 60)


# ----------------------------------------
# Plot
# ----------------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    portfolio,
    label="Portfolio Value",
    linewidth=2
)

plt.legend()

plt.title("Portfolio Value")

plt.xlabel("Step")

plt.ylabel("Portfolio")

plt.grid(True)

plt.show()


# ----------------------------------------
# Save CSV
# ----------------------------------------

results = pd.DataFrame({

    "Portfolio": portfolio_history,

    "Reward": reward_history,

    "Action": action_history

})

results.to_csv(

    "backtest_results.csv",

    index=False

)

print()

print("CSV Saved Successfully.")