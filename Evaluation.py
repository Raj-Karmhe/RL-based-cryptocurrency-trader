import numpy as np
import pandas as pd

from sb3_contrib import RecurrentPPO

from data_processor import CryptoDataProcessor
from Environment import Environment

import os
from Performance import generate_report

processor = CryptoDataProcessor(
    "data/BTCUSDT.csv"
)

train_df, val_df, test_df, features = processor.process()

model = RecurrentPPO.load(
    "models/ppo_crypto_realistic"
)

# Validation evaluation (0.5 year)
val_env = Environment(
    df=val_df,
    features=features,
    initial_balance=10000,
    fee=0.001,
    slippage=0.0005,
    stop_loss=0.05,
    take_profit=0.10
)
val_obs, _ = val_env.reset()
val_done = False
episode_start = np.array([True])
val_portfolio_history = []
while not val_done:
    action,
    val_obs, reward, val_done, _, info = val_env.step(action)
    episode_start = np.array([False])
    val_portfolio_history.append({
        "step": val_env.step_idx,
        "portfolio_value": info["portfolio_value"],
        "cash": info["cash"],
        "btc": info["btc"],
        "reward": reward
    })
# Save validation results
os.makedirs("results", exist_ok=True)
val_results_df = pd.DataFrame(val_portfolio_history)
val_results_df.to_csv("results/validation_results.csv", index=False)
print("Validation results saved to results/validation_results.csv")
val_portfolio_values = val_results_df["portfolio_value"].values
val_report = generate_report(val_portfolio_values, val_env.trade_log)
print("\n--- Validation Performance Report ---")
for key, value in val_report.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")

# Test evaluation (0.5 year)

env = Environment(
    df=test_df,
    features=features,

    initial_balance=10000,

    fee=0.001,

    slippage=0.0005,

    stop_loss=0.05,

    take_profit=0.10
)



done = False

observation, info = env.reset()

while not done:

    action = agent.select_action(
        observation,
        epsilon=0.0      # Pure exploitation
    )

    observation, reward, done, _, info = env.step(action)

    portfolio_history.append({
        "step": env.step_idx,
        "portfolio_value": info["portfolio_value"],
        "cash": info["cash"],
        "btc": info["btc"],
        "reward": reward
    })

os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(portfolio_history)
results_df.to_csv("results/evaluation_results.csv", index=False)
print("Evaluation results saved to results/evaluation_results.csv")

portfolio_values = results_df["portfolio_value"].values
report = generate_report(portfolio_values, env.trade_log)

print("\n--- Evaluation Performance Report ---")
for key, value in report.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value}")
