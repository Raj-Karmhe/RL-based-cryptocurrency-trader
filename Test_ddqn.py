"""
test_ddqn.py

Evaluate a trained Double DQN agent.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from Environment import Environment
from DDQN_agent import DDQNAgent
from config import *


# ===================================
# Create Environment
# ===================================

env = Environment(train=False)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n


# ===================================
# Create Agent
# ===================================

agent = DDQNAgent(
    state_size,
    action_size
)

# Load trained weights
agent.online_net.load_state_dict(
    torch.load(
        SAVE_PATH,
        map_location=DEVICE
    )
)

agent.online_net.eval()

# Disable exploration
agent.epsilon = 0.0


# ===================================
# Testing Loop
# ===================================

state, info = env.reset()

done = False

portfolio_values = []
rewards = []
actions = []

while not done:

    action = agent.select_action(state)

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    portfolio_values.append(info["portfolio_value"])

    rewards.append(reward)

    actions.append(action)

    state = next_state


# ===================================
# Performance Metrics
# ===================================

portfolio_values = np.array(portfolio_values)

total_return = (
    portfolio_values[-1]
    -
    portfolio_values[0]
) / portfolio_values[0]

returns = np.diff(portfolio_values) / portfolio_values[:-1]

sharpe = (
    np.sqrt(365)
    *
    returns.mean()
    /
    (returns.std() + 1e-8)
)

running_max = np.maximum.accumulate(
    portfolio_values
)

drawdown = (
    portfolio_values
    -
    running_max
) / running_max

max_drawdown = drawdown.min()


print("=" * 50)

print(
    f"Total Return : {total_return*100:.2f}%"
)

print(
    f"Sharpe Ratio : {sharpe:.3f}"
)

print(
    f"Maximum Drawdown : {max_drawdown*100:.2f}%"
)

print("=" * 50)


# ===================================
# Plot Equity Curve
# ===================================

plt.figure(figsize=(12,6))

plt.plot(portfolio_values)

plt.title("Portfolio Value")

plt.xlabel("Time")

plt.ylabel("Portfolio Value ($)")

plt.grid(True)

plt.show()