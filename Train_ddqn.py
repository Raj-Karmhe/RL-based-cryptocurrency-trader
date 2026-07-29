"""
train_ddqn.py

Training script for Double DQN
"""

import torch
import numpy as np

from Environment import Environment
from DDQN_agent import DDQNAgent
from config import *


# ==============================
# Create Environment
# ==============================

env = Environment()

state_size = env.observation_space.shape[0]

action_size = env.action_space.n


# ==============================
# Create Agent
# ==============================

agent = DDQNAgent(
    state_size,
    action_size
)


# ==============================
# Training Loop
# ==============================

episode_rewards = []

for episode in range(TRAIN_EPISODES):

    state, info = env.reset()

    done = False

    total_reward = 0

    while not done:

        # Select Action
        action = agent.select_action(state)

        # Environment Step
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Store Experience
        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        # Learn
        agent.learn()

        state = next_state

        total_reward += reward

    episode_rewards.append(total_reward)

    print(
        f"Episode {episode+1}/{TRAIN_EPISODES} | "
        f"Reward: {total_reward:.4f} | "
        f"Epsilon: {agent.epsilon:.4f}"
    )


# ==============================
# Save Model
# ==============================

torch.save(
    agent.online_net.state_dict(),
    SAVE_PATH
)

print("Training Complete.")
