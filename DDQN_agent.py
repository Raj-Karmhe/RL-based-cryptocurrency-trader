"""
ddqn_agent.py

Double Deep Q-Network Agent
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer
from config import *


class DDQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        # Online Network
        self.online_net = QNetwork(
            state_size,
            action_size,
            HIDDEN_SIZE
        ).to(DEVICE)

        # Target Network
        self.target_net = QNetwork(
            state_size,
            action_size,
            HIDDEN_SIZE
        ).to(DEVICE)

        # Copy weights
        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=LEARNING_RATE
        )

        # Replay Buffer
        self.memory = ReplayBuffer(BUFFER_SIZE)

        # Loss
        self.criterion = nn.SmoothL1Loss()

        # Exploration
        self.epsilon = EPSILON_START

        self.learn_step = 0
