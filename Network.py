"""
network.py

Q-Network for Double Deep Q Network (DDQN)

Input:
    State vector from CryptoTradingEnv

Output:
    Q-values for each discrete action
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Neural Network that approximates the Q-function.

    Input:
        state_size

    Output:
        Q-value for every action.
    """

    def __init__(
        self,
        state_size,
        action_size,
        hidden_size=256
    ):
        super(QNetwork, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(state_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, action_size)

        )

    def forward(self, state):

        return self.network(state)
