"""
replay_buffer.py

Experience Replay Buffer for Double DQN
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
    Stores past experiences and randomly samples mini-batches
    for training.
    """

    def __init__(self, buffer_size):

        # Maximum number of experiences
        self.memory = deque(maxlen=buffer_size)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store one experience.
        """

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Randomly sample a mini-batch.
        """

        experiences = random.sample(
            self.memory,
            batch_size
        )

        states = np.array(
            [e[0] for e in experiences],
            dtype=np.float32
        )

        actions = np.array(
            [e[1] for e in experiences],
            dtype=np.int64
        )

        rewards = np.array(
            [e[2] for e in experiences],
            dtype=np.float32
        )

        next_states = np.array(
            [e[3] for e in experiences],
            dtype=np.float32
        )

        dones = np.array(
            [e[4] for e in experiences],
            dtype=np.float32
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        """
        Current number of stored experiences.
        """

        return len(self.memory)
