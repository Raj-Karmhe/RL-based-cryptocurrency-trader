"""
Configuration file for Double DQN
"""

import torch

# ===============================
# Device
# ===============================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===============================
# Environment
# ===============================

INITIAL_BALANCE = 10000

# ===============================
# DDQN Hyperparameters
# ===============================

GAMMA = 0.99

LEARNING_RATE = 1e-4

BATCH_SIZE = 64

BUFFER_SIZE = 100000

MIN_REPLAY_SIZE = 5000

TARGET_UPDATE_FREQ = 1000

TRAIN_EPISODES = 200

MAX_STEPS = 100000

# ===============================
# Exploration
# ===============================

EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY = 100000

# ===============================
# Neural Network
# ===============================

HIDDEN_SIZE = 256

# ===============================
# Optimizer
# ===============================

GRADIENT_CLIP = 10.0

# ===============================
# Model Saving
# ===============================

SAVE_PATH = "saved_models/ddqn_model.pth"