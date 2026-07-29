"""
config.py - Configuration Module for PPO Crypto Trading Bot

This file holds all constants, directory paths, hyperparameter values, and environment
settings. It centralizes variables so that changes can be made without touching implementation code.
"""

import os

# --------------------------------------------------------
# 1. Dataset & Directory Settings
# --------------------------------------------------------
SYMBOL = "BTC/USDT"
SYMBOL_FILE = "BTC_USDT"

MULTI_COIN_MODE = False
MULTI_COINS = ["BTC/USDT"]

TIMEFRAMES = ["1h", "4h", "1d"]
BASE_TF = "1h"

# Lookback period: 5 years + 250 days indicator warm-up buffer
LOOKBACK_DAYS = 5 * 365 + 250

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure required directories exist
for folder in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# File Paths
MERGED_DATA_PATH = os.path.join(DATA_DIR, f"{SYMBOL_FILE}_merged_all_tfs.csv")
TRAIN_FEAT_PATH = os.path.join(DATA_DIR, "train_features.csv")
VAL_FEAT_PATH = os.path.join(DATA_DIR, "val_features.csv")
TEST_FEAT_PATH = os.path.join(DATA_DIR, "test_features.csv")
SCALER_PATH = os.path.join(DATA_DIR, "feature_scaler.pkl")
GOLDEN_FEATURES_PATH = os.path.join(DATA_DIR, "golden_features.json")

MODEL_NAME = "ppo_trading_agent"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

# --------------------------------------------------------
# 2. Indicator Parameters
# --------------------------------------------------------
RSI_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
CCI_PERIOD = 20
MFI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
SAR_STEP = 0.02
SAR_MAX = 0.2
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
KELTNER_PERIOD = 20
KELTNER_ATR_MULT = 2
DONCHIAN_PERIOD = 20
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26, 50]
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0
AO_SHORT = 5
AO_LONG = 34
VWAP_PERIOD = 14
Z_SCORE_WINDOW = 50

# --------------------------------------------------------
# 3. Feature Selection Settings
# --------------------------------------------------------
INDICATOR_SET = 22
FORWARD_K = 5
MI_DROP_BOTTOM_PCT = 0.30
CORR_THRESHOLD = 0.80
N_GOLDEN_FEATURES = (4, 8)

# --------------------------------------------------------
# 4. Model Architecture Settings
# --------------------------------------------------------
SEQ_LEN = 24
TCN_HIDDEN_SIZE = 64           # Channel width per TCN residual block
TCN_N_LAYERS = 4               # Number of stacked residual blocks (dilations: 1,2,4,8)
MLP_HIDDEN_SIZE = 256
DROPOUT_RATE = 0.0              # Disabled: TCN feature extractor converges more stably without dropout noise

# Backwards-compatible aliases for scripts referencing old names
LSTM_HIDDEN_SIZE = TCN_HIDDEN_SIZE

# --------------------------------------------------------
# 5. PPO Hyperparameters
# --------------------------------------------------------
TOTAL_TIMESTEPS = 40_000  # 40k timesteps for training run
LEARNING_RATE = 1e-4       # Lowered to stabilize TCN policy gradients
N_STEPS = 2048
BATCH_SIZE = 128
N_EPOCHS = 5               # Lowered to prevent policy update overshoot
GAMMA = 0.98               # Reduced discount to stabilize hourly value targets
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.1           # Tightened clip range to stabilize step updates
ENT_COEF = 0.02            # Increased to maintain policy exploration (prevent std collapse)
TARGET_KL = 0.015          # Early stopping threshold to prevent policy collapse
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 42
FORCE_RETRAIN = True
N_ENVS = 4

# --------------------------------------------------------
# 6. Sizing Limits & Operational Friction
# --------------------------------------------------------
INITIAL_BALANCE = 100_000.0
TRANSACTION_FEE = 0.001       # 0.1% Binance spot fee
SLIPPAGE = 0.0005             # 0.05% slippage estimate
MIN_POSITION_CHANGE = 0.10    # Reduced to 0.10 to allow more frequent trades
MAX_POSITION_LEVERAGE = 1.0   # No leverage (fully backed cash/spot)
POSITION_STEP_SIZE = 0.05     # Resolution for discrete actions if needed
STOP_LOSS_ATR_MULT = 2.0
TAKE_PROFIT_ATR_MULT = 3.0
PREVENT_SAME_DIRECTION_REENTRY = True
MAX_DRAWDOWN_KILL = 0.30       # Episode terminates if portfolio drops by 30%
REWARD_SCALE = 1.0
ALLOW_SHORT = True
