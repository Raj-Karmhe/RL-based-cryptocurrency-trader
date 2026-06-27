"""
config.py — Central Configuration for the CLSTM-RL Pipeline
============================================================
All tunable parameters in one place.  Change values here; no need to
hunt through multiple scripts.
"""
import os

# ──────────────────────────────────────────────────────────────────────────────
# DATA SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
SYMBOL      = "BTC/USDT"           # Trading pair on Binance (Single coin mode)
SYMBOL_FILE = "BTC_USDT"          # Filesystem-safe version (no '/')

MULTI_COIN_MODE = True
MULTI_COINS     = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

TIMEFRAMES  = ["1h", "4h", "1d"]  # All three timeframes to download
BASE_TF     = "1h"                 # Base (finest-grain) timeframe

# Total lookback: 5 years (in days)
LOOKBACK_DAYS = 5 * 365            # = 1825 days

# Chronological split ratios  (must sum to 1.0)
TRAIN_RATIO = 4 / 5               # 4 years  (~80%)
VAL_RATIO   = 0.5 / 5             # 6 months (~10%)
TEST_RATIO  = 0.5 / 5             # 6 months (~10%)

# ──────────────────────────────────────────────────────────────────────────────
# FILE / DIRECTORY PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Raw OHLCV CSVs for each timeframe
RAW_DATA_PATHS = {
    tf: os.path.join(DATA_DIR, f"{SYMBOL_FILE}_{tf}_raw.csv")
    for tf in TIMEFRAMES
}

# Merged & aligned dataset (1h base, 4h and 1d forward-filled onto it)
MERGED_DATA_PATH   = os.path.join(DATA_DIR, "merged_all_tfs.csv")

# Feature-engineered splits
TRAIN_FEAT_PATH    = os.path.join(DATA_DIR, "train_features.csv")
VAL_FEAT_PATH      = os.path.join(DATA_DIR, "val_features.csv")
TEST_FEAT_PATH     = os.path.join(DATA_DIR, "test_features.csv")

# Selected feature names (output of Phase 2)
GOLDEN_FEATURES_PATH = os.path.join(DATA_DIR, "golden_features.json")

# Registry of versioned feature lists — maps n_features -> JSON path.
# Used by phase4/phase5 to auto-detect which feature set a saved model needs.
FEATURE_REGISTRY = {
    10: os.path.join(DATA_DIR, "golden_features_10.json"),  # Current (reduced) set
    22: os.path.join(DATA_DIR, "golden_features_22.json"),  # Legacy (original) set
}


# Scaler
SCALER_PATH  = os.path.join(DATA_DIR, "feature_scaler.pkl")

# Model checkpoint
MODEL_NAME   = "clstm_ppo_strategy_0_optimized_no_exit_only_long"
MODEL_PATH   = os.path.join(MODELS_DIR, MODEL_NAME)

# Create all required directories up front
for _d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — INDICATOR PERIODS
# ──────────────────────────────────────────────────────────────────────────────
RSI_PERIOD         = 14
STOCH_K, STOCH_D   = 14, 3
CCI_PERIOD         = 20
MFI_PERIOD         = 14
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL        = 9
ADX_PERIOD         = 14
SAR_STEP           = 0.02
SAR_MAX            = 0.2
BB_PERIOD          = 20
BB_STD             = 2
ATR_PERIOD         = 14
KELTNER_PERIOD     = 20
KELTNER_ATR_MULT   = 2
DONCHIAN_PERIOD    = 20
SMA_PERIODS        = [20, 50, 200]  # SMAs for pct-distance features
EMA_PERIODS        = [12, 26, 50]   # EMAs for pct-distance features
SUPERTREND_PERIOD  = 10
SUPERTREND_MULT    = 3.0
AO_SHORT           = 5              # Awesome Oscillator fast/slow
AO_LONG            = 34
VWAP_PERIOD        = 14             # Rolling VWAP window (hours)
Z_SCORE_WINDOW     = 50             # Window for rolling z-score stationarity

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION (Phase 2)
# ──────────────────────────────────────────────────────────────────────────────
FORWARD_K          = 5             # Forward log-return horizon (k periods on 1h TF)
MI_DROP_BOTTOM_PCT = 0.30          # Drop bottom 30% by MI score
CORR_THRESHOLD     = 0.80          # Spearman |corr| threshold for dropping redundant features
N_GOLDEN_FEATURES  = (4, 8)        # Min and max number of final selected features

# ──────────────────────────────────────────────────────────────────────────────
# CLSTM MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN            = 24            # Rolling window size (hours) fed to the LSTM
LSTM_HIDDEN_SIZE   = 64            # Hidden units in each LSTM layer
LSTM_N_LAYERS      = 2             # Number of cascaded LSTM layers
MLP_HIDDEN_SIZE    = 256           # Output MLP hidden size (= features_dim for PPO)
DROPOUT_RATE       = 0.1           # Dropout between LSTM layers

# ──────────────────────────────────────────────────────────────────────────────
# PPO HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS    = 500_000
LEARNING_RATE      = 5.2699e-05
N_STEPS            = 2048          # was 2048 — collect more experience per rollout
BATCH_SIZE         = 64            # was 128 — saturate GPU memory bandwidth
N_EPOCHS           = 10
GAMMA              = 0.9503
GAE_LAMBDA         = 0.9196
CLIP_RANGE         = 0.2
ENT_COEF           = 1.5726e-05
VF_COEF            = 0.5299
MAX_GRAD_NORM      = 0.5
SEED               = 42
FORCE_RETRAIN      = True          # Set False to load a saved model
N_ENVS             = 4             # Parallel environments for SubprocVecEnv

# ──────────────────────────────────────────────────────────────────────────────
# TRADING ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────
INITIAL_BALANCE         = 100_000.0   # Starting capital in USD
TRANSACTION_FEE         = 0.001       # 0.1% Binance maker/taker fee
SLIPPAGE                = 0.0005      # 0.05% price slippage
STOP_LOSS_ATR_MULT      = 2.0         # Stop-loss: entry_price ± ATR × 2
PREVENT_SAME_DIRECTION_REENTRY = True # Lock out same-direction trade after TP/SL until flip
TAKE_PROFIT_ATR_MULT    = 3.0         # Take-profit: entry_price ± ATR × 3
COOLDOWN_STEPS          = 0          # Wait 12 steps (hours) after force-exit before re-entry
TURBULENCE_PERCENTILE   = 100          # Top 5% turbulence triggers forced exit
REWARD_SCALE            = 1.0         # Scale raw reward into PPO-friendly range
MAX_DRAWDOWN_KILL       = 0.70        # was 0.50 — let episodes survive longer early in training

# 0: Default (Sharpe-adjusted log return with transaction & drawdown penalties)
# 1: Sideways Penalty (Penalizes holding in sideways markets + exponential drawdown penalty)
# 2: Sharpe Focused (Rewards increasing Sharpe ratio + linear drawdown penalty)
# 3: Mixed Sharpe + Sortino (0.6 × Sortino + 0.4 × Sharpe — rewards upside, punishes drawdowns)
REWARD_STRATEGY         = 0

# Weight of Sortino vs Sharpe in Strategy 3. 0.0 = pure Sharpe, 1.0 = pure Sortino.
# Can be tuned by Optuna between [0.3, 0.9].
SORTINO_WEIGHT          = 0.6

ALLOW_SHORT             = True       # Set to False to restrict the agent to Long/Flat only
HOLDING_INCENTIVE       = 0.000      # Small reward added for holding a long position to counteract short bias

# ──────────────────────────────────────────────────────────────────────────────
# META-AGENT ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────────
META_AGENT_TIMESTEPS    = 300_000
META_AGENT_LR           = 3e-4
META_SWITCH_PENALTY     = 0.0005
META_AGENT_PATH         = os.path.join(MODELS_DIR, "meta_agent")