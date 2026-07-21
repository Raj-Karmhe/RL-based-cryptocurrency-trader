import os

# ==============================================================================
# CONFIGURATION SETTINGS — PAIRS TRADING RL AGENT
# All tunable parameters for the pairs trading system.
# Mirrors the structure of the Mid-eval single-asset config, adapted for
# two-asset spread trading.
# ==============================================================================

# ── Asset Pair Settings ───────────────────────────────────────────────────────
# We trade the spread between two cointegrated crypto assets.
# ETH/BTC is the most liquid and historically strongest cointegrated pair.
ASSET_A         = "XRP/USDT"       # The "dependent" asset (long leg)
ASSET_B         = "BTC/USDT"       # The "independent" asset (short leg)
ASSET_A_LABEL   = "XRP"
ASSET_B_LABEL   = "BTC"
TIMEFRAME       = "1h"

# How many years of hourly data to fetch (5 years ≈ 43,800 candles per asset)
DATA_YEARS      = 5

# ── File Paths ─────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RESULTS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

ASSET_A_PATH    = os.path.join(DATA_DIR, f"{ASSET_A.replace('/', '_')}_{TIMEFRAME}.csv")
ASSET_B_PATH    = os.path.join(DATA_DIR, f"{ASSET_B.replace('/', '_')}_{TIMEFRAME}.csv")
MERGED_PATH     = os.path.join(DATA_DIR, "merged_pair_data.csv")
SCALER_PATH     = os.path.join(DATA_DIR, "pairs_scaler.pkl")

MODEL_NAME      = f"clstm_ppo_pairs_trading_{ASSET_A_LABEL}_{ASSET_B_LABEL}"
MODEL_PATH      = os.path.join(MODEL_DIR, MODEL_NAME)

# Create directories
for d in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Cointegration & Pair Validation ────────────────────────────────────────────
COINT_PVALUE_THRESHOLD    = 0.05   # Max p-value for Engle-Granger cointegration
ADF_PVALUE_THRESHOLD      = 0.05   # Max p-value for ADF test on spread
HURST_THRESHOLD           = 0.5    # Max Hurst exponent (< 0.5 = mean reverting)
ROLLING_COINT_WINDOW      = 720    # 30 days of hourly data for rolling cointegration
ROLLING_COINT_MIN_PASS    = 0.70   # At least 70% of rolling windows must be cointegrated

# ── Spread Construction ────────────────────────────────────────────────────────
HEDGE_RATIO_WINDOW        = 168    # 1 week (168 hours) rolling OLS window for hedge ratio
SPREAD_ZSCORE_WINDOW      = 72     # 3 days rolling window for z-score calculation
HALF_LIFE_WINDOW          = 168    # 1 week window for half-life estimation
MIN_ADAPTIVE_WINDOW       = 24     # Minimum window size for OU adaptive window
MAX_ADAPTIVE_WINDOW       = 336    # Maximum window size for OU adaptive window

# ── Trading Simulator Settings ─────────────────────────────────────────────────
INITIAL_BALANCE           = 100_000      # Starting capital in USD
TRANSACTION_FEE           = 0.006        # 0.6% transaction fee
SLIPPAGE_BASE             = 0.004        # 0.4% base slippage
SLIPPAGE_IMPACT_FACTOR    = 0.0010       # 0.10% additional slippage per $100k traded
MAX_TRADE_NOTIONAL        = 50000.0      # Maximum dollar amount per leg per trade
MAX_POSITION_FRACTION     = 0.20         # Kelly fraction limit: max 20% of portfolio per trade
MARGIN_REQUIREMENT        = 0.5          # 50% margin for short positions

# ── Risk Management ────────────────────────────────────────────────────────────
# Z-Score Take Profit: Exit when the spread mean-reverts back near 0
ZSCORE_TAKE_PROFIT        = 0.5          
# Z-Score Stop Loss: Exit when the spread statistically breaks (anomaly)
ZSCORE_STOP_LOSS          = 4.0          
# Cointegration breakdown: if rolling p-value > this, force-close
COINT_BREAKDOWN_PVALUE    = 0.20
# Maximum position hold time (hours) — force-close to prevent regime drift
MAX_HOLD_HOURS            = 336          # 2 weeks
# Z-score emergency exit — if |z| exceeds this, something is broken
ZSCORE_EMERGENCY_EXIT     = 5.0
# Z-Score Cooldown (Regime Blocking) — Block new trades until spread normalizes
ZSCORE_COOLDOWN_TRIGGER   = 3.5
ZSCORE_COOLDOWN_RELEASE   = 2.75
# Hard price-based circuit breaker: Force close if any asset moves > X% against us
PRICE_CIRCUIT_BREAKER_PCT = 0.15

# ── CLSTM Neural Network Architecture ──────────────────────────────────────────
TIME_WINDOW               = 72           # Look-back window (hours)
LSTM_HIDDEN_SIZE          = 256          # LSTM hidden state size
LSTM_OUT_FEATURES         = 128          # Output feature vector size
N_LSTM_LAYERS             = 1            # LSTM depth (1 to avoid overfitting)
DUAL_STREAM               = True         # Use dual-stream LSTM (separate for each asset + spread)

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
TOTAL_TIMESTEPS           = 3_000_000
LEARNING_RATE             = 5e-5
N_STEPS                   = 8192
BATCH_SIZE                = 128
N_EPOCHS                  = 10
GAMMA                     = 0.99
CLIP_RANGE                = 0.2
ENT_COEF                  = 0.001
VF_COEF                   = 0.5
MAX_GRAD_NORM             = 0.5
FORCE_RETRAIN             = False

# ── Feature Engineering Settings ───────────────────────────────────────────────
RSI_PERIOD                = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD         = 20, 2
ATR_PERIOD                = 14
SMA_SHORT                 = 10
SMA_LONG                  = 50
VOLATILITY_WINDOW         = 20

# ── Feature Columns ───────────────────────────────────────────────────────────
# These are the final features fed into the RL agent.
# Grouped by category for clarity.

# Spread-derived features (primary signals)
SPREAD_FEATURES = [
    'Spread_ZScore',
    'Spread_ZScore_Velocity',
    'Spread_ZScore_Accel',
    'Half_Life',
    'Hedge_Ratio_Change',
    'Spread_Volatility',
    'Spread_RSI',
    'Spread_BB_Position',
    'Spread_MACD',
    'Spread_MACD_Signal',
    'Spread_CDF_KDE',
    'Spread_ZScore_4h',
    'Spread_ZScore_1d',
    'Cointegration_P_Value_4h',
    'Cointegration_P_Value_1d',
    'Hedge_Ratio_4h',
    'Hedge_Ratio_1d',
]

# Per-asset features (suffixed with _A and _B during processing)
PER_ASSET_FEATURES_BASE = [
    'Log_Return',
    'RSI',
    'ATR_Pct',
    'Volume_Ratio',
    'Volatility_20d',
    'Volatility_20h',
    'Funding_Rate',
]

# Cross-asset / relative features
CROSS_FEATURES = [
    'Return_Diff',
    'Volatility_Ratio',
    'Volume_Corr',
    'Price_Ratio_ZScore',
]

# Build full feature list
ASSET_A_FEATURES = [f"{f}_A" for f in PER_ASSET_FEATURES_BASE]
ASSET_B_FEATURES = [f"{f}_B" for f in PER_ASSET_FEATURES_BASE]

FEATURE_COLUMNS = SPREAD_FEATURES + ASSET_A_FEATURES + ASSET_B_FEATURES + CROSS_FEATURES
N_FEATURES      = len(FEATURE_COLUMNS)

# For dual-stream model: define which features belong to which stream
N_SPREAD_FEATURES = len(SPREAD_FEATURES)
N_ASSET_A_FEATURES = len(ASSET_A_FEATURES)
N_ASSET_B_FEATURES = len(ASSET_B_FEATURES)
N_CROSS_FEATURES = len(CROSS_FEATURES)

# Portfolio state features appended to observation
N_PORTFOLIO_FEATURES = 4  # [position, unrealized_pnl, current_zscore, hedge_ratio]

# ── Data Split Ratios ──────────────────────────────────────────────────────────
TRAIN_RATIO               = 0.80         # 4 years
VAL_RATIO                 = 0.10         # 6 months
TEST_RATIO                = 0.10         # 6 months

# ── Turbulence Index ───────────────────────────────────────────────────────────
TURBULENCE_LOOKBACK       = 252          # ~10.5 days of hourly data
TURBULENCE_PERCENTILE     = 95
TURBULENCE_THRESHOLD      = 100.0        # Hardcoded ceiling to prevent whipsawing
