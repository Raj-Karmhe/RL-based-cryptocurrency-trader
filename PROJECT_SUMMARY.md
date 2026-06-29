# CLSTM-RL Pipeline: Project Summary & Architecture

This document outlines the complete architecture, features, and implemented systems of the `clstm_rl_pipeline` based on the original Problem Statement (`PS.md`). It serves as a master reference for what is actually running in the codebase.

## 1. SYSTEM OBJECTIVE
An end-to-end Python pipeline for a Continuous-action Long Short-Term Memory Reinforcement Learning (CLSTM-RL) cryptocurrency trading agent. It supports multi-coin parallel training (BTC, ETH, SOL), strict multi-timeframe feature engineering without lookahead bias, and a custom Proximal Policy Optimization (PPO) agent.

## 2. TECH STACK
- **Data & Indicators:** `ccxt` (Binance), `pandas`, `numpy`, `pandas_ta`
- **Machine Learning (Feature Selection):** `scikit-learn` (Mutual Info), `xgboost`, `shap`
- **Deep Reinforcement Learning:** `stable-baselines3`, `PyTorch`
- **Optimization:** `optuna` (Hyperparameter Tuning)
- **Visualization:** `matplotlib`, `plotly`

---

## 3. PIPELINE BREAKDOWN

### Phase 1: Data Extraction & Feature Engineering
- **Multi-Coin Support:** Extracts data for BTC/USDT, ETH/USDT, and SOL/USDT.
- **Multi-Timeframe Alignment:** 1h (base), 4h, and 1d. Higher timeframes are properly shifted and forward-filled to strictly prevent lookahead bias.
- **Stationarity:** Bounded oscillators (RSI) are kept raw. Unbounded price data (SMA, Ichimoku) are converted into percentage distances or rolling z-scores. 
- **Splitting:** 4 years Train, 6 months Validation, 6 months Test. A unified `StandardScaler` is fit only on the training set.

### Phase 2: Systematic Feature Selection (The Golden State Space)
To prevent state-space bloat, the pipeline automatically selected the **Top 8 Golden Features** using a rigorous mathematical funnel:
1. **Mutual Information:** Kept features with highest non-linear correlation to the 5-period forward log return.
2. **Spearman Correlation:** Dropped highly collinear indicators (>0.80 threshold).
3. **XGBoost + SHAP:** A tree-based model extracted the top 8 most influential features via SHAP importance:
   - `1h_atr_pct` (Volatility)
   - `4h_sma200_dist` (Macro Trend)
   - `4h_mfi` (Momentum/Volume)
   - `1h_macd` (Short-term Trend)
   - `1d_adx` (Daily Trend Strength)
   - `1d_ichi_kijun_dist` (Daily Equilibrium)
   - `4h_high_low_pct` (Macro Volatility)
   - `4h_kc_width` (Keltner Channel Squeeze)

### Phase 3: The CLSTM-RL PPO Agent
- **Custom Gym Environment (`CryptoTradingEnv`)**
  - **State Space:** A 24-hour rolling window of the 8 Golden Features.
  - **Action Space:** Continuous `[-1.0, 1.0]` mapping to Short/Flat/Long positions. Sizing is interpolated (e.g. 0.5 = 50% Long).
  - **Reward Strategy:** Supports configurable modes. Default is a Sharpe-focused reward that penalizes extreme drawdown and accounts for Binance fees (0.1%) and dynamic slippage (0.05%).
  - **Advanced Risk Manager:** 
    1. *Dynamic ATR:* Configurable trailing Stop-Loss and Take-Profit based on Average True Range.
    2. *Macro Turbulence Exit:* If the broader crypto market (BTC) exceeds the 95th percentile of historical volatility, the agent forces an emergency flat position.

- **Architecture (`CLSTMFeatureExtractor`)**
  - **Input:** 8 features × 24 timesteps
  - **LSTM:** 2 Layers, 64 Hidden Units (per Optuna sweep)
  - **MLP Head:** 256 Hidden Units mapping the temporal state to the PPO Actor/Critic.
  - **PPO Hyperparameters:** Highly optimized via Bayesian search (`learning_rate=5.2e-05`, `n_steps=2048`, `batch_size=64`, `gamma=0.95`).

### Phase 3b & 4: Validation and Out-of-Sample Testing
- Evaluates the model on unseen data.
- **Interactive Plotly Dashboards:** Outputs a synchronized 4-panel HTML dashboard containing:
  1. Price Action, Entry/Exit markers (Green Stars = Take Profit, Red Crosses = Stop Loss, Orange Diamonds = Turbulence Exit).
  2. Continuous Position sizing graph (-1 to 1).
  3. Total Portfolio Value vs Buy & Hold baseline.
  4. Real-time visualization of the Golden Features the agent is analyzing.
- **Dynamic Overrides:** Supports CLI arguments to change slippage, fees, or the target coin (`--coin ETH/USDT`) without retraining the model.

### Phase 3c: Hyperparameter Optimization
- A dedicated `optuna` sweep script that iteratively trains mini-versions of the agent (50k steps) on the validation set to mathematically converge on the absolute best PPO hyperparameters.
