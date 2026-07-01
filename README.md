# CLSTM-RL PPO: End-to-End Reinforcement Learning Cryptocurrency Trader

> **IITISoC 2026 — Advanced Problem Statement 1**
> *End-to-End RL-Based Strategy in Cryptocurrency Markets*

---

## Overview

This repository implements a complete, end-to-end pipeline for training a
**Cascaded LSTM Reinforcement Learning (CLSTM-RL)** agent to trade multiple cryptocurrencies (BTC, ETH, and SOL)
using historical OHLCV data. The agent uses **Proximal Policy Optimization (PPO)**
with a continuous action space.

---

## Motivation & Research Foundation

Traditional deep reinforcement learning (DRL) methods were largely developed for gaming environments. When directly adapted to financial data—which is highly noisy, uneven, and influenced by dynamic external factors—standard models often fall short. 

To overcome these limitations, this project implements a **Cascaded LSTM (CLSTM)** architecture, heavily inspired by recent academic research. 

### The Reference Paper
The foundational concepts for our architecture are derived from the following paper, which is also included in this repository:

* **Source:** Jie Zou et al., ["A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks" (View/Download PDF)](https://arxiv.org/pdf/2212.02721.pdf)
* **Local File:** `2212.02721v2.pdf`

### How This Paper Helped Us
By adapting the methodology from this paper, we achieved significant architectural improvements over standard neural networks (MLPs):

* **Handling Partial Observability:** Financial trading is a partially observable Markov decision process (POMDP) because raw features do not represent the complete state of the market. The paper demonstrated that using an LSTM as a feature extractor uncovers hidden time-series patterns, making the environment behave much closer to a fully observable MDP.
* **The Cascaded Advantage:** Instead of a standard stacked LSTM, the paper introduced a *cascaded* approach. An initial LSTM extracts time-series features from raw daily data, and these refined, encoded features are passed alongside the raw data to the PPO agent (which utilizes a second LSTM). This improves gradient flow and allows learning at multiple temporal scales.
* **Proven Superiority:** The research proved that CLSTM-PPO models consistently outperform standard PPO, MLP baselines, and complex ensemble strategies in key metrics like cumulative returns, maximum earning rates, and Sharpe ratios.

---

## Architecture Summary

```
Multi-Timeframe OHLCV (1h, 4h, 1d)
         ↓
  25-50 Technical Indicators (pandas_ta)
         ↓
  Stationarity Enforcement (rolling z-scores, pct-distances)
         ↓
  3-Stage Feature Selection
    Stage 1: Mutual Information Filter   → drop bottom 30%
    Stage 2: Spearman Correlation Filter → drop redundant (|ρ| > 0.80)
    Stage 3: LightGBM + SHAP           → select top 4-8 "Golden Features"
         ↓
  Custom Gymnasium Environment
    • Continuous action: [-1=short, 0=flat, +1=long]
    • ATR stop-loss & take-profit
    • Turbulence gate (force-exit in market crashes)
    • Transaction fees + slippage
         ↓
  Cascaded LSTM Feature Extractor (PyTorch)
    LSTM L1 → [L1 output | raw input] → LSTM L2 → MLP
         ↓
  PPO Actor / Critic Networks (Stable Baselines 3)
         ↓
  Interactive Plotly Dashboard (synchronized crosshair)
```

---

## Folder Structure

```
clstm_rl_pipeline/
├── config.py                      # All hyperparameters
├── phase1_data_extraction.py      # CCXT multi-TF download & merging
├── phase1_feature_engineering.py  # 25-50 indicators + stationarity
├── phase2_feature_selection.py    # MI → Spearman → SHAP pipeline
├── phase3_environment.py          # Custom Gymnasium environment
├── phase3_model.py                # CLSTM PyTorch architecture
├── phase3_train.py                # PPO training loop + validation
├── phase4_test_and_visualize.py   # Backtest + interactive charts
├── run_full_pipeline.py           # ← Run this to start everything
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional):** Install PyTorch with CUDA support for faster training:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Run the full pipeline

```bash
cd clstm_rl_pipeline
python run_full_pipeline.py
```

This will:
1. Download 5 years of historical OHLCV data (1h, 4h, 1d) for all three assets (BTC, ETH, and SOL) from Binance
2. Engineer 25-50 indicators per timeframe
3. Run 3-stage feature selection → print "Golden State Space"
4. Train the CLSTM-PPO agent for 500,000 timesteps
5. Backtest on the unseen test set
6. Generate separate interactive backtest dashboards (`results/interactive_backtest_<symbol>.html`) and metrics for each asset

### 3. Command-line options

```bash
# Force fresh data download (ignore cached CSVs)
python run_full_pipeline.py --redownload

# Skip training, load saved model
python run_full_pipeline.py --skip-train

# Start from Phase 2 (data already downloaded)
python run_full_pipeline.py --phase 2

# Start from Phase 3 (features already computed)
python run_full_pipeline.py --phase 3

# Only re-run the visualization
python run_full_pipeline.py --phase 4
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/interactive_backtest.html` | 🌐 **Interactive dashboard** — open in browser |
| `results/test_metrics.json` | Return, Sharpe, Drawdown, Win Rate |
| `results/test_trades.csv` | Annotated trade log with entry/exit reasons |
| `results/mi_scores.png` | Stage 1: Mutual Information bar chart |
| `results/correlation_heatmap.png` | Stage 2: Spearman correlation heatmap |
| `results/shap_beeswarm.png` | Stage 3: SHAP beeswarm (feature impact) |
| `results/shap_bar.png` | Stage 3: Mean SHAP importance ranking |
| `results/training_reward_curve.png` | PPO training progress |
| `models/clstm_ppo_btcusdt.zip` | Trained PPO model checkpoint |
| `data/golden_features.json` | Final selected feature names |

---

## Interactive Visualization

Open `results/interactive_backtest.html` in any browser.

**Features:**
- 🖱️ **Synchronized crosshair**: move your mouse over any panel to see a
  vertical line intersecting **all 3 panels simultaneously**, showing the
  exact values at that timestamp
- 📈 **Panel 1**: Asset price (BTC, ETH, or SOL) + Fibonacci retracement levels + trade
  entry/exit markers (▲ long, ▼ short, ✕ stop-loss, ★ take-profit, ⬡ turbulence)
- 📊 **Panel 2**: Agent position over time (−1 to +1, filled green/red)
- 💼 **Panel 3**: Portfolio value vs Buy & Hold baseline

---

## Data Pipeline Details

### Multi-Timeframe Merging (Anti-Lookahead)

The 4h and 1d candles are shifted by 1 period **before** forward-filling onto
the 1h timeline. This ensures the agent at time `t` only sees the **previously
completed** candle from higher timeframes — never the candle still forming.

```python
# 4h data: shift by 1 (one 4h candle = 4h of look-back protection)
htf_df = htf_df.shift(1)
htf_df = htf_df.reindex(h1_index, method="ffill")
```

### Data Split

| Split | Duration | Bars (1h) |
|-------|----------|-----------|
| Train | 4 years  | ~35,040   |
| Val   | 6 months | ~4,380    |
| Test  | 6 months | ~4,380    |

### Stationarity Enforcement

| Feature Type | Transformation |
|---|---|
| Bounded oscillators (RSI, Stoch, MFI) | Kept as-is |
| Price levels (SMA, EMA, Ichimoku) | % distance from moving average |
| Unbounded series (OBV, ADL) | Rolling z-score (50-bar window) |
| Scaling | StandardScaler fit on TRAIN only |

---

## CLSTM Architecture

The **Cascaded LSTM** differs from standard stacked LSTM:

```
Standard Stacked LSTM:   x → LSTM1 → LSTM2 → output
CLSTM (this paper):      x → LSTM1 → [LSTM1_out | x] → LSTM2 → output
                                          ↑ CASCADE ↑
```

The cascade allows Layer 2 to directly access raw features, improving
gradient flow and enabling learning at multiple temporal scales.

Reference: *Cascaded LSTM for Financial Time Series* (arXiv 2212.02721v2)

---

## Reward Function

```
reward = Sharpe_component
       - transaction_cost_penalty
       - drawdown_penalty
       - reversal_penalty
```

- **Sharpe component**: Rolling 20-step Sharpe ratio (risk-adjusted return)
- **Cost penalty**: Discourages excessive trading
- **Drawdown penalty**: Punishes losing from portfolio peak
- **Reversal penalty**: Discourages rapid position flip-flopping

---

## Risk Management

| Mechanism | Description |
|---|---|
| ATR Stop-Loss | Exit if price moves `2 × ATR` against position |
| ATR Take-Profit | Exit if price moves `3 × ATR` in favour |
| Turbulence Gate | Force-flatten all positions during market stress (top 10% turbulence) |
| Max Drawdown Kill | Episode ends early if portfolio drops 50% |

**Fibonacci levels** (23.6%, 38.2%, 50%, 61.8%, 78.6%) are displayed as
reference lines on the visualization, computed from the most recent 200-bar
swing high and low.

---

## Hyperparameter Reference

Key settings in `config.py`:

| Parameter | Value | Description |
|---|---|---|
| `SEQ_LEN` | 24 | Rolling window (hours) |
| `LSTM_HIDDEN_SIZE` | 256 | CLSTM hidden units |
| `MLP_HIDDEN_SIZE` | 128 | Output MLP size |
| `TOTAL_TIMESTEPS` | 500,000 | PPO training steps |
| `LEARNING_RATE` | 3e-4 | Adam optimizer LR |
| `STOP_LOSS_ATR_MULT` | 2.0 | ATR multiplier for SL |
| `TAKE_PROFIT_ATR_MULT` | 3.0 | ATR multiplier for TP |
| `TRANSACTION_FEE` | 0.1% | Binance maker fee |
| `SLIPPAGE` | 0.05% | Execution slippage |

---

## Limitations & Future Scope

* **Asset Isolation (No Joint Portfolio Rebalancing):** While the pipeline processes BTC, ETH, and SOL in a single run, the agent evaluates and trades each asset independently. It does not perform joint portfolio optimization, cross-asset correlation analysis, or dynamic capital reallocation across the basket.
* **Simplified Market Microstructure:** Backtests rely purely on historical OHLCV data. The environment does not account for order book dynamics (L2/L3 market depth), execution latency, or funding rate costs for leveraged positions.
* **Optimization & Training Convergence:** The default configuration of 500,000 steps is intended as a baseline and may underfit the deep Cascaded LSTM layers. Production deployment requires longer training cycles and extensive hyperparameter sweeps.

---

## Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---|---|
| `ccxt` | Exchange data download |
| `pandas_ta` | 25-50 technical indicators |
| `scikit-learn` | MI filter, StandardScaler |
| `lightgbm` | SHAP proxy model |
| `shap` | Feature importance |
| `gymnasium` | RL environment interface |
| `stable-baselines3` | PPO implementation |
| `torch` | CLSTM neural network |
| `plotly` | Interactive visualization |
