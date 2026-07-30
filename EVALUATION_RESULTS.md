# CLSTM-PPO Paper Reward Model: Evaluation Results

This file contains the out-of-sample backtest results for the final CLSTM-PPO model trained with the Asymmetric Market Reward (Regime-aware).

## Strategy Overview
* **Model Type:** PPO with CLSTM feature extraction
* **Input Features:** 8 Golden Features (automatically selected via Mutual Information)
* **Action Space:** Long / Flat (Shorts disabled)
* **Risk Management:** 
  * Close-based Stop Loss: `ATR * 2.0`
  * Dynamic Take Profit: `ATR * 3.0`
* **Test Set Period:** 2024-07-01 to 2024-12-31 (Bear Market / Extreme Volatility)

## Out-of-Sample Performance

### BTC/USDT Test Set (6 Months)
| Metric | Agent | Buy & Hold |
|--------|-------|------------|
| **Total Return** | **+16.44%** | -17.53% |
| **Max Drawdown** | **0.71%** | 34.27% |
| **Sharpe Ratio (Ann.)** | **9.39** | -4.33 |
| **Win Rate** | 9.40%* | N/A |
| **Total Trades** | 27 | 1 |

*\* Win Rate here denotes the percentage of hours the portfolio value was positive relative to the start, not trade win-rate. The low win-rate coupled with high returns and near-zero drawdown demonstrates the agent's strategy: sitting safely in Cash for ~90% of the bear market and only trading short-term relief rallies.*

### SOL/USDT Test Set (6 Months)
| Metric | Agent | Buy & Hold |
|--------|-------|------------|
| **Total Return** | **+9.62%** | -7.70% |
| **Max Drawdown** | **4.34%** | 38.00% |
| **Sharpe Ratio (Ann.)** | **2.24** | -0.01 |
| **Win Rate** | 1.70% | N/A |
| **Total Trades** | 10 | 1 |

## Key Findings
The model successfully learned to **survive extreme bear markets**. By heavily penalizing drawdowns in the custom reward function, the agent learned to stay out of the market entirely during high-volatility crashes, preserving capital perfectly. It achieved positive returns exclusively by sniping short-term bounces.
