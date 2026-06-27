# CLSTM-PPO Pipeline Improvements Roadmap

## 🟢 Quick Wins (Low Effort, High Impact)

1. **Hyperparameter Optimization (Optuna)**
   Integrate Optuna to automate the search for the best PPO hyperparameters (Learning Rate, Gamma, Entropy Coefficient) and neural network sizes on the validation set.

2. **Refine the Reward Function**
   Experiment with alternative reward strategies:
   - **Sortino Reward:** Penalize only downside volatility.
   - **Holding Incentives:** Add a small positive reward scalar for holding a winning position over multiple steps.
   - **Overtrading Penalties:** Increase the penalty for flipping positions to account for double transaction fees and spread.

3. **Dynamic Stop-Loss (Trailing Stop)**
   Upgrade the Risk Manager to trail the stop-loss upward as the position moves into profit (e.g., moving the stop-loss up by 0.5 ATR if price rises by 1 ATR).

## 🟡 Intermediate Improvements (Medium Effort)

4. **Incorporate Alternative Data Sources**
   Add new feature inputs to `phase1_feature_engineering.py`:
   - **Funding Rates & Open Interest:** To detect over-leveraged markets.
   - **Orderbook Imbalance (L2 Data):** To detect short-term support/resistance.
   - **Fear & Greed Index / Sentiment:** Macro-indicator for market context.

5. **Dynamic Slippage Modeling**
   Replace the flat percentage slippage with a volume-based dynamic slippage model that scales slippage with order size and simulated orderbook liquidity.

## 🔴 Advanced Upgrades (High Effort)

6. **Ensemble Models & RL Meta-Agent**
   Currently, the ensemble uses a Hidden Markov Model (HMM) with basic smoothing and confidence thresholds. A massive upgrade is to replace the HMM entirely with a true **RL Meta-Agent**. 
   - **How it works:** Build a new top-level environment where the available "actions" are: `[Use Bull Expert, Use Bear Expert, Use Crab Expert, Hold Cash]`.
   - **Advantage:** Unlike an HMM (which just predicts statistical states), an RL Meta-Agent optimizes directly for *profit*. It learns exactly when each expert fails or succeeds, and naturally learns to avoid switching too often if transaction fees are high.

7. **Paper Trading / Live Execution Engine**
   Build `phase5_live_trader.py` using `CCXT` to connect to a live Testnet (e.g., Binance), pulling real-time websockets, formatting the data through the `StandardScaler`, and placing simulated live orders.
