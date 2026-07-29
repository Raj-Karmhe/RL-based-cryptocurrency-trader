"""
phase4_test_and_visualize.py - VectorBT-Based Backtesting & Evaluation

This script evaluates the trained recurrent PPO agent on the Validation and Test sets
using VectorBT as the external backtesting engine (as required by the problem statement).
The PPO agent generates position signals, which are then executed by VectorBT with
realistic transaction costs (fees + slippage). Results are saved as CSV files.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive plotting
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

try:
    import vectorbt as vbt
except ImportError:
    raise ImportError("VectorBT is required for backtesting. Install with: pip install vectorbt")

warnings.filterwarnings("ignore")

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv


def generate_agent_signals(model, df_split, golden_cols):
    """
    Runs the PPO agent through the trading environment in eval mode
    and collects the position allocation signals at each step.
    
    Returns: (timestamps, prices, positions, trade_history_from_env, initial_step)
    """
    env = CryptoTradingEnv(df_split, golden_cols, is_eval=True)
    obs, _ = env.reset()
    
    # Save the starting step index for trade marker alignment in plotting
    initial_step = env.current_step
    
    timestamps = []
    prices = []
    positions = []
    
    current_alloc = 0.0
    
    done = False
    truncated = False
    
    while not (done or truncated):
        current_step_before = env.current_step
        trades_before = env.trade_count
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        if env.trade_count > trades_before:
            last_trade = env.trade_history[-1]
            current_alloc = last_trade["next_pos"]
        
        # After step, record the state AT the step the trade was executed!
        step_idx = min(current_step_before, len(df_split) - 1)
        timestamps.append(df_split.index[step_idx])
        prices.append(env.prices_array[step_idx])
        positions.append(current_alloc)
        
    # Add the final state after the episode terminates for final evaluation
    final_step = min(env.current_step, len(df_split) - 1)
    timestamps.append(df_split.index[final_step])
    prices.append(env.prices_array[final_step])
    positions.append(current_alloc)
    
    return timestamps, prices, positions, env.trade_history, initial_step


def run_vectorbt_backtest(timestamps, prices, positions, initial_balance):
    """
    Executes a backtest using VectorBT's Portfolio.from_orders() based on
    position allocation signals from the PPO agent.
    
    Applies realistic transaction costs (fees + slippage).
    
    Returns: (vbt_portfolio, metrics_dict)
    """
    # Create aligned price and position series
    price_series = pd.Series(prices, index=timestamps, name="Close")
    position_series = pd.Series(positions, index=timestamps, name="Position")
    
    has_shorts = (position_series < -0.01).any()
    
    if has_shorts:
        # Manual portfolio simulation for short positions (Bug #10 fix)
        # VectorBT's targetpercent does not reliably support negative (short) allocations.
        print("  [Info] Short positions detected. Using manual portfolio simulation.")
        nav = np.zeros(len(prices))
        nav[0] = initial_balance
        cash = initial_balance
        crypto = 0.0
        prev_pos = 0.0
        total_trades = 0
        wins = 0
        trade_entry_nav = initial_balance
        peak_nav = initial_balance
        max_dd = 0.0
        
        for i in range(1, len(prices)):
            # Mark-to-market: update portfolio with price change
            portfolio_val = cash + crypto * prices[i]
            
            if portfolio_val <= 0:
                cash = 0.0
                crypto = 0.0
                nav[i] = 0.0
                peak_nav = max(peak_nav, 0.0)
                max_dd = 1.0
                continue
                
            # Check if position changed
            curr_pos = positions[i]
            if abs(curr_pos - prev_pos) > 0.001:
                # Execute rebalance
                target_crypto = (curr_pos * portfolio_val) / prices[i] if prices[i] > 0 else 0.0
                delta = target_crypto - crypto
                trade_val = abs(delta) * prices[i]
                fee_cost = trade_val * config.TRANSACTION_FEE
                slip_cost = trade_val * config.SLIPPAGE
                
                if delta > 0:  # Buying
                    cash -= (delta * prices[i] + fee_cost + slip_cost)
                    crypto += delta
                else:  # Selling
                    cash += (abs(delta) * prices[i] - fee_cost - slip_cost)
                    crypto += delta
                
                # Bug #6 fix: Track win/loss on position close AND reversals
                # A reversal (e.g., long->short) counts as closing the old trade
                is_closing = abs(prev_pos) > 0.02 and abs(curr_pos) < 0.02
                is_reversal = abs(prev_pos) > 0.02 and abs(curr_pos) > 0.02 and (np.sign(prev_pos) != np.sign(curr_pos))
                
                if is_closing or is_reversal:
                    if portfolio_val > trade_entry_nav:
                        wins += 1
                    total_trades += 1
                
                # Record entry NAV for new position or reversal
                if (abs(prev_pos) < 0.02 and abs(curr_pos) > 0.02) or is_reversal:
                    trade_entry_nav = portfolio_val
                    
                prev_pos = curr_pos
            
            portfolio_val = cash + crypto * prices[i]
            nav[i] = max(0.0, portfolio_val)
            peak_nav = max(peak_nav, nav[i])
            dd = (peak_nav - nav[i]) / peak_nav if peak_nav > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        # Bug #6 fix: Count open-at-end position as a trade
        if abs(prev_pos) > 0.02:
            final_val = nav[-1] if len(nav) > 0 else initial_balance
            if final_val > trade_entry_nav:
                wins += 1
            total_trades += 1
        
        total_return = (nav[-1] - initial_balance) / initial_balance
        # Annualized Sharpe from hourly returns
        nav_series = pd.Series(nav)
        hourly_returns = nav_series.pct_change().dropna()
        sharpe_ratio = (hourly_returns.mean() / (hourly_returns.std() + 1e-12)) * np.sqrt(8760)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        
        buy_hold_return = (prices[-1] - prices[0]) / prices[0]
        
        metrics = {
            "Initial Balance ($)": initial_balance,
            "Final Balance ($)": float(nav[-1]),
            "Total Return (%)": float(total_return * 100),
            "Buy & Hold Return (%)": float(buy_hold_return * 100),
            "Annualized Sharpe Ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
            "Max Drawdown (%)": float(max_dd * 100),
            "Total Trades": int(total_trades),
            "Win Rate (%)": float(win_rate)
        }
        
        # Create a lightweight portfolio-like object for plotting compatibility
        class _ManualPortfolio:
            def __init__(self, nav_arr, idx):
                self._nav = pd.Series(nav_arr, index=idx)
            def value(self):
                return self._nav
            def final_value(self):
                return self._nav.iloc[-1]
        
        portfolio = _ManualPortfolio(nav, timestamps)
        return portfolio, metrics
    
    # Standard VectorBT path for long-only strategies
    # Bug #5 fix: Clip any residual negative positions to 0 before sending to VectorBT,
    # since targetpercent does not handle negative (short) allocations.
    position_series = position_series.clip(lower=0.0)
    
    # Avoid continuous hourly rebalancing to target percentages on price fluctuations.
    # We only place orders when the PPO agent's target allocation actually changes.
    vbt_positions = position_series.copy()
    diffs = vbt_positions.diff().fillna(0.0)
    
    # Mask steps where the position target has not changed with NaN
    mask = diffs == 0.0
    # Keep the first step if the initial position is non-zero
    mask.iloc[0] = False if vbt_positions.iloc[0] != 0.0 else True
    vbt_positions[mask] = np.nan
    
    # Build VectorBT portfolio from target allocations
    # size_type='targetpercent' treats positions (0.0 to 1.0) as fractional targets
    portfolio = vbt.Portfolio.from_orders(
        close=price_series,
        size=vbt_positions,
        size_type='targetpercent',
        fees=config.TRANSACTION_FEE,
        slippage=config.SLIPPAGE,
        init_cash=initial_balance,
        freq="1h"
    )
    
    # Extract mandatory metrics
    total_return = portfolio.total_return()
    max_drawdown = portfolio.max_drawdown()
    sharpe_ratio = portfolio.sharpe_ratio()
    
    # Win rate from trades
    trades = portfolio.trades
    try:
        records = trades.records_readable
        if len(records) > 0:
            win_rate = trades.win_rate()
            total_trades = trades.count()
        else:
            win_rate = 0.0
            total_trades = 0
    except Exception:
        win_rate = 0.0
        total_trades = 0
        
    # Calculate Buy & Hold Return for comparison
    buy_hold_return = (prices[-1] - prices[0]) / prices[0]
    
    metrics = {
        "Initial Balance ($)": initial_balance,
        "Final Balance ($)": float(portfolio.final_value()),
        "Total Return (%)": float(total_return * 100),
        "Buy & Hold Return (%)": float(buy_hold_return * 100),
        "Annualized Sharpe Ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
        "Max Drawdown (%)": float(abs(max_drawdown) * 100),
        "Total Trades": int(total_trades),
        "Win Rate (%)": float(win_rate * 100) if not np.isnan(win_rate) else 0.0
    }
    
    return portfolio, metrics


def plot_performance_charts(symbol_name, split_name, timestamps, prices, positions, portfolio, trades_from_env, step_offset=0):
    """
    Generates performance comparison charts for a given split.
    Uses VectorBT portfolio values for the NAV curve.
    step_offset: the env's starting step index, used to convert absolute trade steps to list indices.
    """
    portfolio_values = portfolio.value().values
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Price panel with trade markers
    ax1.plot(timestamps, prices, color="gray", label="Close Price", alpha=0.7)
    
    long_entries_x, long_entries_y = [], []
    short_entries_x, short_entries_y = [], []
    exits_x, exits_y = [], []
    
    for t in trades_from_env:
        # Convert absolute env step to 0-based list index
        list_idx = t["step"] - step_offset
        if 0 <= list_idx < len(timestamps):
            trade_time = timestamps[list_idx]
            trade_price = t["price"]
            
            if abs(t["prev_pos"]) < 0.02 and t["next_pos"] > 0:
                long_entries_x.append(trade_time)
                long_entries_y.append(trade_price)
            elif abs(t["prev_pos"]) < 0.02 and t["next_pos"] < 0:
                short_entries_x.append(trade_time)
                short_entries_y.append(trade_price)
            elif abs(t["next_pos"]) < 0.02:
                exits_x.append(trade_time)
                exits_y.append(trade_price)
                
    if long_entries_x:
        ax1.scatter(long_entries_x, long_entries_y, color="green", marker="^", s=80, label="Long Entry", zorder=5)
    if short_entries_x:
        ax1.scatter(short_entries_x, short_entries_y, color="red", marker="v", s=80, label="Short Entry", zorder=5)
    if exits_x:
        ax1.scatter(exits_x, exits_y, color="black", marker="x", s=60, label="Exit", zorder=5)
        
    ax1.set_title(f"{symbol_name} {split_name.capitalize()} Price History & Executions")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Position allocation panel
    ax2.plot(timestamps, positions, color="purple", label="Allocation")
    ax2.fill_between(timestamps, 0, positions, where=np.array(positions) >= 0, color="green", alpha=0.15)
    ax2.fill_between(timestamps, 0, positions, where=np.array(positions) < 0, color="red", alpha=0.15)
    ax2.set_title(f"{symbol_name} {split_name.capitalize()} Position Allocation")
    ax2.set_ylabel("Allocation Fraction")
    ax2.grid(True, alpha=0.3)
    
    # Portfolio value panel (VectorBT NAV vs Buy & Hold)
    ax3.plot(timestamps, portfolio_values, color="blue", label="PPO Agent (VectorBT)")
    
    buy_hold_value = config.INITIAL_BALANCE * (np.array(prices) / prices[0])
    ax3.plot(timestamps, buy_hold_value, color="orange", linestyle="--", label="Buy & Hold")
    
    ax3.set_title(f"{symbol_name} {split_name.capitalize()} Portfolio Growth vs Buy & Hold")
    ax3.set_ylabel("NAV ($)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    chart_path = os.path.join(config.RESULTS_DIR, f"{symbol_name}_{split_name}_backtest_performance.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved performance chart to {chart_path}")


def run_backtest_on_split(symbol_name, split_name, features_csv_path, model, golden_cols):
    """
    Runs evaluation for a specific split (validation or test):
    1. PPO agent generates position signals via the env
    2. VectorBT executes trades with realistic costs
    3. Metrics are computed and saved as CSV + JSON
    """
    print(f"\nEvaluating PPO Agent on {symbol_name} ({split_name.upper()} Set) via VectorBT")
    
    if not os.path.exists(features_csv_path):
        print(f"  [Warning] Features file not found: {features_csv_path}. Skipping.")
        return None
        
    df_split = pd.read_csv(features_csv_path, index_col="Date", parse_dates=True)
    
    # Step 1: Generate agent signals
    print(f"  Generating PPO agent signals...")
    timestamps, prices, positions, trade_history, initial_step = generate_agent_signals(model, df_split, golden_cols)
    
    # Step 2: Run VectorBT backtest
    print(f"  Running VectorBT backtest with fees={config.TRANSACTION_FEE}, slippage={config.SLIPPAGE}...")
    portfolio, metrics = run_vectorbt_backtest(timestamps, prices, positions, config.INITIAL_BALANCE)
    
    # Print metrics
    print(f"  {symbol_name} {split_name.capitalize()} Performance Summary (VectorBT):")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"    {key:<28}: {val:.4f}")
        else:
            print(f"    {key:<28}: {val}")
    
    # Step 3: Save metrics as JSON
    metrics_path = os.path.join(config.RESULTS_DIR, f"{symbol_name}_{split_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved {split_name} metrics to {metrics_path}")
    
    # Step 4: Save results as CSV (mandatory deliverable)
    csv_path = os.path.join(config.RESULTS_DIR, f"{symbol_name}_{split_name}_backtest_results.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved {split_name} results CSV to {csv_path}")
    
    # Step 5: Save trade log from agent
    if trade_history:
        trades_df = pd.DataFrame(trade_history)
        trades_csv_path = os.path.join(config.RESULTS_DIR, f"{symbol_name}_{split_name}_trades.csv")
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"  Saved {split_name} trade logs to {trades_csv_path}")
    
    # Step 6: Plot performance charts
    plot_performance_charts(symbol_name, split_name, timestamps, prices, positions, portfolio, trade_history, step_offset=initial_step)
    
    return metrics


def run_all_evaluations():
    """
    Loads PPO agent and runs validation and test evaluations using VectorBT.
    """
    # Check model path
    if not os.path.exists(config.MODEL_PATH + ".zip"):
         raise FileNotFoundError(f"Trained model checkpoint not found at {config.MODEL_PATH}")
         
    # Load model
    model = PPO.load(config.MODEL_PATH)
    print(f"Successfully loaded PPO agent from {config.MODEL_PATH}")
    
    # Load golden features
    with open(config.GOLDEN_FEATURES_PATH, "r") as f:
        golden_cols = json.load(f)
    
    symbol = config.SYMBOL
    symbol_file = config.SYMBOL_FILE
    
    all_results = []
    
    for split in ["validation", "test"]:
        split_suffix = "val" if split == "validation" else split
        feat_csv_path = os.path.join(config.DATA_DIR, f"{symbol_file}_{split_suffix}_features.csv")
        
        metrics = run_backtest_on_split(symbol_file, split, feat_csv_path, model, golden_cols)
        if metrics is not None:
            metrics["Split"] = split.capitalize()
            metrics["Symbol"] = symbol
            all_results.append(metrics)
    
    # Save combined results CSV with all splits
    if all_results:
        combined_df = pd.DataFrame(all_results)
        # Reorder columns for clarity
        col_order = ["Symbol", "Split", "Initial Balance ($)", "Final Balance ($)", 
                      "Total Return (%)", "Buy & Hold Return (%)", "Annualized Sharpe Ratio", "Max Drawdown (%)", 
                      "Total Trades", "Win Rate (%)"]
        combined_df = combined_df[[c for c in col_order if c in combined_df.columns]]
        
        combined_csv_path = os.path.join(config.RESULTS_DIR, "backtest_results_summary.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n  Saved combined results summary to {combined_csv_path}")
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        print(combined_df.to_string(index=False))


if __name__ == "__main__":
    run_all_evaluations()
