"""
phase3_environment.py - Custom Gymnasium Trading Environment

This script implements a custom trading environment using the Gymnasium API.
It simulates real-world execution friction (transaction fees and slippage),
manages position sizing based on continuous actions, and enforces volatility-based
risk controls (ATR stop-loss and take-profit exits) to ensure capital preservation.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import os
import sys

# Add workspace directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

class TradingRiskManager:
    """
    Manages volatility-based risk targets (ATR stop-loss and take-profit thresholds).
    Calculates exit boundaries upon trade entry and monitors them at each step.
    """
    def __init__(self, sl_multiplier: float, tp_multiplier: float):
        self.sl_mult = sl_multiplier
        self.tp_mult = tp_multiplier
        self.reset()
        
    def reset(self):
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.active = False
        self.side = 0  # +1 for Long, -1 for Short
        self.exits_sl = 0
        self.exits_tp = 0
        
    def check_entry(self, entry_price: float, atr: float, side: int):
        """
        Initializes risk boundaries when a position is opened or reversed.
        """
        self.entry_price = entry_price
        self.side = side
        self.active = True
        
        if side == 1:  # Long position
            self.stop_loss = entry_price - (self.sl_mult * atr)
            self.take_profit = entry_price + (self.tp_mult * atr)
        else:          # Short position
            self.stop_loss = entry_price + (self.sl_mult * atr)
            self.take_profit = entry_price - (self.tp_mult * atr)
            
    def evaluate_exit(self, price_now: float) -> tuple:
        """
        Checks if stop-loss or take-profit boundaries are breached.
        Returns: (should_exit: bool, reason: str or None)
        """
        if not self.active:
            return False, None
            
        if self.side == 1:  # Long
            if price_now <= self.stop_loss:
                self.exits_sl += 1
                self.active = False
                return True, "stop_loss"
            elif price_now >= self.take_profit:
                self.exits_tp += 1
                self.active = False
                return True, "take_profit"
        elif self.side == -1:  # Short
            if price_now >= self.stop_loss:
                self.exits_sl += 1
                self.active = False
                return True, "stop_loss"
            elif price_now <= self.take_profit:
                self.exits_tp += 1
                self.active = False
                return True, "take_profit"
                
        return False, None
        
    def deactivate(self):
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.active = False
        self.side = 0

class CryptoTradingEnv(gym.Env):
    """
    Continuous trading environment for reinforcement learning agents.
    Simulates a margin trading portfolio where target position allocation is in [-1.0, 1.0].
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        initial_balance: float = config.INITIAL_BALANCE,
        fee_rate: float = config.TRANSACTION_FEE,
        slippage_rate: float = config.SLIPPAGE,
        seq_len: int = config.SEQ_LEN,
        allow_short: bool = config.ALLOW_SHORT,
        is_eval: bool = False
    ):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.seq_len = seq_len
        self.allow_short = allow_short
        self.is_eval = is_eval
        
        # Bug #16 fix: Validate that all golden feature columns exist in the DataFrame
        missing_cols = [col for col in self.feature_cols if col not in self.df.columns]
        if missing_cols:
            raise KeyError(
                f"Golden feature columns missing from DataFrame: {missing_cols}. "
                f"Available columns: {list(self.df.columns)}. "
                f"Re-run phase1_feature_engineering.py and phase2_feature_selection.py."
            )
        
        # Precompute arrays for step iterations
        self.features_array = self.df[self.feature_cols].values.astype(np.float32)
        self.prices_array = self.df["Close"].values.astype(np.float64)
        self.atrs_array = self.df["ATR"].values.astype(np.float64)
        self.n_steps = len(self.df)
        
        # Space Definitions
        obs_dim = self.seq_len * self.n_features + 2  # Features window + current position + unrealized pnl
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=-1.0 if self.allow_short else 0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        self.risk_manager = TradingRiskManager(config.STOP_LOSS_ATR_MULT, config.TAKE_PROFIT_ATR_MULT)
        self.obs_buffer = np.zeros((self.seq_len, self.n_features), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        """
        Resets environment.
        """
        super().reset(seed=seed)
        
        # Start index: randomize in training to prevent timeline overfitting
        min_episode_len = 500  # Guarantee minimum 500-step episodes for meaningful learning
        max_start = self.n_steps - min_episode_len
        if max_start <= self.seq_len:
            # Dataset too small for randomization; always start at earliest valid index
            print(f"  [Warning] Dataset has {self.n_steps} rows, insufficient for {min_episode_len}-step episodes + {self.seq_len} warm-up. Training starts at step {self.seq_len}.")
            self.current_step = self.seq_len
        elif not self.is_eval:
            self.current_step = np.random.randint(self.seq_len, max_start)
        else:
            self.current_step = self.seq_len
            
        self.balance = float(self.initial_balance)
        self.crypto_held = 0.0
        self.position_allocation = 0.0
        self.portfolio_value = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)
        self.peak_portfolio_value = float(self.initial_balance)
        self.entry_price = None
        self.locked_side = None  # Lockout direction after SL/TP exits
        self.locked_side_step = 0  # Step counter for auto-clearing lockout
        
        # Logs
        self.trade_count = 0
        self.fees_paid = 0.0
        self.slippage_costs = 0.0
        self.trade_history = []
        
        self.risk_manager.reset()
        
        # Fill features buffer
        start_idx = self.current_step - self.seq_len + 1
        self.obs_buffer = self.features_array[start_idx : self.current_step + 1].copy()
        
        return self._generate_obs(), {}
        
    def step(self, action: np.ndarray):
        """
        Executes one trading step.
        """
        action_val = float(action[0])
        target_allocation = np.clip(action_val, -1.0 if self.allow_short else 0.0, 1.0)
        
        current_price = self.prices_array[self.current_step]
        current_atr = self.atrs_array[self.current_step]
        
        # 1. Handle lockout if configured (Evaluation only — training explores freely)
        if self.is_eval and config.PREVENT_SAME_DIRECTION_REENTRY and self.locked_side is not None:
            # Auto-clear lockout after 24 steps (1 day of 1h bars) to prevent deadlock
            self.locked_side_step += 1
            if self.locked_side_step >= 24:
                self.locked_side = None
                self.locked_side_step = 0
            else:
                attempt_side = 1 if target_allocation > config.MIN_POSITION_CHANGE else (-1 if target_allocation < -config.MIN_POSITION_CHANGE else 0)
                if attempt_side == self.locked_side:
                    target_allocation = 0.0  # Force flat
                elif attempt_side == -self.locked_side and attempt_side != 0:
                    self.locked_side = None  # Clear lockout
                    self.locked_side_step = 0
                
        # 2. Risk check exits — Bug #10 fix: enabled during BOTH training and eval
        #    so the agent learns risk-aware behavior in the same environment it's evaluated in.
        force_exit = False
        exit_reason = None
        
        if abs(self.position_allocation) > 0.01:
            should_exit, reason = self.risk_manager.evaluate_exit(current_price)
            if should_exit:
                force_exit = True
                exit_reason = reason
                target_allocation = 0.0
                if self.is_eval and config.PREVENT_SAME_DIRECTION_REENTRY:
                    self.locked_side = 1 if self.position_allocation > 0 else -1
                    self.locked_side_step = 0
                    
        # 3. Execute Trade Transaction
        position_delta = target_allocation - self.position_allocation
        trade_costs = 0.0
        # Cache pre-trade portfolio value for reward cost penalty (Bug #7 fix)
        pre_trade_portfolio_value = self.balance + self.crypto_held * current_price
        
        is_significant = abs(position_delta) > config.MIN_POSITION_CHANGE
        is_close_trade = (abs(target_allocation) < 0.02) and (abs(self.position_allocation) >= 0.02)
        
        if is_significant or force_exit or is_close_trade:
            # Capture the pre-trade allocation for logging (before it gets overwritten by sync)
            pre_trade_allocation = self.position_allocation
            # Compute fresh portfolio value at CURRENT price before sizing
            current_portfolio_value = pre_trade_portfolio_value
            if current_portfolio_value <= 0:
                current_portfolio_value = 1e-8  # Prevent division by zero
            
            # Valuation based on fresh portfolio
            trade_value = abs(position_delta) * current_portfolio_value
            fee = trade_value * self.fee_rate
            slip = trade_value * self.slippage_rate
            trade_costs = fee + slip
            
            self.fees_paid += fee
            self.slippage_costs += slip
            
            # Rebalance using fresh portfolio value
            target_crypto = (target_allocation * current_portfolio_value) / current_price
            crypto_delta = target_crypto - self.crypto_held
            
            if crypto_delta > 0:  # Buying
                cash_needed = crypto_delta * current_price + trade_costs
                if cash_needed <= self.balance:
                    self.crypto_held += crypto_delta
                    self.balance -= cash_needed
                else:
                    # Buy fractional maximum; recalculate costs from scratch for actual fill
                    available_for_purchase = self.balance
                    if available_for_purchase > 0:
                        # Solve: possible_crypto * price + possible_crypto * price * (fee_rate + slippage_rate) = available
                        effective_price = current_price * (1.0 + self.fee_rate + self.slippage_rate)
                        possible_crypto = available_for_purchase / effective_price
                        actual_trade_value = possible_crypto * current_price
                        actual_fee = actual_trade_value * self.fee_rate
                        actual_slip = actual_trade_value * self.slippage_rate
                        self.crypto_held += possible_crypto
                        self.balance -= (possible_crypto * current_price + actual_fee + actual_slip)
                        # Correct the aggregate cost trackers for the actual fill
                        self.fees_paid += (actual_fee - fee)
                        self.slippage_costs += (actual_slip - slip)
                        trade_costs = actual_fee + actual_slip
                    # else: insufficient cash; skip fill entirely
            elif crypto_delta < 0:  # Selling / Closing Long / Opening Short
                sell_amount = abs(crypto_delta)
                remaining_to_short = 0.0
                
                if self.crypto_held > 0:
                    # Selling actual holdings (closing long or reducing position)
                    actual_sell = min(sell_amount, self.crypto_held)
                    # Bug #1 fix: split costs proportionally between close-long and open-short legs
                    sell_fraction = actual_sell / sell_amount if sell_amount > 0 else 1.0
                    sell_leg_costs = trade_costs * sell_fraction
                    self.crypto_held -= actual_sell
                    self.balance += (actual_sell * current_price - sell_leg_costs)
                    remaining_to_short = sell_amount - actual_sell
                else:
                    remaining_to_short = sell_amount
                
                if remaining_to_short > 0 and self.allow_short:
                    # Bug #5 fix: Credit cash for short position opening (margin simulation)
                    # In a real margin short, the broker credits the sale proceeds to the account.
                    # The short P&L is then realized via portfolio mark-to-market.
                    short_leg_fraction = remaining_to_short / sell_amount if sell_amount > 0 else 1.0
                    short_leg_costs = trade_costs * short_leg_fraction
                    short_proceeds = remaining_to_short * current_price
                    self.balance += (short_proceeds - short_leg_costs)
                    self.crypto_held -= remaining_to_short
                
            # Bug #2 fix: Guard against negative balance instead of silently clamping.
            # If balance went negative (costs exceeded cash), log it and floor at a tiny
            # positive value to avoid division-by-zero while keeping the cost visible.
            if self.balance < 0.0:
                self.balance = 0.0
            
            # Risk manager status (uses pre-trade allocation captured above)
            was_flat = abs(pre_trade_allocation) < 0.02
            is_open = abs(target_allocation) >= 0.02
            side_changed = (np.sign(pre_trade_allocation) != np.sign(target_allocation)) and not was_flat and is_open
            
            if (was_flat and is_open) or side_changed:
                trade_side = 1 if target_allocation > 0 else -1
                self.risk_manager.check_entry(current_price, current_atr, trade_side)
                self.entry_price = current_price
            elif is_close_trade:
                self.risk_manager.deactivate()
                self.entry_price = None
                
            # Bug #2 fix: Sync position allocation BEFORE logging so the trade log
            # reflects actual achieved allocation (handles partial fills correctly)
            actual_portfolio = self.balance + self.crypto_held * current_price
            if actual_portfolio > 0:
                self.position_allocation = (self.crypto_held * current_price) / actual_portfolio
            else:
                self.position_allocation = 0.0
            
            # Bug #4 fix: Clear entry_price whenever position drops below threshold
            # to prevent stale unrealized PnL in observations during gradual reductions
            if abs(self.position_allocation) < 0.02 and self.entry_price is not None:
                self.risk_manager.deactivate()
                self.entry_price = None
                
            self.trade_count += 1
            self.trade_history.append({
                "step": self.current_step,
                "price": current_price,
                "action": action_val,
                "prev_pos": pre_trade_allocation,  # original pre-trade allocation
                "next_pos": self.position_allocation,  # Bug #2: log actual achieved allocation
                "costs": trade_costs,
                "force_exit": force_exit,
                "reason": exit_reason
            })
                
        # 4. Step Time
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        if not terminated:
            new_price = self.prices_array[self.current_step]
        else:
            # On terminal step, use the last valid price for final valuation (Bug #2: current_step == n_steps-1 here)
            new_price = self.prices_array[self.n_steps - 1]
        
        # Always update portfolio value so the reward reflects the true final movement
        self.prev_portfolio_value = self.portfolio_value
        raw_portfolio = self.balance + (self.crypto_held * new_price)
        # Bug #17 fix: If portfolio goes negative (e.g., sharp move against short),
        # floor at a small positive value instead of zero to prevent log(0) reward spikes
        self.portfolio_value = max(1.0, raw_portfolio)
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)
        
        # Bug #6 fix: Always recalculate position_allocation after price update
        # to prevent stale allocation values in observations on non-trade steps
        updated_portfolio = self.balance + self.crypto_held * new_price
        if updated_portfolio > 0:
            self.position_allocation = (self.crypto_held * new_price) / updated_portfolio
        else:
            self.position_allocation = 0.0
        
        if not terminated:
            # Update sequence buffer only when episode continues
            self.obs_buffer = np.roll(self.obs_buffer, -1, axis=0)
            self.obs_buffer[-1] = self.features_array[self.current_step]
        
        # Bug #11 fix: Drawdown kill now active during BOTH training and eval
        # so the agent learns to avoid catastrophic drawdowns instead of only seeing them in eval.
        if self.portfolio_value < self.peak_portfolio_value * (1.0 - config.MAX_DRAWDOWN_KILL):
            terminated = True
            
        # Hard stop if portfolio goes to zero (prevent division by zero/overflow)
        if self.portfolio_value <= 0.0:
            terminated = True
            
        # 5. Reward Calculation
        reward = self._compute_step_reward(trade_costs, position_delta, pre_trade_portfolio_value)
        
        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "crypto_held": self.crypto_held,
            "position": self.position_allocation,
            "trades": self.trade_count,
            "fees": self.fees_paid,
            "slippage": self.slippage_costs,
            "force_exit": force_exit,
            "exit_reason": exit_reason,
            "price": current_price
        }
        
        return self._generate_obs(), reward, terminated, truncated, info
        
    def _generate_obs(self) -> np.ndarray:
        """
        Builds the flattened observation vector: [features_seq, position, unrealized_pnl]
        """
        step = min(self.current_step, self.n_steps - 1)
        unrealized_pnl = 0.0
        
        if self.entry_price is not None and self.entry_price > 0.0:
            price_now = self.prices_array[step]
            if self.position_allocation > 0.02:  # Long
                unrealized_pnl = (price_now - self.entry_price) / self.entry_price
            elif self.position_allocation < -0.02:  # Short
                unrealized_pnl = (self.entry_price - price_now) / self.entry_price
                
        state_vars = np.array([self.position_allocation, unrealized_pnl], dtype=np.float32)
        obs = np.concatenate([self.obs_buffer.flatten(), state_vars])
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        
    def _compute_step_reward(self, trade_costs: float, position_delta: float, pre_trade_pv: float) -> float:
        """
        Computes the step reward based on returns, fees, drawdowns, and trading stability.
        """
        if self.prev_portfolio_value <= 0.0 or self.portfolio_value <= 0.0:
            return 0.0
            
        # Log return reward component (which already natively accounts for trade_costs via portfolio_value drops)
        step_return = np.log(self.portfolio_value / (self.prev_portfolio_value + 1e-8))
        
        # Reversal penalty (scaled to 0.002 to permit adjustments while penalizing noise jitter)
        reversal_penalty = 0.002 * abs(position_delta)
        
        # Drawdown penalty (scaled by 0.1 to penalize deviation from lifetime portfolio peak)
        drawdown = 0.0
        if self.peak_portfolio_value > 0.0:
            drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        drawdown_penalty = drawdown * 0.1
        
        # Compound reward (pure return minus transition overhead and drawdown penalty)
        reward_score = step_return - reversal_penalty - drawdown_penalty
        
        # Scale and clip reward (stable PPO target bounds)
        reward = float(np.clip(reward_score * config.REWARD_SCALE, -1.0, 1.0))
        return reward
        
    def render(self, mode="human"):
        step = min(self.current_step, self.n_steps - 1)
        price = self.prices_array[step]
        print(f"Tick {step:5d} | Price: ${price:10,.2f} | Portfolio: ${self.portfolio_value:12,.2f} | Position: {self.position_allocation:+.2f} | Trades: {self.trade_count}")

if __name__ == "__main__":
    # Environment Standalone Test
    import json
    
    if not os.path.exists(config.TRAIN_FEAT_PATH):
        print("Features not found. Please run feature extraction and engineering first.")
        sys.exit(1)
        
    df_features = pd.read_csv(config.TRAIN_FEAT_PATH, index_col="Date", parse_dates=True)
    with open(config.GOLDEN_FEATURES_PATH, "r") as f:
        golden_cols = json.load(f)
        
    print(f"Running standalone test with {len(golden_cols)} features...")
    test_env = CryptoTradingEnv(df_features, golden_cols)
    observation, _ = test_env.reset()
    
    print(f"Observation Vector Shape: {observation.shape}")
    print(f"Action Space Shape: {test_env.action_space}")
    
    total_reward = 0.0
    for tick in range(50):
        dummy_action = test_env.action_space.sample()
        obs, reward, done, trunc, info = test_env.step(dummy_action)
        total_reward += reward
        if done or trunc:
            break
            
    test_env.render()
    print(f"Random Rollout Total Reward: {total_reward:.4f}")
    print("Gym environment standalone test PASSED!")
