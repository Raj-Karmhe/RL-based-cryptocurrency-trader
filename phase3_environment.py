"""
phase3_environment.py — Custom Gymnasium Trading Environment
=============================================================
Phase 3, Component 1 of the CLSTM-PPO Agent.

ENVIRONMENT DESIGN
------------------
The environment simulates a live crypto trading account.  At each hourly step:

  • The agent observes a ROLLING WINDOW of the last SEQ_LEN (24h) candles,
    represented as the Golden State Space features → fed to the LSTM.
  • The agent outputs a CONTINUOUS action in [-1, +1]:
        +1  = 100% long (fully invested in BTC)
         0  = fully in cash
        -1  = 100% short (borrow and sell BTC)
  • The environment executes the trade, applies:
        - Transaction fees (0.1%)
        - Slippage (0.05%)
        - ATR-based stop-loss and take-profit
        - Turbulence gate (force-exit in extreme market dislocations)
  • Reward is a combination of:
        - Log portfolio return (primary signal)
        - Rolling Sharpe component (risk-adjusted reward)
        - Drawdown penalty (punish losing from peak)
        - Transaction cost penalty (discourage over-trading)
        - Position reversal penalty (punish flip-flopping)

STATE SPACE
-----------
obs = [flattened (SEQ_LEN × N_FEATURES) window] + [position, unrealized_pnl]
Shape: (SEQ_LEN * N_FEATURES + 2,)

ACTION SPACE
------------
Box(-1, +1, shape=(1,), dtype=float32)
    Positive → long allocation fraction
    Negative → short allocation fraction
    |action| → position size as fraction of portfolio
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import importlib
from reward_functions import classify_regimes, AdaptiveRiskControlReward, AsymmetricMarketReward


# ──────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class RiskManager:
    """
    ATR-based dynamic stop-loss and take-profit manager.

    When a position is entered, the manager calculates:
        Long  entry at price P, ATR = A:
            Stop-Loss    = P - SL_MULT × A
            Take-Profit  = P + TP_MULT × A
        Short entry at price P, ATR = A:
            Stop-Loss    = P + SL_MULT × A
            Take-Profit  = P - TP_MULT × A

    Every step, check_exit() tests whether the current price has crossed
    either boundary.  If so, the position is force-closed regardless of the
    agent's action.
    """

    def __init__(self,
                 sl_mult: float = config.STOP_LOSS_ATR_MULT,
                 tp_mult: float = config.TAKE_PROFIT_ATR_MULT):
        self.sl_mult   = sl_mult
        self.tp_mult   = tp_mult
        self.reset()

    def reset(self):
        """Clears all state at the start of an episode."""
        self.entry_price     = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.in_position     = False
        self.position_side   = 0     # +1 long, -1 short
        self.sl_count        = 0     # cumulative stop-loss triggers
        self.tp_count        = 0     # cumulative take-profit triggers

    def on_entry(self, entry_price: float, atr: float, side: int):
        """
        Called when a new position is opened.

        Parameters
        ----------
        entry_price : Execution price of the trade
        atr         : ATR at the time of entry (volatility measure)
        side        : +1 for long, -1 for short
        """
        self.entry_price   = entry_price
        self.position_side = side
        self.in_position   = True

        if side == 1:   # Long
            self.stop_loss_price   = entry_price  - self.sl_mult * atr
            self.take_profit_price = entry_price  + self.tp_mult * atr
        else:           # Short
            self.stop_loss_price   = entry_price  + self.sl_mult * atr
            self.take_profit_price = entry_price  - self.tp_mult * atr

    def check_exit(self, current_price: float, current_atr: float = 0.0) -> tuple:
        """
        Returns (should_exit: bool, reason: str or None).
        Called every step to check if a forced exit is needed.
        Also updates the trailing stop-loss if enabled.
        """
        if not self.in_position:
            return False, None

        if self.position_side == 1:   # Long position
            # Update trailing stop loss
            if getattr(config, "DYNAMIC_STOP_LOSS", False) and current_atr > 0:
                new_sl = current_price - self.sl_mult * current_atr
                if new_sl > self.stop_loss_price:
                    self.stop_loss_price = new_sl

            if current_price <= self.stop_loss_price:
                self.sl_count    += 1
                self.in_position  = False
                return True, "stop_loss"
            if current_price >= self.take_profit_price:
                self.tp_count    += 1
                self.in_position  = False
                return True, "take_profit"
        else:                          # Short position
            # Update trailing stop loss
            if getattr(config, "DYNAMIC_STOP_LOSS", False) and current_atr > 0:
                new_sl = current_price + self.sl_mult * current_atr
                if new_sl < self.stop_loss_price:
                    self.stop_loss_price = new_sl

            if current_price >= self.stop_loss_price:
                self.sl_count    += 1
                self.in_position  = False
                return True, "stop_loss"
            if current_price <= self.take_profit_price:
                self.tp_count    += 1
                self.in_position  = False
                return True, "take_profit"

        return False, None

    def on_exit(self):
        """Called when the agent voluntarily closes a position."""
        self.entry_price       = None
        self.stop_loss_price   = None
        self.take_profit_price = None
        self.in_position       = False
        self.position_side     = 0


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENVIRONMENT
# ──────────────────────────────────────────────────────────────────────────────

class CryptoTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for the CLSTM-PPO cryptocurrency trader.

    Parameters
    ----------
    df              : Scaled feature DataFrame (includes 'Close' and 'ATR')
    feature_cols    : List of feature column names for the observation
    initial_balance : Starting USD balance
    transaction_fee : Percentage fee per trade (e.g. 0.001 = 0.1%)
    slippage        : Additional cost from execution price drift (e.g. 0.0005)
    turb_threshold  : Turbulence index above which we force-exit all positions
    seq_len         : Number of historical bars in the observation window
    allow_short     : If True, agent can take short positions (action < 0)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df:               pd.DataFrame,
        feature_cols:     list,
        initial_balance:  float = config.INITIAL_BALANCE,
        transaction_fee:  float = config.TRANSACTION_FEE,
        slippage:         float = config.SLIPPAGE,
        turb_threshold:   float = None,
        seq_len:          int   = config.SEQ_LEN,
        allow_short:      bool  = True,
        is_eval:          bool  = False
    ):
        super().__init__()

        # Removed importlib.reload(config) because it overwrites runtime overrides 
        # made by phase6_optuna_tune.py with the default values from config.py.

        self.df              = df.copy()
        self.feature_cols    = feature_cols
        self.n_features      = len(feature_cols)
        self.initial_balance = initial_balance
        self.fee_rate        = transaction_fee
        self.slippage_rate   = slippage
        self.turb_threshold  = turb_threshold if turb_threshold is not None \
                               else getattr(config, "TURBULENCE_THRESHOLD", 1e9)
        self.seq_len         = seq_len
        self.allow_short     = getattr(config, "ALLOW_SHORT", allow_short)
        self.is_eval         = is_eval

        # Precompute numpy arrays for speed inside step()
        self.features    = self.df[feature_cols].values.astype(np.float32)
        self.closes      = self.df["Close"].values.astype(np.float64)
        mean_vol = getattr(config, "MEAN_VOL_TRAINING", None)
        self.regimes, self.sigmas = classify_regimes(self.df, mean_vol_training=mean_vol)
        self.adaptive_risk_reward = AdaptiveRiskControlReward()
        self.asymmetric_reward = AsymmetricMarketReward()
        self.atrs        = self.df["ATR"].values.astype(np.float64)
        self.turbulences = (
            self.df["Turbulence"].values.astype(np.float64)
            if "Turbulence" in self.df.columns
            else np.zeros(len(self.df), dtype=np.float64)
        )
        self.n_steps = len(self.df)

        # Observation: flattened SEQ_LEN×N_FEATURES window + [position, pnl]
        self.n_portfolio_state = 2
        obs_dim = seq_len * self.n_features + self.n_portfolio_state

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        # Continuous action: [0.0=flat, +1=full long] if not allow_short else [-1=full short, 0=flat, +1=full long]
        self.action_space = spaces.Box(
            low=-1.0 if self.allow_short else 0.0, high=1.0,
            shape=(1,), dtype=np.float32
        )

        self.risk_manager = RiskManager()
        # Pre-allocate the rolling window buffer
        self._obs_buf = np.zeros((seq_len, self.n_features), dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        # Start from the first step where we have a full SEQ_LEN and 480-hour regime history
        # During training, randomize the starting step to prevent timeline overfitting
        min_start = max(self.seq_len, 480)
        max_start = self.n_steps - 200 # Ensure at least 200 steps remaining
        
        if not self.is_eval and max_start > min_start:
            self.current_step = np.random.randint(min_start, max_start)
        else:
            self.current_step = min_start
        self.balance              = float(self.initial_balance)
        self.btc_held             = 0.0         # positive = long, negative = short
        self.position             = 0.0         # current allocation [-1, +1]
        self.portfolio_value      = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)
        
        self.adaptive_risk_reward.reset()
        self.asymmetric_reward.reset()

        self.peak_value           = float(self.initial_balance)
        self.entry_price          = None
        self.cooldown_counter     = 0           # Steps remaining before allowed to trade
        self.locked_direction     = None        # Tracks direction (1=Long, -1=Short) locked out by TP/SL
        
        # Reward tracking
        self.prev_sharpe          = 0.0
        self.prev_drawdown        = 0.0         # FIX: explicitly reset between episodes
        self._rolling_log_rets    = []          # FIX: reset rolling return buffer each episode

        # Logging
        self.total_trades         = 0
        self.total_fees_paid      = 0.0
        self.total_slippage       = 0.0
        self.turb_exits           = 0
        self.trade_log            = []          # full annotated trade history
        self._rolling_log_rets    = []

        self.risk_manager.reset()

        # Fill the observation buffer with the past SEQ_LEN rows (including current_step to prevent missing a bar)
        start_idx = self.current_step - self.seq_len + 1
        self._obs_buf = self.features[start_idx:self.current_step + 1].copy()

        return self._get_obs(), {}

    # ──────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        """
        Executes one trading step.

        1. Decode action → desired portfolio allocation in [-1, +1]
        2. Check turbulence gate (force-flatten if extreme market stress)
        3. Check ATR stop-loss / take-profit (force-exit if threshold hit)
        4. Execute trade if allocation changed significantly (> 2%)
        5. Advance time, revalue portfolio
        6. Compute reward
        """
        action_val      = float(np.clip(action[0], -1.0, 1.0))
        target_position = action_val if self.allow_short else max(0.0, action_val)

        # ── Handle Directional Lockout ──────────────────────────────────────
        if getattr(config, "PREVENT_SAME_DIRECTION_REENTRY", False):
            if self.locked_direction is not None:
                # Use MIN_POSITION_CHANGE threshold
                min_pos = getattr(config, "MIN_POSITION_CHANGE", 0.2)
                new_side = 1 if target_position > min_pos else -1 if target_position < -min_pos else 0
                if new_side == self.locked_direction:
                    # Attempting to enter same direction: override to flat
                    target_position = 0.0
                elif new_side == -self.locked_direction and new_side != 0:
                    # Flipped to opposite direction with a meaningful trade: lift the lock
                    self.locked_direction = None

        current_price  = self.closes[self.current_step]
        current_atr    = self.atrs[self.current_step]
        force_exit     = False
        exit_reason    = None

        # ── Cooldown Check ──────────────────────────────────────────────────
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            target_position = 0.0  # Override agent's action and force flat

        # ── Turbulence Gate ─────────────────────────────────────────────────
        turb = self.turbulences[self.current_step]
        if turb > self.turb_threshold and abs(self.position) > 0.01:
            target_position = 0.0
            force_exit      = True
            exit_reason     = "turbulence"
            self.turb_exits += 1

        # ── ATR Risk Management ─────────────────────────────────────────────
        if not force_exit and abs(self.position) > 0.01:
            rm_exit, rm_reason = self.risk_manager.check_exit(current_price, current_atr)
            if rm_exit:
                if getattr(config, "PREVENT_SAME_DIRECTION_REENTRY", False):
                    self.locked_direction = 1 if self.position > 0 else -1
                    
                target_position = 0.0
                force_exit      = True
                exit_reason     = rm_reason

        # ── Trigger Cooldown ────────────────────────────────────────────────
        if force_exit:
            self.cooldown_counter = getattr(config, "COOLDOWN_STEPS", 0)

        # ── Execute Trade ───────────────────────────────────────────────────
        position_delta = target_position - self.position
        trade_cost     = 0.0

        # A trade is only a 'closing trade' if we actually held a position and are now flattening it
        is_closing_trade = (abs(target_position) < 0.02) and (abs(self.position) >= 0.02)

        # Execute trade if it's a significant change (>MIN_POSITION_CHANGE), a forced exit, or closing the position
        min_pos_change = getattr(config, "MIN_POSITION_CHANGE", 0.2)
        if abs(position_delta) > min_pos_change or force_exit or is_closing_trade:
            # Bug 3 fix: compute true BTC delta FIRST so fees are charged on the
            # actual notional traded (abs(btc_delta) * price), not on the proxy
            # (abs(position_delta) * portfolio_value) which drifts after PnL.
            target_btc  = (target_position * self.portfolio_value) / current_price
            btc_delta   = target_btc - self.btc_held
            trade_value = abs(btc_delta) * current_price   # true notional
            fee         = trade_value * self.fee_rate
            slip        = trade_value * self.slippage_rate
            trade_cost  = fee + slip
            self.total_fees_paid += fee
            self.total_slippage  += slip

            if btc_delta > 0:           # Buying
                total_cost = btc_delta * current_price + trade_cost
                if total_cost <= self.balance:
                    self.btc_held += btc_delta
                    self.balance  -= total_cost
                else:                   # Buy as much as possible
                    affordable = max(0.0, (self.balance - trade_cost) / current_price)
                    self.btc_held += affordable
                    self.balance  -= (affordable * current_price + trade_cost)
            else:                       # Selling / shorting
                # For simplicity: short selling uses margin (balance as collateral)
                self.btc_held += btc_delta
                self.balance  += (abs(btc_delta) * current_price - trade_cost)

            # Prevent floating-point drift
            self.balance = max(0.0, self.balance)

            # ── Update Risk Manager ──────────────────────────────────────────
            was_flat   = abs(self.position) < 0.02
            is_now_open= abs(target_position) >= 0.02
            
            # If the agent flips from Long directly to Short (or vice versa), 
            # the sign of the position changes.
            old_side = 1 if self.position > 0 else -1 if self.position < 0 else 0
            new_side = 1 if target_position > 0 else -1 if target_position < 0 else 0
            side_flipped = (old_side != new_side) and (old_side != 0) and (new_side != 0)

            if (was_flat and is_now_open) or side_flipped:
                side = 1 if target_position > 0 else -1
                self.risk_manager.on_entry(current_price, current_atr, side)
                self.entry_price = current_price
            elif is_closing_trade:
                self.risk_manager.on_exit()
                self.entry_price = None

            self.total_trades += 1
            self.trade_log.append({
                "step":            self.current_step,
                # Bug 1 fix: store the exact DataFrame timestamp so parse_trades()
                # can place markers on the correct bar without index arithmetic.
                # Strip timezone at source to prevent downstream tz-naive/aware mixing.
                "date":            str(self.df.index[self.current_step].tz_localize(None))
                                   if hasattr(self.df.index[self.current_step], 'tzinfo')
                                   and self.df.index[self.current_step].tzinfo is not None
                                   else str(self.df.index[self.current_step]),
                "price":           current_price,
                "action":          action_val,
                "prev_position":   self.position,
                "target_position": target_position,
                "position_delta":  position_delta,
                "trade_cost":      trade_cost,
                "force_exit":      force_exit,
                "exit_reason":     exit_reason,
            })

            # Only update the model's tracked position if the trade actually executed
            # Sync self.position with actual reality in case 'affordable' fallback was triggered
            if self.portfolio_value > 0:
                self.position = (self.btc_held * current_price) / self.portfolio_value
            else:
                self.position = target_position
        # ── Advance Time ────────────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        # Early termination if portfolio is catastrophically blown up
        if self.portfolio_value < self.initial_balance * (1 - config.MAX_DRAWDOWN_KILL):
            terminated = True

        # Timing fix: revaluation_price is the bar T+1 close used to mark the
        # portfolio to market each step.  Exposing it in info lets run_backtest
        # build a prices[] list that is perfectly aligned with portfolio_values[].
        # Falls back to current_price (bar T) when the episode terminates.
        revaluation_price = current_price
        if not terminated:
            new_price = self.closes[self.current_step]
            revaluation_price = new_price
            self.prev_portfolio_value = self.portfolio_value

            # Portfolio value = cash + market value of BTC position
            # For shorts: if btc_held < 0, we owe BTC at current_price
            raw_pv = self.balance + self.btc_held * new_price
            self.portfolio_value = max(0.0,
                min(raw_pv, self.initial_balance * 1000))
            self.peak_value = max(self.peak_value, self.portfolio_value)

            # Slide the rolling observation buffer by one step
            self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
            self._obs_buf[-1] = self.features[self.current_step]

        # ── Compute Reward ──────────────────────────────────────────────────
        reward = self._compute_reward(trade_cost, position_delta)

        info = {
            "portfolio_value":   self.portfolio_value,
            "balance":           self.balance,
            "btc_held":          self.btc_held,
            "position":          self.position,
            "total_trades":      self.total_trades,
            "total_fees":        self.total_fees_paid,
            "total_slippage":    self.total_slippage,
            "force_exit":        force_exit,
            "exit_reason":       exit_reason,
            "trade_log":         self.trade_log,
            "current_price":     current_price,       # bar T — execution price
            "revaluation_price": revaluation_price,   # bar T+1 — mark-to-market price
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """Builds the flat observation vector."""
        step = min(self.current_step, self.n_steps - 1)

        # Unrealised PnL as a fraction of portfolio
        unrealized_pnl = 0.0
        if self.entry_price and self.entry_price > 0:
            price_now = self.closes[step]
            if self.position > 0.02:    # Long
                unrealized_pnl = (price_now - self.entry_price) / self.entry_price
            elif self.position < -0.02: # Short
                unrealized_pnl = (self.entry_price - price_now) / self.entry_price

        portfolio_state = np.array([self.position, unrealized_pnl],
                                   dtype=np.float32)

        obs = np.concatenate([self._obs_buf.flatten(), portfolio_state])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, trade_cost: float,
                        position_delta: float) -> float:
        """
        Reward = Sharpe-adjusted log return
               − transaction cost penalty
               − drawdown penalty
               − position reversal penalty

        This multi-component reward encourages:
        ✓ Growing the portfolio (log return)
        ✓ Doing so efficiently (Sharpe)
        ✓ Not losing too much from peaks (drawdown penalty)
        ✓ Not trading too frequently (cost penalty + reversal penalty)
        """
        if self.prev_portfolio_value <= 0 or self.portfolio_value <= 0:
            return 0.0

        # Instantaneous log return
        log_ret = np.log(self.portfolio_value / (self.prev_portfolio_value + 1e-8))

        # Rolling Sharpe component over last 20 steps
        self._rolling_log_rets.append(log_ret)
        if len(self._rolling_log_rets) > 20:
            self._rolling_log_rets.pop(0)

        arr = np.array(self._rolling_log_rets)

        if len(self._rolling_log_rets) >= 5:
            sharpe_component = arr.mean() / (arr.std() + 1e-8)
            # Sortino: only penalize downside deviations
            downside = arr[arr < 0]
            downside_std = downside.std() if len(downside) >= 2 else arr.std()
            sortino_component = arr.mean() / (downside_std + 1e-8)
        else:
            sharpe_component = log_ret
            sortino_component = log_ret

        # Transaction cost penalty
        cost_penalty = (trade_cost / (self.portfolio_value + 1e-8))

        # Drawdown computation
        drawdown   = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        
        # CRITICAL FIX: Only penalize NEW drawdown depth.
        # If we penalize absolute drawdown at every step, the agent learns to intentionally
        # blow up the account to terminate the episode early and escape infinite punishment!
        dd_delta = max(0.0, drawdown - getattr(self, "prev_drawdown", 0.0))
        self.prev_drawdown = drawdown
        
        # Reversal penalty (punish flip-flopping)
        rev_penalty = 0.01 * abs(position_delta)

        # Get the configured reward strategy
        strategy = getattr(config, "REWARD_STRATEGY", 0)

        if strategy == 0:
            # 0: Default (Sharpe-adjusted log return with transaction & drawdown penalties)
            dd_penalty = dd_delta * 1.0  # Penalize new drawdown drops
            raw_reward = sharpe_component - cost_penalty - dd_penalty - rev_penalty

        elif strategy == 1:
            # 1: Sideways Penalty (Penalizes holding in sideways markets + exponential drawdown penalty)
            sideways_penalty = 0.0
            # If position is held but return is extremely small, penalize
            if abs(self.position) > 0.1 and abs(log_ret) < 0.0005:
                sideways_penalty = 0.05
            
            dd_penalty = (dd_delta ** 2) * 5.0  # exponential penalty for sharp drops
            raw_reward = log_ret - cost_penalty - dd_penalty - rev_penalty - sideways_penalty

        elif strategy == 2:
            # 2: Sharpe Focused (Rewards increasing Sharpe ratio + linear drawdown penalty)
            dd_penalty = dd_delta * 2.0
            # Emphasize sharpe heavily
            raw_reward = (sharpe_component * 2.0) - cost_penalty - dd_penalty - rev_penalty

        elif strategy == 3:
            # 3: Mixed Sharpe + Sortino
            # Sortino rewards upside volatility; Sharpe smooths out erratic wins.
            # Blend = SORTINO_WEIGHT × Sortino + (1 - SORTINO_WEIGHT) × Sharpe
            sortino_w = getattr(config, "SORTINO_WEIGHT", 0.6)
            blended   = sortino_w * sortino_component + (1.0 - sortino_w) * sharpe_component
            dd_penalty = dd_delta * 1.5   # moderate drawdown penalty
            raw_reward = blended - cost_penalty - dd_penalty - rev_penalty

        elif strategy == 4:
            # 4: Adaptive Risk Control Reward (Mathematics 2026 Paper)
            pnl_t = self.portfolio_value - self.prev_portfolio_value
            current_regime = self.regimes[self.current_step]
            current_sigma = self.sigmas[self.current_step]
            
            raw_reward = self.adaptive_risk_reward.compute(
                pnl=pnl_t, 
                action=0.0, 
                regime=current_regime,
                sigma_480=current_sigma,
                position=self.position,
                position_delta=position_delta,
                cooldown_counter=self.cooldown_counter
            )

        elif strategy == 5:
            # 5: Asymmetric Market Reward (Mathematics 2026 Paper)
            pnl_t = self.portfolio_value - self.prev_portfolio_value
            current_regime = self.regimes[self.current_step]
            current_sigma = self.sigmas[self.current_step]
            
            raw_reward = self.asymmetric_reward.compute(
                pnl=pnl_t,
                action=0.0,
                regime=current_regime,
                sigma_480=current_sigma,
                position=self.position,
                position_delta=position_delta,
                cooldown_counter=self.cooldown_counter
            )

            
        else:
            raw_reward = log_ret

        # Holding Incentive (Counteracts short bias by rewarding long holdings)
        holding_inc = getattr(config, "HOLDING_INCENTIVE", 0.0)
        if self.position > 0:
            raw_reward += holding_inc

        # Clip to a reasonable range for PPO stability
        reward_scale = getattr(config, "REWARD_SCALE", 1.0)
        reward = float(np.clip(raw_reward * reward_scale, -1.0, 1.0))
        return reward

    # ──────────────────────────────────────────────────────────────────────────
    def render(self, mode="human"):
        step = self.current_step
        price = self.closes[min(step, self.n_steps - 1)]
        print(f"Step {step:5d} | Price ${price:10,.2f} | "
              f"Portfolio ${self.portfolio_value:12,.2f} | "
              f"Position {self.position:+.2f} | "
              f"Trades {self.total_trades:4d}")


# ──────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    features_path = getattr(config, "FEATURE_REGISTRY", {}).get(
        getattr(config, "ACTIVE_FEATURE_SET", 22), 
        config.GOLDEN_FEATURES_PATH
    )

    if not os.path.exists(config.TRAIN_FEAT_PATH):
        raise FileNotFoundError("Run Phases 1 & 2 first to generate feature CSVs.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Run Phase 2 first to generate {features_path}.")

    df = pd.read_csv(config.TRAIN_FEAT_PATH, index_col=0, parse_dates=True)
    with open(features_path) as fp:
        golden = json.load(fp)

    env = CryptoTradingEnv(df, feature_cols=golden)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Random rollout
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        total_reward += reward
        if done or trunc:
            break

    env.render()
    print(f"Random rollout reward: {total_reward:.4f}")
    print("Environment standalone test PASSED [OK]")
