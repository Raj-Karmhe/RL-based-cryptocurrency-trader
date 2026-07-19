import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import os
import sys

# ==============================================================================
# PAIRS TRADING GYMNASIUM ENVIRONMENT
# The RL agent interacts with this simulated market to learn spread trading.
# Key difference from single-asset: the agent trades the SPREAD between
# two assets, not individual assets directly.
# ==============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


class SpreadRiskManager:
    """
    Risk manager for spread-based positions.
    Stop-loss and take-profit are applied to the SPREAD P&L, not individual prices.
    Also monitors cointegration breakdown and maximum hold time.
    """

    def __init__(self,
                 zscore_sl: float = config.ZSCORE_STOP_LOSS,
                 zscore_tp: float = config.ZSCORE_TAKE_PROFIT,
                 max_hold: int = config.MAX_HOLD_HOURS,
                 zscore_emergency: float = config.ZSCORE_EMERGENCY_EXIT):
        self.sl_mult = zscore_sl
        self.tp_mult = zscore_tp
        self.max_hold = max_hold
        self.zscore_emergency = zscore_emergency
        self.reset()

    def reset(self):
        """Reset all risk state for a new episode."""
        self.entry_spread = None
        self.entry_zscore = None
        self.in_position = False
        self.position_type = 0       # +1 = long spread, -1 = short spread
        self.hold_duration = 0
        self.sl_count = 0
        self.tp_count = 0
        self.timeout_count = 0
        self.emergency_count = 0

    def on_entry(self, spread_value: float, zscore_value: float, position_type: int):
        """Called when opening a new spread position."""
        self.entry_spread = spread_value
        self.entry_zscore = zscore_value
        self.position_type = position_type
        self.in_position = True
        self.hold_duration = 0

    def check_exit(self, current_spread: float, current_zscore: float):
        """
        Check all risk exit conditions.

        Returns:
            (should_exit: bool, reason: str or None)
        """
        if not self.in_position:
            return False, None

        self.hold_duration += 1

        # 1. Stop-Loss & Take-Profit check (Z-Score based)
        if self.position_type == 1:  # Long spread (entered when Z-score was very negative)
            if current_zscore <= -self.sl_mult: # e.g. -4.0
                self.sl_count += 1
                self.in_position = False
                return True, 'stop_loss'
            if current_zscore >= -self.tp_mult and self.entry_zscore < -self.tp_mult: # e.g. -0.5
                self.tp_count += 1
                self.in_position = False
                return True, 'take_profit'
        else:  # Short spread (entered when Z-score was very positive)
            if current_zscore >= self.sl_mult:  # e.g. 4.0
                self.sl_count += 1
                self.in_position = False
                return True, 'stop_loss'
            if current_zscore <= self.tp_mult and self.entry_zscore > self.tp_mult:  # e.g. 0.5
                self.tp_count += 1
                self.in_position = False
                return True, 'take_profit'

        # 2. Maximum hold duration
        if self.hold_duration >= self.max_hold:
            self.timeout_count += 1
            self.in_position = False
            return True, 'timeout'

        # 3. Emergency z-score exit (spread has blown out)
        if abs(current_zscore) > self.zscore_emergency:
            self.emergency_count += 1
            self.in_position = False
            return True, 'zscore_emergency'

        return False, None

    def on_exit(self):
        """Clear position state on manual exit."""
        self.entry_spread = None
        self.entry_spread_atr = None
        self.stop_loss_level = None
        self.take_profit_level = None
        self.in_position = False
        self.position_type = 0
        self.hold_duration = 0


class PairsTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for RL-based pairs trading.

    The agent trades the SPREAD between two cointegrated assets.

    Action Space: Continuous [-1, +1]
        -1.0 = Full short spread (short A, long B)
         0.0 = Flat (no position)
        +1.0 = Full long spread (long A, short B)

    Observation Space: Flattened vector of:
        - Rolling window (TIME_WINDOW steps) of all feature columns
        - Portfolio state: [position, unrealized_pnl, current_zscore, hedge_ratio]

    Reward: Multi-component based on spread P&L, risk-adjusted returns,
            transaction costs, and mean-reversion bonuses.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list = None,
        initial_balance: float = config.INITIAL_BALANCE,
        transaction_fee: float = config.TRANSACTION_FEE,
        slippage: float = config.SLIPPAGE,
        time_window: int = config.TIME_WINDOW,
        use_risk_management: bool = True,
        use_turbulence: bool = True,
        turbulence_threshold: float = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns or config.FEATURE_COLUMNS
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.time_window = time_window
        self.use_risk_management = use_risk_management
        self.use_turbulence = use_turbulence
        self.turb_threshold = turbulence_threshold or getattr(
            config, 'TURBULENCE_THRESHOLD', None
        ) or 90.0

        # ── Precompute arrays for fast access ─────────────────────────────
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        self.close_a = self.df['Close_A'].values.astype(np.float64)
        self.close_b = self.df['Close_B'].values.astype(np.float64)
        self.spreads = self.df['Spread'].values.astype(np.float64)
        self.hedge_ratios = self.df['Hedge_Ratio'].values.astype(np.float64)
        self.n_steps = len(self.df)

        # ATR for spread (approximate: use spread rolling std as a proxy)
        spread_series = pd.Series(self.spreads)
        self.spread_atrs = spread_series.rolling(config.ATR_PERIOD).std().fillna(
            spread_series.std()
        ).values.astype(np.float64)

        # Z-scores (precomputed from features if available, else compute)
        if 'Spread_ZScore' in self.df.columns:
            # Use the RAW (unscaled) z-score for risk management decisions
            # We need to get it from the unscaled data, but if scaled, just
            # use the spread directly
            spread_mean = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).mean()
            spread_std = spread_series.rolling(config.SPREAD_ZSCORE_WINDOW).std()
            self.zscores = ((spread_series - spread_mean) / (spread_std + 1e-10)).fillna(0).values
        else:
            self.zscores = np.zeros(self.n_steps)

        self.turbulences = self.df['Turbulence'].values.astype(np.float64) \
            if 'Turbulence' in self.df.columns else np.zeros(self.n_steps)

        # Funding rates and KDE CDF
        self.funding_a = self.df['Funding_Rate_A'].values.astype(np.float64) if 'Funding_Rate_A' in self.df.columns else np.zeros(self.n_steps)
        self.funding_b = self.df['Funding_Rate_B'].values.astype(np.float64) if 'Funding_Rate_B' in self.df.columns else np.zeros(self.n_steps)
        self.cdf_kde = self.df['Spread_CDF_KDE'].values.astype(np.float64) if 'Spread_CDF_KDE' in self.df.columns else np.full(self.n_steps, 0.5)

        # ── Observation and Action spaces ─────────────────────────────────
        n_features = len(self.feature_columns)
        self.n_portfolio_features = config.N_PORTFOLIO_FEATURES
        obs_dim = time_window * n_features + self.n_portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.risk_manager = SpreadRiskManager()
        self._obs_buffer = np.zeros(
            (time_window, n_features), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment at the start of a new episode."""
        super().reset(seed=seed)

        self.current_step = self.time_window
        self.balance = float(self.initial_balance)
        self.position = 0.0          # Current spread position: [-1, +1]
        self.portfolio_value = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)
        self.peak_value = float(self.initial_balance)

        # Position tracking
        self.entry_spread_value = None
        self.entry_price_a = None
        self.entry_price_b = None
        self.units_a = 0.0           # Number of units of asset A
        self.units_b = 0.0           # Number of units of asset B
        self.notional_per_leg = 0.0  # Dollar value allocated per leg
        self.zscore_cooldown_active = False # Block trades after extreme Z-score

        # Logging
        self.total_trades = 0
        self.total_fees_paid = 0.0
        self.total_funding_paid = 0.0
        self.turb_exits = 0
        self.trade_log = []
        self.rolling_log_returns = []

        self.risk_manager.reset()

        # Fill observation buffer
        self._obs_buffer = self.features[self.current_step - self.time_window + 1 : self.current_step + 1].copy()

        return self._get_obs(), {}

    def step(self, action):
        """Execute one trading step."""
        # ── 1. Decode action ──────────────────────────────────────────────
        action_val = float(np.clip(action[0], -1.0, 1.0))

        # Discretize into zones: short spread / flat / long spread
        # Dead zone around 0 to reduce whipsawing
        if abs(action_val) < 0.1:
            target_position = 0.0
        else:
            target_position = action_val

        current_price_a = self.close_a[self.current_step]
        current_price_b = self.close_b[self.current_step]
        current_spread = self.spreads[self.current_step]
        current_zscore = self.zscores[self.current_step]
        current_hedge_ratio = self.hedge_ratios[self.current_step]
        current_spread_atr = self.spread_atrs[self.current_step]

        force_exit = False
        exit_reason = None

        # ── 1.5 Z-Score Cooldown (Regime Blocking) ───────────────────────
        if abs(current_zscore) >= getattr(config, 'ZSCORE_COOLDOWN_TRIGGER', 3.5):
            self.zscore_cooldown_active = True
        elif abs(current_zscore) <= getattr(config, 'ZSCORE_COOLDOWN_RELEASE', 2.75):
            self.zscore_cooldown_active = False

        if self.zscore_cooldown_active and abs(self.position) < 0.05 and abs(target_position) > 0.05:
            # Block new entries while cooldown is active
            target_position = 0.0

        # ── 2. Turbulence gate ────────────────────────────────────────────
        if self.use_turbulence and abs(self.position) > 0.05:
            turb = self.turbulences[self.current_step]
            if turb > self.turb_threshold:
                target_position = 0.0
                force_exit = True
                exit_reason = 'turbulence'
                self.turb_exits += 1

        # ── 3. Risk management checks ────────────────────────────────────
        if self.use_risk_management and not force_exit and abs(self.position) > 0.05:
            rm_exit, rm_reason = self.risk_manager.check_exit(
                current_spread, current_zscore
            )
            
            # ── 3.5 Price Circuit Breaker (Black Swan Protection) ────────
            if not rm_exit and self.entry_price_a is not None and self.entry_price_b is not None:
                pct_a = (current_price_a - self.entry_price_a) / self.entry_price_a
                pct_b = (current_price_b - self.entry_price_b) / self.entry_price_b
                
                # Check if asset moves > X% against us
                if self.position > 0: # Long spread (long A, short B)
                    if pct_a < -config.PRICE_CIRCUIT_BREAKER_PCT or pct_b > config.PRICE_CIRCUIT_BREAKER_PCT:
                        rm_exit, rm_reason = True, 'price_circuit_breaker'
                else: # Short spread (short A, long B)
                    if pct_a > config.PRICE_CIRCUIT_BREAKER_PCT or pct_b < -config.PRICE_CIRCUIT_BREAKER_PCT:
                        rm_exit, rm_reason = True, 'price_circuit_breaker'

            if rm_exit:
                target_position = 0.0
                force_exit = True
                exit_reason = rm_reason

        # ── 4. Execute position change ────────────────────────────────────
        position_change = target_position - self.position
        trade_cost = 0.0

        if abs(position_change) > 0.05:  # Minimum trade threshold
            # Calculate trade value for BOTH legs
            trade_fraction = abs(position_change)
            trade_value_per_leg = trade_fraction * self.portfolio_value * \
                config.MAX_POSITION_FRACTION / 2.0

            # Costs apply to BOTH legs
            fee_a = trade_value_per_leg * self.transaction_fee
            slip_a = trade_value_per_leg * self.slippage
            fee_b = trade_value_per_leg * self.transaction_fee
            slip_b = trade_value_per_leg * self.slippage
            trade_cost = fee_a + slip_a + fee_b + slip_b
            self.total_fees_paid += trade_cost

            # Update position
            was_flat = abs(self.position) < 0.05
            is_entering = abs(target_position) > 0.05
            is_exiting = abs(target_position) < 0.05

            if is_exiting:
                # Close position: realize P&L
                if self.entry_price_a is not None and self.entry_price_b is not None:
                    # P&L from long leg
                    if self.position > 0:  # Was long spread (long A, short B)
                        pnl_a = self.units_a * (current_price_a - self.entry_price_a)
                        pnl_b = self.units_b * (self.entry_price_b - current_price_b)
                    else:  # Was short spread (short A, long B)
                        pnl_a = self.units_a * (self.entry_price_a - current_price_a)
                        pnl_b = self.units_b * (current_price_b - self.entry_price_b)

                    total_pnl = pnl_a + pnl_b - trade_cost
                    self.balance += self.notional_per_leg * 2 + total_pnl

                self.units_a = 0.0
                self.units_b = 0.0
                self.entry_price_a = None
                self.entry_price_b = None
                self.entry_spread_value = None
                self.notional_per_leg = 0.0
                self.risk_manager.on_exit()

            elif was_flat and is_entering:
                # Open new position
                self.notional_per_leg = min(
                    trade_value_per_leg,
                    (self.balance - trade_cost) / 2.0
                )
                self.notional_per_leg = max(0.0, self.notional_per_leg)

                self.units_a = self.notional_per_leg / current_price_a
                self.units_b = self.notional_per_leg / current_price_b

                self.entry_price_a = current_price_a
                self.entry_price_b = current_price_b
                self.entry_spread_value = current_spread
                self.balance -= (self.notional_per_leg * 2 + trade_cost)
                self.balance = max(0.0, self.balance)

                ptype = 1 if target_position > 0 else -1
                self.risk_manager.on_entry(current_spread, current_zscore, ptype)

            else:
                # Position adjustment (partial close/open)
                # For simplicity, close and reopen at new size
                # Close existing
                if self.entry_price_a is not None:
                    if self.position > 0:
                        pnl_a = self.units_a * (current_price_a - self.entry_price_a)
                        pnl_b = self.units_b * (self.entry_price_b - current_price_b)
                    else:
                        pnl_a = self.units_a * (self.entry_price_a - current_price_a)
                        pnl_b = self.units_b * (current_price_b - self.entry_price_b)
                    total_pnl = pnl_a + pnl_b
                    self.balance += self.notional_per_leg * 2 + total_pnl
                    self.risk_manager.on_exit()

                # Reopen at new size
                new_notional = abs(target_position) * self.portfolio_value * \
                    config.MAX_POSITION_FRACTION / 2.0
                new_notional = min(new_notional, (self.balance - trade_cost) / 2.0)
                new_notional = max(0.0, new_notional)

                self.notional_per_leg = new_notional
                self.units_a = new_notional / current_price_a
                self.units_b = new_notional / current_price_b
                self.entry_price_a = current_price_a
                self.entry_price_b = current_price_b
                self.entry_spread_value = current_spread
                self.balance -= (new_notional * 2 + trade_cost)
                self.balance = max(0.0, self.balance)

                ptype = 1 if target_position > 0 else -1
                self.risk_manager.on_entry(current_spread, current_zscore, ptype)

            self.total_trades += 1
            self.trade_log.append({
                'step': self.current_step,
                'price_a': current_price_a,
                'price_b': current_price_b,
                'spread': current_spread,
                'zscore': current_zscore,
                'action': action_val,
                'target_position': target_position,
                'prev_position': self.position,
                'position_change': position_change,
                'trade_cost': trade_cost,
                'exit_reason': exit_reason,
                'portfolio_value': self.portfolio_value,
            })

        self.position = target_position

        # ── 5. Advance time ───────────────────────────────────────────────
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        if not terminated:
            new_price_a = self.close_a[self.current_step]
            new_price_b = self.close_b[self.current_step]

            # Update portfolio value (mark-to-market)
            self.prev_portfolio_value = self.portfolio_value
            unrealized_pnl = 0.0

            if self.entry_price_a is not None and abs(self.position) > 0.05:
                # Apply hourly funding cost
                hourly_fund_a = self.funding_a[self.current_step] / 8.0
                hourly_fund_b = self.funding_b[self.current_step] / 8.0
                notional_a = self.units_a * new_price_a
                notional_b = self.units_b * new_price_b
                
                if self.position > 0:  # Long spread
                    pnl_a = self.units_a * (new_price_a - self.entry_price_a)
                    pnl_b = self.units_b * (self.entry_price_b - new_price_b)
                    funding_cost = (notional_a * hourly_fund_a) + (notional_b * -hourly_fund_b)
                else:  # Short spread
                    pnl_a = self.units_a * (self.entry_price_a - new_price_a)
                    pnl_b = self.units_b * (new_price_b - self.entry_price_b)
                    funding_cost = (notional_a * -hourly_fund_a) + (notional_b * hourly_fund_b)
                
                unrealized_pnl = pnl_a + pnl_b
                self.balance -= funding_cost
                self.total_funding_paid += funding_cost

            raw_pv = self.balance + self.notional_per_leg * 2 + unrealized_pnl
            self.portfolio_value = max(0.0, min(raw_pv, self.initial_balance * 1000))
            self.peak_value = max(self.peak_value, self.portfolio_value)

            # Slide observation buffer
            self._obs_buffer = np.roll(self._obs_buffer, -1, axis=0)
            self._obs_buffer[-1] = self.features[self.current_step]

        # ── 6. Compute reward ─────────────────────────────────────────────
        reward = self._compute_reward(trade_cost, position_change, current_zscore)

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'spread': current_spread,
            'zscore': current_zscore,
            'hedge_ratio': current_hedge_ratio,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'total_funding': self.total_funding_paid,
            'force_exit': force_exit,
            'exit_reason': exit_reason,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Construct the flat observation vector."""
        step = min(self.current_step, self.n_steps - 1)

        # Unrealized P&L as fraction of portfolio
        unrealized_pnl_frac = 0.0
        if self.entry_price_a is not None and abs(self.position) > 0.05:
            if self.position > 0:
                pnl_a = self.units_a * (self.close_a[step] - self.entry_price_a)
                pnl_b = self.units_b * (self.entry_price_b - self.close_b[step])
            else:
                pnl_a = self.units_a * (self.entry_price_a - self.close_a[step])
                pnl_b = self.units_b * (self.close_b[step] - self.entry_price_b)
            unrealized_pnl_frac = (pnl_a + pnl_b) / (self.portfolio_value + 1e-8)

        portfolio_state = np.array([
            self.position,
            unrealized_pnl_frac,
            self.zscores[step],
            self.hedge_ratios[step],
        ], dtype=np.float32)

        obs = np.concatenate([self._obs_buffer.flatten(), portfolio_state])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    def _compute_reward(self, trade_cost: float, position_change: float,
                         current_zscore: float) -> float:
        """
        Multi-component reward function for pairs trading.

        Components:
        1. Portfolio log-return (Sharpe-like)
        2. Transaction cost penalty
        3. Drawdown penalty
        4. Reversal penalty (reduces whipsawing)
        5. Mean-reversion bonus (encourages correct spread positioning)
        """
        if self.prev_portfolio_value <= 0 or self.portfolio_value <= 0:
            return 0.0

        # 1. Log return of portfolio value
        log_return = np.log(self.portfolio_value / self.prev_portfolio_value)

        # Rolling Sharpe component
        self.rolling_log_returns.append(log_return)
        if len(self.rolling_log_returns) > 20:
            self.rolling_log_returns.pop(0)

        if len(self.rolling_log_returns) >= 5:
            r_arr = np.array(self.rolling_log_returns)
            sharpe_component = r_arr.mean() / (r_arr.std() + 1e-8)
        else:
            sharpe_component = log_return

        # 2. Transaction cost penalty (applied to both legs)
        cost_penalty = (trade_cost / self.portfolio_value) \
            if self.portfolio_value > 0 else 0.0

        # 3. Drawdown penalty
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        dd_penalty = max(0.0, drawdown) * 0.1

        # 4. Reversal penalty
        rev_penalty = 0.01 * abs(position_change)

        mr_bonus = 0.0
        current_cdf = self.cdf_kde[self.current_step]
        if abs(self.position) > 0.05:
            # Reward correct alignment with KDE extremes
            # cdf < 0.2 means spread is unusually low -> expect rise -> long spread (+1)
            # cdf > 0.8 means spread is unusually high -> expect fall -> short spread (-1)
            if current_cdf < 0.2 and self.position > 0:
                mr_bonus = 0.005 * (0.2 - current_cdf) * 5.0
            elif current_cdf > 0.8 and self.position < 0:
                mr_bonus = 0.005 * (current_cdf - 0.8) * 5.0
            elif current_cdf < 0.2 and self.position < 0:
                mr_bonus = -0.002 * (0.2 - current_cdf) * 5.0
            elif current_cdf > 0.8 and self.position > 0:
                mr_bonus = -0.002 * (current_cdf - 0.8) * 5.0

        # Combine
        raw_reward = sharpe_component - cost_penalty - dd_penalty - rev_penalty + mr_bonus

        # Scale and clip
        reward = np.clip(raw_reward * config.REWARD_SCALING * 1e4, -1.0, 1.0)
        return float(reward)
