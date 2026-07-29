"""
phase3_environment.py — Custom Gymnasium Trading Environment
=============================================================
Phase 3, Component 1 of the CLSTM-PPO Agent.
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


class RiskManager:
    """ATR-based dynamic stop-loss and take-profit manager."""

    def __init__(self,
                 sl_mult: float = config.STOP_LOSS_ATR_MULT,
                 tp_mult: float = config.TAKE_PROFIT_ATR_MULT):
        self.sl_mult   = sl_mult
        self.tp_mult   = tp_mult
        self.reset()

    def reset(self):
        self.entry_price     = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.in_position     = False
        self.position_side   = 0
        self.sl_count        = 0
        self.tp_count        = 0

    def on_entry(self, entry_price: float, atr: float, side: int):
        self.entry_price   = entry_price
        self.position_side = side
        self.in_position   = True

        if side == 1:
            self.stop_loss_price   = entry_price  - self.sl_mult * atr
            self.take_profit_price = entry_price  + self.tp_mult * atr
        else:
            self.stop_loss_price   = entry_price  + self.sl_mult * atr
            self.take_profit_price = entry_price  - self.tp_mult * atr

    def check_exit(self, current_price: float) -> tuple:
        if not self.in_position:
            return False, None

        if self.position_side == 1:
            if current_price <= self.stop_loss_price:
                self.sl_count    += 1
                self.in_position  = False
                return True, "stop_loss"
            if current_price >= self.take_profit_price:
                self.tp_count    += 1
                self.in_position  = False
                return True, "take_profit"
        else:
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
        self.entry_price       = None
        self.stop_loss_price   = None
        self.take_profit_price = None
        self.in_position       = False
        self.position_side     = 0


class CryptoTradingEnv(gym.Env):
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

        self.df              = df.reset_index(drop=True)
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

        self.features    = self.df[feature_cols].values.astype(np.float32)
        self.closes      = self.df["Close"].values.astype(np.float64)
        self.atrs        = self.df["ATR"].values.astype(np.float64)
        self.turbulences = (
            self.df["Turbulence"].values.astype(np.float64)
            if "Turbulence" in self.df.columns
            else np.zeros(len(self.df), dtype=np.float64)
        )
        self.n_steps = len(self.df)

        self.n_portfolio_state = 2
        obs_dim = seq_len * self.n_features + self.n_portfolio_state

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)
        self.action_map = {
            0: -1.00,
            1: -0.75,
            2: -0.50,
            3: -0.25,
            4:  0.00,
            5:  0.25,
            6:  0.50,
            7:  0.75,
            8:  1.00,
        }

        self.risk_manager = RiskManager()
        self._obs_buf = np.zeros((seq_len, self.n_features), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = self.n_steps - 200
        if not self.is_eval and max_start > self.seq_len:
            self.current_step = np.random.randint(self.seq_len, max_start)
        else:
            self.current_step = self.seq_len
        self.balance              = float(self.initial_balance)
        self.btc_held             = 0.0
        self.position             = 0.0
        self.portfolio_value      = float(self.initial_balance)
        self.prev_portfolio_value = float(self.initial_balance)
        self.peak_value           = float(self.initial_balance)
        self.entry_price          = None
        self.cooldown_counter     = 0
        self.locked_direction     = None

        self.prev_sharpe          = 0.0
        self.prev_drawdown        = 0.0

        self.total_trades         = 0
        self.total_fees_paid      = 0.0
        self.total_slippage       = 0.0
        self.turb_exits           = 0
        self.trade_log            = []
        self._rolling_log_rets    = []

        self.risk_manager.reset()

        start_idx = self.current_step - self.seq_len
        self._obs_buf = self.features[start_idx:self.current_step].copy()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = int(action)

        action_val = self.action_map[action]

        target_position = (
            action_val
            if self.allow_short
            else max(0.0, action_val)
        )
        if getattr(config, "PREVENT_SAME_DIRECTION_REENTRY", False):
            if self.locked_direction is not None:
                min_pos = getattr(config, "MIN_POSITION_CHANGE", 0.2)
                new_side = 1 if target_position > min_pos else -1 if target_position < -min_pos else 0
                if new_side == self.locked_direction:
                    target_position = 0.0
                elif new_side == -self.locked_direction and new_side != 0:
                    self.locked_direction = None

        current_price  = self.closes[self.current_step]
        current_atr    = self.atrs[self.current_step]
        force_exit     = False
        exit_reason    = None

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            target_position = 0.0

        turb = self.turbulences[self.current_step]
        if turb > self.turb_threshold and abs(self.position) > 0.01:
            target_position = 0.0
            force_exit      = True
            exit_reason     = "turbulence"
            self.turb_exits += 1

        if not force_exit and abs(self.position) > 0.01:
            rm_exit, rm_reason = self.risk_manager.check_exit(current_price)
            if rm_exit:
                if getattr(config, "PREVENT_SAME_DIRECTION_REENTRY", False):
                    self.locked_direction = 1 if self.position > 0 else -1

                target_position = 0.0
                force_exit      = True
                exit_reason     = rm_reason

        if force_exit:
            self.cooldown_counter = getattr(config, "COOLDOWN_STEPS", 0)

        position_delta = target_position - self.position
        trade_cost     = 0.0

        is_closing_trade = (abs(target_position) < 0.02) and (abs(self.position) >= 0.02)

        min_pos_change = getattr(config, "MIN_POSITION_CHANGE", 0.2)
        if abs(position_delta) > min_pos_change or force_exit or is_closing_trade:
            trade_value = abs(position_delta) * self.portfolio_value
            fee         = trade_value * self.fee_rate
            slip        = trade_value * self.slippage_rate
            trade_cost  = fee + slip
            self.total_fees_paid += fee
            self.total_slippage  += slip

            target_btc = (target_position * self.portfolio_value) / current_price
            btc_delta  = target_btc - self.btc_held

            if btc_delta > 0:
                total_cost = btc_delta * current_price + trade_cost
                if total_cost <= self.balance:
                    self.btc_held += btc_delta
                    self.balance  -= total_cost
                else:
                    affordable = max(0.0, (self.balance - trade_cost) / current_price)
                    self.btc_held += affordable
                    self.balance  -= (affordable * current_price + trade_cost)
            else:
                self.btc_held += btc_delta
                self.balance  += (abs(btc_delta) * current_price - trade_cost)

            self.balance = max(0.0, self.balance)

            was_flat   = abs(self.position) < 0.02
            is_now_open= abs(target_position) >= 0.02

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
                "price":           current_price,
                "action":          action_val,
                "prev_position":   self.position,
                "target_position": target_position,
                "position_delta":  position_delta,
                "trade_cost":      trade_cost,
                "force_exit":      force_exit,
                "exit_reason":     exit_reason,
            })

            if self.portfolio_value > 0:
                self.position = (self.btc_held * current_price) / self.portfolio_value
            else:
                self.position = target_position

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1
        truncated  = False

        if self.portfolio_value < self.initial_balance * (1 - config.MAX_DRAWDOWN_KILL):
            terminated = True

        if not terminated:
            new_price = self.closes[self.current_step]
            self.prev_portfolio_value = self.portfolio_value

            raw_pv = self.balance + self.btc_held * new_price
            self.portfolio_value = max(0.0,
                min(raw_pv, self.initial_balance * 1000))
            self.peak_value = max(self.peak_value, self.portfolio_value)

            self._obs_buf = np.roll(self._obs_buf, -1, axis=0)
            self._obs_buf[-1] = self.features[self.current_step]

        reward = self._compute_reward(trade_cost, position_delta)

        info = {
            "portfolio_value": self.portfolio_value,
            "balance":         self.balance,
            "btc_held":        self.btc_held,
            "position":        self.position,
            "total_trades":    self.total_trades,
            "total_fees":      self.total_fees_paid,
            "total_slippage":  self.total_slippage,
            "force_exit":      force_exit,
            "exit_reason":     exit_reason,
            "current_price":   current_price,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        step = min(self.current_step, self.n_steps - 1)

        unrealized_pnl = 0.0
        if self.entry_price and self.entry_price > 0:
            price_now = self.closes[step]
            if self.position > 0.02:
                unrealized_pnl = (price_now - self.entry_price) / self.entry_price
            elif self.position < -0.02:
                unrealized_pnl = (self.entry_price - price_now) / self.entry_price

        portfolio_state = np.array([self.position, unrealized_pnl],
                                   dtype=np.float32)

        obs = np.concatenate([self._obs_buf.flatten(), portfolio_state])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    def _compute_reward(self, trade_cost: float,
                        position_delta: float) -> float:
        if self.prev_portfolio_value <= 0 or self.portfolio_value <= 0:
            return 0.0

        log_ret = np.log(self.portfolio_value / (self.prev_portfolio_value + 1e-8))

        self._rolling_log_rets.append(log_ret)
        if len(self._rolling_log_rets) > 20:
            self._rolling_log_rets.pop(0)

        arr = np.array(self._rolling_log_rets)

        if len(self._rolling_log_rets) >= 5:
            sharpe_component = arr.mean() / (arr.std() + 1e-8)
            downside = arr[arr < 0]
            downside_std = downside.std() if len(downside) >= 2 else arr.std()
            sortino_component = arr.mean() / (downside_std + 1e-8)
        else:
            sharpe_component = log_ret
            sortino_component = log_ret

        cost_penalty = (trade_cost / (self.portfolio_value + 1e-8))

        drawdown   = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)

        dd_delta = max(0.0, drawdown - getattr(self, "prev_drawdown", 0.0))
        self.prev_drawdown = drawdown

        rev_penalty = 0.01 * abs(position_delta)

        strategy = getattr(config, "REWARD_STRATEGY", 0)

        if strategy == 0:
            dd_penalty = dd_delta * 1.0
            raw_reward = sharpe_component - cost_penalty - dd_penalty - rev_penalty

        elif strategy == 1:
            sideways_penalty = 0.0
            if abs(self.position) > 0.1 and abs(log_ret) < 0.0005:
                sideways_penalty = 0.05

            dd_penalty = (dd_delta ** 2) * 5.0
            raw_reward = log_ret - cost_penalty - dd_penalty - rev_penalty - sideways_penalty

        elif strategy == 2:
            dd_penalty = dd_delta * 2.0
            raw_reward = (sharpe_component * 2.0) - cost_penalty - dd_penalty - rev_penalty

        elif strategy == 3:
            sortino_w = getattr(config, "SORTINO_WEIGHT", 0.6)
            blended   = sortino_w * sortino_component + (1.0 - sortino_w) * sharpe_component
            dd_penalty = dd_delta * 1.5
            raw_reward = blended - cost_penalty - dd_penalty - rev_penalty

        else:
            raw_reward = log_ret

        holding_inc = getattr(config, "HOLDING_INCENTIVE", 0.0)
        if self.position > 0:
            raw_reward += holding_inc

        reward_scale = getattr(config, "REWARD_SCALE", 1.0)
        reward = float(np.clip(raw_reward * reward_scale, -1.0, 1.0))
        return reward

    def render(self, mode="human"):
        step = self.current_step
        price = self.closes[min(step, self.n_steps - 1)]
        print(f"Step {step:5d} | Price ${price:10,.2f} | "
              f"Portfolio ${self.portfolio_value:12,.2f} | "
              f"Position {self.position:+.2f} | "
              f"Trades {self.total_trades:4d}")


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