import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from phase3_environment import CryptoTradingEnv

class MetaTradingEnv(gym.Env):
    """
    Meta-Environment for training a Regime Selector RL Agent.
    
    Action Space: Discrete(4)
        0 = Delegate to Bull Expert
        1 = Delegate to Bear Expert
        2 = Delegate to Crab Expert
        3 = Cash (force close positions)
        
    Observation Space: Box(9,)
        - smoothed 24h log-return
        - smoothed 24h volatility
        - current position (-1 to 1)
        - unrealized PnL
        - drawdown
        - time since last switch (normalized)
        - one-hot of active expert (3 values, cash is all zeros)
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self,
                 base_env: CryptoTradingEnv,
                 model_bull,
                 model_bear,
                 model_crab):
        super().__init__()
        self.base_env = base_env
        self.model_bull = model_bull
        self.model_bear = model_bear
        self.model_crab = model_crab
        
        # Meta-action: 4 discrete choices
        self.action_space = spaces.Discrete(4)
        
        # Meta-observation: 9 dimensions
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(9,), dtype=np.float32
        )
        
        # For smoothing market features
        self.ret_buffer = []
        self.vol_buffer = []
        self.smoothing_window = 24
        
        self.active_expert = 3 # 0: Bull, 1: Bear, 2: Crab, 3: Cash
        self.steps_since_switch = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.base_obs, info = self.base_env.reset()
        
        self.ret_buffer = []
        self.vol_buffer = []
        
        # Prime the smoothing buffer
        for i in range(self.base_env.current_step - self.smoothing_window, self.base_env.current_step):
            idx = max(0, i)
            self.ret_buffer.append(self.base_env.df.iloc[idx].get("1d_log_return", 0.0))
            self.vol_buffer.append(self.base_env.df.iloc[idx].get("1d_hvol_20", 0.0))
            
        self.active_expert = 3
        self.steps_since_switch = 0
        
        return self._get_meta_obs(), info

    def step(self, meta_action: int):
        # 1. Update active expert
        if meta_action != self.active_expert:
            self.active_expert = meta_action
            self.steps_since_switch = 0
            switched = True
        else:
            self.steps_since_switch += 1
            switched = False
            
        # 2. Get the chosen expert's action
        if meta_action == 0:
            expert_action, _ = self.model_bull.predict(self.base_obs, deterministic=True)
        elif meta_action == 1:
            expert_action, _ = self.model_bear.predict(self.base_obs, deterministic=True)
        elif meta_action == 2:
            expert_action, _ = self.model_crab.predict(self.base_obs, deterministic=True)
        else:
            # Cash action
            expert_action = np.array([0.0], dtype=np.float32)
            
        # 3. Step the underlying environment
        next_base_obs, reward, terminated, truncated, info = self.base_env.step(expert_action)
        self.base_obs = next_base_obs
        
        # 4. Update smoothing buffers
        current_step = self.base_env.current_step
        if current_step < len(self.base_env.df):
            self.ret_buffer.append(self.base_env.df.iloc[current_step].get("1d_log_return", 0.0))
            self.vol_buffer.append(self.base_env.df.iloc[current_step].get("1d_hvol_20", 0.0))
            if len(self.ret_buffer) > self.smoothing_window:
                self.ret_buffer.pop(0)
                self.vol_buffer.pop(0)
                
        # 5. Calculate Meta-Reward
        # The meta-reward is the underlying environment's reward, minus a penalty for switching 
        # regimes too often (to prevent erratic flip-flopping)
        meta_reward = reward
        if switched:
            meta_reward -= config.META_SWITCH_PENALTY
            
        return self._get_meta_obs(), float(meta_reward), terminated, truncated, info

    def _get_meta_obs(self):
        smooth_ret = np.mean(self.ret_buffer) if self.ret_buffer else 0.0
        smooth_vol = np.mean(self.vol_buffer) if self.vol_buffer else 0.0
        
        pos = self.base_env.position
        
        # Calculate unrealized pnl roughly
        unrealized_pnl = 0.0
        if self.base_env.entry_price and self.base_env.entry_price > 0:
            step = min(self.base_env.current_step, self.base_env.n_steps - 1)
            price_now = self.base_env.closes[step]
            if pos > 0.02:
                unrealized_pnl = (price_now - self.base_env.entry_price) / self.base_env.entry_price
            elif pos < -0.02:
                unrealized_pnl = (self.base_env.entry_price - price_now) / self.base_env.entry_price
                
        drawdown = (self.base_env.peak_value - self.base_env.portfolio_value) / (self.base_env.peak_value + 1e-8)
        norm_time_since_switch = min(1.0, self.steps_since_switch / 100.0)
        
        expert_one_hot = [0.0, 0.0, 0.0]
        if self.active_expert < 3:
            expert_one_hot[self.active_expert] = 1.0
            
        obs = [
            smooth_ret,
            smooth_vol,
            pos,
            unrealized_pnl,
            drawdown,
            norm_time_since_switch,
            expert_one_hot[0],
            expert_one_hot[1],
            expert_one_hot[2]
        ]
        return np.array(obs, dtype=np.float32)
