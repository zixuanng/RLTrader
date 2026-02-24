import gymnasium as gym
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A simple custom trading environment following OpenAI gymnasium interface.
    """
    def __init__(self, data: pd.DataFrame, initial_balance=10000.0, features=None):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.features = features or ['Close']
        
        # Action space: 0: hold, 1: buy, 2: sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: features * 5 history days
        obs_shape = len(self.features) * 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        
        self.current_step = 4
        self.balance = self.initial_balance
        self.position = 0
        self.avg_cost = 0.0
        
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 4 # Start at index 4 to have 5 days of history
        self.balance = self.initial_balance
        self.position = 0
        self.avg_cost = 0.0
        return self._get_obs(), {}

    def step(self, action):
        # We assume the user's action at time t is performed at the close price
        # Extract scalar value from dataframe
        price = float(self.data.iloc[self.current_step]['Close'].item())
        reward = 0.0
        done = False
        
        transaction_cost = 0.001 * price # 0.1% transaction fee
        
        if action == 1:  # Buy
            if self.balance >= price + transaction_cost:
                self.position += 1
                self.balance -= (price + transaction_cost)
                if self.position == 1:
                    self.avg_cost = price
                else:
                    self.avg_cost = ((self.avg_cost * (self.position - 1)) + price) / self.position
                reward = -transaction_cost
        elif action == 2:  # Sell
            if self.position > 0:
                self.position -= 1
                self.balance += (price - transaction_cost)
                reward = price - self.avg_cost - transaction_cost # Reward is profit - fee
                if self.position == 0:
                    self.avg_cost = 0.0
        else: # Hold
            # Minor penalty to incentivize trading, or 0. We'll leave it as 0
            reward = 0.0
            
        self.current_step += 1
        
        if self.current_step >= len(self.data) - 1:
            done = True
            
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        start = max(0, self.current_step - 4)
        
        obs_list = []
        for feature in self.features:
            if feature in self.data.columns:
                history = self.data[feature].iloc[start:self.current_step+1].values
                history = history.ravel()
                if len(history) < 5:
                    history = np.pad(history, (5 - len(history), 0), 'edge')
                # Simple normalization by first value in window for this feature
                history_norm = history / history[0] if history[0] != 0 else history
                obs_list.extend(history_norm.astype(np.float32))
            else:
                # If feature is missing, pad with zeros
                obs_list.extend([0.0]*5)
                
        return np.array(obs_list, dtype=np.float32)
