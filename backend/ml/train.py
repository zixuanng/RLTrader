import os
import io
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from ml.env import TradingEnv

def fetch_data(symbol: str, features: list):
    print(f"Downloading data for {symbol}...")
    data = yf.download(symbol, start='2020-01-01', end='2025-01-01')
    if len(data) == 0:
        raise Exception(f"Failed to download data for {symbol}.")
    
    # Calculate additional features if requested
    if 'RSI' in features:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
    if 'MACD' in features:
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        
    data = data.fillna(0)
    return data

class WebSocketCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    """
    def __init__(self, verbose=0, notify_callback=None):
        super(WebSocketCallback, self).__init__(verbose)
        self.notify_callback = notify_callback

    def _on_step(self) -> bool:
        if self.notify_callback and self.num_timesteps % 1000 == 0:
            # We call the async notify_callback using loop.call_soon_threadsafe or asyncio.run_coroutine_threadsafe 
            # if we are running in a separate thread. For simplicity in BackgroundTasks, we can just track progress 
            # globally and let the websocket poll it, or use queue.
            pass
        return True

async def train_agent(job_id: str, symbol: str, features: list, progress_dict: dict):
    try:
        progress_dict[job_id].update({"status": "downloading", "progress": 0})
        
        def log(msg):
            progress_dict[job_id]["logs"].append(msg)
            print(f"[{job_id}] {msg}")
            
        log(f"Starting training job for {symbol} with features {features}")
        data = fetch_data(symbol, features)
        
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        
        log(f"Initializing Environment...")
        env = DummyVecEnv([lambda: TradingEnv(train_data, features=features)])
        
        log(f"Initializing PPO Model...")
        model = PPO('MlpPolicy', env, verbose=0)
        
        total_steps = 10000
        progress_dict[job_id]["status"] = "training"
        
        class ProgressCallback(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps % 1000 == 0:
                    pct = int((self.num_timesteps / total_steps) * 100)
                    progress_dict[job_id]["progress"] = pct
                    log(f"Training... {pct}% ({self.num_timesteps}/{total_steps})")
                return True
                
        log(f"Training for {total_steps} timesteps...")
        model.learn(total_timesteps=total_steps, callback=ProgressCallback())
        
        log("Saving model...")
        model_path = f"backend/data/{job_id}_model"
        model.save(model_path)
        progress_dict[job_id]["status"] = "completed"
        progress_dict[job_id]["progress"] = 100
        progress_dict[job_id]["model_path"] = model_path
        log("Training finished successfully!")
        
    except Exception as e:
        progress_dict[job_id]["status"] = "failed"
        progress_dict[job_id]["error"] = str(e)
        print(f"Job {job_id} failed: {e}")

async def evaluate_agent(job_id: str, symbol: str, features: list, model_path: str):
    data = fetch_data(symbol, features)
    train_size = int(len(data) * 0.8)
    test_data = data.iloc[train_size:]
    
    env = DummyVecEnv([lambda: TradingEnv(test_data, features=features)])
    base_env = env.envs[0]
    
    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    
    portfolio_values = []
    dates = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done_array, info = env.step(action)
        done = done_array[0]
        
        if base_env.current_step < len(base_env.data):
            current_price = float(base_env.data.iloc[base_env.current_step]['Close'].item())
            portfolio_value = base_env.balance + (base_env.position * current_price)
            portfolio_values.append(portfolio_value)
            
            # Using index converted to string for JSON serialization
            dates.append(str(base_env.data.index[base_env.current_step].date()))
            
    initial_balance = base_env.initial_balance
    final_value = portfolio_values[-1] if portfolio_values else initial_balance
    profit = final_value - initial_balance
    
    return {
        "initial_balance": initial_balance,
        "final_balance": final_value,
        "profit": profit,
        "return_pct": (profit / initial_balance) * 100,
        "equity_curve": {
            "dates": dates,
            "values": portfolio_values
        }
    }
