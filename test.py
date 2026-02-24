import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import TradingEnv

def main():
    print("Downloading data for testing...")
    data = yf.download('AAPL', start='2020-01-01', end='2025-01-01')
    
    train_size = int(len(data) * 0.8)
    test_data = data.iloc[train_size:]
    
    if not os.path.exists("ppo_trading_agent.zip"):
        print("Model file 'ppo_trading_agent.zip' not found. Train the model first.")
        return

    print("Loading test environment...")
    # Using dummy env to match training dimensions
    env = DummyVecEnv([lambda: TradingEnv(test_data)])
    
    # Access the base TradingEnv instance to tap into its variables
    base_env = env.envs[0]
    
    print("Loading model...")
    model = PPO.load("ppo_trading_agent")
    
    obs = env.reset()
    done = False
    
    # Start tracking values
    portfolio_values = []
    initial_balance = base_env.initial_balance
    dates = []
    
    print("Running evaluation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # DummyVecEnv returns arrays: obs, reward, done, info
        obs, reward, done_array, info = env.step(action)
        done = done_array[0]
        
        # Calculate current net portfolio value
        # Make sure not to go out of bounds
        if base_env.current_step < len(base_env.data):
            current_price = float(base_env.data.iloc[base_env.current_step]['Close'].item())
            portfolio_value = base_env.balance + (base_env.position * current_price)
            portfolio_values.append(portfolio_value)
            dates.append(base_env.data.index[base_env.current_step])
            
    # Print summary
    final_value = portfolio_values[-1] if portfolio_values else initial_balance
    profit = final_value - initial_balance
    return_pct = (profit / initial_balance) * 100
    
    print(f"--- Evaluation Complete ---")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${final_value:.2f}")
    print(f"Total Profit: ${profit:.2f} ({return_pct:.2f}%)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(dates, portfolio_values, label="Portfolio Value")
    plt.title("RL Trading Agent Performance on AAPL (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("portfolio_value.png")
    print("Saved plot to 'portfolio_value.png'.")

if __name__ == '__main__':
    main()
