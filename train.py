import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import TradingEnv

def main():
    print("Downloading data...")
    # Using specific ticker and date ranges
    data = yf.download('AAPL', start='2020-01-01', end='2025-01-01')
    
    if len(data) == 0:
        print("Failed to download data.")
        return
        
    print(f"Downloaded {len(data)} rows of OHLCV data.")
    
    # Using 80% for training
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    
    print("Initializing Environment...")
    # Wrap in DummyVecEnv
    env = DummyVecEnv([lambda: TradingEnv(train_data)])
    
    print("Initializing PPO Model...")
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
    
    print("Training the agent (this may take a minute)...")
    # Using a 50k steps for reasonable training time
    model.learn(total_timesteps=50000)
    
    # Save the model
    print("Saving model to ppo_trading_agent.zip...")
    model.save("ppo_trading_agent")
    print("Training finished!")

if __name__ == '__main__':
    main()
