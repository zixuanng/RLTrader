# RLTrader

RLTrader is a full-stack web application designed for quantitative finance enthusiasts to build, train, and evaluate Reinforcement Learning (RL) agents for algorithmic trading.

## 🚀 Features

- **Custom RL Environment**: A Gymnasium-based trading environment that simulates market dynamics.
- **Dynamic Feature Selection**: Choose from various technical indicators like RSI and MACD to include in the agent's state.
- **Real-time Training Logs**: Monitor the agent's learning progress via WebSockets.
- **Backtesting & Evaluation**: Visualize the performance of trained agents on historical data with equity curves.
- **Modern UI**: A responsive, dark-mode dashboard built with React and TailwindCSS.

## 🛠️ Tech Stack

- **Frontend**: React (Vite), TailwindCSS 4, Recharts, Lucide React.
- **Backend**: FastAPI, Uvicorn, WebSockets.
- **Machine Learning**: Stable-Baselines3 (PPO), Gymnasium, PyTorch.
- **Data**: yfinance for historical market data.

## 📦 Installation

### Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python -m uvicorn main:app --reload --port 8000
   ```

### Frontend

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the Vite dev server:
   ```bash
   npm run dev
   ```

## 📈 Usage

1. Open your browser and navigate to `http://localhost:5173`.
2. Select an asset symbol (e.g., AAPL).
3. Choose the technical indicators you want the agent to use.
4. Click **Start Training**.
5. Once training is complete, view the evaluation results and equity curve.

## 🛡️ Disclaimer

This platform is for educational and research purposes only. Trading involves significant risk, and reinforcement learning models can behave unpredictably. Never trade with money you cannot afford to lose.

## 📄 License

MIT
