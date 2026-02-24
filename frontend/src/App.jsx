import React from 'react';
import TrainingDashboard from './components/TrainingDashboard';
import { Bot } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen p-8 flex flex-col items-center">
      <header className="w-full max-w-6xl mb-8 border-b border-slate-800 pb-4">
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Bot className="text-primary" size={36} />
          RLTrader
          <span className="text-sm bg-primary/20 text-primary px-2 py-1 rounded-full border border-primary/30">Alpha</span>
        </h1>
        <p className="text-slate-400 mt-2 text-sm">Design, Train, and Evaluate Reinforcement Learning Trading Agents</p>
      </header>
      
      <main className="w-full flex-1">
        <TrainingDashboard />
      </main>
    </div>
  );
}

export default App;
