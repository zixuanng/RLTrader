import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, Play, CheckCircle, AlertCircle } from 'lucide-react';

const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

function TrainingDashboard() {
  const [symbol, setSymbol] = useState('AAPL');
  const [features, setFeatures] = useState({ Close: true, RSI: false, MACD: false });
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, training, completed, failed
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);

  const handleFeatureToggle = (feature) => {
    setFeatures(prev => ({ ...prev, [feature]: !prev[feature] }));
  };

  const startTraining = async () => {
    try {
      setStatus('starting');
      setLogs([]);
      setResults(null);
      setProgress(0);
      
      const selectedFeatures = Object.entries(features)
        .filter(([_, isSelected]) => isSelected)
        .map(([f]) => f);
        
      const res = await axios.post(`${API_BASE}/api/train`, {
        symbol,
        features: selectedFeatures
      });
      
      setJobId(res.data.job_id);
    } catch (err) {
      console.error(err);
      setStatus('failed');
      setLogs(prev => [...prev, "Failed to start training job."]);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const ws = new WebSocket(`${WS_BASE}/ws/progress/${jobId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status) setStatus(data.status);
      if (data.progress !== undefined) setProgress(data.progress);
      if (data.new_logs && data.new_logs.length > 0) {
        setLogs(prev => [...prev, ...data.new_logs]);
      }
      if (data.status === 'completed') {
        fetchResults(jobId);
        ws.close();
      }
    };

    return () => ws.close();
  }, [jobId]);

  const fetchResults = async (id) => {
    try {
      const res = await axios.get(`${API_BASE}/api/evaluate/${id}`);
      if (res.data.results) {
        // Transform data for recharts
        const chartData = res.data.results.equity_curve.dates.map((date, index) => ({
          date,
          value: res.data.results.equity_curve.values[index]
        }));
        setResults({ ...res.data.results, chartData });
      }
    } catch (err) {
      console.error("Failed to fetch evaluation results", err);
    }
  };

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto w-full">
      {/* Configuration Card */}
      <div className="bg-surface rounded-xl p-6 border border-slate-700 shadow-xl">
        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Activity className="text-primary" />
          Train RL Agent
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Asset Symbol</label>
            <input 
              type="text" 
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-primary transition-colors"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">State Features</label>
            <div className="flex flex-wrap gap-3">
              {Object.keys(features).map(feat => (
                <label key={feat} className="flex items-center gap-2 cursor-pointer bg-slate-800 px-3 py-2 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors">
                  <input 
                    type="checkbox" 
                    checked={features[feat]} 
                    onChange={() => handleFeatureToggle(feat)}
                    disabled={feat === 'Close'} // Close is mandatory typically
                    className="accent-primary w-4 h-4"
                  />
                  <span className="text-sm">{feat}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        <button 
          onClick={startTraining}
          disabled={['starting', 'training', 'downloading'].includes(status)}
          className="bg-primary hover:bg-blue-600 disabled:bg-slate-700 disabled:text-slate-400 text-white font-semibold py-2 px-6 rounded-lg transition-colors flex items-center gap-2"
        >
          {['starting', 'training', 'downloading'].includes(status) ? (
            <span className="animate-pulse">Training... {progress}%</span>
          ) : (
            <><Play size={18} /> Start Training</>
          )}
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Terminal / Logs Card */}
        <div className="lg:col-span-1 bg-surface rounded-xl p-6 border border-slate-700 flex flex-col h-[500px]">
          <h3 className="font-semibold mb-3">Training Logs</h3>
          <div className="flex-1 bg-black rounded-lg p-4 overflow-y-auto font-mono text-sm text-green-400">
            {logs.length === 0 ? (
              <span className="text-slate-600">Waiting to start...</span>
            ) : (
              logs.map((log, i) => <div key={i}>{log}</div>)
            )}
          </div>
        </div>

        {/* Results / Charts Card */}
        <div className="lg:col-span-2 bg-surface rounded-xl p-6 border border-slate-700 h-[500px] flex flex-col">
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            Evaluation Results
            {status === 'completed' && <CheckCircle size={18} className="text-success" />}
            {status === 'failed' && <AlertCircle size={18} className="text-danger" />}
          </h3>
          
          <div className="flex-1 flex flex-col justify-center border border-slate-700 rounded-lg p-4 bg-slate-800/50">
            {status === 'training' || status === 'downloading' ? (
              <div className="text-center text-slate-400">
                <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                Agent is learning...
              </div>
            ) : results ? (
              <div className="h-full flex flex-col">
                <div className="flex justify-between mb-4 bg-slate-800 p-4 rounded-lg">
                  <div>
                    <div className="text-slate-400 text-xs uppercase font-bold">Initial Balance</div>
                    <div className="text-lg font-mono">${results.initial_balance.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-slate-400 text-xs uppercase font-bold">Final Balance</div>
                    <div className="text-lg font-mono">${results.final_balance.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-slate-400 text-xs uppercase font-bold">Return</div>
                    <div className={`text-lg font-mono ${results.return_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                      {results.return_pct >= 0 ? '+' : ''}{results.return_pct.toFixed(2)}%
                    </div>
                  </div>
                </div>
                
                <div className="flex-1 w-full min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={results.chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} tickFormatter={(val) => val.split('-').slice(1).join('/')} />
                      <YAxis stroke="#94a3b8" fontSize={12} domain={['auto', 'auto']} tickFormatter={(val) => `$${val}`} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569' }}
                        itemStyle={{ color: '#e2e8f0' }}
                      />
                      <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="text-center text-slate-500">Run training to see evaluation results here.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrainingDashboard;
