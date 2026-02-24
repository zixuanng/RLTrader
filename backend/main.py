import uuid
import asyncio
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.train import train_agent, evaluate_agent

app = FastAPI(title="RLTrader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for MVP
jobs = {}

class TrainRequest(BaseModel):
    symbol: str
    features: List[str]

@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "logs": [],
        "symbol": req.symbol,
        "features": req.features
    }
    background_tasks.add_task(train_agent, job_id, req.symbol, req.features, jobs)
    return {"job_id": job_id, "status": "started"}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    return jobs

@app.get("/api/evaluate/{job_id}")
async def evaluate_job(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job_info = jobs[job_id]
    if job_info["status"] != "completed":
        return {"error": "Job is not completed yet"}
        
    model_path = job_info.get("model_path")
    if not model_path:
        return {"error": "Model path not found"}
        
    try:
        results = await evaluate_agent(job_id, job_info["symbol"], job_info["features"], model_path)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in jobs:
        await websocket.send_json({"error": "Job not found"})
        await websocket.close()
        return
        
    try:
        last_log_idx = 0
        while True:
            job_info = jobs[job_id]
            
            # Send new logs
            new_logs = job_info["logs"][last_log_idx:]
            if new_logs or job_info["status"] in ["completed", "failed"]:
                await websocket.send_json({
                    "status": job_info["status"],
                    "progress": job_info["progress"],
                    "new_logs": new_logs,
                    "error": job_info.get("error")
                })
                last_log_idx = len(job_info["logs"])
                
            if job_info["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        print(f"Client disconnected for job {job_id}")
