from fastapi import FastAPI, BackgroundTasks
from orchestrator.planner import OrchestratorPlanner
from orchestrator.redis_bus import BlackboardBus
from workers.tasks import run_rag_worker, run_grag_worker # Giả định file tasks
from typing import List
import uuid

app = FastAPI()
planner = OrchestratorPlanner()
bus = BlackboardBus()

@app.post("/ask")
async def handle_request(question: str):
    session_id = str(uuid.uuid4())
    
    # 1. Thực hiện Task Decomposition [cite: 648, 949]
    plan = planner.create_plan(question, session_id)
    
    # 2. Đẩy tác vụ vào Message Broker (Producer role) [cite: 650, 902]
    task_ids = []
    for task in plan.tasks:
        task_ids.append(task.task_id)
        if task.task_type == "RAG":
            run_rag_worker.delay(task.task_id, task.query_intent, task.parameters)
        elif task.task_type == "GRAG":
            run_grag_worker.delay(task.task_id, task.query_intent, task.parameters)
            
    return {"session_id": session_id, "task_ids": task_ids, "status": "PROCESSING"}

@app.get("/sync/{session_id}")
async def sync_results(session_id: str, task_ids: List[str]):
    # 3. Kiểm tra rào cản đồng bộ (Synchronization Barrier) [cite: 910, 931]
    all_results = []
    for t_id in task_ids:
        status = bus.get_task_status(t_id)
        if status != "SUCCESS":
            return {"status": "WAITING", "completed": len(all_results)}
        all_results.append(bus.get_task_result(t_id))
    
    # 4. Khi tất cả đã xong, Generator sẽ tổng hợp câu trả lời [cite: 659, 911]
    return {"status": "SUCCESS", "data": all_results}