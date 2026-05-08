from fastapi import FastAPI, HTTPException, Query, Depends
from orchestrator.planner import OrchestratorPlanner
from orchestrator.redis_bus import BlackboardBus
from workers.tasks import process_rag_task, process_grag_task 
from typing import List
import uuid
from core.security import verify_token
from core.security import verify_token, oauth2_scheme
from api import auth


app = FastAPI(title="Thesis Advising System API")
app.include_router(auth.router)
planner = OrchestratorPlanner()
bus = BlackboardBus()

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@app.post("/ask")
async def handle_request(question: str, mssv: str = Depends(get_current_mssv)):
    # 1. Tạo session_id chuyên nghiệp hơn có gắn MSSV
    session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    
    # 2. Task Decomposition 
    plan = planner.create_plan(question, session_id)
    
    # 2. Đẩy tác vụ vào Message Broker (V-Signal)
    task_ids = []
    for task in plan.tasks:
        task_ids.append(task.task_id)
        if task.task_type == "RAG":
            process_rag_task.delay(session_id, task.task_id, task.query_intent, task.parameters)
        elif task.task_type == "GRAG":
            process_grag_task.delay(session_id, task.task_id, task.parameters)
    # 3. Phản hồi         
    return {
        "session_id": session_id, 
        "task_ids": task_ids, 
        "status": "PROCESSING"
    }

@app.get("/sync/{session_id}")
async def sync_results(session_id: str, task_ids: List[str] = Query(...), mssv: str = Depends(get_current_mssv)):
    
    # Kiểm tra xem session_id này có thuộc về MSSV này không
    if not session_id.startswith(f"SES_{mssv}"):
        raise HTTPException(status_code=403, detail="Bạn không có quyền xem kết quả này!")
    all_results = []
    for t_id in task_ids:
        # Lấy kết quả từ Redis State Store thông qua RedisBus/BlackboardBus
        result = bus.get_task_result(session_id, t_id)
        
        if result is None:
            return {
                "status": "WAITING", 
                "completed": len(all_results), 
                "total": len(task_ids)
            }
        all_results.append(result)
    
    # 3. Khi đủ tri thức, hệ thống sẵn sàng cho Generator tổng hợp
    return {
        "status": "SUCCESS", 
        "data": all_results
    }

