from fastapi import APIRouter, Depends
import uuid
from core.security import verify_token, oauth2_scheme
from api.upload_infor import STUDENT_TRANSCRIPT_STORE

from orchestrator.planner import OrchestratorPlanner
from orchestrator.executor import TaskExecutor

router = APIRouter()
planner = OrchestratorPlanner()
executor = TaskExecutor()

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@router.post("/chat")
async def handle_chat(question: str, mssv: str = Depends(get_current_mssv)):
    session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    
    # 1. Gọi Orchestrator lấy Task Plan
    plan = planner.create_plan(question, session_id)
    
    # Lấy dữ liệu điểm của sinh viên
    student_data = STUDENT_TRANSCRIPT_STORE.get(mssv, {})
    
    # 2. Định tuyết Task Plan cho Executor thực thi
    execution_result = await executor.execute_plan(plan, question, mssv, student_data)
    
    # 3. Trả về cho Frontend
    return {
        "session_id": session_id,
        "status": "SUCCESS", 
        "answer": execution_result["answer"],
        "debug_data": execution_result["debug_data"]
    }