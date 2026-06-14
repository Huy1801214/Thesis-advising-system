from fastapi import APIRouter, Depends
import uuid
from core.security import verify_token, oauth2_scheme
from api.upload_infor import STUDENT_TRANSCRIPT_STORE

from orchestrator.planner import OrchestratorPlanner
from orchestrator.executor import TaskExecutor
from orchestrator.synthesizer import FinalSynthesizer
from orchestrator.critic import CriticAgent

router = APIRouter()
planner = OrchestratorPlanner()
executor = TaskExecutor()
synthesizer = FinalSynthesizer()
critic = CriticAgent()

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
    trace_data = execution_result["tasks_execution_info"]
    
    final_answer = await synthesizer.synthesize(question, trace_data)
    report = await critic.review(
        original_question=question, 
        final_answer=final_answer, 
        trace=trace_data
    )

    # 3. Trả về cho Frontend
    if not report.passed:
        print(f"⚠️ Critic đã bắt lỗi: {report.issues}")
        if report.revised_answer:
            print("🔄 Đang ghi đè bằng câu trả lời đã được Critic sửa đổi.")
            final_answer = report.revised_answer

    return {
        "answer": final_answer,
        "critic_score": report.score,
        "debug_trace": trace_data
    }