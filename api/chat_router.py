from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import uuid
import json
import asyncio
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

@router.post("/chat/stream")
async def handle_chat_stream(question: str, mssv: str = Depends(get_current_mssv)):
    async def event_generator():
        session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
        
        # 1. Gọi Orchestrator lấy Task Plan
        yield "data: " + json.dumps({"event": "planner_start", "message": "Đang phân tích câu hỏi và lập kế hoạch..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        try:
            plan = planner.create_plan(question, session_id)
            tasks_list = [{"task_id": t.task_id, "task_type": t.task_type, "query_intent": t.query_intent} for t in plan.tasks]
            yield "data: " + json.dumps({
                "event": "planner_done", 
                "message": f"Đã lập kế hoạch gồm {len(tasks_list)} tác vụ.",
                "tasks": tasks_list
            }) + "\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Lỗi lập kế hoạch: {str(e)}"}) + "\n\n"
            return

        # 2. Định tuyến Task Plan cho Executor thực thi
        student_data = STUDENT_TRANSCRIPT_STORE.get(mssv, {})
        MAX_TASKS = 5
        safe_tasks = plan.tasks[:MAX_TASKS]
        
        yield "data: " + json.dumps({
            "event": "executor_running", 
            "message": "Đang chạy song song các RAG & GRAG Agent...",
            "running_tasks": [t.task_id for t in safe_tasks]
        }) + "\n\n"
        await asyncio.sleep(0.1)
        
        try:
            task_coroutines = [
                executor._execute_single_task(task, question, mssv, student_data) 
                for task in safe_tasks
            ]
            
            trace_data = []
            if task_coroutines:
                for future in asyncio.as_completed(task_coroutines):
                    task_res = await future
                    trace_data.append(task_res)
                    yield "data: " + json.dumps({
                        "event": "task_completed", 
                        "message": f"Agent {task_res['task_id']} ({task_res['task_type']}) hoàn thành.",
                        "task": task_res
                    }) + "\n\n"
                    await asyncio.sleep(0.1)
            else:
                yield "data: " + json.dumps({
                    "event": "executor_done", 
                    "message": "Không có Agent nào cần thực thi."
                }) + "\n\n"
                await asyncio.sleep(0.1)
                
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Lỗi thực thi Agent: {str(e)}"}) + "\n\n"
            return
            
        # 3. Tổng hợp dữ liệu (Synthesizer)
        yield "data: " + json.dumps({"event": "synthesizer_start", "message": "Đang tổng hợp thông tin câu trả lời..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        try:
            final_answer = await synthesizer.synthesize(question, trace_data)
            yield "data: " + json.dumps({"event": "synthesizer_done", "message": "Đã tổng hợp xong câu trả lời sơ bộ."}) + "\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Lỗi tổng hợp: {str(e)}"}) + "\n\n"
            return
            
        # 4. Kiểm duyệt và phản biện (Critic)
        yield "data: " + json.dumps({"event": "critic_start", "message": "Đang gửi qua Critic Agent để kiểm duyệt..."}) + "\n\n"
        await asyncio.sleep(0.1)
        
        try:
            report = await critic.review(
                original_question=question, 
                final_answer=final_answer, 
                trace=trace_data
            )
            
            if not report.passed:
                print(f"⚠️ Critic đã bắt lỗi: {report.issues}")
                if report.revised_answer:
                    print("🔄 Đang ghi đè bằng câu trả lời đã được Critic sửa đổi.")
                    final_answer = report.revised_answer
                    
            yield "data: " + json.dumps({
                "event": "critic_done", 
                "message": "Kiểm duyệt hoàn tất.",
                "critic_score": report.score
            }) + "\n\n"
            await asyncio.sleep(0.1)
        except Exception as e:
            yield "data: " + json.dumps({"event": "error", "message": f"Lỗi kiểm duyệt: {str(e)}"}) + "\n\n"
            return
            
        # Gửi kết quả cuối cùng
        yield "data: " + json.dumps({
            "event": "final_result",
            "answer": final_answer,
            "critic_score": report.score,
            "debug_trace": trace_data
        }) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")