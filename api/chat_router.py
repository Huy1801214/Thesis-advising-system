from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
import uuid
import hashlib
import json
from sqlalchemy.orm import Session
from core.database import get_db, SessionLocal
import asyncio
from core.security import verify_token, oauth2_scheme, limiter
from api.upload_infor import STUDENT_TRANSCRIPT_STORE
from model.chat_message import ChatMessage, TaskStatus
from model.student_profile import StudentProfile

from orchestrator.planner import OrchestratorPlanner
from orchestrator.executor import TaskExecutor
from orchestrator.synthesizer import FinalSynthesizer
from orchestrator.critic import CriticAgent
from sqlalchemy import desc, func

router = APIRouter()
planner = OrchestratorPlanner()
executor = TaskExecutor()
synthesizer = FinalSynthesizer()
critic = CriticAgent()
ACTIVE_SESSIONS = set()

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@router.post("/chat")
async def handle_chat(question: str, mode: str = "agent", mssv: str = Depends(get_current_mssv), db: Session = Depends(get_db)):
    session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    
    if mode == "rag":
        # Pure RAG Baseline pipeline
        final_answer = await executor.rag_agent.search_and_answer(question)
        trace_data = [{
            "task_id": "RAG_BASE",
            "task_type": "RAG",
            "query_intent": question,
            "raw_data": f"--- Dữ liệu từ Sổ tay ({question}) ---\n{final_answer}",
            "status": "SUCCESS"
        }]
        critic_score = 1.0
    else:
        # 1. Gọi Orchestrator lấy Task Plan
        plan = planner.create_plan(question, session_id)
        
        # Lấy dữ liệu điểm của sinh viên từ database
        profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
        student_data = {}
        if profile:
            student_data = {
                "cumulative_gpa": profile.cumulative_gpa,
                "total_earned_credits": profile.total_earned_credits,
                "passed_courses": profile.passed_courses,
                "current_courses": profile.current_courses,
                "major": profile.major,
                "target_career": profile.target_career,
                "interests": profile.interests
            }
        
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
        critic_score = report.score

    return {
        "answer": final_answer,
        "critic_score": critic_score,
        "debug_trace": trace_data
    }

@router.post("/chat/stream")
@limiter.limit("3/minute") 
async def handle_chat_stream(request: Request, question: str, session_id: str = None, mode: str = "agent", mssv: str = Depends(get_current_mssv), db: Session = Depends(get_db)):
    print(f"[RateLimit] Đang kiểm soát lưu lượng cho sinh viên: {mssv}")
    query_hash = f"{mssv}_{hashlib.md5(question.strip().lower().encode()).hexdigest()}"
    
    # Chặn nếu sinh viên bấm F5
    if query_hash in ACTIVE_SESSIONS:
        raise HTTPException(status_code=409, 
            detail="Hệ thống đang xử lý câu hỏi này, vui lòng đợi trong giây lát để tránh quá tải."
        )
    
    ACTIVE_SESSIONS.add(query_hash)

    # 1. Quản lý Session ID
    if not session_id:
        current_session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    else:
        current_session_id = session_id

    # Lấy lịch sử hội thoại 
    history_msgs = db.query(ChatMessage).filter(
        ChatMessage.session_id == current_session_id,
        ChatMessage.mssv == mssv
    ).order_by(ChatMessage.created_at.asc()).all()[-6:]
    chat_history_str = "\n".join([f"{m.role.upper()}: {m.content}" for m in history_msgs])

    # 2. Lưu câu hỏi của User vào Database
    user_msg = ChatMessage(
        mssv=mssv, 
        session_id=current_session_id, 
        role="user", 
        content=question,
        status=TaskStatus.COMPLETED
    )
    db.add(user_msg)
    db.commit()

    async def event_generator():
        
        try:
            if mode == "rag":
                # Pure RAG Baseline pipeline stream
                if await request.is_disconnected(): return
                yield "data: " + json.dumps({"event": "planner_start", "message": "Đang kết nối cơ sở dữ liệu (Chế độ Thuần RAG)..."}) + "\n\n"
                await asyncio.sleep(0.1)
                
                tasks_list = [{"task_id": "RAG_BASE", "task_type": "RAG", "query_intent": question}]
                yield "data: " + json.dumps({
                    "event": "planner_done",
                    "message": "Truy cập công cụ tìm kiếm RAG.",
                    "tasks": tasks_list
                }) + "\n\n"
                await asyncio.sleep(0.1)
                
                if await request.is_disconnected(): return
                yield "data: " + json.dumps({
                    "event": "executor_running",
                    "message": "Đang tìm kiếm tài liệu học vụ bằng Vector DB...",
                    "running_tasks": ["RAG_BASE"]
                }) + "\n\n"
                await asyncio.sleep(0.1)
                
                # Gọi thẳng RAG Agent
                final_answer = await executor.rag_agent.search_and_answer(question)
                
                task_res = {
                    "task_id": "RAG_BASE",
                    "task_type": "RAG",
                    "query_intent": question,
                    "raw_data": f"--- Dữ liệu từ Sổ tay ({question}) ---\n{final_answer}",
                    "status": "SUCCESS"
                }
                
                yield "data: " + json.dumps({
                    "event": "task_completed",
                    "message": "Đã tìm kiếm xong tài liệu.",
                    "task": task_res
                }) + "\n\n"
                await asyncio.sleep(0.1)
                
                # Synthesizer dummy events to match step styles
                yield "data: " + json.dumps({"event": "synthesizer_start", "message": "Đang hoàn thiện câu trả lời..."}) + "\n\n"
                await asyncio.sleep(0.1)
                yield "data: " + json.dumps({"event": "synthesizer_done", "message": "Hoàn tất."}) + "\n\n"
                await asyncio.sleep(0.1)
                
                if await request.is_disconnected(): return
                with SessionLocal() as stream_db:
                    ai_msg = ChatMessage(
                        mssv=mssv,
                        session_id=current_session_id,
                        role="assistant",
                        content=final_answer,
                        status=TaskStatus.COMPLETED
                    )
                    stream_db.add(ai_msg)
                    stream_db.commit()
                    
                yield "data: " + json.dumps({
                    "event": "final_result",
                    "answer": final_answer,
                    "critic_score": 1.0,
                    "debug_trace": [task_res],
                    "session_id": current_session_id
                }) + "\n\n"
                return

            #  1. Gọi Orchestrator lấy Task Plan
            if await request.is_disconnected(): return 
            
            yield "data: " + json.dumps({"event": "planner_start", "message": "Đang phân tích câu hỏi và lập kế hoạch..."}) + "\n\n"
            await asyncio.sleep(0.1)
            
            standalone_question = planner.condense_question(question, chat_history_str)
            
            plan = planner.create_plan(standalone_question, current_session_id)
            tasks_list = [{"task_id": t.task_id, "task_type": t.task_type, "query_intent": t.query_intent} for t in plan.tasks]
            
            yield "data: " + json.dumps({
                "event": "planner_done", 
                "message": f"Đã lập kế hoạch gồm {len(tasks_list)} tác vụ.",
                "tasks": tasks_list
            }) + "\n\n"
            await asyncio.sleep(0.1)

            # 2. Định tuyến Task Plan cho Executor thực thi
            if await request.is_disconnected(): return
            
            profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
            student_data = {}
            if profile:
                student_data = {
                    "cumulative_gpa": profile.cumulative_gpa,
                    "total_earned_credits": profile.total_earned_credits,
                    "passed_courses": profile.passed_courses,
                    "current_courses": profile.current_courses,
                    "major": profile.major,
                    "target_career": profile.target_career,
                    "interests": profile.interests
                }
            MAX_TASKS = 5
            safe_tasks = plan.tasks[:MAX_TASKS]
            
            yield "data: " + json.dumps({
                "event": "executor_running", 
                "message": "Đang chạy song song các Agent thực thi câu hỏi của bạn...",
                "running_tasks": [t.task_id for t in safe_tasks]
            }) + "\n\n"
            await asyncio.sleep(0.1)
            
            task_coroutines = [
                executor._execute_single_task(task, standalone_question, mssv, student_data) 
                for task in safe_tasks
            ]
            
            trace_data = []
            if task_coroutines:
                for future in asyncio.as_completed(task_coroutines):
                    # Nếu sinh viên tắt web -> hủy luồng
                    if await request.is_disconnected(): 
                        return 
                        
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

            # 3. Tổng hợp dữ liệu (Synthesizer)
            if await request.is_disconnected(): return
            
            yield "data: " + json.dumps({"event": "synthesizer_start", "message": "Đang tổng hợp thông tin câu trả lời..."}) + "\n\n"
            await asyncio.sleep(0.1)
            
            final_answer = await synthesizer.synthesize(standalone_question, trace_data)
            
            yield "data: " + json.dumps({"event": "synthesizer_done", "message": "Đã tổng hợp xong câu trả lời sơ bộ."}) + "\n\n"
            await asyncio.sleep(0.1)

            # 4. Kiểm duyệt và phản biện (Critic)
            if await request.is_disconnected(): return
            
            yield "data: " + json.dumps({"event": "critic_start", "message": "Đang gửi qua Critic Agent để kiểm duyệt..."}) + "\n\n"
            await asyncio.sleep(0.1)
            
            report = await critic.review(
                original_question=standalone_question, 
                final_answer=final_answer, 
                trace=trace_data
            )
            
            if not report.passed:
                if report.revised_answer:
                    final_answer = report.revised_answer
                    
            yield "data: " + json.dumps({
                "event": "critic_done", 
                "message": "Kiểm duyệt hoàn tất.",
                "critic_score": report.score
            }) + "\n\n"
            await asyncio.sleep(0.1)

            # 5. Gửi kết quả cuối cùng
            if await request.is_disconnected(): return
            with SessionLocal() as stream_db:
                ai_msg = ChatMessage(
                    mssv=mssv,
                    session_id=current_session_id,
                    role="assistant",
                    content=final_answer,
                    status=TaskStatus.COMPLETED
                )
                stream_db.add(ai_msg)
                stream_db.commit()
            yield "data: " + json.dumps({
                "event": "final_result",
                "answer": final_answer,
                "critic_score": report.score,
                "debug_trace": trace_data,
                "session_id": current_session_id
            }) + "\n\n"

        except Exception as e:
            print(f"❌ Lỗi Stream ngầm: {e}")
            yield "data: " + json.dumps({"event": "error", "message": f"Lỗi hệ thống: {str(e)}"}) + "\n\n"
        finally:
            if query_hash in ACTIVE_SESSIONS:
                ACTIVE_SESSIONS.remove(query_hash)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/history/sessions")
async def get_user_sessions(mssv: str = Depends(get_current_mssv), db: Session = Depends(get_db)):
    sessions = db.query(
        ChatMessage.session_id,
    ).filter(ChatMessage.mssv == mssv)\
     .group_by(ChatMessage.session_id)\
     .order_by(desc(func.max(ChatMessage.created_at))).all()

    result = []
    for (sess_id,) in sessions:
        first_msg = db.query(ChatMessage).filter(
            ChatMessage.session_id == sess_id, 
            ChatMessage.role == "user"
        ).order_by(ChatMessage.created_at.asc()).first()
        
        if first_msg:
            title = first_msg.content[:40] + "..." if len(first_msg.content) > 40 else first_msg.content
            result.append({
                "session_id": sess_id,
                "title": title
            })
            
    return {"sessions": result}

@router.get("/history/messages/{session_id}")
async def get_session_messages(session_id: str, mssv: str = Depends(get_current_mssv), db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id,
        ChatMessage.mssv == mssv
    ).order_by(ChatMessage.created_at.asc()).all()
    
    if not messages:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử chat")
        
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at
            } for msg in messages
        ]
    }