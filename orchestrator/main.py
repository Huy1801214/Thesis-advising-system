from fastapi import FastAPI, HTTPException, Query, Depends
from orchestrator.planner import OrchestratorPlanner
from typing import List
import uuid
from core.security import verify_token, oauth2_scheme
from api import auth
from workers.rag_worker import RAGEngine
from workers.grag_worker import GRAGEngine
from api.upload_infor import router as grag_router, STUDENT_TRANSCRIPT_STORE

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from workers import grag_tools 
from orchestrator.llm_extractor import EntityExtractor

app = FastAPI(title="Thesis Advising System API")
app.include_router(auth.router)
app.include_router(grag_router, prefix="/api/grag", tags=["GRAG"])
planner = OrchestratorPlanner()
rag_agent = RAGEngine()
grag_agent = GRAGEngine()
extractor = EntityExtractor()

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@app.post("/chat")
async def handle_chat(question: str, mssv: str = Depends(get_current_mssv)):
    session_id = f"SES_{mssv}_{uuid.uuid4().hex[:8]}"
    print(f"🚀 [FastAPI] Bắt đầu xử lý câu hỏi từ {mssv}: {question}")
    
    # 1. Lên kế hoạch (Task Decomposition)
    plan = planner.create_plan(question, session_id)
    all_results = []
    student_data = STUDENT_TRANSCRIPT_STORE.get(mssv, {})
    
    # 2. Chạy trực tiếp các AI Agent 
    for task in plan.tasks:
        if not task.parameters:
            task.parameters = {}
            
        task.parameters["student_id"] = mssv
        task.parameters["query"] = question
        task.parameters["passed_courses"] = student_data.get("passed_courses", [])
        task.parameters["cumulative_gpa"] = student_data.get("cumulative_gpa")
        
        print(mssv, question, student_data.get("passed_courses", []), student_data.get("cumulative_gpa"))
        current_intent = task.parameters.get("intent", "")
        
        if task.task_type == "GRAG" and current_intent in ["registration_check", "course_info"]:
            planned_codes = extractor.extract_course_codes(question)
            print(f"🎯 [LLM Extractor] Bóc tách thành công các mã môn: {planned_codes}")
            
            if current_intent == "registration_check":
                task.parameters["planned_courses"] = planned_codes
            elif current_intent == "course_info" and planned_codes:
                task.parameters["course_code"] = planned_codes[0]
        
        # try:
        if task.task_type == "RAG":
            print(f"🔍 [Agent] Đang truy xuất Qdrant và gọi Gemini (RAG)...")
            res = rag_agent.search_and_answer(question) 
            all_results.append(res)
                
        elif task.task_type == "GRAG":
            print(f"🔍 [Agent] Đang truy xuất Neo4j (GRAG)...")
            
            # Chặn nếu hỏi điều kiện mà chưa up bảng điểm
            if not student_data and current_intent in ["registration_check", "graduation_check"]:
                all_results.append("💡 Để hệ thống tư vấn chính xác về điều kiện học vụ, bạn vui lòng đính kèm file Bảng điểm (Excel/CSV) vào khung chat trước nhé!")
                continue
            
            # Gọi Neo4j
            res = grag_agent.query_graph(task.parameters)
            all_results.append(res)
                
        # except Exception as e:
        #     print(f"❌ [Lỗi] Agent {task.task_type} gặp sự cố: {str(e)}")
        #     all_results.append(f"Xin lỗi, có lỗi xảy ra khi xử lý {task.task_type}.")

    print(f"✅ [FastAPI] Trả câu trả lời về cho Streamlit!")
    
    return {
        "session_id": session_id,
        "status": "SUCCESS", 
        "data": all_results
    }