from fastapi import FastAPI, HTTPException, Query, Depends
from orchestrator.planner import OrchestratorPlanner
from typing import List
import uuid
from core.security import verify_token, oauth2_scheme
from api import auth
from workers.rag_worker import RAGEngine
from workers.grag_worker import GRAGEngine


app = FastAPI(title="Thesis Advising System API")
app.include_router(auth.router)
planner = OrchestratorPlanner()
rag_agent = RAGEngine()
grag_agent = GRAGEngine()

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
    
    # 2. Chạy trực tiếp các AI Agent (Không qua Redis/Celery nữa)
    for task in plan.tasks:
        if not task.parameters:
            task.parameters = {}
        task.parameters["student_id"] = mssv
        task.parameters["query"] = question
        
        try:
            if task.task_type == "RAG":
                print(f"🔍 [Agent] Đang truy xuất Qdrant và gọi Gemini (RAG)...")
                # Gọi thẳng hàm của RAGEngine
                res = rag_agent.search_and_answer(question) 
                all_results.append(res)
                
            elif task.task_type == "GRAG":
                print(f"🔍 [Agent] Đang truy xuất Neo4j (GRAG)...")
                # Gọi thẳng hàm của GRAG Engine
                res = grag_agent.query_graph(task.parameters)
                all_results.append(res)
                
        except Exception as e:
            print(f"❌ [Lỗi] Agent {task.task_type} gặp sự cố: {str(e)}")
            all_results.append(f"Xin lỗi, có lỗi xảy ra khi xử lý {task.task_type}.")

    print(f"✅ [FastAPI] Trả câu trả lời về cho Streamlit!")
    
    # 3. Trả kết quả ngay lập tức
    return {
        "session_id": session_id,
        "status": "SUCCESS", 
        "data": all_results
    }


