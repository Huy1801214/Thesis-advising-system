from celery import Celery
import os
from dotenv import load_dotenv
from .rag_worker import RAGEngine
from .grag_worker import grag_engine
from orchestrator.redis_bus import BlackboardBus

load_dotenv()

app = Celery('worker_agents', 
             broker='redis://localhost:6379/0', # [cite: 632]
             backend='redis://localhost:6379/0') # [cite: 634]

rag_engine = RAGEngine() 
redis_bus = BlackboardBus()

@app.task(name="process_rag_task")
def process_rag_task(session_id: str, task_id: str, query: str, parameters: dict): 
    context_result = rag_engine.search_and_answer(query)
    
    redis_bus.set_result(session_id, task_id, context_result)
    return f"RAG Task {task_id} completed"

@app.task(name="process_grag_task")
def process_grag_task(session_id: str, task_id: str, parameters: dict):
    try:
        # Gọi "bộ não" GRAG để xử lý
        result = grag_engine.query_graph(parameters)
        
        # Lưu kết quả vào Redis để API /sync lấy về
        redis_bus.set_result(session_id, task_id, result)
        
        return {"status": "SUCCESS", "task_id": task_id}
    except Exception as e:
        error_msg = f"Lỗi GRAG Worker: {str(e)}"
        redis_bus.set_task_result(session_id, task_id, error_msg)
        return {"status": "FAILED", "error": error_msg}