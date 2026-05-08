from celery import Celery
import os
from dotenv import load_dotenv
from .rag_worker import RAGEngine
from orchestrator.redis_bus import BlackboardBus

load_dotenv()

# Khởi tạo Celery đóng vai trò Consumer
app = Celery('worker_agents', 
             broker='redis://localhost:6379/0', # [cite: 632]
             backend='redis://localhost:6379/0') # [cite: 634]

rag_engine = RAGEngine() 
redis_bus = BlackboardBus()

@app.task(name="process_rag_task")
def process_rag_task(session_id: str, task_id: str, query: str, parameters: dict): # Thêm query vào đây
    # Giờ Huy không cần extract từ parameters nữa, dùng trực tiếp query luôn
    context_result = rag_engine.search_and_answer(query)
    
    redis_bus.set_result(session_id, task_id, context_result)
    return f"RAG Task {task_id} completed"

@app.task(name="process_grag_task")
def process_grag_task(session_id: str, task_id: str, parameters: dict):
    """
    Worker xử lý tra cứu đồ thị (Neo4j)
    """
    # Logic tương tự cho GRAG: Truy vấn Cypher -> Trả về context -> Ghi vào Redis
    # result = grag_engine.query_graph(parameters)
    # redis_bus.set_result(session_id, task_id, result)
    return f"GRAG Task {task_id} completed"