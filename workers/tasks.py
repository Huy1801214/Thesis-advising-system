from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

# Khởi tạo Celery đóng vai trò Consumer
app = Celery('worker_agents', 
             broker='redis://localhost:6379/0', # [cite: 632]
             backend='redis://localhost:6379/0') # [cite: 634]

@app.task(name="run_rag_worker")
def run_rag_worker(task_id: str, query: str, parameters: dict):
    # Đây là nơi code RAG cũ của Huy sẽ chạy [cite: 788-792]
    return f"Kết quả từ RAG cho task {task_id}"

@app.task(name="run_grag_worker")
def run_grag_worker(task_id: str, query: str, parameters: dict):
    # Đây là nơi code GRAG (Neo4j) sẽ chạy [cite: 799-803]
    return f"Kết quả từ GRAG cho task {task_id}"