from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Task(BaseModel):
    task_id: str = Field(description="Mã định danh duy nhất cho tác vụ, ví dụ: TASK_01")
    task_type: str = Field(description="Loại worker: 'RAG' hoặc 'GRAG'")
    query_intent: str = Field(description="Câu hỏi đã được viết lại tối ưu cho việc tra cứu")
    parameters: Dict[str, Any] = Field(description="Các thực thể bóc tách như tên môn học, mã sinh viên")

class TaskPlan(BaseModel):
    session_id: str
    tasks: List[Task]