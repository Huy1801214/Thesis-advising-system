import redis
import json
from typing import Any

class BlackboardBus:
    def __init__(self):
        # Redis đóng vai trò Blackboard và Message Broker [cite: 955, 1035]
        self.client = redis.Redis(host='localhost', port=6379, db=0)

    def get_task_status(self, task_id: str):
        # Thao tác P-Wait: Polling trạng thái từ kho lưu trữ tập trung [cite: 657, 808]
        status = self.client.get(f"status:{task_id}")
        return status.decode() if status else "PENDING"

    def get_task_result(self, session_id: str, task_id: str):
            # Key phải khớp với cách Worker ghi vào: result:{session_id}:{task_id}
            key = f"result:{session_id}:{task_id}"
            result = self.client.get(key)
            
            if result:
                return json.loads(result)
            return None
    
    def set_result(self, session_id: str, task_id: str, data: Any):
        key = f"result:{session_id}:{task_id}"
        self.client.set(key, json.dumps(data), ex=3600)