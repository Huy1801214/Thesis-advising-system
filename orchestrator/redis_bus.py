import redis
import json

class BlackboardBus:
    def __init__(self):
        # Redis đóng vai trò Blackboard và Message Broker [cite: 955, 1035]
        self.client = redis.Redis(host='localhost', port=6379, db=0)

    def get_task_status(self, task_id: str):
        # Thao tác P-Wait: Polling trạng thái từ kho lưu trữ tập trung [cite: 657, 808]
        status = self.client.get(f"status:{task_id}")
        return status.decode() if status else "PENDING"

    def get_task_result(self, task_id: str):
        result = self.client.get(f"result:{task_id}")
        return json.loads(result) if result else None