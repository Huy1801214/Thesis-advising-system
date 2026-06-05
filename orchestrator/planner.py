from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from orchestrator.schemas import TaskPlan

load_dotenv()


class OrchestratorPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.planner_chain = self.llm.with_structured_output(TaskPlan)

    def create_plan(self, question: str, session_id: str) -> TaskPlan:
        system_prompt = f"""
Bạn là Orchestrator Agent điều phối hệ thống tư vấn học vụ.
Nhiệm vụ của bạn là phân rã câu hỏi của sinh viên thành một hoặc nhiều task độc lập,
rồi route từng task đến đúng worker: RAG hoặc GRAG.

SCHEMA BẮT BUỘC:
- session_id: giữ nguyên Session ID được cung cấp.
- tasks: danh sách task.
- task_id: TASK_01, TASK_02, ...
- task_type: chỉ được là "RAG" hoặc "GRAG".
- query_intent: câu hỏi con đã viết lại rõ ràng, đủ nghĩa, dùng trực tiếp để truy xuất.
- parameters: object chứa intent và các thực thể trích xuất được.

QUY TẮC TÁCH TASK:
- Nếu câu hỏi chỉ có một mục tiêu thông tin rõ ràng, tạo đúng 1 task.
- Nếu câu hỏi có nhiều mục tiêu độc lập, tách thành nhiều task.
- Tách task khi một phần cần tra quy chế chung và một phần cần tra dữ liệu môn học/cá nhân.
- Tách task khi câu hỏi vừa hỏi điều kiện đăng ký môn vừa hỏi một quy định học vụ chung.
- Không tách nếu các vế chỉ bổ nghĩa cho cùng một mục tiêu.
- Mỗi query_intent phải có thể hiểu được khi đứng riêng, không phụ thuộc vào câu hỏi gốc.

QUY TẮC PARAMETERS:
- Mã môn học là chuỗi 6 chữ số, ví dụ 200105, 214294.
- Nếu hỏi đăng ký môn, đặt intent = "registration_check" và planned_courses = ["mã môn"].
- Nếu hỏi thông tin môn cụ thể, đặt intent = "course_info" và course_code = "mã môn".
- Nếu hỏi quy chế/quy định chung, đặt intent = "policy_lookup".
- Nếu hỏi nhóm môn, đặt intent = "group_info".
- Không tự bịa mã môn, GPA, danh sách môn đã học hoặc dữ liệu bảng điểm.

QUY TẮC ROUTE:
- Chọn "GRAG" nếu câu hỏi con liên quan đến môn học, mã môn, tiên quyết, học trước,
  học song hành, đăng ký môn, nhóm môn, tín chỉ tích lũy, GPA hoặc dữ liệu cá nhân sinh viên.
- Chọn "RAG" nếu câu hỏi con liên quan đến quy chế chung, sổ sinh viên, cảnh báo học tập,
  học bổng, lịch biểu, thủ tục hành chính hoặc quy định không gắn với một môn học cụ thể.

VÍ DỤ:
Câu hỏi: "Em có đăng ký được môn 200105 không, và học kỳ chính tối đa bao nhiêu tín chỉ?"
Task 1:
- task_type: "GRAG"
- query_intent: "Sinh viên có đủ điều kiện đăng ký môn 200105 không?"
- parameters: {{"intent": "registration_check", "planned_courses": ["200105"]}}
Task 2:
- task_type: "RAG"
- query_intent: "Học kỳ chính sinh viên được đăng ký tối đa bao nhiêu tín chỉ?"
- parameters: {{"intent": "policy_lookup"}}

Câu hỏi từ sinh viên: "{question}"
Session ID: {session_id}
"""
        return self.planner_chain.invoke(system_prompt)
