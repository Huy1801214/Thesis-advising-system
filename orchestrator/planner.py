from langchain_google_genai import ChatGoogleGenerativeAI
from orchestrator.schemas import TaskPlan
from dotenv import load_dotenv

load_dotenv()

class OrchestratorPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.planner_chain = self.llm.with_structured_output(TaskPlan)

    def create_plan(self, question: str, session_id: str) -> TaskPlan:
        system_prompt = f"""
        Bạn là Orchestrator Agent điều phối hệ thống tư vấn học vụ.
        Nhiệm vụ: Phân rã câu hỏi thành các tác vụ RAG hoặc GRAG.

        QUY TẮC TRÍCH XUẤT THAM SỐ (Parameters):
        1. Mã môn học: Luôn là chuỗi 6 chữ số (VD: 200105, 214294). 
           Trích xuất vào danh sách 'planned_courses'.
        2. Mã sinh viên: Thường bắt đầu bằng 'SV' hoặc chuỗi số MSSV (VD: SV001, 22130099).
           Trích xuất vào 'student_id'.
        3. Intent: 
           - Nếu hỏi về điều kiện, đăng ký môn -> intent: 'registration_check'.
           - Nếu hỏi thông tin môn học cụ thể -> intent: 'course_info'.

        LỰA CHỌN CÔNG CỤ:
        - 'RAG': Tra cứu quy chế, thông báo, văn bản pháp quy.
        - 'GRAG': Tra cứu điều kiện tiên quyết, học trước, song hành hoặc lộ trình học tập trên đồ thị.

        Câu hỏi từ sinh viên: "{question}"
        Session ID: {session_id}
        """
        return self.planner_chain.invoke(system_prompt)