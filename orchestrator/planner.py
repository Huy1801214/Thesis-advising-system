from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from orchestrator.schemas import TaskPlan

load_dotenv()


class OrchestratorPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
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

        Quy tắc phân loại nhiệm vụ (Task Type):
            1. Chọn "GRAG" (Graph RAG) NẾU câu hỏi liên quan đến:
               - Tên môn học, mã môn học (VD: Cơ sở dữ liệu, CTDL, Toán cao cấp...)
               - Điều kiện tiên quyết, học trước, học song hành.
               - Đăng ký môn học.
               - Số tín chỉ, tính toán tổng tín chỉ đã tích lũy, GPA.
               - Nhóm môn học (tự chọn, bắt buộc).

            2. Chọn "RAG" (Vector RAG) NẾU câu hỏi liên quan đến:
               - Các quy chế chung (VD: Bao nhiêu điểm thì bị đuổi học? Quy định học bổng?).
               - Lịch biểu, thủ tục hành chính không dính tới môn học cụ thể.

        Câu hỏi từ sinh viên: "{question}"
        Session ID: {session_id}
        """
        return self.planner_chain.invoke(system_prompt)