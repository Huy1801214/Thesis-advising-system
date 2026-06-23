from dotenv import load_dotenv
from core.llm import build_chat_model
from orchestrator.schemas import TaskPlan

load_dotenv()


class OrchestratorPlanner:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.planner_chain = self.llm.with_structured_output(TaskPlan, method="function_calling")

    def create_plan(self, question: str, session_id: str) -> TaskPlan:
        system_prompt = f"""
        Bạn là Orchestrator Agent điều phối hệ thống tư vấn học vụ.
        Nhiệm vụ: Phân tích câu hỏi của sinh viên và phân rã thành một bản Kế hoạch Tác vụ (Task Plan) tối ưu nhất.

        QUY TẮC TRÍCH XUẤT THAM SỐ (Parameters):
        1. Mã môn học: Luôn là chuỗi 6 chữ số (VD: 200105, 214294). 
           Trích xuất vào danh sách 'planned_courses'.
        2. Mã sinh viên: Thường bắt đầu bằng 'SV' hoặc chuỗi số MSSV (VD: SV001, 22130099).
           Trích xuất vào 'student_id'.
        3. Intent: 
           - Nếu hỏi về điều kiện, đăng ký môn -> intent: 'registration_check'.
           - Nếu hỏi thông tin môn học cụ thể -> intent: 'course_info'.

         GIỚI HẠN TỐI ĐA (CRITICAL):
        - Chỉ sinh ra TỐI ĐA 5 tác vụ (tasks) cho mỗi câu hỏi. Tuyệt đối không sinh nhiều hơn để tránh nghẽn I/O hệ thống.

        Quy tắc phân loại nhiệm vụ (Task Type):
         1. Chọn "GRAG" (Graph RAG) NẾU câu hỏi ĐÒI HỎI SUY LUẬN LOGIC HOẶC RÀNG BUỘC MÔN HỌC:
         - Gợi ý lộ trình, hỏi "nên học môn gì tiếp theo", tư vấn dựa trên danh sách môn đã học.
         - Tên môn học, mã môn học (VD: Cơ sở dữ liệu, CTDL, Toán cao cấp...).
         - Điều kiện tiên quyết, học trước, học song hành.
         - Đăng ký môn học, xung đột lịch học.
         - Số tín chỉ, tính toán tổng tín chỉ đã tích lũy, GPA.

        2. Chọn "RAG" (Vector RAG) NẾU câu hỏi CHỈ TRA CỨU QUY CHẾ/VĂN BẢN (Không dính tới môn học cụ thể):
           - Các quy chế chung (VD: Bao nhiêu điểm thì bị đuổi học? Điều kiện xét học bổng? Quy định bảo lưu?).
           - Lịch biểu chung, thủ tục hành chính giấy tờ.

        3. Chọn "CLARIFY" (Hỏi lại) NẾU CÂU HỎI QUÁ MƠ HỒ HOẶC THIẾU NGỮ CẢNH TRẦM TRỌNG:
           - Ví dụ SV hỏi: "Em đăng ký môn đó được không?" (Không rõ 'môn đó' là môn nào).
           - Hành động: Chọn task_type="CLARIFY", ghi câu hỏi cần làm rõ vào trường 'clarification_message', và set 'needs_clarification'=True.

         QUY TẮC ĐÁNH GIÁ (Metadata Logging):
        - reasoning_summary: Viết 1-2 câu giải thích tại sao bạn lại chọn các Agent này.
        - confidence_score: Chấm điểm từ 0.0 đến 1.0 (1.0 nếu câu hỏi cực kỳ rõ ràng, dưới 0.6 nếu câu hỏi lủng củng hoặc phải dùng CLARIFY).

        Câu hỏi từ sinh viên: "{question}"
        Session ID: {session_id}
        """
        return self.planner_chain.invoke(system_prompt)
