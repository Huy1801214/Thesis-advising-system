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

        QUY TẮC PHÂN RÃ CÂU HỎI (query_intent):
        - Trường `query_intent` trong mỗi Task phải là một câu hỏi con (sub-question) bằng tiếng Việt tự nhiên đã được viết lại độc lập, rõ nghĩa, sẵn sàng để làm câu truy vấn cho RAG/GRAG.
        - TUYỆT ĐỐI KHÔNG điền các nhãn phân loại (như 'course_info', 'registration_check') vào trường `query_intent`. Nhãn phân loại chỉ được điền vào trường `parameters.intent`.
        - Ví dụ:
          + Đúng: query_intent = "Quy định về việc tham gia lớp học và điều kiện cấm thi là gì?"
          + Sai: query_intent = "course_info" hoặc "registration_check".

        QUY TẮC TRÍCH XUẤT THAM SỐ (Parameters):
        1. Mã môn học: NẾU TRONG CÂU HỎI có chứa mã số 6 chữ số, trích xuất vào 'planned_courses'. TUYỆT ĐỐI KHÔNG TỰ BỊA RA MÃ MÔN HỌC HOẶC LẤY TỪ VÍ DỤ NẾU NGƯỜI DÙNG KHÔNG CUNG CẤP! Bỏ trống nếu không có.
        2. Tên môn học: Trích xuất TÊN môn học vào trường 'course_name' (VD: "Cơ sở dữ liệu", "Phân tích thiết kế hệ thống thông tin", "Khóa luận tốt nghiệp"). Nếu sinh viên chỉ nói tên môn mà không nói mã, hãy bỏ trống 'planned_courses' và 'course_code', chỉ điền 'course_name'.
        3. Mã sinh viên: Thường bắt đầu bằng 'SV' hoặc chuỗi số MSSV (VD: SV001, 22130099).
           Trích xuất vào 'student_id'.
        4. Intent (BẮT BUỘC ĐIỀN vào `parameters.intent`): 
           - Nếu hỏi về số tín chỉ, số tiết, đặc điểm của MỘT môn học cụ thể (kể cả Khóa luận) -> intent: 'course_info'.
           - Nếu hỏi về điều kiện đăng ký môn, kiểm tra lộ trình, tư vấn học phần -> intent: 'registration_check'.

         GIỚI HẠN TỐI ĐA (CRITICAL):
        - Chỉ sinh ra TỐI ĐA 5 tác vụ (tasks) cho mỗi câu hỏi. Tuyệt đối không sinh nhiều hơn để tránh nghẽn I/O hệ thống.

        Quy tắc phân loại nhiệm vụ (Task Type):
         1. Chọn "CAREER_ADVISE" (Tư vấn nghề nghiệp) NẾU câu hỏi liên quan đến:
         - Định hướng công việc hoặc vị trí tuyển dụng (ví dụ: DevOps, Software Engineer, Data Scientist, Cloud, An toàn thông tin...).
         - Đề xuất lộ trình môn học định hướng công việc cụ thể.
         - Hỏi về kỹ năng cần có, xu hướng công nghệ toàn cầu hoặc chứng chỉ quốc tế liên quan.

         2. Chọn "GRAG" (Graph RAG) NẾU câu hỏi ĐÒI HỎI SUY LUẬN LOGIC, RÀNG BUỘC MÔN HỌC HOẶC THÔNG TIN VỀ 1 MÔN CỤ THỂ (không bàn về định hướng nghề nghiệp):
         - BẤT KỲ câu hỏi nào về "Khóa luận tốt nghiệp", "Đồ án", "Thực tập" hoặc 1 môn học có tên cụ thể (Hỏi về số tín chỉ, điều kiện, mã môn).
         - Gợi ý lộ trình, hỏi "nên học môn gì tiếp theo", tư vấn dựa trên danh sách môn đã học nói chung.
         - Tên môn học, mã môn học (VD: Cơ sở dữ liệu, CTDL, Toán cao cấp...).
         - Điều kiện tiên quyết, học trước, học song hành.
         - Đăng ký môn học, xung đột lịch học.
         - Tính toán tổng tín chỉ đã tích lũy, GPA (dựa trên bảng điểm).

         3. Chọn "RAG" (Vector RAG) NẾU câu hỏi CHỈ TRA CỨU QUY CHẾ/VĂN BẢN (Không dính tới môn học cụ thể hay nghề nghiệp):
            - Các quy chế chung (VD: Bao nhiêu điểm thì bị đuổi học? Điều kiện xét học bổng? Quy định bảo lưu?).
            - Lịch biểu chung, thủ tục hành chính giấy tờ.

         4. Chọn "CLARIFY" (Hỏi lại) NẾU CÂU HỎI QUÁ MƠ HỒ HOẶC THIẾU NGỮ CẢNH TRẦM TRỌNG:
            - Ví dụ SV hỏi: "Em đăng ký môn đó được không?" (Không rõ 'môn đó' là môn nào).
            - Hành động: Chọn task_type="CLARIFY", ghi câu hỏi cần làm rõ vào trường 'clarification_message', và set 'needs_clarification'=True.

         QUY TẮC ĐÁNH GIÁ (Metadata Logging):
        - reasoning_summary: Viết 1-2 câu giải thích tại sao bạn lại chọn các Agent này.
        - confidence_score: Chấm điểm từ 0.0 đến 1.0 (1.0 nếu câu hỏi cực kỳ rõ ràng, dưới 0.6 nếu câu hỏi lủng củng hoặc phải dùng CLARIFY).

        Câu hỏi từ sinh viên: "{question}"
        Session ID: {session_id}
        """
        return self.planner_chain.invoke(system_prompt)
