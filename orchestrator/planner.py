from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from orchestrator.schemas import TaskPlan
from dotenv import load_dotenv

load_dotenv()

class OrchestratorPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        
        self.system_prompt = """
        Bạn là Nhạc trưởng Điều phối (Orchestrator) siêu việt của Hệ thống Tư vấn Học vụ Đại học. Nhiệm vụ của bạn là đọc câu hỏi của sinh viên, PHÂN RÃ nó thành các tác vụ (Tasks) độc lập và điền vào cấu trúc dữ liệu được yêu cầu.

        QUY TẮC CHỌN WORKER (task_type):
        - Chọn "GRAG": Nếu tác vụ cần truy vấn cơ sở dữ liệu đồ thị về Môn học (mã môn, số tín chỉ, nhóm môn), Đăng ký môn (điều kiện tiên quyết, học trước, học song hành), hoặc Điểm số/GPA.
        - Chọn "RAG": Nếu tác vụ cần tra cứu văn bản về Quy chế (bảo lưu, thôi học, học bổng), Thủ tục hành chính, Lịch biểu, hay Hướng dẫn sinh viên.

        HƯỚNG DẪN ĐIỀN THÔNG TIN TÁC VỤ:
        - task_id: Đánh số thứ tự (Ví dụ: TASK_01, TASK_02).
        - query_intent: Viết lại ý định của sinh viên thành một câu lệnh tìm kiếm tối ưu, rõ ràng, nhắm thẳng vào mục tiêu.
        - parameters: Bóc tách các thực thể có trong câu hỏi.
          + Nếu là GRAG: Bắt buộc xác định tham số "intent" (có thể là: "registration_check", "course_info", hoặc "group_info").
          + Cố gắng trích xuất tên môn học (ví dụ: "course_name": "Cơ sở dữ liệu") nếu có.

        LUÔN CẢNH GIÁC: Một câu hỏi của sinh viên có thể chứa ĐA Ý ĐỊNH (Ví dụ: vừa hỏi điểm, vừa hỏi thủ tục). Hãy chia thành nhiều Task tương ứng.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Session ID: {session_id}\nCâu hỏi của sinh viên: {question}")
        ])
        
        self.planner_chain = self.prompt | self.llm.with_structured_output(TaskPlan)

    def create_plan(self, question: str, session_id: str) -> TaskPlan:
        plan = self.planner_chain.invoke({
            "question": question, 
            "session_id": session_id
        })
        print(f"[Planner] Đã chia thành {len(plan.tasks)} tác vụ!")
        return plan