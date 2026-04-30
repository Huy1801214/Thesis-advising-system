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
        Bạn là Orchestrator Agent. Nhiệm vụ của bạn là phân rã câu hỏi sinh viên thành các tác vụ độc lập:
        - Sử dụng 'RAG' cho tra cứu quy chế, sổ tay (văn bản phẳng).
        - Sử dụng 'GRAG' cho tra cứu môn tiên quyết, lộ trình học tập (đồ thị).
        Đảm bảo các tác vụ con không có sự phụ thuộc chéo để thực thi song song[cite: 975, 976].
        
        Câu hỏi: {question}
        Session ID: {session_id}
        """
        return self.planner_chain.invoke(system_prompt)