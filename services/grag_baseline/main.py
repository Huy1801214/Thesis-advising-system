import os
from langchain_groq import ChatGroq
from config import Config
from graph_retriever import GraphRetriever
from embeddings import VectorStoreManager

class GRAGAgent:
    def __init__(self):
        # Khởi tạo Graph và Vector như cũ
        self.graph = GraphRetriever()
        self.vector_manager = VectorStoreManager()
        self.vector_db = self.vector_manager.get_retriever()
        
        # Khởi tạo Groq LLM
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Giữ mức thấp để trả lời chính xác theo Quy chế
            groq_api_key=Config.GROQ_API_KEY
        )

    def _call_llm(self, prompt: str):
        """Sử dụng phương thức invoke của LangChain"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"❌ Lỗi Groq: {str(e)}"

    def chat(self, user_query: str):
        # Bước 1: Truy xuất Graph (Lấy logic tiên quyết)
        course_info = self.graph.search_course(user_query)
        graph_context = "Không tìm thấy thông tin môn học cụ thể trong đồ thị."
        if course_info:
            pres = self.graph.get_direct_prerequisites(course_info['ma_hp'])
            graph_context = f"Môn {course_info['ten_hp']} ({course_info['ma_hp']}) yêu cầu môn học trước: {pres}"

        # Bước 2: Truy xuất Vector (Lấy quy chế/hỗ trợ)
        docs = self.vector_db.invoke(user_query)
        vector_context = "\n".join([doc.page_content for doc in docs])

        # Bước 3: Hybrid Context Fusion
        prompt = f"""
            Bạn là Cố vấn học tập AI của khoa CNTT - Đại học Nông Lâm TP.HCM. 
            Dựa vào ngữ cảnh, hãy trả lời chính xác câu hỏi của sinh viên.

            LƯU Ý QUY CHẾ NLU:
            1. Điểm đạt (qua môn) thông thường là điểm D (>= 4.0 hệ 10).
            2. Điểm để "công nhận/bảo lưu" khi chuyển ngành/hệ thường >= 5.5.
            Hãy phân biệt rõ hai khái niệm này dựa trên câu hỏi.

            NGỮ CẢNH ĐỒ THỊ: {graph_context}
            NGỮ CẢNH VĂN BẢN: {vector_context}

            CÂU HỎI: {user_query}
            TRẢ LỜI:
            """
        return self._call_llm(prompt)

if __name__ == "__main__":
    agent = GRAGAgent()
    while True:
        query = input("Sinh viên hỏi: ")
        if not query.strip() or query.lower() == 'exit': break
        print(f"\n{agent.chat(query)}\n")