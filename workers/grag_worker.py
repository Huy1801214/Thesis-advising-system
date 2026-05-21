import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from workers import grag_tools

class GRAGEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.system_prompt = """Bạn là trợ lý tư vấn học vụ chuyên nghiệp.
        Chỉ trả lời dựa trên dữ liệu GRAG_CONTEXT cung cấp. 
        Nếu không có dữ liệu hoặc không đủ thông tin, hãy yêu cầu sinh viên cung cấp thêm mã sinh viên hoặc mã môn học.
        Trả lời bằng tiếng Việt, thân thiện và chính xác."""

    def query_graph(self, parameters: Dict[str, Any]) -> str:
        # 1. Lấy đúng câu hỏi (phòng hờ cả 2 key 'query' và 'question')
        question = parameters.get("query") 
        student_id = parameters.get("student_id")
        
        # 2. Xử lý Intent
        intent = parameters.get("intent", "registration_check") 
        context = {}
        
        try:
            if intent == "registration_check":
                # Lấy danh sách môn học, nếu Planner không tìm thấy thì gán rỗng
                planned = parameters.get("planned_courses", [])
                if not student_id:
                    return "Bạn vui lòng cung cấp mã sinh viên để mình kiểm tra nhé."
                
                context = {
                    "registration_report": grag_tools.validate_registration(student_id, planned),
                    "student_credits": grag_tools.sum_credits_by_group(student_id),
                    "prerequisite_chains": {code: grag_tools.get_prerequisite_chain(code) for code in planned}
                }

            elif intent == "course_info":
                course_code = parameters.get("course_code")
                if course_code:
                    context = {"course_detail": grag_tools.describe_course(course_code)}

            elif intent == "group_info":
                group_name = parameters.get("group_name")
                if group_name:
                    context = {"group_detail": grag_tools.describe_course_group(group_name)}
                    
        except Exception as e:
            print(f"❌ [Lỗi GRAG Tools]: {str(e)}")
            context = {"error": f"Lỗi truy xuất đồ thị: {str(e)}"}

        # 3. LLM Synthesizer: Đưa câu hỏi và ngữ cảnh cho Gemini
        prompt = f"""{self.system_prompt}

        GRAG_CONTEXT (Dữ liệu từ hệ thống đồ thị Neo4j):
        {json.dumps(context, ensure_ascii=False, indent=2)}

        CÂU HỎI CỦA SINH VIÊN:
        {question}
        """
        
        print(f"🧠 [GRAG Prompt gửi Gemini]: Đang xử lý câu hỏi '{question}'...")
        response = self.llm.invoke(prompt)
        return response.content

grag_engine = GRAGEngine()