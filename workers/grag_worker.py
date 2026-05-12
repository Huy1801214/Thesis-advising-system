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

        question = parameters.get("question", "")
        student_id = parameters.get("student_id")
        intent = parameters.get("intent", "registration_check") 
        
        # 1. Thu thập Context từ Neo4j dựa trên Intent
        context = {}
        
        if intent == "registration_check":
            planned = parameters.get("planned_courses", [])
            if not student_id:
                return "Bạn vui lòng cung cấp mã sinh viên để mình kiểm tra điều kiện đăng ký nhé."
            
            context = {
                "registration_report": grag_tools.validate_registration(student_id, planned),
                "student_credits": grag_tools.sum_credits_by_group(student_id),
                "prerequisite_chains": {code: grag_tools.get_prerequisite_chain(code) for code in planned}
            }

        elif intent == "course_info":
            course_code = parameters.get("course_code")
            context = {"course_detail": grag_tools.describe_course(course_code)}

        elif intent == "group_info":
            group_name = parameters.get("group_name")
            context = {"group_detail": grag_tools.describe_course_group(group_name)}

        # 2. LLM Synthesizer: Diễn giải dữ liệu Graph thành lời văn
        prompt = f"""{self.system_prompt}

        GRAG_CONTEXT:
        {json.dumps(context, ensure_ascii=False, indent=2)}

        CÂU HỎI:
        {question}
        """
        
        response = self.llm.invoke(prompt)
        return response.content

grag_engine = GRAGEngine()