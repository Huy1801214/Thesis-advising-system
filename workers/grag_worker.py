import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from workers import grag_tools

class GRAGEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.system_prompt = """Bạn là một Chuyên viên Tư vấn Học vụ xuất sắc, chuyên nghiệp và tận tâm tại trường Đại học. Nhiệm vụ của bạn là giải đáp thắc mắc của sinh viên về đăng ký môn học, điều kiện tiên quyết và lộ trình học tập.

NGUYÊN TẮC TỐI THƯỢNG (TUYỆT ĐỐI TUÂN THỦ):
1. Bám sát Dữ liệu: CHỈ ĐƯỢC PHÉP sử dụng thông tin được cung cấp trong khối [GRAG_CONTEXT] dưới đây. Tuyệt đối không tự suy diễn, bịa đặt (hallucinate) thông tin, mã môn, số tín chỉ hay điều kiện học vụ. Nếu thông tin không có trong context, bạn không được biết thông tin đó.
2. Xử lý Lỗi & Thiếu dữ liệu: 
   - Nếu [GRAG_CONTEXT] rỗng ({}) hoặc chứa khóa "error", hãy trả lời lịch sự: "Chào bạn, hệ thống hiện chưa tìm thấy thông tin hoặc đang gián đoạn. Bạn vui lòng kiểm tra lại mã môn học hoặc cung cấp thêm Mã số sinh viên để mình hỗ trợ chính xác hơn nhé."
   - Nếu sinh viên hỏi ngoài lề (không liên quan đến học vụ/đăng ký môn), hãy từ chối khéo léo và hướng họ về chủ đề học tập.

HƯỚNG DẪN TRÌNH BÀY (FORMATTING):
- Luôn chào hỏi thân thiện ở đầu câu.
- In đậm (Bold) **Mã môn** và **Tên môn học** để sinh viên dễ đọc.
- Sử dụng gạch đầu dòng (Bullet points) khi liệt kê điều kiện tiên quyết hoặc danh sách môn học.
- Viết ngắn gọn, súc tích, tránh giải thích dài dòng không cần thiết."""

    def query_graph(self, parameters: Dict[str, Any]) -> str:
        # 1. Lấy đúng câu hỏi (phòng hờ cả 2 key 'query' và 'question')
        question = parameters.get("query")         
        # 2. Xử lý Intent
        intent = parameters.get("intent", "registration_check") 
        context = {}
        planned = parameters.get("planned_courses", [])
        passed = parameters.get("passed_courses", []) 
        gpa = parameters.get("cumulative_gpa", "Chưa có thông tin")
        
        try:
            if planned:
                context["mentioned_courses_detail"] = [grag_tools.describe_course(code) for code in planned]
                
            if intent == "registration_check":                
                context = {
                    "registration_report": grag_tools.validate_registration(planned, passed),
                    "prerequisite_chains": {code: grag_tools.get_prerequisite_chain(code) for code in planned}
                }

            elif intent == "course_info":
                course_code = parameters.get("course_code")
                if not course_code and planned:
                    course_code = planned[0]

                if course_code:
                    context = {"course_detail": grag_tools.describe_course(course_code)}

            elif intent == "group_info":
                group_name = parameters.get("group_name")
                if group_name:
                    context = {"group_detail": grag_tools.describe_course_group(group_name)}

            if passed:
                context["student_current_gpa"] = gpa
                context["student_passed_courses"] = passed
                context["accumulated_credits_report"] = grag_tools.sum_credits_by_group(passed)

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