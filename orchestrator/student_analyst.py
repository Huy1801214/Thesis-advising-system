import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from core.graph_db import graph_db
from core.llm import build_chat_model

class CompetencyAcquired(BaseModel):
    name: str = Field(description="Tên kỹ năng/năng lực đã học")
    description: str = Field(description="Mô tả tóm tắt kỹ năng")

class StudentCompetencyProfile(BaseModel):
    mssv: str
    gpa: float
    total_credits: int
    major: str | None = None
    target_career: str | None = None
    interests: str | None = None
    acquired_competencies: List[CompetencyAcquired] = Field(default_factory=list)
    academic_summary: str = Field(description="Nhận xét tổng quan kết quả học tập và năng lực tích lũy đối chiếu với mục tiêu nghề nghiệp")

class StudentAnalysisAgent:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.analyst_chain = self.llm.with_structured_output(StudentCompetencyProfile, method="function_calling")

    async def get_acquired_competencies(self, passed_courses: List[str]) -> List[Dict[str, str]]:
        if not passed_courses:
            return []
            
        cypher = """
        MATCH (c:Course)-[:TEACHES_SKILL]->(comp:Competency)
        WHERE c.code IN $passed
        RETURN DISTINCT comp.name AS name, comp.description AS description
        """
        records = await graph_db.run_query_async(cypher, {"passed": passed_courses})
        return [{"name": r["name"], "description": r["description"]} for r in records]

    async def analyze_student(self, mssv: str, student_data: Dict[str, Any]) -> StudentCompetencyProfile:
        passed = student_data.get("passed_courses", [])
        gpa = student_data.get("cumulative_gpa", 0.0)
        # Ép kiểu GPA thành float hợp lệ
        try:
            gpa = float(gpa) if gpa else 0.0
        except ValueError:
            gpa = 0.0
            
        total_credits = student_data.get("total_earned_credits", 0)
        try:
            total_credits = int(total_credits) if total_credits else 0
        except ValueError:
            total_credits = 0

        major = student_data.get("major", "Chưa xác định")
        target_career = student_data.get("target_career", "Chưa xác định")
        interests = student_data.get("interests", "Chưa xác định")

        # 1. Tra cứu năng lực đã học từ Neo4j
        acquired_comps = await self.get_acquired_competencies(passed)

        # 2. Gọi LLM để sinh phân tích nhận xét học thuật dựa trên kết quả tích lũy và định hướng
        prompt = f"""
Bạn là Student Analysis Agent chịu trách nhiệm phân tích học thuật sinh viên.
Hãy dựa vào các thông tin sau để viết một nhận xét học thuật súc tích (dưới 150 từ) bằng tiếng Việt đánh giá thế mạnh học tập và độ phù hợp của bảng điểm hiện có với mục tiêu chuyên ngành và nghề nghiệp mong muốn.

Thông tin sinh viên:
- Mã số sinh viên (MSSV): {mssv}
- Chuyên ngành đang học: {major}
- Định hướng nghề nghiệp mong muốn: {target_career}
- Sở thích công nghệ/Mục tiêu cá nhân: {interests}
- Điểm trung bình tích lũy (GPA): {gpa}
- Tín chỉ tích lũy: {total_credits}
- Danh sách năng lực/kỹ năng đã tích lũy từ các môn đã qua:
{json.dumps(acquired_comps, ensure_ascii=False, indent=2)}

Yêu cầu nhận xét:
- Đánh giá xem với bảng điểm hiện tại, sinh viên đã có được các kỹ năng cốt lõi nào hỗ trợ cho nghề '{target_career}' chưa.
- Lời khuyên định hướng và mang tính động viên sinh viên.
- Trả về cấu trúc JSON chứa đầy đủ thông tin yêu cầu.
"""
        try:
            # Gọi LLM sinh structured profile
            profile = await self.analyst_chain.ainvoke(prompt)
            # Điền lại các giá trị thô để đảm bảo chính xác tuyệt đối
            profile.mssv = mssv
            profile.gpa = gpa
            profile.total_credits = total_credits
            profile.major = major
            profile.target_career = target_career
            profile.interests = interests
            profile.acquired_competencies = [CompetencyAcquired(**c) for c in acquired_comps]
            return profile
        except Exception as e:
            # Fallback nếu LLM gặp sự cố
            print(f"[StudentAnalyst] Lỗi phân tích: {e}")
            return StudentCompetencyProfile(
                mssv=mssv,
                gpa=gpa,
                total_credits=total_credits,
                major=major,
                target_career=target_career,
                interests=interests,
                acquired_competencies=[CompetencyAcquired(**c) for c in acquired_comps],
                academic_summary=f"Chào bạn, hệ thống đã ghi nhận chuyên ngành {major} và đích đến là {target_career}. Bảng điểm của bạn cho thấy bạn sẵn sàng cho các gợi ý môn học tiếp theo."
            )
