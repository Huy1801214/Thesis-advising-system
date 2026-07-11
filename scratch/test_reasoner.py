import asyncio
import json
from core.database import SessionLocal
from model.student_profile import StudentProfile
from orchestrator.student_analyst import StudentAnalysisAgent
from orchestrator.reasoning_agent import KnowledgeReasoningAgent

async def test():
    db = SessionLocal()
    try:
        profile_db = db.query(StudentProfile).filter(StudentProfile.mssv == "22130093").first()
        student_data = {
            "cumulative_gpa": profile_db.cumulative_gpa,
            "total_earned_credits": profile_db.total_earned_credits,
            "passed_courses": profile_db.passed_courses,
            "current_courses": profile_db.current_courses,
            "major": profile_db.major,
            "target_career": profile_db.target_career,
            "interests": profile_db.interests
        }
        
        analyst = StudentAnalysisAgent()
        profile = await analyst.analyze_student(profile_db.mssv, student_data)
        
        reasoner = KnowledgeReasoningAgent()
        
        question = "Tư vấn lộ trình học để làm Kỹ sư Điện toán đám mây"
        
        # Gọi bước 1, 2, 3, 4 để xem recommended_courses có những gì
        job_info = await reasoner.find_matching_job(question)
        print("Job matched:", job_info.get("name"))
        
        req_comps = await reasoner.get_job_requirements(job_info["name"])
        print("\nRequired Competencies:")
        for r in req_comps:
            print(f"- {r['name']}")
            
        student_comps = {c.name for c in profile.acquired_competencies}
        print("\nStudent Acquired Competencies:")
        for c in student_comps:
            print(f"- {c}")
            
        gap_comps = [req for req in req_comps if req["name"] not in student_comps]
        print("\nGap Competencies:")
        for g in gap_comps:
            print(f"- {g['name']}")
            
        recommended_courses = []
        for comp in gap_comps:
            courses = await reasoner.get_courses_teaching_competency(comp["name"])
            for c in courses:
                code = c["code"]
                name = c["name"]
                if code in student_data["passed_courses"]:
                    continue
                prereqs = await reasoner.get_course_prerequisites(code)
                missing_prereqs = [p for p in prereqs if p not in student_data["passed_courses"]]
                status = "Sẵn sàng đăng ký học ngay" if not missing_prereqs else f"Cần tích lũy môn học trước: {', '.join(missing_prereqs)}"
                recommended_courses.append({
                    "competency": comp["name"],
                    "course_code": code,
                    "course_name": name,
                    "credits": c["credits"],
                    "status": status
                })
                
        print("\nRecommended Courses list:")
        print(json.dumps(recommended_courses, indent=2, ensure_ascii=False))
        
        # Xem LLM sinh gì
        res = await reasoner.reason_and_recommend(question, profile, student_data["passed_courses"])
        print("\n=== CHATBOT RESPONSE ===")
        print(res)
        
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test())
