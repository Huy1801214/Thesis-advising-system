import asyncio
import json
from core.database import SessionLocal
from model.student_profile import StudentProfile
from orchestrator.student_analyst import StudentAnalysisAgent
from orchestrator.reasoning_agent import KnowledgeReasoningAgent

async def test_personalized_advising():
    print("=== CHẠY THỬ NGHIỆM TƯ VẤN CÁ NHÂN HÓA THEO HỒ SƠ SINH VIÊN ===")
    
    # 1. Khởi tạo mock student data
    mssv_test = "SV_TEST_PROFILE"
    student_data = {
        "cumulative_gpa": 3.4,
        "total_earned_credits": 15,
        "passed_courses": ["200101", "202108", "214241"],  # Triết học, Anh văn, Điện toán đám mây
        "major": "Kỹ thuật phần mềm",
        "target_career": "Kỹ sư Điện toán đám mây (Cloud Engineer)",
        "interests": "Tự động hóa hạ tầng (IaC), Kubernetes, Bảo mật đám mây"
    }

    # 2. Chạy Student Analysis Agent
    analyst = StudentAnalysisAgent()
    print("\n[Step 1] Đang phân tích năng lực cá nhân hóa dựa trên hồ sơ...")
    profile = await analyst.analyze_student(mssv_test, student_data)
    
    print(f"-> GPA: {profile.gpa}")
    print(f"-> Chuyên ngành: {profile.major}")
    print(f"-> Định hướng công việc: {profile.target_career}")
    print(f"-> Sở thích: {profile.interests}")
    print(f"-> Nhận xét học thuật từ Agent:")
    print(f"   \"{profile.academic_summary}\"")
    print(f"-> Số kỹ năng tích lũy: {len(profile.acquired_competencies)}")
    for c in profile.acquired_competencies:
        print(f"   + {c.name}: {c.description}")

    # 3. Chạy Knowledge Reasoning Agent để lấy lộ trình
    reasoner = KnowledgeReasoningAgent()
    print(f"\n[Step 2] Đang lập luận tư vấn nghề nghiệp cá nhân hóa cho câu hỏi...")
    question = "Em nên đăng ký học những môn nào tiếp theo?"
    res = await reasoner.reason_and_recommend(
        question=question,
        profile=profile,
        passed_courses=student_data["passed_courses"]
    )
    
    print("\n[Step 3] KẾT QUẢ TƯ VẤN CÁ NHÂN HÓA:")
    print("============================================================")
    print(res)
    print("============================================================")

if __name__ == "__main__":
    asyncio.run(test_personalized_advising())
