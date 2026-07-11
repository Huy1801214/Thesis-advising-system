import asyncio
from orchestrator.student_analyst import StudentAnalysisAgent
from orchestrator.reasoning_agent import KnowledgeReasoningAgent

async def main():
    print("=== BẮT ĐẦU CHẠY THỬ NGHIỆM TƯ VẤN HƯỚNG NGHIỆP TRỰC TIẾP ===")
    
    # 1. Khởi tạo các Agent
    analyst = StudentAnalysisAgent()
    reasoner = KnowledgeReasoningAgent()
    
    # 2. Dữ liệu giả lập của một sinh viên đã học một số môn
    # Môn 200101: Triết học Mác Lênin, 202108: Toán cao cấp A1, 200105: Cơ sở dữ liệu (tùy thuộc database)
    student_data = {
        "cumulative_gpa": 3.2,
        "total_earned_credits": 15,
        "passed_courses": ["200101", "202108", "214241"] # Giả sử đã qua các môn này
    }
    mssv = "22130093"
    
    print("\n[Step 1] Đang phân tích năng lực sinh viên...")
    profile = await analyst.analyze_student(mssv, student_data)
    print(f"-> GPA: {profile.gpa}")
    print(f"-> Tín chỉ: {profile.total_credits}")
    print(f"-> Nhận xét học thuật: {profile.academic_summary}")
    print(f"-> Số kỹ năng đã tích lũy: {len(profile.acquired_competencies)}")
    for comp in profile.acquired_competencies:
        print(f"   + {comp.name}: {comp.description}")

    # 3. Chạy lập luận tư vấn nghề nghiệp
    question = "Em muốn làm Kỹ sư Điện toán đám mây (Cloud Engineer) thì nên chọn học những môn học nào tiếp theo?"
    print(f"\n[Step 2] Đang lập luận tư vấn nghề nghiệp cho câu hỏi: '{question}'...")
    
    recommendation = await reasoner.reason_and_recommend(
        question=question,
        profile=profile,
        passed_courses=student_data["passed_courses"]
    )
    
    print("\n[Step 3] KẾT QUẢ TƯ VẤN:")
    print("=" * 60)
    print(recommendation)
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
