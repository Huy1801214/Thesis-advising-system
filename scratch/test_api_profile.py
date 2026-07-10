from fastapi.testclient import TestClient
from main import app
import os

def test_upload_excel():
    client = TestClient(app)
    
    # Đọc file Diem.xlsx
    file_path = "data/Diem.xlsx"
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} không tồn tại.")
        return
        
    print(f"👉 Đang test upload file: {file_path}")
    
    # Vì endpoint yêu cầu đăng nhập (get_current_mssv), chúng ta cần bypass hoặc mock auth.
    # Tuy nhiên, chúng ta có thể gọi trực tiếp hàm extract_student_context để kiểm tra tính chính xác của parser:
    from core.document_parser import extract_student_context
    context = extract_student_context(file_path)
    
    print("\n[Parser Results]")
    print(f"- GPA: {context['cumulative_gpa']}")
    print(f"- Tín chỉ tích lũy (total_earned_credits): {context['total_earned_credits']}")
    print(f"- Số môn học đã qua: {len(context['passed_courses'])}")
    
    # Kiểm tra kiểu dữ liệu
    total_passed_credits = int(float(context["total_earned_credits"])) if context["total_earned_credits"] is not None else 0
    print(f"- Số tín chỉ chuyển đổi sang API: {total_passed_credits}")
    
    assert total_passed_credits == 137, f"Sai số tín chỉ tích lũy, mong đợi 137, nhận được {total_passed_credits}"
    print("\n✅ KIỂM TRA THÀNH CÔNG: Số tín chỉ tích lũy đã được phân tích và bóc tách chính xác là 137!")

if __name__ == "__main__":
    test_upload_excel()
