from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from core.security import verify_token, oauth2_scheme  # Dùng lại hàm Auth của bạn
import shutil
import os

# Import hàm đọc file mà chúng ta đã làm ở bước trước
# (Giả sử bạn để file document_parser.py ở thư mục utils hoặc thư mục gốc)
from core.document_parser import extract_student_context
router = APIRouter()

# ==========================================
# BỘ NHỚ TẠM (SESSION MEMORY)
# Lưu trữ bảng điểm của sinh viên theo dạng: { "20110001": {"passed_courses": [...], ...} }
# ==========================================
STUDENT_TRANSCRIPT_STORE = {}

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@router.post("/upload-transcript")
async def upload_transcript(
    file: UploadFile = File(...), 
    mssv: str = Depends(get_current_mssv)  # Bắt buộc sinh viên phải đăng nhập
):
    print(f"📥 [API] Nhận file bảng điểm {file.filename} từ sinh viên {mssv}")
    
    # Lưu file tạm với tên có chứa mssv để tránh trùng lặp nếu nhiều người up cùng lúc
    temp_file_path = f"temp_{mssv}_{file.filename}"
    
    try:
        # 1. Lưu file tạm vào ổ cứng
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Trích xuất dữ liệu bằng hàm của chúng ta
        context = extract_student_context(temp_file_path)
        
        if context["status"] != "success":
            raise HTTPException(status_code=400, detail=context["message"])
            
        # 3. LƯU VÀO BỘ NHỚ TẠM (Gán thẳng vào MSSV)
        STUDENT_TRANSCRIPT_STORE[mssv] = {
            "cumulative_gpa": context["cumulative_gpa"],
            "total_earned_credits": context["total_earned_credits"],
            "passed_courses": context["passed_courses"],
            "current_courses": context["current_courses"]
        }
        
        return {
            "status": "SUCCESS",
            "message": "Đã đồng bộ dữ liệu bảng điểm thành công!",
            "data": {
                "gpa": context["cumulative_gpa"],
                "total_passed": len(context["passed_courses"])
            }
        }
        
    finally:
        # 4. Xóa file ngay lập tức để bảo mật thông tin và tiết kiệm ổ cứng
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)