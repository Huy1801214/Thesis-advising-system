from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from core.security import verify_token, oauth2_scheme  # Dùng lại hàm Auth của bạn
from core.database import get_db
from model.student_profile import StudentProfile
import shutil
import os
from pydantic import BaseModel
from core.graph_db import graph_db

# Import hàm đọc file mà chúng ta đã làm ở bước trước
# (Giả sử bạn để file document_parser.py ở thư mục utils hoặc thư mục gốc)
from core.document_parser import extract_student_context
router = APIRouter()

class ProfileUpdateRequest(BaseModel):
    major: str | None = None
    target_career: str | None = None
    interests: str | None = None

# ==========================================
# BỘ NHỚ TẠM (SESSION MEMORY) - Dùng để tương thích ngược nếu cần
# Lưu trữ bảng điểm của sinh viên theo dạng: { "20110001": {"passed_courses": [...], ...} }
# ==========================================
STUDENT_TRANSCRIPT_STORE = {}

def get_current_mssv(token: str = Depends(oauth2_scheme)):
    user_info = verify_token(token) 
    return user_info.get("sub")

@router.post("/upload-transcript")
async def upload_transcript(
    file: UploadFile = File(...), 
    mssv: str = Depends(get_current_mssv),  # Bắt buộc sinh viên phải đăng nhập
    db: Session = Depends(get_db)
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
            
        # 3. LƯU VÀO CƠ SỞ DỮ LIỆU (Tìm kiếm và cập nhật hoặc tạo mới)
        profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
        if not profile:
            profile = StudentProfile(mssv=mssv)
            db.add(profile)
            
        profile.cumulative_gpa = context["cumulative_gpa"]
        profile.total_earned_credits = context["total_earned_credits"]
        profile.passed_courses = context["passed_courses"]
        profile.current_courses = context["current_courses"]
        
        db.commit()
        db.refresh(profile)
        
        # Cập nhật bộ nhớ tạm tương thích ngược
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
                "total_passed": int(float(context["total_earned_credits"])) if context["total_earned_credits"] is not None else 0
            }
        }
        
    finally:
        # 4. Xóa file ngay lập tức để bảo mật thông tin và tiết kiệm ổ cứng
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/student-profile")
async def get_student_profile(
    mssv: str = Depends(get_current_mssv),
    db: Session = Depends(get_db)
):
    profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
    if not profile:
        return {
            "status": "NOT_FOUND",
            "message": "Sinh viên chưa tải lên bảng điểm.",
            "data": None
        }
    
    # Đồng bộ sang bộ nhớ tạm
    STUDENT_TRANSCRIPT_STORE[mssv] = {
        "cumulative_gpa": profile.cumulative_gpa,
        "total_earned_credits": profile.total_earned_credits,
        "passed_courses": profile.passed_courses,
        "current_courses": profile.current_courses,
        "major": profile.major,
        "target_career": profile.target_career,
        "interests": profile.interests
    }
    
    return {
        "status": "SUCCESS",
        "data": {
            "gpa": profile.cumulative_gpa,
            "total_passed": int(profile.total_earned_credits) if profile.total_earned_credits is not None else 0,
            "major": profile.major,
            "target_career": profile.target_career,
            "interests": profile.interests
        }
    }

@router.post("/student-profile/update")
async def update_student_profile(
    data: ProfileUpdateRequest,
    mssv: str = Depends(get_current_mssv),
    db: Session = Depends(get_db)
):
    profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
    if not profile:
        profile = StudentProfile(mssv=mssv)
        db.add(profile)
        
    if data.major is not None:
        profile.major = data.major
    if data.target_career is not None:
        profile.target_career = data.target_career
    if data.interests is not None:
        profile.interests = data.interests
        
    db.commit()
    db.refresh(profile)
    
    # Đồng bộ sang bộ nhớ tạm
    STUDENT_TRANSCRIPT_STORE[mssv] = {
        "cumulative_gpa": profile.cumulative_gpa,
        "total_earned_credits": profile.total_earned_credits,
        "passed_courses": profile.passed_courses,
        "current_courses": profile.current_courses,
        "major": profile.major,
        "target_career": profile.target_career,
        "interests": profile.interests
    }
    
    return {
        "status": "SUCCESS",
        "message": "Cập nhật hồ sơ cá nhân thành công!",
        "data": {
            "major": profile.major,
            "target_career": profile.target_career,
            "interests": profile.interests
        }
    }

@router.get("/careers")
async def get_all_careers():
    # Lấy danh sách các nghề đang lưu trong Neo4j để làm dropdown động
    cypher = "MATCH (j:JobRole) RETURN j.name AS name ORDER BY j.name"
    try:
        records = await graph_db.run_query_async(cypher)
        return {
            "status": "SUCCESS",
            "careers": [r["name"] for r in records]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi truy vấn cơ sở dữ liệu: {e}")

@router.delete("/student-profile")
async def delete_student_profile(
    mssv: str = Depends(get_current_mssv),
    db: Session = Depends(get_db)
):
    profile = db.query(StudentProfile).filter(StudentProfile.mssv == mssv).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Không tìm thấy bảng điểm để xóa.")
    
    db.delete(profile)
    db.commit()
    
    # Xóa khỏi bộ nhớ tạm
    if mssv in STUDENT_TRANSCRIPT_STORE:
        del STUDENT_TRANSCRIPT_STORE[mssv]
        
    return {
        "status": "SUCCESS",
        "message": "Đã xóa dữ liệu bảng điểm thành công!"
    }