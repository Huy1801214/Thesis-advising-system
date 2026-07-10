import os
import pandas as pd
import numpy as np
from markitdown import MarkItDown
from core.llm import build_chat_model
from pydantic import BaseModel, Field

class ExtractedTranscript(BaseModel):
    cumulative_gpa: float = Field(description="Điểm trung bình tích lũy hệ 4 (ví dụ: 2.64, 3.2). Nếu chỉ có hệ 10, hãy tự quy đổi sang hệ 4 bằng cách chia cho 2.5.")
    total_earned_credits: int = Field(description="Tổng số tín chỉ tích lũy/tín chỉ đạt (ví dụ: 137)")
    passed_courses: list[str] = Field(description="Danh sách mã môn học đã đạt (điểm chữ từ D trở lên, hoặc P, hoặc M)")
    failed_courses: list[str] = Field(description="Danh sách mã môn học bị rớt (điểm F, F+, V)")
    current_courses: list[str] = Field(description="Danh sách mã môn học chưa có điểm hoặc đang học trong kỳ này")

def extract_student_context(file_path):
    print(f"⏳ Đang đọc file: {file_path}...\n")
    
    file_lower = file_path.lower()
    
    # 1. Thử dùng Pandas parser nhanh cho Excel/CSV chuẩn của trường Nông Lâm
    if file_lower.endswith(('.xlsx', '.xls', '.csv')):
        try:
            if file_lower.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='utf-16')
                    
            # Check if columns look like default NLU format
            expected_cols = {'STT', 'Mã MH', 'Tên môn học'}
            if expected_cols.issubset(df.columns.tolist()) or ('Mã MH' in df.columns):
                passed_courses = []
                failed_courses = []
                current_courses = []
                cumulative_gpa = None
                total_earned_credits = None
                
                for index, row in df.iterrows():
                    stt = str(row.get('STT', '')).strip()
                    ma_mh = str(row.get('Mã MH', '')).strip()
                    
                    if stt == '- Điểm trung bình tích lũy hệ 4:' and cumulative_gpa is None:
                        cumulative_gpa = ma_mh
                    if stt == '- Số tín chỉ tích lũy:' and total_earned_credits is None:
                        total_earned_credits = ma_mh
                        
                    if not stt.isdigit():
                        continue
                    if pd.isna(row.get('Mã MH')) or ma_mh.lower() == 'nan' or not ma_mh:
                        continue
                        
                    diem_chu = str(row.get('Điểm TK (C)', '')).strip().upper()
                    if diem_chu in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'P', 'M']: 
                        passed_courses.append(ma_mh)
                    elif diem_chu in ['F', 'F+', 'V']: 
                        failed_courses.append(ma_mh)
                    else:
                        current_courses.append(ma_mh)
                
                if cumulative_gpa is not None and total_earned_credits is not None:
                    return {
                        "status": "success",
                        "cumulative_gpa": float(cumulative_gpa),
                        "total_earned_credits": int(float(total_earned_credits)),
                        "passed_courses": passed_courses,
                        "failed_courses": failed_courses,
                        "current_courses": current_courses
                    }
        except Exception as e:
            print(f"[Parser Warning] Pandas parsing skipped or failed: {e}. Falling back to MarkItDown.")
            
    # 2. Sử dụng MarkItDown + LLM cho các định dạng khác (PDF, Word, Doc, hoặc Excel có format lạ)
    try:
        print(f"👉 Khởi chạy MarkItDown + LLM cho file: {file_path}")
        md = MarkItDown()
        result = md.convert(file_path)
        markdown_text = result.text_content
        
        # Gọi LLM trích xuất thông tin
        llm = build_chat_model(temperature=0)
        structured_llm = llm.with_structured_output(ExtractedTranscript, method="function_calling")
        
        prompt = f"""
Bạn là trợ lý AI chuyên trách trích xuất thông tin bảng điểm học tập của sinh viên trường Đại học Nông Lâm.
Hãy phân tích nội dung Markdown dưới đây và trích xuất các thông tin chính xác theo định dạng yêu cầu.

Nội dung bảng điểm:
\"\"\"
{markdown_text}
\"\"\"
"""
        extracted = structured_llm.invoke(prompt)
        
        return {
            "status": "success",
            "cumulative_gpa": extracted.cumulative_gpa,
            "total_earned_credits": extracted.total_earned_credits,
            "passed_courses": extracted.passed_courses,
            "failed_courses": extracted.failed_courses,
            "current_courses": extracted.current_courses
        }
    except Exception as e:
        print(f"❌ [Parser Error] Cả hai phương pháp phân tích đều thất bại: {e}")
        return {"status": "error", "message": f"Không thể đọc file bảng điểm: {e}"}
