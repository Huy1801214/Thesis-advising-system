import pandas as pd
import numpy as np

def extract_student_context(file_path):
    print(f"⏳ Đang đọc file: {file_path}...\n")
    
    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='utf-16')

        passed_courses = []
        failed_courses = []
        current_courses = []
        
        cumulative_gpa = None
        total_earned_credits = None

        for index, row in df.iterrows():
            stt = str(row.get('STT', '')).strip()
            ma_mh = str(row.get('Mã MH', '')).strip()
            ten_mh = str(row.get('Tên môn học', '')).strip()
            
            # 1. BẮT CHỈ SỐ TÍCH LŨY
            if stt == '- Điểm trung bình tích lũy hệ 4:' and cumulative_gpa is None:
                cumulative_gpa = ma_mh
            if stt == '- Số tín chỉ tích lũy:' and total_earned_credits is None:
                total_earned_credits = ma_mh

            # 2. BỘ LỌC MÔN HỌC
            if not stt.isdigit():
                continue

            if pd.isna(row.get('Mã MH')) or ma_mh.lower() == 'nan' or not ma_mh:
                continue

            # 3. XỬ LÝ ĐIỂM CHỮ VÀ PHÂN LOẠI
            diem_chu = str(row.get('Điểm TK (C)', '')).strip().upper()

            # Thêm 'M' (Miễn) vào danh sách điểm đậu
            if diem_chu in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'P', 'M']: 
                passed_courses.append(ma_mh)
            elif diem_chu in ['F', 'F+', 'V']: 
                failed_courses.append(ma_mh)
            else:
                current_courses.append(ma_mh)

        return {
            "status": "success",
            "cumulative_gpa": cumulative_gpa,
            "total_earned_credits": total_earned_credits,
            "passed_courses": passed_courses,
            "failed_courses": failed_courses,
            "current_courses": current_courses
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
