import os
import re
import json
import urllib3
import requests
import asyncio
from pathlib import Path
from core.llm import build_chat_model

# Tắt cảnh báo chứng chỉ SSL không hợp lệ
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "subject.json"

# Danh sách 37 môn học công khai trên web khoa CNTT Đại học Nông Lâm
SYLLABUS_LINKS = [
    {"name": "An toàn và bảo mật hệ thống", "url": "https://fit.hcmuaf.edu.vn/data/file/An%20toan%20bao%20mat%20he%20thong.doc"},
    {"name": "Bảo mật mạng và hệ thống", "url": "https://fit.hcmuaf.edu.vn/data/file/BaoMatMangVaHeThong.doc"},
    {"name": "Cấu trúc máy tính", "url": "https://fit.hcmuaf.edu.vn/data/file/Cau%20Truc%20May%20Tinh.doc"},
    {"name": "Cấu trúc dữ liệu", "url": "https://fit.hcmuaf.edu.vn/data/file/CauTrucDuLieu.doc"},
    {"name": "Chuyên đề hệ thống thông tin", "url": "https://fit.hcmuaf.edu.vn/data/file/Chuyen%20De%20He%20Thong%20Thong%20Tin.doc"},
    {"name": "Chuyên đề Web", "url": "https://fit.hcmuaf.edu.vn/data/file/Chuyen%20De%20Web.doc"},
    {"name": "Chuyên đề mạng máy tính", "url": "https://fit.hcmuaf.edu.vn/data/file/ChuyenDeMangMayTinh.doc"},
    {"name": "Đảm bảo chất lượng và kiểm thử phần mềm", "url": "https://fit.hcmuaf.edu.vn/data/file/Dam%20bao%20chat%20luong%20va%20kiem%20thu%20phan%20mem.doc"},
    {"name": "Đồ họa máy tính", "url": "https://fit.hcmuaf.edu.vn/data/file/Do%20Hoa%20May%20Tinh.doc"},
    {"name": "Giao tiếp người máy", "url": "https://fit.hcmuaf.edu.vn/data/file/GiaoTiepNguoiMay.doc"},
    {"name": "Hệ quản trị cơ sở dữ liệu", "url": "https://fit.hcmuaf.edu.vn/data/file/He%20Quan%20Tri%20Co%20So%20Du%20Lieu.doc"},
    {"name": "Hệ thống thông tin địa lý", "url": "https://fit.hcmuaf.edu.vn/data/file/He%20Thong%20Thong%20Tin%20Dia%20Ly.doc"},
    {"name": "Hệ thống thông tin quản lý", "url": "https://fit.hcmuaf.edu.vn/data/file/He%20thong%20thong%20tinquan%20ly.doc"},
    {"name": "Hệ điều hành nâng cao", "url": "https://fit.hcmuaf.edu.vn/data/file/HeDieuHanhNangCao.doc"},
    {"name": "Lập trình .Net", "url": "https://fit.hcmuaf.edu.vn/data/file/Lap%20Trinh%20_Net.doc"},
    {"name": "Lập trình cơ bản", "url": "https://fit.hcmuaf.edu.vn/data/file/Lap%20Trinh%20Co%20Ban.doc"},
    {"name": "Lập trình hệ thống nhúng", "url": "https://fit.hcmuaf.edu.vn/data/file/Lap%20Trinh%20He%20Thong%20Nhung.doc"},
    {"name": "Lập trình mạng nâng cao", "url": "https://fit.hcmuaf.edu.vn/data/file/Lap%20Trinh%20Mang%20Nang%20Cao.doc"},
    {"name": "Lập trình ứng dụng web", "url": "https://fit.hcmuaf.edu.vn/data/file/Lap%20trinh%20ung%20dung%20web.doc"},
    {"name": "Lập trình mạng cơ bản", "url": "https://fit.hcmuaf.edu.vn/data/file/LapTringMangCoBan.doc"},
    {"name": "Lập trình C++Linux", "url": "https://fit.hcmuaf.edu.vn/data/file/LapTrinhC++Linux.doc"},
    {"name": "Lập trình J2EE", "url": "https://fit.hcmuaf.edu.vn/data/file/LapTrinhJ2EE.doc"},
    {"name": "Lập trình nâng cao", "url": "https://fit.hcmuaf.edu.vn/data/file/LapTrinhNangCao.doc"},
    {"name": "Lý thuyết đồ thị", "url": "https://fit.hcmuaf.edu.vn/data/file/Ly%20Thuyet%20Do%20Thi.doc"},
    {"name": "Mã nguồn mở", "url": "https://fit.hcmuaf.edu.vn/data/file/Ma%20Nguon%20Mo.doc"},
    {"name": "Mạng máy tính cơ bản", "url": "https://fit.hcmuaf.edu.vn/data/file/MangMayTinhCoBan.doc"},
    {"name": "Mạng máy tính nâng cao", "url": "https://fit.hcmuaf.edu.vn/data/file/MangMayTinhNangCao.doc"},
    {"name": "Nhập môn cơ cở dữ liệu", "url": "https://fit.hcmuaf.edu.vn/data/file/Nhap%20Mon%20Co%20So%20Du%20Lieu.doc"},
    {"name": "Nhập môn công nghệ phần mềm", "url": "https://fit.hcmuaf.edu.vn/data/file/Nhap%20mon%20Cong%20nghe%20Phan%20Mem.doc"},
    {"name": "Nhập môn hệ điều hành", "url": "https://fit.hcmuaf.edu.vn/data/file/NhapMonHeDieuHanh.doc"},
    {"name": "Phân tích và thiết kế hệ thống", "url": "https://fit.hcmuaf.edu.vn/data/file/Phan%20Tich%20va%20Thiet%20Ke%20He%20Thong.doc"},
    {"name": "Quản lý dự án phần mềm", "url": "https://fit.hcmuaf.edu.vn/data/file/Quan%20Ly%20Du%20An%20Phan%20Mem.doc"},
    {"name": "Quản trị mạng", "url": "https://fit.hcmuaf.edu.vn/data/file/QuanTriMang.doc"},
    {"name": "Thị trường CNTT", "url": "https://fit.hcmuaf.edu.vn/data/file/Thi%20Truong%20CNTT.doc"},
    {"name": "Thiết kế hướng đối tượng", "url": "https://fit.hcmuaf.edu.vn/data/file/ThietKeHuongDoiTuong.doc"},
    {"name": "Tin học đại cương", "url": "https://fit.hcmuaf.edu.vn/data/file/Tin%20hoc%20dai%20cuong.doc"},
    {"name": "Trí tuệ nhân tạo", "url": "https://fit.hcmuaf.edu.vn/data/file/Tri%20Tue%20Nhan%20Tao.doc"}
]

class NLUSyllabusCrawler:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.temp_dir = BASE_DIR / "scratch" / "syllabi"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_doc(self, file_path: Path) -> str:
        # Sử dụng logic bóc tách stream unicode (UTF-16LE) đã được test thành công
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            text = data.decode('utf-16le', errors='ignore')
            # Lọc bỏ các ký tự điều khiển, giữ chữ tiếng Việt, alphabet, số và dấu câu cơ bản
            clean_text = re.sub(r'[^\x20-\x7E\u00C0-\u1EF9\n\t\r]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text
        except Exception as e:
            print(f"Lỗi trích xuất {file_path}: {e}")
            return ""

    def download_doc_file(self, url: str, name: str) -> Path:
        local_path = self.temp_dir / f"{name.replace(' ', '_')}.doc"
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            resp = requests.get(url, headers=headers, verify=False, timeout=15)
            if resp.status_code == 200:
                local_path.write_bytes(resp.content)
                return local_path
        except Exception as e:
            print(f"Lỗi tải {name}: {e}")
        return None

    def clean_string_for_match(self, s: str) -> str:
        s = s.lower().strip()
        # Loại bỏ các từ đại cương hoặc hậu tố (A), (B) để matching chuẩn xác
        s = re.sub(r'\(.*?\)|\[.*?\]', '', s)
        s = re.sub(r'[^a-z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', s)
        return s.strip()

    async def generate_summary(self, name: str, raw_text: str) -> str:
        # Gửi 2500 ký tự đầu tiên (đủ chứa thông tin giảng viên, mục tiêu môn học, giáo trình)
        context = raw_text[:2500]
        prompt = f"""
Dưới đây là phần nội dung đề cương thô bóc tách từ file Word chính thức của môn học '{name}' tại Khoa CNTT Trường Đại học Nông Lâm TP.HCM.

Nội dung đề cương thô:
---
{context}
---

Nhiệm vụ:
- Hãy viết một đoạn mô tả môn học súc tích (khoảng 35-50 từ) bằng tiếng Việt.
- Mô tả phải bao gồm: Các kiến thức trọng tâm được dạy (ví dụ: thiết kế database, cấu hình Cisco, giải thuật tìm kiếm, OOP) và các công cụ thực hành nếu có.
- Chỉ trả về ĐÚNG đoạn mô tả môn học. Không viết thêm lời chào hay giải thích gì khác ngoài đoạn mô tả.
"""
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Lỗi sinh mô tả bằng LLM cho {name}: {e}")
            return ""

    async def run(self):
        if not JSON_PATH.exists():
            print(f"❌ Không tìm thấy subject.json tại: {JSON_PATH}")
            return

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            subjects = json.load(f)

        print(f"🚀 Bắt đầu tải và phân tích {len(SYLLABUS_LINKS)} đề cương chi tiết từ Khoa CNTT Nông Lâm...")
        
        updated_count = 0
        for item in SYLLABUS_LINKS:
            name = item["name"]
            url = item["url"]
            
            # A. Tìm kiếm môn học tương ứng trong subject.json
            matched_subject = None
            clean_item_name = self.clean_string_for_match(name)
            
            for s in subjects:
                clean_subj_name = self.clean_string_for_match(s["course_name"])
                # So khớp gần đúng (nếu tên này là con của tên kia hoặc ngược lại)
                if clean_item_name in clean_subj_name or clean_subj_name in clean_item_name:
                    matched_subject = s
                    break
                    
            if not matched_subject:
                print(f"⚠️ Bỏ qua '{name}' (Không có môn tương ứng trong subject.json)")
                continue

            print(f"⚡ Đang xử lý: '{name}' -> Môn học trong DB: '{matched_subject['course_name']}'")
            
            # B. Tải file đề cương (.doc)
            file_path = self.download_doc_file(url, name)
            if not file_path:
                continue
                
            # C. Trích xuất text
            raw_text = self.extract_text_from_doc(file_path)
            if not raw_text or len(raw_text) < 100:
                print(f"⚠️ Không có nội dung text hợp lệ cho {name}")
                continue
                
            # D. Gọi LLM sinh mô tả chuẩn
            description = await self.generate_summary(name, raw_text)
            if description:
                matched_subject["description"] = description
                updated_count += 1
                print(f"   -> Đã cập nhật đề cương gốc: {description[:60]}...")
                
            # Lưu lại sau mỗi môn học đề phòng đứt quãng mạng
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(subjects, f, ensure_ascii=False, indent=2)
                
            # Dọn dẹp file tạm
            if file_path.exists():
                file_path.unlink()

        print(f"\n✨ Đã cào và cập nhật thành công đề cương thực tế cho {updated_count} môn học của trường Nông Lâm!")

if __name__ == "__main__":
    crawler = NLUSyllabusCrawler()
    asyncio.run(crawler.run())
