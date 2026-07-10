import os
import json
import asyncio
from pathlib import Path
from core.llm import build_chat_model

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "subject.json"

class SubjectEnricher:
    def __init__(self):
        self.llm = build_chat_model(temperature=0.1)

    async def enrich_descriptions(self):
        if not JSON_PATH.exists():
            print(f"❌ Không tìm thấy file subject.json tại: {JSON_PATH}")
            return

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            subjects = json.load(f)

        print(f"🚀 Tìm thấy {len(subjects)} môn học trong subject.json. Bắt đầu tạo mô tả...")

        # Chia thành các batch nhỏ (mỗi batch 10 môn) để gửi cho LLM xử lý nhanh
        batch_size = 10
        for i in range(0, len(subjects), batch_size):
            batch = subjects[i:i+batch_size]
            
            # Chỉ tạo mô tả cho những môn chưa có description
            todo_batch = [s for s in batch if "description" not in s or not s["description"]]
            if not todo_batch:
                print(f"   [Batch {i//batch_size + 1}] Đã có sẵn mô tả. Bỏ qua.")
                continue

            print(f"   [Batch {i//batch_size + 1}] Đang xử lý {len(todo_batch)} môn học...")
            
            courses_str = "\n".join([f"- {s['course_code']}: {s['course_name']}" for s in todo_batch])
            
            prompt = f"""
Bạn là Chuyên gia Đào tạo Học thuật tại trường Đại học Nông Lâm TP.HCM (NLU).
Nhiệm vụ của bạn là viết mô tả môn học (syllabus/đề cương chi tiết tóm tắt) bằng tiếng Việt cho danh sách các môn học dưới đây.

Yêu cầu về mô tả:
- Súc tích (khoảng 30-50 từ cho mỗi môn).
- Thể hiện rõ các chủ đề kiến thức cốt lõi và kỹ năng thực hành sinh viên sẽ đạt được sau khi học xong.
- Định dạng mô tả chính xác theo khung chương trình đào tạo của ngành Công nghệ thông tin / Kỹ thuật phần mềm / Hệ thống thông tin tại NLU.

Danh sách môn học cần viết mô tả (Mã môn: Tên môn):
{courses_str}

Hãy trả về kết quả dưới dạng một đối tượng JSON có khóa là mã môn học (course_code) và giá trị là chuỗi mô tả môn học đó.
Ví dụ:
{{
  "200105": "Môn học cung cấp kiến thức về thiết kế cơ sở dữ liệu quan hệ, thiết kế sơ đồ ERD, ngôn ngữ truy vấn SQL, và các mức chuẩn hóa cơ sở dữ liệu.",
  "214271": "Môn học giảng dạy các khái niệm cơ bản về mạng máy tính, quản trị mạng, cấu hình thiết bị Switch/Router và xử lý sự cố mạng."
}}

Tuyệt đối không viết thêm lời giải thích gì ngoài chuỗi JSON hợp lệ.
"""
            try:
                response = await self.llm.ainvoke(prompt)
                content = response.content.strip()
                
                # Loại bỏ định dạng markdown code block nếu có
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                    
                descriptions_map = json.loads(content)
                
                # Cập nhật mô tả vào danh sách môn học gốc
                for s in subjects:
                    code = s["course_code"]
                    if code in descriptions_map:
                        s["description"] = descriptions_map[code]
                        
                print(f"   [Batch {i//batch_size + 1}] Thành công!")
                
                # Lưu tạm sau mỗi batch để tránh mất mát dữ liệu nếu lỗi
                with open(JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(subjects, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"❌ Lỗi xử lý tại Batch {i//batch_size + 1}: {e}")
                # Đợi một chút rồi thử lại hoặc tiếp tục
                await asyncio.sleep(2)

        print("✨ Hoàn tất tiến trình làm giàu mô tả môn học!")

if __name__ == "__main__":
    enricher = SubjectEnricher()
    asyncio.run(enricher.enrich_descriptions())
