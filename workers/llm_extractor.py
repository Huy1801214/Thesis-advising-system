# 

import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from workers import grag_tools 

CUSTOM_ALIASES = {
    "python": "Lập trình Python",
    "ctdl & gt": "Cấu trúc dữ liệu",
    "ctdl&gt": "Cấu trúc dữ liệu"
}

class EntityExtractor:
    def __init__(self):
        # 1. Khai báo rỗng (Lazy Loading). Tuyệt đối không tính toán gì trong __init__
        self.course_cache = None
        self.sorted_keywords = None
        
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    async def _build_dynamic_cache(self):
        print("[EntityExtractor] Đang xây dựng cache từ khóa môn học từ đồ thị...")
        cache = {}
        try:
            # Đã bổ sung await
            all_courses = await grag_tools.search_courses("", limit=2000)
            if not all_courses:
                print("Neo4j trả về rỗng. Vui lòng kiểm tra lại Database.")
        except Exception as e:
            print(f"Không thể lấy dữ liệu từ Neo4j để tạo Cache: {e}")
            all_courses = []

        for course in all_courses:
            code = str(course.get("code", "")).lower().strip()
            name = str(course.get("name", "")).lower().strip()
            
            if not code or not name:
                continue
            
            # 1. Thêm Tên đầy đủ -> Mã môn
            cache[name] = code
            
            # 2. Thêm Mã môn -> Mã môn
            cache[code] = code
            
            # 3. TỰ ĐỘNG SINH TỪ VIẾT TẮT 
            name_without_parens = re.sub(r'\(.*?\)|\[.*?\]', '', name)
            clean_name = re.sub(r'[^\w\s]', '', name_without_parens)
            words = clean_name.split()
            if len(words) >= 2:
                abbr = "".join([word[0] for word in words if word])
                cache[abbr] = code

        # 4. Gộp các từ dị biệt vào Cache (Đã fix logic bug)
        for alias, real_name in CUSTOM_ALIASES.items():
            # Truy cập trực tiếp tên môn học để lấy mã, O(1) thay vì lặp O(N)
            real_code = cache.get(real_name.lower())
            if real_code:
                cache[alias.lower()] = real_code

        # 5. Khởi tạo dữ liệu cho class
        self.course_cache = cache
        self.sorted_keywords = sorted(cache.keys(), key=len, reverse=True)
        print(f"[EntityExtractor] Cache hoàn tất với {len(self.sorted_keywords)} từ khóa.")

    async def extract_course_codes(self, question: str) -> list:
        # KÍCH HOẠT LAZY LOADING: Chỉ build cache ở lần hỏi đầu tiên
        if self.course_cache is None or self.sorted_keywords is None:
            await self._build_dynamic_cache()
        
        question_lower = question.lower()
        found_codes = set() 
        
        # --- BƯỚC 1: EXACT MATCH ---
        for kw in self.sorted_keywords:
            escaped_kw = re.escape(kw)
            if re.search(r'(?<!\w)' + escaped_kw + r'(?!\w)', question_lower):
                found_codes.add(self.course_cache[kw])
                question_lower = question_lower.replace(kw, "") 
                
        if found_codes:
            return list(found_codes)

        # --- BƯỚC 2: AI FALLBACK ---
        try:
            prompt = f"""
            Trích xuất TÊN môn học trong câu sau. 
            Tự dịch các từ viết tắt lạ sang tiếng Việt đầy đủ.
            Chỉ trả về mảng JSON. Ví dụ: ["Phân tích thiết kế hệ thống"]
            Câu hỏi: "{question}"
            """
            
            # Thay đổi thành ainvoke để tương thích Native Async
            res = await self.llm.ainvoke(prompt)
            res_ext = res.content
            match = re.search(r'\[.*\]', res_ext, re.DOTALL)
            
            if match:
                ai_names = json.loads(match.group(0))
                planned_codes = []
                for n in ai_names:
                    # Bổ sung await vì search_courses là hàm bất đồng bộ
                    f = await grag_tools.search_courses(n, limit=1)
                    if f: 
                        planned_codes.append(f[0]["code"])
                return planned_codes
            return []
            
        except Exception as e:
            print(f"[Lỗi AI Extractor]: {e}")
            return []