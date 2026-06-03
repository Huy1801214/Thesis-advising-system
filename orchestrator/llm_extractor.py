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
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # 1. Tự động xây dựng Từ điển từ Neo4j khi khởi động
        self.course_cache = self._build_dynamic_cache()
        # 2. Sắp xếp từ khóa theo độ dài giảm dần 
        self.sorted_keywords = sorted(self.course_cache.keys(), key=len, reverse=True)

    def _build_dynamic_cache(self):
        cache = {}
        try:
            all_courses = grag_tools.search_courses("", limit=2000)
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

        # 4. Gộp các từ dị biệt vào Cache
        for alias, real_name in CUSTOM_ALIASES.items():
            for c_code, c_name in cache.items(): 
                if c_name == real_name.lower():
                    cache[alias.lower()] = c_code
                    break
                    
        return cache

    def extract_course_codes(self, question: str) -> list:
        question_lower = question.lower()
        found_codes = set() 
        
        for kw in self.sorted_keywords:
            escaped_kw = re.escape(kw)
            if re.search(r'(?<!\w)' + escaped_kw + r'(?!\w)', question_lower):
                found_codes.add(self.course_cache[kw])
                question_lower = question_lower.replace(kw, "") 
                
        if found_codes:
            return list(found_codes)

        try:
            prompt = f"""
            Trích xuất TÊN môn học trong câu sau. 
            Tự dịch các từ viết tắt lạ sang tiếng Việt đầy đủ.
            Chỉ trả về mảng JSON. Ví dụ: ["Phân tích thiết kế hệ thống"]
            Câu hỏi: "{question}"
            """
            res_ext = self.llm.invoke(prompt).content
            match = re.search(r'\[.*\]', res_ext, re.DOTALL)
            
            if match:
                ai_names = json.loads(match.group(0))
                planned_codes = []
                for n in ai_names:
                    f = grag_tools.search_courses(n, limit=1)
                    if f: planned_codes.append(f[0]["code"])
                return planned_codes
            return []
        except Exception as e:
            print(f"[Lỗi AI Extractor]: {e}")
            return []
