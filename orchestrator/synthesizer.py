import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class FinalSynthesizer:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    async def synthesize(self, original_question: str, trace: List[Dict[str, Any]]) -> str:
        prompt = f"""
Bạn là bộ tổng hợp câu trả lời cuối cùng cho hệ thống tư vấn học vụ.

Nhiệm vụ:
- Dựa trên câu hỏi gốc và kết quả từ các worker RAG/GRAG.
- Viết một câu trả lời liền mạch, dễ hiểu cho sinh viên.
- Không liệt kê máy móc tên worker nếu không cần.
- Nếu một worker báo thiếu dữ liệu, nói rõ sinh viên cần bổ sung gì.
- Không tự thêm thông tin ngoài các kết quả worker.

Câu hỏi gốc:
{original_question}

Kết quả trung gian:
{json.dumps(trace, ensure_ascii=False, indent=2)}
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
