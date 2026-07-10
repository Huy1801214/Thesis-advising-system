import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from core.llm import build_chat_model

load_dotenv()


class FinalSynthesizer:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)

    async def synthesize(self, original_question: str, trace: List[Dict[str, Any]]) -> str:
        # Kiểm tra xem có tác vụ tư vấn nghề nghiệp trong trace không
        is_career_advise = any(t.get("task_type") == "CAREER_ADVISE" for t in trace)
        
        if is_career_advise:
            prompt = f"""
Bạn là bộ tổng hợp câu trả lời hướng nghiệp cho hệ thống tư vấn học vụ.

Nhiệm vụ:
- Đây là kết quả tư vấn hướng nghiệp học tập (CAREER_ADVISE).
- Hãy giữ nguyên cấu trúc 3 phần từ kết quả của agent CAREER_ADVISE cung cấp:
  1. **📌 ĐÁNH GIÁ NĂNG LỰC HIỆN TẠI**
  2. **📚 LỘ TRÌNH MÔN HỌC ĐỀ XUẤT**
  3. **🌐 XU HƯỚNG THỊ TRƯỜNG & CHỨNG CHỈ QUỐC TẾ**
- Bạn chỉ cần thêm một câu chào hỏi thân thiện ở đầu và lời chúc học tập thành công ở cuối một cách tự nhiên.
- Tuyệt đối không tự ý bịa thêm môn học hay chứng chỉ hãng ngoài những gì kết quả của agent cung cấp.

Câu hỏi gốc của sinh viên:
{original_question}

Kết quả của Agent CAREER_ADVISE:
{json.dumps(trace, ensure_ascii=False, indent=2)}
"""
        else:
            prompt = f"""
Bạn là bộ tổng hợp câu trả lời cuối cùng cho hệ thống tư vấn học vụ.

Nhiệm vụ:
- Dựa trên câu hỏi gốc và kết quả từ các worker RAG/GRAG.
- Trình bày câu trả lời một cách khoa học, rõ ràng và mạch lạc:
  + Tách biệt các ý bằng các đoạn văn ngắn hoặc sử dụng danh sách gạch đầu dòng (bullet points) để dễ theo dõi.
  + Luôn sử dụng xuống dòng thích hợp để cấu trúc câu trả lời thoáng và chuyên nghiệp. Tuyệt đối không viết dồn mọi ý vào một khối văn bản dài liên tục.
- Định dạng cấu trúc câu trả lời thành 2 phần rõ rệt bằng Markdown:
  1. **📌 CÂU TRẢ LỜI CHI TIẾT**: Trình bày rõ ràng câu trả lời tương ứng với câu hỏi của sinh viên.
  2. **📚 DẪN CHỨNG / NGUỒN TRÍCH DẪN**: Liệt kê rõ các căn cứ pháp lý, quy chế (ví dụ: "Điều 10, Chương II, Sổ tay sinh viên", "Trích từ quy chế học vụ...", v.v.).
- Nếu một worker báo thiếu dữ liệu (ví dụ thiếu bảng điểm), hãy ghi chú rõ sinh viên cần bổ sung thông tin gì để hệ thống trả lời chính xác hơn.
- Không liệt kê máy móc tên các worker (như "worker RAG", "GRAG") trong câu trả lời.
- Không tự ý bịa đặt hoặc thêm thông tin nằm ngoài kết quả từ các worker cung cấp.

Câu hỏi gốc:
{original_question}

Kết quả trung gian:
{json.dumps(trace, ensure_ascii=False, indent=2)}
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
