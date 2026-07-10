import json
from typing import Any, Dict, List

from dotenv import load_dotenv
from core.llm import build_chat_model

from orchestrator.schemas import CriticReport

load_dotenv()


class CriticAgent:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.critic_chain = self.llm.with_structured_output(CriticReport, method="function_calling")

    async def review(
        self,
        original_question: str,
        final_answer: str,
        trace: List[Dict[str, Any]],
    ) -> CriticReport:
        prompt = f"""
Bạn là Critic Agent kiểm định chất lượng câu trả lời cuối cùng của hệ thống tư vấn học vụ.

NGUỒN SỰ THẬT DUY NHẤT:
- original_question: câu hỏi gốc của sinh viên.
- trace: câu hỏi con và câu trả lời của từng worker/agent.
- final_answer: câu trả lời cuối cùng cần kiểm định.

TUYỆT ĐỐI KHÔNG dùng kiến thức ngoài trace để đánh giá đúng/sai.
Bạn chỉ được kiểm tra xem final_answer có bám sát trace và trả lời đủ câu hỏi gốc không.

RUBRIC:
1. Mỗi trace[].sub_question phải được final_answer trả lời, hoặc giải thích vì sao chưa trả lời được.
2. final_answer không được kết luận vượt quá những gì trace[].raw_data hoặc trace[].answer cung cấp.
3. Nếu trace báo thiếu bảng điểm hoặc nhắc nhở về bảng điểm, final_answer phải nhắc rõ sinh viên.
4. Đối với câu hỏi hướng nghiệp/lộ trình (CAREER_ADVISE), final_answer phải hiển thị đúng lộ trình học tập, điều kiện tiên quyết và chứng chỉ hãng/xu hướng công nghệ lấy từ kết quả của agent.
5. final_answer không được mâu thuẫn với bất kỳ thông tin nào trong trace.
6. Nếu trace có worker/agent status = ERROR hoặc FAILED, final_answer phải nói rõ phần đó chưa xử lý được.
7. Nếu có nhiều worker/agent, final_answer phải tổng hợp liền mạch, không chỉ copy rời rạc.
8. Nếu dữ liệu chưa đủ, final_answer không được tự tin quá mức.

ISSUE TYPES:
- missing_sub_answer: thiếu trả lời một câu hỏi con.
- unsupported_claim: có kết luận không được trace hỗ trợ.
- contradiction: mâu thuẫn với trace.
- missing_required_context: không nhắc dữ liệu bắt buộc cần bổ sung hoặc bảng điểm.
- low_clarity: diễn đạt khó hiểu hoặc rời rạc.
- unsafe_overconfidence: khẳng định chắc chắn khi dữ liệu chưa đủ.
- other: lỗi khác.

OUTPUT:
- Nếu final_answer đạt yêu cầu: passed=true, score từ 0.85 đến 1.0, issues=[], revised_answer=null.
- Nếu chưa đạt: passed=false, score dưới 0.85, liệt kê issues cụ thể.
- Khi passed=false, revised_answer phải là câu trả lời đã sửa, bám sát trace, không tự thêm dữ liệu.

original_question:
{original_question}

trace:
{json.dumps(trace, ensure_ascii=False, indent=2)}

final_answer:
{final_answer}
"""
        return await self.critic_chain.ainvoke(prompt)


def fallback_critic_report(error: str = "critic_unavailable") -> CriticReport:
    return CriticReport(
        passed=True,
        score=1.0,
        issues=[],
        revised_answer=None,
        error=error,
    )
