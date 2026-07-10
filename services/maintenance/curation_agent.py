import os
import json
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path

from core.llm import build_chat_model
from core.graph_db import graph_db
from services.maintenance.config import SUPPORTED_CAREERS, TRUSTED_SOURCES, CANDIDATE_UPDATES_FILE

# Pydantic models for structured LLM extraction
class CompetencyExtraction(BaseModel):
    name: str = Field(description="Tên kỹ năng hoặc năng lực học thuật/kỹ thuật (ví dụ: Lập trình Python, Quản lý Cloud)")
    description: str = Field(description="Mô tả ngắn gọn về năng lực này và những kiến thức cốt lõi cần đạt được")

class MarketArticle(BaseModel):
    title: str = Field(description="Tiêu đề bài viết về xu hướng thị trường, tác động công nghệ toàn cầu hoặc chứng chỉ liên quan")
    content: str = Field(description="Nội dung chi tiết phân tích xu hướng, tác động (ví dụ: AI ảnh hưởng thế nào đến nghề này) hoặc lộ trình học chứng chỉ hãng")
    source: str = Field(description="Tên nguồn tin cậy xuất xứ")

class CareerProfile(BaseModel):
    name: str = Field(description="Tên ngành nghề/vị trí công việc (ví dụ: Kỹ sư DevOps, Khoa học Dữ liệu)")
    description: str = Field(description="Mô tả tổng quan về công việc và vai trò của vị trí này trong thực tế")
    market_demand: str = Field(description="Nhu cầu thị trường hiện tại (High, Medium, Low)")
    market_outlook: str = Field(description="Đánh giá triển vọng nghề nghiệp toàn cầu và các xu hướng tuyển dụng mới nhất")
    competencies: List[CompetencyExtraction] = Field(description="Danh sách các kỹ năng/năng lực cốt lõi bắt buộc phải có cho nghề nghiệp này")
    market_articles: List[MarketArticle] = Field(description="Các tài liệu/bài viết tóm tắt xu hướng thị trường, chứng chỉ quốc tế và biến động công nghệ toàn cầu liên quan")

class CurationResult(BaseModel):
    careers: List[CareerProfile] = Field(description="Danh sách hồ sơ nghề nghiệp đã được thu thập và cấu trúc hóa")

class KnowledgeCurationAgent:
    def __init__(self):
        self.llm = build_chat_model(temperature=0.1)
        self.curator_chain = self.llm.with_structured_output(CurationResult, method="function_calling")
        self.crawled_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "crawled_articles"

    def load_crawled_articles(self) -> str:
        if not self.crawled_dir.exists():
            return "Không có bài viết cào được."
            
        articles = []
        for file_path in self.crawled_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    articles.append(f.read())
            except Exception as e:
                print(f"[CurationAgent] Lỗi đọc file {file_path}: {e}")
                
        if not articles:
            return "Không có bài viết cào được."
            
        # Nối các bài viết thành một khối ngữ cảnh phân tích
        return "\n\n=== BÀI VIẾT TIẾP THEO ===\n\n".join(articles)

    async def get_existing_careers(self) -> List[str]:
        # Truy vấn các JobRole hiện tại từ Neo4j để làm nguồn tri thức cũ đối chiếu
        cypher = "MATCH (j:JobRole) RETURN j.name AS name"
        try:
            records = await graph_db.run_query_async(cypher)
            return [r["name"] for r in records]
        except Exception as e:
            print(f"[CurationAgent] Không thể lấy các JobRole cũ từ Neo4j: {e}")
            return SUPPORTED_CAREERS

    async def curate_career_knowledge(self) -> CurationResult:
        print("[CurationAgent] Đang nạp các bài báo đã cào...")
        crawled_content = self.load_crawled_articles()
        
        print("[CurationAgent] Đang truy vấn tri thức nghề nghiệp hiện có trong Neo4j...")
        existing_careers = await self.get_existing_careers()
        
        print(f"[CurationAgent] Tri thức hiện có gồm {len(existing_careers)} ngành nghề: {existing_careers}")
        
        prompt = f"""
Bạn là Knowledge Curation Agent chịu trách nhiệm thu thập, phân tích và cấu trúc hóa tri thức hướng nghiệp.

Nhiệm vụ cốt lõi:
1. Đọc nội dung các bài báo công nghệ mới cào được ở phần dưới.
2. So sánh với danh sách ngành nghề hiện có trong hệ thống: {existing_careers}.
3. Thực hiện phân tích:
   - **Đánh giá cập nhật**: Nếu bài viết đề cập đến xu hướng mới, kỹ năng mới, hoặc chứng chỉ mới liên quan đến một ngành nghề hiện có, hãy cập nhật hồ sơ ngành nghề đó.
   - **Phát hiện ngành nghề mới (Emerging Careers)**: Nếu bài viết thảo luận về một vị trí công việc công nghệ mới đang thịnh hành (ví dụ: Kỹ sư AI, Kỹ sư Dữ liệu lớn, Kỹ sư Điện toán đám mây, v.v.) chưa có trong hệ thống, hãy phân tích để **TỰ ĐỘNG TẠO MỚI** hồ sơ cho ngành nghề đó.
4. Trích xuất các competencies cụ thể (ví dụ: thay vì ghi chung chung 'Programming', hãy ghi rõ kỹ năng 'Lập trình Python', 'Kiến trúc mạng', 'Quản trị đám mây') để có thể đối chiếu chính xác sang các môn học đại học sau này.

Nội dung các bài báo mới cào được:
---
{crawled_content}
---

Hãy trả về kết quả cấu trúc CurationResult (Tiếng Việt) chứa tất cả các hồ sơ ngành nghề cần cập nhật hoặc thêm mới.
"""
        try:
            result = await self.curator_chain.ainvoke(prompt)
            print(f"[CurationAgent] Đã phân tích xong. Trích xuất được {len(result.careers)} hồ sơ ngành nghề để cập nhật/thêm mới.")
            return result
        except Exception as e:
            print(f"[CurationAgent] Lỗi chạy LLM Curation: {e}")
            raise e

    def save_candidate_updates(self, result: CurationResult):
        CANDIDATE_UPDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = result.model_dump()
        with open(CANDIDATE_UPDATES_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        print(f"[CurationAgent] Đã lưu dữ liệu cập nhật vào: {CANDIDATE_UPDATES_FILE}")

async def main():
    agent = KnowledgeCurationAgent()
    res = await agent.curate_career_knowledge()
    agent.save_candidate_updates(res)

if __name__ == "__main__":
    asyncio.run(main())
