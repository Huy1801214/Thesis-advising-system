import json
from typing import List, Dict, Any
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings

from core.graph_db import graph_db
from core.llm import build_chat_model
from services.maintenance.config import QDRANT_COLLECTION, QDRANT_URL
from orchestrator.student_analyst import StudentCompetencyProfile

class KnowledgeReasoningAgent:
    def __init__(self):
        self.llm = build_chat_model(temperature=0.2)
        
        # Thiết lập Qdrant Vector Store giống như RAG Engine
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large", 
            cache_folder="./hf_cache"
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        
        try:
            self.vectorstore = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode="hybrid",
                collection_name=QDRANT_COLLECTION,
                url=QDRANT_URL
            )
        except Exception as e:
            print(f"[ReasoningAgent] Qdrant connection warning: {e}")
            self.vectorstore = None

    async def find_matching_job(self, query: str) -> Dict[str, Any]:
        # Tìm kiếm JobRole gần nhất trong Neo4j
        cypher = """
        MATCH (j:JobRole)
        RETURN j.name AS name, j.description AS description, 
               j.market_demand AS market_demand, j.market_outlook AS market_outlook
        """
        records = await graph_db.run_query_async(cypher)
        if not records:
            return {}
            
        # Tìm match gần nhất dùng LLM hoặc keyword
        for r in records:
            if r["name"].lower() in query.lower() or query.lower() in r["name"].lower():
                return r
                
        # Dùng LLM làm fallback để chọn JobRole khớp nhất từ câu hỏi
        job_options = [r["name"] for r in records]
        prompt = f"""
Hãy chọn ra đúng 1 ngành nghề từ danh sách dưới đây khớp nhất với nội dung câu hỏi của sinh viên.
Nếu không có ngành nào khớp, trả về 'None'.
Danh sách ngành nghề: {job_options}
Câu hỏi: "{query}"

Trả về đúng tên ngành nghề được chọn hoặc 'None'. Không trả lời gì thêm ngoài tên ngành nghề.
"""
        try:
            res = await self.llm.ainvoke(prompt)
            selected = res.content.strip()
            for r in records:
                if r["name"] == selected:
                    return r
        except Exception:
            pass
            
        return records[0] # Mặc định trả về ngành đầu tiên nếu lỗi

    async def get_job_requirements(self, job_name: str) -> List[Dict[str, str]]:
        cypher = """
        MATCH (j:JobRole {name: $job_name})-[:REQUIRES_SKILL]->(c:Competency)
        RETURN c.name AS name, c.description AS description
        """
        records = await graph_db.run_query_async(cypher, {"job_name": job_name})
        return [{"name": r["name"], "description": r["description"]} for r in records]

    async def get_courses_teaching_competency(self, comp_name: str) -> List[Dict[str, Any]]:
        cypher = """
        MATCH (co:Course)-[:TEACHES_SKILL]->(c:Competency {name: $comp_name})
        RETURN co.code AS code, co.name AS name, co.credits AS credits
        """
        records = await graph_db.run_query_async(cypher, {"comp_name": comp_name})
        return [{"code": r["code"], "name": r["name"], "credits": r["credits"]} for r in records]

    async def get_course_prerequisites(self, course_code: str) -> List[str]:
        cypher = """
        MATCH (prev:Course)-[:PREVIOUS_OF]->(co:Course {code: $course_code})
        RETURN prev.code AS code
        """
        records = await graph_db.run_query_async(cypher, {"course_code": course_code})
        return [r["code"] for r in records]

    async def retrieve_market_insights(self, job_name: str) -> str:
        if not self.vectorstore:
            return "Không thể kết nối Vector DB."
        try:
            # Tìm kiếm hybrid trong Qdrant
            query_str = f"query: {job_name} xu hướng công nghệ chứng chỉ và lộ trình"
            docs = self.vectorstore.similarity_search(query_str, k=3)
            return "\n\n".join([f"[{d.metadata.get('source', 'Unknown')}] {d.page_content}" for d in docs])
        except Exception as e:
            return f"Lỗi truy xuất Qdrant: {str(e)}"

    async def reason_and_recommend(self, question: str, profile: StudentCompetencyProfile, passed_courses: List[str]) -> str:
        # 1. Nhận dạng ngành nghề mong muốn
        job_info = await self.find_matching_job(question)
        if not job_info:
            return "Chào bạn, hệ thống hiện tại chưa có dữ liệu đào tạo cho ngành nghề này. Bạn vui lòng chọn ngành khác hoặc liên hệ văn phòng khoa nhé."

        job_name = job_info["name"]
        
        # 2. Lấy danh sách kỹ năng cần có của ngành này
        req_comps = await self.get_job_requirements(job_name)
        
        # 3. Tính toán khoảng trống kỹ năng (Skill Gap)
        student_comps = {c.name for c in profile.acquired_competencies}
        gap_comps = []
        for req in req_comps:
            if req["name"] not in student_comps:
                gap_comps.append(req)

        # 4. Gợi ý môn học tương ứng và kiểm tra tiên quyết
        recommended_courses = []
        for comp in gap_comps:
            courses = await self.get_courses_teaching_competency(comp["name"])
            for c in courses:
                code = c["code"]
                name = c["name"]
                
                # Bỏ qua nếu sinh viên học môn này rồi
                if code in passed_courses:
                    continue
                    
                prereqs = await self.get_course_prerequisites(code)
                missing_prereqs = [p for p in prereqs if p not in passed_courses]
                
                status = "Sẵn sàng đăng ký học ngay" if not missing_prereqs else f"Cần tích lũy môn học trước: {', '.join(missing_prereqs)}"
                
                recommended_courses.append({
                    "competency": comp["name"],
                    "course_code": code,
                    "course_name": name,
                    "credits": c["credits"],
                    "status": status
                })

        # 5. Truy xuất xu hướng thị trường từ Qdrant
        market_insights = await self.retrieve_market_insights(job_name)

        # 6. Tổng hợp toàn bộ dữ liệu bằng LLM
        prompt = f"""
Bạn là Chuyên viên Tư vấn Học vụ & Hướng nghiệp đại học (Knowledge Reasoning Agent) của trường Đại học Nông Lâm TP.HCM (NLU).
Nhiệm vụ của bạn là tổng hợp các thông tin dưới đây để đưa ra lời khuyên lộ trình học tập và xu hướng thị trường chi tiết, thuyết phục và mang tính hành động cao cho sinh viên.

CÂU HỎI CỦA SINH VIÊN: "{question}"

1. HỒ SƠ SINH VIÊN:
- MSSV: {profile.mssv}
- GPA hiện tại: {profile.gpa}
- Số tín chỉ đã đạt: {profile.total_credits}
- Nhận xét học thuật: {profile.academic_summary}

2. THÔNG TIN NGÀNH NGHỀ ĐỊNH HƯỚNG:
- Tên nghề: {job_name}
- Mô tả: {job_info.get('description', '')}
- Nhu cầu tuyển dụng: {job_info.get('market_demand', 'Medium')}
- Triển vọng thị trường: {job_info.get('market_outlook', '')}

3. KHOẢNG TRỐNG KỸ NĂNG CỦA SINH VIÊN (Skill Gap):
{json.dumps(gap_comps, ensure_ascii=False, indent=2)}

4. GỢI Ý CÁC MÔN HỌC TẠI TRƯỜNG & TRẠNG THÁI TIÊN QUYẾT (Đây là nguồn môn học thực tế duy nhất có trong CSDL đào tạo):
{json.dumps(recommended_courses, ensure_ascii=False, indent=2)}

5. XU HƯỚNG THỊ TRƯỜNG VÀ CHỨNG CHỈ CÔNG NGHỆ (Từ các tài liệu thu thập):
{market_insights}

RÀNG BUỘC CỰC KỲ QUAN TRỌNG VỀ MÔN HỌC:
- Trong phần "LỘ TRÌNH MÔN HỌC ĐỀ XUẤT", bạn CHỈ ĐƯỢC PHÉP liệt kê các môn học có trong danh sách "4. GỢI Ý CÁC MÔN HỌC TẠI TRƯỜNG" ở trên.
- Tuyệt đối KHÔNG tự sáng tạo hay tự dịch các môn học khác không có trong danh sách trên.
- Nếu danh sách môn học gợi ý ở trên trống (tức là sinh viên đã hoàn thành tất cả các môn học liên quan đến nghề nghiệp này có trong chương trình học), hãy chúc mừng sinh viên đã tích lũy đầy đủ các học phần cốt lõi cho nghề '{job_name}', đề xuất họ tập trung vào học phần thực tập tốt nghiệp, làm khóa luận tốt nghiệp, hoặc chuẩn bị thi các chứng chỉ quốc tế liên quan. KHÔNG ĐƯỢC BIA RA MÔN HỌC NÀO KHÁC.

Yêu cầu định dạng câu trả lời bằng Markdown:
- Chia câu trả lời thành 3 phần rõ rệt:
  1. **📌 ĐÁNH GIÁ NĂNG LỰC HIỆN TẠI**: Nhận xét điểm mạnh của sinh viên dựa trên bảng điểm và so sánh mức độ tương thích với ngành nghề mong muốn.
  2. **📚 LỘ TRÌNH MÔN HỌC ĐỀ XUẤT**: Liệt kê các môn học cụ thể từ danh sách gợi ý ở trên để bù đắp kỹ năng thiếu hụt. Ghi rõ mã môn, số tín chỉ và trạng thái tiên quyết của môn đó. Nếu không có môn nào cần học thêm, hãy ghi nhận và chúc mừng sinh viên như hướng dẫn ở trên.
  3. **🌐 XU HƯỚNG THỊ TRƯỜNG & CHỨNG CHỈ QUỐC TẾ**: Tư vấn về xu hướng thị trường, tác động của công nghệ mới lên ngành này và khuyên học các chứng chỉ quốc tế phù hợp (AWS, MS Learn, roadmap.sh v.v.) dựa trên thông tin xu hướng.
- Sử dụng lối diễn đạt chuyên nghiệp, giàu thông tin và súc tích.
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
