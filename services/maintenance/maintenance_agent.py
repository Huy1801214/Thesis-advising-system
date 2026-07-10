import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse

from core.graph_db import graph_db
from core.llm import build_chat_model
from services.maintenance.config import (
    CANDIDATE_UPDATES_FILE,
    MAINTENANCE_LOG_FILE,
    QDRANT_COLLECTION,
    QDRANT_URL
)

class KnowledgeMaintenanceAgent:
    def __init__(self):
        self.llm = build_chat_model(temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large", 
            cache_folder="./hf_cache"
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    def _log_message(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(message)
        
        MAINTENANCE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MAINTENANCE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)

    async def get_db_state(self) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[str, str]]]:
        # 1. Lấy tất cả JobRole hiện tại
        jobs = await graph_db.run_query_async("MATCH (j:JobRole) RETURN j.name AS name, j.description AS desc")
        jobs_dict = {j["name"]: j["desc"] for j in jobs}
        
        # 2. Lấy tất cả Competency hiện tại
        comps = await graph_db.run_query_async("MATCH (c:Competency) RETURN c.name AS name, c.description AS desc")
        comps_dict = {c["name"]: c["desc"] for c in comps}
        
        # 3. Lấy tất cả quan hệ REQUIRES_SKILL
        rels = await graph_db.run_query_async("""
        MATCH (j:JobRole)-[:REQUIRES_SKILL]->(c:Competency) 
        RETURN j.name AS job, c.name AS comp
        """)
        rels_list = [(r["job"], r["comp"]) for r in rels]
        
        return jobs_dict, comps_dict, rels_list

    async def get_all_courses(self) -> List[Dict[str, str]]:
        cypher = "MATCH (c:Course) RETURN c.code AS code, c.name AS name, c.description AS desc"
        records = await graph_db.run_query_async(cypher)
        return [{"code": r["code"], "name": r["name"], "description": r["desc"]} for r in records]

    async def map_competency_to_courses(self, competency_name: str, competency_desc: str, courses: List[Dict[str, str]]) -> List[str]:
        courses_str = "\n".join([f"- {c['code']}: {c['name']} (Đề cương: {c.get('description', '')})" for c in courses])
        
        prompt = f"""
Nhiệm vụ: Đối chiếu kỹ năng nghề nghiệp với chương trình môn học.
Hãy chọn ra từ danh sách môn học dưới đây những môn trực tiếp giảng dạy hoặc cung cấp kiến thức nền tảng bắt buộc để đạt được kỹ năng '{competency_name}' (Mô tả: {competency_desc}).
Danh sách môn học:
{courses_str}

Yêu cầu đầu ra:
- Chọn tối đa 5 môn học trực tiếp nhất. Nếu không có môn nào liên quan, trả về [].
- Chỉ trả về MẢNG JSON các mã môn học (ví dụ: ["200105", "200107"]).
- Tuyệt đối không giải thích gì thêm ngoài mảng JSON.
"""
        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            matched_codes = json.loads(content)
            return matched_codes
        except Exception as e:
            self._log_message(f"[Error] Không thể ánh xạ competency '{competency_name}' sang môn học: {e}")
            return []

    def generate_change_report(self, candidate_data: Dict[str, Any], db_jobs: Dict[str, str], db_comps: Dict[str, str], db_rels: List[Tuple[str, str]]) -> Dict[str, Any]:
        report = {
            "new_careers": [],
            "updated_careers": [],
            "new_competencies": [],
            "new_relationships": []
        }
        
        for career in candidate_data.get("careers", []):
            name = career["name"]
            desc = career.get("description", "")
            
            # Kiểm tra xem ngành nghề đã có trong database chưa
            if name not in db_jobs:
                report["new_careers"].append({
                    "name": name,
                    "description": desc
                })
            else:
                # Ngành đã có, kiểm tra xem mô tả có thay đổi không
                if db_jobs[name] != desc:
                    report["updated_careers"].append({
                        "name": name,
                        "old_description": db_jobs[name],
                        "new_description": desc
                    })
            
            # Kiểm tra năng lực/kỹ năng của ngành
            for comp in career.get("competencies", []):
                comp_name = comp["name"]
                comp_desc = comp.get("description", "")
                
                # Kiểm tra Competency mới
                if comp_name not in db_comps:
                    report["new_competencies"].append({
                        "name": comp_name,
                        "description": comp_desc
                    })
                
                # Kiểm tra quan hệ REQUIRES_SKILL mới
                if (name, comp_name) not in db_rels:
                    report["new_relationships"].append({
                        "job": name,
                        "competency": comp_name
                    })
                    
        return report

    async def apply_neo4j_updates(self, career_data: Dict[str, Any], course_catalog: List[Dict[str, str]]):
        career_name = career_data["name"]
        
        # 1. MERGE JobRole
        await graph_db.run_query_async("""
        MERGE (j:JobRole {name: $name})
        SET j.description = $description,
            j.market_demand = $market_demand,
            j.market_outlook = $market_outlook
        """, {
            "name": career_name,
            "description": career_data.get("description", ""),
            "market_demand": career_data.get("market_demand", "Medium"),
            "market_outlook": career_data.get("market_outlook", "")
        })

        # 2. MERGE Competency & REQUIRES_SKILL, TEACHES_SKILL
        for comp in career_data.get("competencies", []):
            comp_name = comp["name"]
            comp_desc = comp["description"]
            
            await graph_db.run_query_async("""
            MERGE (c:Competency {name: $name})
            SET c.description = $description
            """, {
                "name": comp_name,
                "description": comp_desc
            })
            
            await graph_db.run_query_async("""
            MATCH (j:JobRole {name: $job_name}), (c:Competency {name: $comp_name})
            MERGE (j)-[:REQUIRES_SKILL]->(c)
            """, {
                "job_name": career_name,
                "comp_name": comp_name
            })
            
            # Ánh xạ kỹ năng sang môn học đại học
            matched_courses = await self.map_competency_to_courses(comp_name, comp_desc, course_catalog)
            for course_code in matched_courses:
                await graph_db.run_query_async("""
                MATCH (co:Course {code: $course_code}), (c:Competency {name: $comp_name})
                MERGE (co)-[:TEACHES_SKILL]->(c)
                """, {
                    "course_code": course_code,
                    "comp_name": comp_name
                })

    async def update_qdrant(self, all_articles: List[Document]):
        if not all_articles:
            return
        try:
            QdrantVectorStore.from_documents(
                documents=all_articles,
                embedding=self.embeddings,
                sparse_embedding=self.sparse_embeddings,
                retrieval_mode="hybrid",
                url=QDRANT_URL,
                collection_name=QDRANT_COLLECTION,
                force_recreate=True,
                batch_size=50,
                timeout=300
            )
            self._log_message("-> Đồng bộ Qdrant thành công!")
        except Exception as e:
            self._log_message(f"[Error] Không thể nạp Qdrant: {e}")

    async def run_maintenance(self):
        if not CANDIDATE_UPDATES_FILE.exists():
            self._log_message(f"[Abort] Không tìm thấy file updates: {CANDIDATE_UPDATES_FILE}")
            return

        self._log_message("=== KHỞI CHẠY TIẾN TRÌNH KNOWLEDGE MAINTENANCE AGENT ===")
        
        # 1. Đọc dữ liệu cập nhật candidate
        with open(CANDIDATE_UPDATES_FILE, "r", encoding="utf-8") as f:
            candidate_data = json.load(f)
            
        careers = candidate_data.get("careers", [])
        if not careers:
            self._log_message("[Abort] Dữ liệu cập nhật rỗng.")
            return

        # 2. Lấy trạng thái hiện tại của Database để so sánh
        db_jobs, db_comps, db_rels = await self.get_db_state()
        
        # 3. Tạo báo cáo so sánh thay đổi tri thức (Change Report)
        change_report = self.generate_change_report(candidate_data, db_jobs, db_comps, db_rels)
        
        self._log_message("\n" + "="*50)
        self._log_message("📊 BÁO CÁO THAY ĐỔI TRI THỨC (KNOWLEDGE CHANGE REPORT)")
        self._log_message("="*50)
        self._log_message(f"-> Ngành nghề mới phát hiện: {len(change_report['new_careers'])}")
        for nc in change_report["new_careers"]:
            self._log_message(f"   [Mới] Nghề nghiệp: {nc['name']}")
        self._log_message(f"-> Ngành nghề có cập nhật: {len(change_report['updated_careers'])}")
        for uc in change_report["updated_careers"]:
            self._log_message(f"   [Cập nhật] Nghề nghiệp: {uc['name']}")
        self._log_message(f"-> Năng lực/Kỹ năng mới phát hiện: {len(change_report['new_competencies'])}")
        for ncomp in change_report["new_competencies"]:
            self._log_message(f"   [Mới] Năng lực: {ncomp['name']}")
        self._log_message(f"-> Quan hệ Yêu cầu kỹ năng mới (REQUIRES_SKILL): {len(change_report['new_relationships'])}")
        for nr in change_report["new_relationships"]:
            self._log_message(f"   [Mới] Quan hệ: ({nr['job']}) -> REQUIRES_SKILL -> ({nr['competency']})")
        self._log_message("="*50 + "\n")

        # 4. Thực thi áp dụng các thay đổi vào Neo4j
        course_catalog = await self.get_all_courses()
        all_documents = []

        for career in careers:
            await self.apply_neo4j_updates(career, course_catalog)
            career_name = career["name"]
            for art in career.get("market_articles", []):
                enriched_content = (
                    f"passage: [Xu hướng thị trường tuyển dụng | Ngành nghề: {career_name} | Nguồn: {art['source']} | Đề tài: {art['title']}]\n"
                    f"Nội dung chi tiết: {art['content']}"
                )
                doc = Document(
                    page_content=enriched_content,
                    metadata={
                        "title": art["title"],
                        "source": art["source"],
                        "job_role": career_name,
                        "ContextPath": f"Xu hướng tuyển dụng | {career_name}"
                    }
                )
                all_documents.append(doc)

        # 5. Cập nhật Vector DB Qdrant
        await self.update_qdrant(all_documents)
        
        self._log_message("=== HOÀN TẤT TIẾN TRÌNH KNOWLEDGE MAINTENANCE ===")

if __name__ == "__main__":
    agent = KnowledgeMaintenanceAgent()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(agent.run_maintenance())
