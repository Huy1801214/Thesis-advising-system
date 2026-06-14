import os
import json
from pathlib import Path
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

JSON_PATH = BASE_DIR / "data" / "subject.json"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jSeeder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def query(self, cypher, params=None):
        with self.driver.session() as session:
            session.run(cypher, params or {})

    def seed(self):
        if not JSON_PATH.exists():
            print(f"❌ Không tìm thấy file JSON tại: {JSON_PATH}")
            return

        with open(JSON_PATH, "r", encoding="utf-8") as f:
            subjects = json.load(f)

        print(f"🚀 Bắt đầu nạp {len(subjects)} môn học vào Neo4j...")

        # A. Xóa dữ liệu cũ & Thiết lập ràng buộc (Constraints)
        print("🧹 Đang dọn dẹp dữ liệu cũ và thiết lập Index...")
        self.query("MATCH (n) DETACH DELETE n")
        self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Course) REQUIRE c.code IS UNIQUE")
        self.query("CREATE CONSTRAINT IF NOT EXISTS FOR (g:CourseGroup) REQUIRE g.name IS UNIQUE")

        # B. Nạp Nodes (Course, CourseGroup) và quan hệ BELONGS_TO
        print("📦 Đang tạo các nút môn học và nhóm học phần...")
        node_cypher = """
        UNWIND $rows AS row
        MERGE (c:Course {code: row.code})
        SET c.name = row.name,
            c.credits = row.credits,
            c.theory_credits = row.theory_credits,      
            c.practice_credits = row.practice_credits,
            c.year = row.year,
            c.semester = row.semester,
            c.is_core_A = row.is_core_A,
            c.is_conditional_star = row.is_conditional_star
        MERGE (g:CourseGroup {name: row.group_name})
        MERGE (c)-[:BELONGS_TO]->(g)
        """
        rows = [{
            "code": s["course_code"],
            "name": s["course_name"],
            "credits": s["credits"]["total"],
            "theory_credits": s["credits"].get("lt", 0),   
            "practice_credits": s["credits"].get("th", 0),
            "year": s["curriculum_position"]["year"],
            "semester": s["curriculum_position"]["semester"],
            "is_core_A": s["classification"]["is_core_A"],
            "is_conditional_star": s["classification"]["is_conditional_star"],
            "group_name": s["classification"]["group_name"]
        } for s in subjects]
        self.query(node_cypher, {"rows": rows})

        # C. Nạp quan hệ Ràng buộc (Relationships)
        print("🔗 Đang thiết lập các quan hệ ràng buộc (Học trước/Tiên quyết/Song hành)...")
        
        prev_edges, prereq_edges, coreq_edges = [], [], []
        for s in subjects:
            tgt = s["course_code"]
            for src in s["requirements"]["previous_courses"]:
                prev_edges.append({"src": src, "tgt": tgt})
            for src in s["requirements"]["prerequisites"]:
                prereq_edges.append({"src": src, "tgt": tgt})
            for src in s["requirements"]["parallel_courses"]:
                coreq_edges.append({"src": src, "tgt": tgt})

        # Thực thi nạp cạnh
        rel_queries = [
            ("PREVIOUS_OF", prev_edges),
            ("PREREQUISITE_OF", prereq_edges),
            ("COREQUISITE_OF", coreq_edges)
        ]

        for rel_type, edges in rel_queries:
            if edges:
                # Với COREQUISITE_OF ta tạo 2 chiều (đối xứng)
                cypher_rel = f"""
                UNWIND $edges AS e
                MATCH (a:Course {{code: e.src}}), (b:Course {{code: e.tgt}})
                MERGE (a)-[:{rel_type}]->(b)
                """
                if rel_type == "COREQUISITE_OF":
                    cypher_rel += f"\nMERGE (b)-[:{rel_type}]->(a)"
                
                self.query(cypher_rel, {"edges": edges})
                print(f"   ✅ Đã nạp {len(edges)} quan hệ {rel_type}")

        print("\n✨ Hoàn tất! Đồ thị tri thức môn học đã sẵn sàng.")

if __name__ == "__main__":
    seeder = Neo4jSeeder()
    try:
        seeder.seed()
    finally:
        seeder.close()