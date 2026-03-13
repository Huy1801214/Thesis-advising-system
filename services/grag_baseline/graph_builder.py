import json
from neo4j import GraphDatabase
from config import Config


class GraphImporter:
    """
    Import dữ liệu Course và PREREQUISITE vào Neo4j
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def import_from_file(self, file_path: str):
        data = self._load_json(file_path)

        with self.driver.session() as session:
            self._create_constraint(session)
            self._import_courses(session, data["courses"])
            self._import_edges(session, data["edges"])

        print("✔ Import hoàn tất")


    def _load_json(self, file_path: str):
        print("📂 Đang đọc file JSON...")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _create_constraint(self, session):
        print("🔒 Tạo Unique Constraint...")
        session.run("""
            CREATE CONSTRAINT IF NOT EXISTS
            FOR (c:HocPhan)
            REQUIRE c.ma_hp IS UNIQUE
        """)

    def _import_courses(self, session, courses):
        print(f"📘 Import {len(courses)} môn học...")

        query = """
        UNWIND $courses AS course
        MERGE (c:HocPhan {ma_hp: course.ma_hp})
        SET c.ten_hp = course.ten_hp,
            c.search_name = course.search_name,
            c.so_tc = course.so_tc,
            c.is_core = course.is_core,
            c.is_conditional = course.is_conditional
        """

        session.run(query, courses=courses)

    def _import_edges(self, session, edges):
        print(f"🔗 Import {len(edges)} quan hệ...")

        query = """
        UNWIND $edges AS edge
        MATCH (from:HocPhan {ma_hp: edge.from})
        MATCH (to:HocPhan {ma_hp: edge.to})
        MERGE (from)-[:PREREQUISITE]->(to)
        """

        session.run(query, edges=edges)


# ================================
# Entry point
# ================================

if __name__ == "__main__":
    FILE_PATH = "data/chunks/graph_data.json"

    importer = GraphImporter()
    importer.import_from_file(FILE_PATH)
    importer.close()