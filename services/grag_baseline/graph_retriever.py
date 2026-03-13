from neo4j import GraphDatabase
from config import Config

class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI, 
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # --- 1. Tìm kiếm môn học (Entity Search) ---
    def search_course(self, search_text: str):
        """Tìm mã môn học dựa trên tên hoặc mã sinh viên nhập vào"""
        # Tránh dùng tên tham số 'query' để không bị TypeError
        cypher = """
        MATCH (c:HocPhan)
        WHERE c.ma_hp = $input_text OR c.search_name CONTAINS $input_lower
        RETURN c.ma_hp AS ma_hp, c.ten_hp AS ten_hp
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(
                cypher, 
                input_text=search_text, 
                input_lower=search_text.lower()
            )
            return result.single()

    # --- 2. Truy vấn 1-hop (Môn điều kiện trực tiếp) ---
    def get_direct_prerequisites(self, ma_hp: str):
        cypher = """
        MATCH (c:HocPhan {ma_hp: $ma_hp})-[:PREREQUISITE]->(pre)
        RETURN pre.ma_hp AS ma_hp, pre.ten_hp AS ten_hp, pre.so_tc AS so_tc
        """
        with self.driver.session() as session:
            result = session.run(cypher, ma_hp=ma_hp)
            return [record.data() for record in result]

    # --- 3. Truy vấn 2-hop (Suy luận lộ trình) ---
    def get_prerequisite_chain(self, ma_hp: str):
        """Lấy chuỗi môn điều kiện sâu hơn (2-3 bước nhảy)"""
        cypher = """
        MATCH p=(c:HocPhan {ma_hp: $ma_hp})-[:PREREQUISITE*1..2]->(pre)
        RETURN pre.ma_hp AS ma_hp, pre.ten_hp AS ten_hp, length(p) AS hop_count
        ORDER BY hop_count
        """
        with self.driver.session() as session:
            result = session.run(cypher, ma_hp=ma_hp)
            return [record.data() for record in result]

# --- Test logic ---
if __name__ == "__main__":
    retriever = GraphRetriever()
    
    # Thử tìm môn "Lập trình nâng cao" (có mã 214331 trong file của bạn)
    test_subject = "Lập trình nâng cao"
    course = retriever.search_course(test_subject)
    
    if course:
        print(f"🔍 Tìm thấy môn: {course['ten_hp']} ({course['ma_hp']})")
        
        # Thử lấy môn điều kiện (Reasoning)
        pres = retriever.get_direct_prerequisites(course['ma_hp'])
        print(f"⛓ Môn điều kiện trực tiếp: {pres}")
        
        chain = retriever.get_prerequisite_chain(course['ma_hp'])
        print(f"🧬 Lộ trình (Multi-hop): {chain}")
    else:
        print(f"❌ Không tìm thấy môn: {test_subject}")
        
    retriever.close()