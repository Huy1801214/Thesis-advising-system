import asyncio
from core.graph_db import graph_db

async def inspect():
    # 1. Xem tất cả JobRole
    print("=== JOB ROLES ===")
    records = await graph_db.run_query_async("MATCH (j:JobRole) RETURN j.name AS name")
    for r in records:
        print(f"- {r['name']}")
        
    # 2. Xem các Competency yêu cầu bởi Kỹ sư Điện toán đám mây (Cloud Engineer)
    print("\n=== CLOUD ENGINEER REQUIREMENTS ===")
    cypher = """
    MATCH (j:JobRole {name: "Kỹ sư Điện toán đám mây (Cloud Engineer)"})-[:REQUIRES_SKILL]->(c:Competency)
    RETURN c.name AS name, c.description AS description
    """
    records = await graph_db.run_query_async(cypher)
    for r in records:
        print(f"- Skill: {r['name']} ({r['description']})")
        
        # Xem các môn dạy competency này
        cypher_courses = """
        MATCH (co:Course)-[:TEACHES_SKILL]->(c:Competency {name: $comp_name})
        RETURN co.code AS code, co.name AS name
        """
        courses = await graph_db.run_query_async(cypher_courses, {"comp_name": r["name"]})
        print("  Dạy bởi các môn:")
        for co in courses:
            print(f"    + {co['code']} - {co['name']}")

if __name__ == "__main__":
    asyncio.run(inspect())
