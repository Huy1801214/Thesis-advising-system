from typing import List, Dict, Any
from core.graph_db import graph_db

# Tool 2: check_previous 
async def check_previous(target_code: str, passed_courses: List[str]) -> List[Dict]:
    cypher = """
    MATCH (prev:Course)-[:PREVIOUS_OF]->(t:Course {code: $target})
    WHERE NOT prev.code IN $passed
    RETURN prev.code AS code, prev.name AS name
    """
    return await graph_db.run_query_async(cypher, {"target": target_code, "passed": passed_courses})

# Tool 4: validate_registration
async def validate_registration(planned_courses: List[str], passed_courses: List[str]) -> Dict:
    if not planned_courses:
        return {"valid": True, "violations": []}

    cypher = """
    UNWIND $planned AS target_code
    MATCH (t:Course {code: target_code}) 
    
    // Chỉ tìm môn Học trước thiếu
    CALL (t) {
        MATCH (p:Course)-[:PREVIOUS_OF]->(t)
        WHERE NOT p.code IN $passed
        RETURN collect({code: p.code, name: p.name, type: 'previous'}) AS missing_previous
    }
     
    RETURN target_code AS course,
           t.name AS course_name,
           missing_previous AS all_violations
    """
    
    rows = await graph_db.run_query_async(cypher, {"planned": planned_courses, "passed": passed_courses})
    
    violations = []
    for row in rows:
        for v in row["all_violations"]:
            msg = f"Môn {row['course']}: Cần ĐÃ HỌC môn {v['code']} - {v['name']} trước"
            violations.append({
                "course": row["course"],
                "type": v['type'],
                "missing": v["code"],
                "missing_name": v["name"],
                "message": msg
            })

    return {
        "planned": planned_courses,
        "valid": len(violations) == 0,
        "violations": violations,
    }

# Tool 5: get_prerequisite_chain 
async def get_prerequisite_chain(target_code: str, max_depth: int = 5) -> List[Dict]:
    cypher = """
    MATCH path = (start:Course)-[:PREVIOUS_OF*1..%d]->(t:Course {code: $target})
    WHERE NOT EXISTS { MATCH (:Course)-[:PREVIOUS_OF]->(start) }
    RETURN [n IN nodes(path) | {code: n.code, name: n.name}] AS chain,
           [r IN relationships(path) | type(r)] AS rel_types,
           length(path) AS depth
    ORDER BY depth DESC
    """ % max_depth
    return await graph_db.run_query_async(cypher, {"target": target_code})

# Tool 6: list_courses_in_group
async def list_courses_in_group(group_name: str) -> List[Dict]:
    cypher = """
    MATCH (c:Course)-[:BELONGS_TO]->(g:CourseGroup {name: $grp})
    RETURN c.code AS code, c.name AS name, c.credits AS credits,
           c.year AS year, c.semester AS semester
    ORDER BY c.year, c.semester, c.code
    """
    return await graph_db.run_query_async(cypher, {"grp": group_name})

# Tool 7: describe_course_group
async def describe_course_group(group_name: str) -> Dict[str, Any]:
    rows = await graph_db.run_query_async("""
    MATCH (c:Course)-[:BELONGS_TO]->(g:CourseGroup {name: $grp})
    RETURN g.name AS group_name,
           g.min_credits_required AS min_credits_required,
           c.code AS code, c.name AS name, c.credits AS credits,
           c.year AS year, c.semester AS semester
    ORDER BY c.year, c.semester, c.code
    """, {"grp": group_name})
    
    if not rows:
        return {"group_name": group_name, "found": False, "num_courses": 0, "courses": []}
    return {
        "group_name": rows[0]["group_name"],
        "found": True,
        "min_credits_required": rows[0]["min_credits_required"],
        "total_available_credits": sum(r["credits"] or 0 for r in rows),
        "num_courses": len(rows),
        "courses": [{"code": r["code"], "name": r["name"], "credits": r["credits"]} for r in rows],
    }

# Tool 8: describe_course
async def describe_course(course_code: str) -> Dict[str, Any]:
    rows = await graph_db.run_query_async("""
    MATCH (c:Course {code: $code})
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(g:CourseGroup)
    OPTIONAL MATCH (prev:Course)-[:PREVIOUS_OF]->(c)
    RETURN c.code AS code, c.name AS name, 
           c.credits AS credits,
           c.theory_credits AS theory_credits,
           c.practice_credits AS practice_credits,
           c.year AS year, c.semester AS semester,
           c.is_core_A AS is_core_A, c.is_conditional_star AS is_conditional_star,
           g.name AS group_name,
           collect(DISTINCT {code: prev.code, name: prev.name}) AS previous_courses
    """, {"code": course_code})
    
    if not rows or not rows[0]["code"]:
        return {"code": course_code, "found": False}
        
    row = rows[0]
    return {
        "found": True,
        "code": row["code"],
        "name": row["name"],
        "credits": row["credits"],
        "theory_credits": row["theory_credits"],     
        "practice_credits": row["practice_credits"],
        "year": row["year"],
        "semester": row["semester"],
        "group_name": row["group_name"],
        "previous_courses": [x for x in row["previous_courses"] if x["code"]],
    }

# Tool 9: search_courses
async def search_courses(keyword: str, limit: int = 10) -> List[Dict]:
    return await graph_db.run_query_async("""
    MATCH (c:Course)
    WHERE toLower(c.code) CONTAINS toLower($kw) OR toLower(c.name) CONTAINS toLower($kw)
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(g:CourseGroup)
    RETURN c.code AS code, c.name AS name, c.credits AS credits, g.name AS group_name
    LIMIT $limit
    """, {"kw": keyword, "limit": limit})

# Tool 10: sum_credits_by_group
async def sum_credits_by_group(passed_courses: List[str]) -> List[Dict]:
    if not passed_courses:
        return []
        
    cypher = """
    UNWIND $passed AS code
    MATCH (c:Course {code: code})-[:BELONGS_TO]->(g:CourseGroup)
    RETURN g.name AS group_name, sum(c.credits) AS earned_credits, count(c) AS num_courses
    ORDER BY group_name
    """
    return await graph_db.run_query_async(cypher, {"passed": passed_courses})