from typing import List, Dict, Any
from core.graph_db import graph_db

# Tool 1: check_prerequisites
def check_prerequisites(student_id: str, target_code: str) -> List[Dict]:
    cypher = """
    MATCH (prereq:Course)-[:PREREQUISITE_OF]->(t:Course {code: $target})
    OPTIONAL MATCH (s:Student {id: $sid})-[r:HAS_TAKEN]->(prereq)
    WITH prereq, r
    WHERE r IS NULL OR r.status <> 'passed'
    RETURN prereq.code AS code,
           prereq.name AS name,
           coalesce(r.status, 'never_taken') AS current_status
    """
    return graph_db.run_query(cypher, {"sid": student_id, "target": target_code})

# Tool 2: check_previous
def check_previous(student_id: str, target_code: str) -> List[Dict]:
    cypher = """
    MATCH (prev:Course)-[:PREVIOUS_OF]->(t:Course {code: $target})
    OPTIONAL MATCH (s:Student {id: $sid})-[r:HAS_TAKEN]->(prev)
    WITH prev, r
    WHERE r IS NULL
    RETURN prev.code AS code, prev.name AS name
    """
    return graph_db.run_query(cypher, {"sid": student_id, "target": target_code})

# Tool 3: check_corequisites
def check_corequisites(target_code: str, planned_courses: List[str]) -> List[Dict]:
    cypher = """
    MATCH (co:Course)-[:COREQUISITE_OF]->(t:Course {code: $target})
    WHERE NOT co.code IN $planned
    RETURN co.code AS code, co.name AS name
    """
    return graph_db.run_query(cypher, {"target": target_code, "planned": planned_courses})

# Tool 4: validate_registration 
def validate_registration(student_id: str, planned_courses: List[str]) -> Dict:
    violations = []
    for code in planned_courses:
        # Tiên quyết
        for v in check_prerequisites(student_id, code):
            violations.append({
                "course": code,
                "type": "prerequisite",
                "missing": v["code"],
                "missing_name": v["name"],
                "current_status": v["current_status"],
                "message": f"Môn {code}: cần đậu môn tiên quyết {v['code']} - {v['name']} (hiện: {v['current_status']})"
            })
        # Học trước
        for v in check_previous(student_id, code):
            violations.append({
                "course": code,
                "type": "previous",
                "missing": v["code"],
                "missing_name": v["name"],
                "message": f"Môn {code}: cần đã từng đăng ký môn học trước {v['code']} - {v['name']}"
            })
        # Song song
        for v in check_corequisites(code, planned_courses):
            violations.append({
                "course": code,
                "type": "corequisite",
                "missing": v["code"],
                "missing_name": v["name"],
                "message": f"Môn {code}: phải đăng ký song song {v['code']} - {v['name']} cùng kỳ"
            })
    return {
        "student_id": student_id,
        "planned": planned_courses,
        "valid": len(violations) == 0,
        "violations": violations,
    }

# Tool 5: get_prerequisite_chain 
def get_prerequisite_chain(target_code: str, max_depth: int = 5) -> List[Dict]:
    cypher = """
    MATCH path = (start:Course)-[:PREREQUISITE_OF|PREVIOUS_OF*1..%d]->(t:Course {code: $target})
    WHERE NOT EXISTS { MATCH (:Course)-[:PREREQUISITE_OF|PREVIOUS_OF]->(start) }
    RETURN [n IN nodes(path) | {code: n.code, name: n.name}] AS chain,
           [r IN relationships(path) | type(r)] AS rel_types,
           length(path) AS depth
    ORDER BY depth DESC
    """ % max_depth
    return graph_db.run_query(cypher, {"target": target_code})

# Tool 6: list_courses_in_group
def list_courses_in_group(group_name: str) -> List[Dict]:
    cypher = """
    MATCH (c:Course)-[:BELONGS_TO]->(g:CourseGroup {name: $grp})
    RETURN c.code AS code, c.name AS name, c.credits AS credits,
           c.year AS year, c.semester AS semester
    ORDER BY c.year, c.semester, c.code
    """
    return graph_db.run_query(cypher, {"grp": group_name})

# Tool 7: describe_course_group
def describe_course_group(group_name: str) -> Dict[str, Any]:
    rows = graph_db.run_query("""
    MATCH (c:Course)-[:BELONGS_TO]->(g:CourseGroup {name: $grp})
    RETURN g.name AS group_name,
           g.min_credits_required AS min_credits_required,
           c.code AS code,
           c.name AS name,
           c.credits AS credits,
           c.year AS year,
           c.semester AS semester
    ORDER BY c.year, c.semester, c.code
    """, {"grp": group_name})
    if not rows:
        return {
            "group_name": group_name,
            "found": False,
            "min_credits_required": None,
            "total_available_credits": 0,
            "num_courses": 0,
            "courses": [],
        }
    return {
        "group_name": rows[0]["group_name"],
        "found": True,
        "min_credits_required": rows[0]["min_credits_required"],
        "total_available_credits": sum(r["credits"] or 0 for r in rows),
        "num_courses": len(rows),
        "courses": [{
            "code": r["code"],
            "name": r["name"],
            "credits": r["credits"],
            "year": r["year"],
            "semester": r["semester"],
        } for r in rows],
    }

# Tool 8: describe_course
def describe_course(course_code: str) -> Dict[str, Any]:
    rows = graph_db.run_query("""
    MATCH (c:Course {code: $code})
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(g:CourseGroup)
    OPTIONAL MATCH (prev:Course)-[:PREVIOUS_OF]->(c)
    OPTIONAL MATCH (pre:Course)-[:PREREQUISITE_OF]->(c)
    OPTIONAL MATCH (co:Course)-[:COREQUISITE_OF]->(c)
    RETURN c.code AS code,
           c.name AS name,
           c.credits AS credits,
           c.year AS year,
           c.semester AS semester,
           c.is_core_A AS is_core_A,
           c.is_conditional_star AS is_conditional_star,
           g.name AS group_name,
           collect(DISTINCT {code: prev.code, name: prev.name}) AS previous_courses,
           collect(DISTINCT {code: pre.code, name: pre.name}) AS prerequisites,
           collect(DISTINCT {code: co.code, name: co.name}) AS corequisites
    """, {"code": course_code})
    if not rows or not rows[0]["code"]:
        return {"code": course_code, "found": False}
    row = rows[0]
    return {
        "found": True,
        "code": row["code"],
        "name": row["name"],
        "credits": row["credits"],
        "year": row["year"],
        "semester": row["semester"],
        "is_core_A": row["is_core_A"],
        "is_conditional_star": row["is_conditional_star"],
        "group_name": row["group_name"],
        "previous_courses": [x for x in row["previous_courses"] if x["code"]],
        "prerequisites": [x for x in row["prerequisites"] if x["code"]],
        "corequisites": [x for x in row["corequisites"] if x["code"]],
    }

# Tool 9: search_courses
def search_courses(keyword: str, limit: int = 10) -> List[Dict]:
    return graph_db.run_query("""
    MATCH (c:Course)
    WHERE toLower(c.code) CONTAINS toLower($kw)
       OR toLower(c.name) CONTAINS toLower($kw)
    OPTIONAL MATCH (c)-[:BELONGS_TO]->(g:CourseGroup)
    RETURN c.code AS code, c.name AS name, c.credits AS credits,
           c.year AS year, c.semester AS semester, g.name AS group_name
    ORDER BY c.code
    LIMIT $limit
    """, {"kw": keyword, "limit": limit})

# Tool 10: sum_credits_by_group - tính tín chỉ SV đã ĐẬU theo từng nhóm
def sum_credits_by_group(student_id: str) -> List[Dict]:
    cypher = """
    MATCH (s:Student {id: $sid})-[r:HAS_TAKEN {status: 'passed'}]->(c:Course)-[:BELONGS_TO]->(g:CourseGroup)
    RETURN g.name AS group_name, sum(c.credits) AS earned_credits, count(c) AS num_courses
    ORDER BY group_name
    """
    return graph_db.run_query(cypher, {"sid": student_id})

