from core.document_parser import extract_student_context

result = extract_student_context("c:/Learn/Thesis/Thesis-advising-system/data/Diem.xlsx")
print("Status:", result.get("status"))
print("cumulative_gpa:", result.get("cumulative_gpa"))
print("total_earned_credits:", result.get("total_earned_credits"))
print("passed_courses count:", len(result.get("passed_courses", [])))
print("failed_courses count:", len(result.get("failed_courses", [])))
print("current_courses count:", len(result.get("current_courses", [])))
print("passed_courses:", result.get("passed_courses"))
