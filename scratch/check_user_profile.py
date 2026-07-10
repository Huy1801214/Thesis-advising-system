from core.database import SessionLocal
from model.student_profile import StudentProfile
from model.user import User

db = SessionLocal()
try:
    users = db.query(User).all()
    print("=== USERS ===")
    for u in users:
        print(f"MSSV: {u.mssv}, Email: {u.email}")
        
    profiles = db.query(StudentProfile).all()
    print("\n=== STUDENT PROFILES ===")
    for p in profiles:
        print(f"MSSV: {p.mssv}")
        print(f"  GPA: {p.cumulative_gpa}")
        print(f"  Credits: {p.total_earned_credits}")
        print(f"  Major: {p.major}")
        print(f"  Target Career: {p.target_career}")
        print(f"  Interests: {p.interests}")
        print(f"  Passed courses: {p.passed_courses}")
        print(f"  Current courses: {p.current_courses}")
finally:
    db.close()
