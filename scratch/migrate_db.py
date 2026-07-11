from core.database import SessionLocal
from sqlalchemy import text

def run_migration():
    print("Running database migrations for student_profiles...")
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS major VARCHAR;"))
        db.execute(text("ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS target_career VARCHAR;"))
        db.execute(text("ALTER TABLE student_profiles ADD COLUMN IF NOT EXISTS interests VARCHAR;"))
        db.commit()
        print("Migration successful! Added major, target_career, and interests columns.")
    except Exception as e:
        db.rollback()
        print(f"Error migrating: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    run_migration()
