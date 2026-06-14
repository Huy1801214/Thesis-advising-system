import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import engine, Base
from model import User, StudentProfile, ChatMessage

def init_db():
    print("Đang kết nối tới PostgreSQL và khởi tạo các bảng...")
    try:
        Base.metadata.create_all(bind=engine)
        print("Đã tạo các bảng thành công trong database 'thesis_advising_db'!")
    except Exception as e:
        print(f"Có lỗi xảy ra khi tạo bảng: {e}")

if __name__ == "__main__":
    init_db()