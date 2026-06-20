from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database import Base
import enum
import hashlib

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    mssv = Column(String, ForeignKey('users.mssv', ondelete="CASCADE"), index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False) 
    role = Column(String, nullable=False) 
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    user = relationship("User", back_populates="chat_history")

    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING, index=True)
    query_hash = Column(String, index=True, nullable=True)

    @staticmethod
    def generate_hash(mssv: str, question: str) -> str:
        raw_str = f"{mssv}_{question.strip().lower()}"
        return hashlib.md5(raw_str.encode()).hexdigest()