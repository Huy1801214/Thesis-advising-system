from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from core.database import Base

class StudentProfile(Base):
    __tablename__ = 'student_profiles'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    mssv = Column(String, ForeignKey('users.mssv', ondelete="CASCADE"), unique=True, index=True, nullable=False)
    cumulative_gpa = Column(Float, nullable=True)
    total_earned_credits = Column(Float, nullable=True)
    passed_courses = Column(JSONB, default=[]) 
    current_courses = Column(JSONB, default=[])
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="profile")