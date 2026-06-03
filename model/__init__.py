from core.database import Base
from .user import User
from .student_profile import StudentProfile
from .chat_message import ChatMessage

metadata = Base.metadata
__all__ = ["Base", "metadata", "User", "StudentProfile", "ChatMessage"]