from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    mssv: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    mssv: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True