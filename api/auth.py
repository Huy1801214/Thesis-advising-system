from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from core.database import get_db
from model.user import User
from core.security import get_password_hash, verify_password, create_access_token
from schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.mssv == user_in.mssv).first()
    if db_user:
        raise HTTPException(status_code=400, detail="MSSV này đã tồn tại!")
    
    new_user = User(
        mssv=user_in.mssv,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.mssv == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu", headers={"WWW-Authenticate": "Bearer"})
    
    token = create_access_token(data={"sub": user.mssv})
    return {"access_token": token, "token_type": "bearer"}