from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import auth, upload_infor, chat_router

app = FastAPI(title="Thesis Advising System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(upload_infor.router, prefix="/api/grag", tags=["GRAG"])
app.include_router(chat_router.router, tags=["Chat"])