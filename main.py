from fastapi import FastAPI
from api import auth, upload_infor, chat_router

app = FastAPI(title="Thesis Advising System API")

app.include_router(auth.router)
app.include_router(upload_infor.router, prefix="/api/grag", tags=["GRAG"])
app.include_router(chat_router.router, tags=["Chat"])