from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import auth, upload_infor, chat_router
from core.security import limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

app = FastAPI(title="Thesis Advising System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(auth.router)
app.include_router(upload_infor.router, prefix="/api/grag", tags=["GRAG"])
app.include_router(chat_router.router, tags=["Chat"])