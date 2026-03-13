import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
    GROQ_MODEL_NAME = "llama-3.3-70b-versatile" 
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    MD_QUYCHE_PATH = "data/raw/quyche.md"