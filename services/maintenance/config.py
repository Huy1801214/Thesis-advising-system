import os
import socket
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

def is_in_docker():
    if os.path.exists('/.dockerenv'):
        return True
    try:
        socket.gethostbyname('nlu_graph_db')
        return True
    except socket.gaierror:
        return False

IN_DOCKER = is_in_docker()

# Predefined target careers to support
SUPPORTED_CAREERS = [
    "Kỹ sư Phần mềm (Software Engineer)",
    "Khoa học Dữ liệu (Data Scientist)",
    "Kỹ sư DevOps (DevOps Engineer)",
    "Kỹ sư Điện toán đám mây (Cloud Engineer)",
    "Chuyên viên Phân tích Dữ liệu (Data Analyst)",
    "Kỹ sư An toàn thông tin (Security Engineer)"
]

# Trusted Sources list for curation reference
TRUSTED_SOURCES = [
    "roadmap.sh",
    "Microsoft Learn",
    "GitHub Guides",
    "AWS Training & Certification",
    "Google Cloud Training",
    "IBM Developer",
    "Oracle Learning"
]

# Paths
CANDIDATE_UPDATES_FILE = BASE_DIR / "data" / "candidate_career_updates.json"
MAINTENANCE_LOG_FILE = BASE_DIR / "data" / "maintenance_history.log"

# Qdrant & Neo4j configuration depending on running environment
if IN_DOCKER:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://nlu_vector_db:6333")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://nlu_graph_db:7687")
else:
    QDRANT_URL = "http://localhost:16333"
    NEO4J_URI = "bolt://localhost:7687"

QDRANT_COLLECTION = "career_market_insights"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "khoaluan")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
