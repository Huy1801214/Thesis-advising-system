import os 
from pathlib import Path
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv 

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class GragDBManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))

    def close(self):
        self.driver.close()
    
    def run_query(self, cypher: str, params: dict = None):
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            return [dict(r) for r in result]
        
graph_db = GragDBManager()


