import os 
from pathlib import Path
from neo4j import AsyncGraphDatabase, basic_auth
from dotenv import load_dotenv 

env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class GragDBManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))

    async def close(self):
        self.driver.close()
    
    async def run_query_async(self, query: str, parameters: dict = None):
            if parameters is None:
                parameters = {}
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                records = await result.data()
                return records
        
graph_db = GragDBManager()


