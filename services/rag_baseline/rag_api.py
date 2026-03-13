from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from services.rag_baseline.llm import get_llm
from services.rag_baseline.store import ChromaStore
from services.rag_baseline.retriever import HierarchicalRetriever
from services.rag_baseline.answer import Answer
import json 
import os

app = FastAPI(title="Advising System API", description="API tư vấn học vụ sử dụng RAG")

store = ChromaStore(
    collection_name="hierarchical",
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

parent_map_data = {}
map_file_path = "parent_map.json"

if os.path.exists(map_file_path):
    with open(map_file_path, "r", encoding="utf-8") as f:
        parent_map_data = json.load(f)
    print(f"[INFO] Đã tải thành công {len(parent_map_data)} đoạn văn bản gốc từ {map_file_path}")
else:
    print(f"[WARNING] Không tìm thấy file {map_file_path}. Hệ thống sẽ không lấy được context!")

retriever = HierarchicalRetriever(store.vectorstore, parent_map_data) 

llm = get_llm()
generator = Answer(llm)

class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/")
def read_root():
    return {"status": "online", "message": "Hệ thống tư vấn học vụ sẵn sàng."}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        docs = retriever.retrieve(request.question, k=request.k)
        
        if not docs:
            return QueryResponse(
                answer="Rất tiếc, tôi không tìm thấy thông tin này trong hệ thống.", 
                sources=[]
            )

        context = "\n\n".join([doc.page_content for doc in docs])

        source_set = set() 
        for doc in docs:
            pid = doc.metadata.get("parent_id", "")
            if "_" in pid:
                file_name = pid.rsplit("_", 1)[0] 
                source_set.add(file_name)
            else:
                source_set.add(pid)

        answer = generator.generate(request.question, context)
        
        return QueryResponse(
            answer=answer,
            sources=list(source_set) 
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
