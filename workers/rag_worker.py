import os
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT_TEMPLATE = """Bạn là một chuyên gia tư vấn học vụ tại ĐH Nông Lâm.
Dựa vào NGỮ CẢNH dưới đây để trả lời CÂU HỎI của sinh viên.
---
NGỮ CẢNH:
{context}   
---
CÂU HỎI: {question}
---
Yêu cầu: Trả lời đầy đủ (dưới 500 từ), chính xác, có dẫn chứng Điều/Chương/ Trích từ Sổ tay sinh viên hay Quy chế học vụ.
Nếu không có thông tin trong ngữ cảnh, hãy nói đúng câu: "Tôi chưa có dữ liệu về vấn đề này".
"""

class RAGEngine:
    def __init__(self):
        self.vector_url = os.getenv("QDRANT_URL", "http://nlu_vector_db:6333")
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large", cache_folder="./hf_cache")
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings, 
            retrieval_mode="hybrid",
            collection_name="nlu_academic_rules",
            url=self.vector_url
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1, max_retries=5)

    async def search_and_answer(self, query: str):
        # 1. Retrieval 
        search_query = f"query: {query}"
        docs = self.vectorstore.similarity_search(search_query, k=4)
        context_blocks = []
        for d in docs:
            context_path = d.metadata.get("ContextPath", "Mục chung")
            block = f"[{context_path}]\n{d.page_content}"
            context_blocks.append(block)
        
        context = "\n\n".join(context_blocks)
        
        # 2. Generation 
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        resp = await self.llm.ainvoke(prompt)
        return resp.content