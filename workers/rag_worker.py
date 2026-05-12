import os
from langchain_qdrant import QdrantVectorStore
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
Yêu cầu: Trả lời ngắn gọn (dưới 200 từ), chính xác, có dẫn chứng Điều/Chương.
Nếu không có thông tin trong ngữ cảnh, hãy nói đúng câu: "Tôi chưa có dữ liệu về vấn đề này".
"""

class RAGEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        self.vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name="nlu_academic_rules",
            url="http://localhost:6333" 
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=5)

    def search_and_answer(self, query: str):
        # 1. Retrieval 
        docs = self.vectorstore.similarity_search(query, k=4)
        context_blocks = []
        for d in docs:
            chuong = d.metadata.get("Chuong", "Không rõ chương")
            dieu = d.metadata.get("Dieu", "Không rõ điều")
            block = f"[{chuong} - {dieu}]\n{d.page_content}"
            context_blocks.append(block)
        
        context = "\n\n".join(context_blocks)
        
        # 2. Generation 
        prompt = PROMPT_TEMPLATE.format(context=context, question=query)
        resp = self.llm.invoke(prompt)
        return resp.content