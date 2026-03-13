import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import Config

class VectorStoreManager:
    def __init__(self):
        # Sử dụng mô hình free chạy local, không tốn phí API
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = "data/vector_db"

    def build_vector_db(self):
        # 1. Khai báo rõ ràng danh sách các file Markdown cần nạp
        # Bạn có thể thêm bao nhiêu file tùy thích vào danh sách này
        markdown_files = [
            "data/raw/quyche.md",      
            "data/raw/sosinhvien.md" 
        ]
        
        all_documents = []

        print("📂 Bắt đầu quá trình nạp tài liệu...")
        for file_path in markdown_files:
            if os.path.exists(file_path):
                print(f"--- Đang nạp: {file_path}")
                loader = TextLoader(file_path, encoding="utf-8")
                all_documents.extend(loader.load())
            else:
                print(f"⚠ Cảnh báo: Không tìm thấy file {file_path}. Vui lòng kiểm tra lại thư mục data/raw/")

        if not all_documents:
            print("❌ Không có tài liệu nào được nạp. Vui lòng kiểm tra lại đường dẫn file.")
            return

        # 2. Chia nhỏ văn bản (Chunking) 
        # Giữ kích thước 1000 để đảm bảo ngữ cảnh của từng điều khoản không bị ngắt quãng
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, # Tăng overlap một chút để các đoạn văn có sự kết nối
            separators=["\n## ", "\n### ", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"✂ Tổng cộng đã chia thành {len(chunks)} đoạn văn bản.")

        # 3. Tạo và lưu Vector Database
        print("🤖 Đang mã hóa (Embedding) và lưu vào ChromaDB. Vui lòng đợi...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # ChromaDB từ phiên bản mới sẽ tự động lưu, nhưng gọi persist() để đảm bảo an toàn
        vector_db.persist() 
        print(f"✔ Hoàn tất! Dữ liệu của cả 2 file đã nằm trong: {self.persist_directory}")
    def get_retriever(self):
        """Hàm này giúp main.py lấy dữ liệu từ ChromaDB lên"""
        if not os.path.exists(self.persist_directory):
            print("⚠ Chưa có dữ liệu vector. Đang tự động build...")
            self.build_vector_db()
            
        vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        # Trả về bộ truy xuất lấy 3 đoạn văn bản liên quan nhất
        return vector_db.as_retriever(search_kwargs={"k": 3})
if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.build_vector_db()