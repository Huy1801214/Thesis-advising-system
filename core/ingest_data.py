import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. Cấu hình các tham số từ Notebook
load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
SOURCE_FILES = [DATA_DIR / "quyche.md", DATA_DIR / "sosinhvien.md"]

COLLECTION_NAME = "nlu_academic_rules"
CHUNK_SIZE = 900         # Khuyến nghị từ thử nghiệm
CHUNK_OVERLAP = 120      # Khuyến nghị từ thử nghiệm
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
QDRANT_URL = "http://localhost:6333"

def load_markdown(path: Path) -> str:
    """Đọc file Markdown với encoding utf-8"""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return path.read_text(encoding="utf-8")

def normalize_metadata(meta: Dict, source_path: Path) -> Dict:
    """Chuẩn hóa key metadata để đồng nhất format [Chương - Điều]"""
    chapter = meta.get("Chuong") or meta.get("CHƯƠNG") or "Không rõ chương"
    article = meta.get("Dieu") or meta.get("ĐIỀU") or "Không rõ điều"
    return {
        "Chuong": str(chapter).strip(),
        "Dieu": str(article).strip(),
        "Source": source_path.name,
    }

def hierarchical_chunking(text: str, source_path: Path) -> List[Document]:
    """Cơ chế chia nhỏ văn bản phân cấp theo tiêu đề Markdown"""
    # Chia theo Header (# và ##)[cite: 1]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Chuong"), ("##", "Dieu")],
        strip_headers=False,
    )
    coarse_docs = md_splitter.split_text(text)

    # Chia nhỏ tiếp bằng Recursive để tối ưu độ dài chunk[cite: 1]
    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for d in coarse_docs:
        meta = normalize_metadata(d.metadata, source_path)
        tiny_docs = rc_splitter.create_documents([d.page_content], metadatas=[meta])
        chunks.extend(tiny_docs)
    return chunks

def run_ingestion():
    print(f"--- Bắt đầu quá trình nạp dữ liệu vào Qdrant ---")
    
    # Khởi tạo Embedding Model[cite: 1]
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Xử lý từng file[cite: 1]
    all_docs = []
    for fp in SOURCE_FILES:
        print(f"Đang xử lý: {fp.name}...")
        raw_text = load_markdown(fp)
        pieces = hierarchical_chunking(raw_text, fp)
        all_docs.extend(pieces)
        print(f"-> Đã chia thành {len(pieces)} chunks.")

    # Nạp vào Qdrant thực tế (Docker)[cite: 1]
    print(f"Đang đẩy {len(all_docs)} chunks vào collection '{COLLECTION_NAME}'...")
    
    # Lưu ý: Sử dụng .from_documents để tạo mới/ghi đè dữ liệu[cite: 1]
    QdrantVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        force_recreate=True  # Làm sạch dữ liệu cũ trước khi nạp mới
    )
    
    print(f"--- Hoàn tất! Dữ liệu đã sẵn sàng tại {QDRANT_URL} ---")

if __name__ == "__main__":
    run_ingestion()