import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse

load_dotenv()
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data" / "raw"
SOURCE_FILES = [DATA_DIR / "quyche.md", DATA_DIR / "sosinhvien.md"]

COLLECTION_NAME = "nlu_academic_rules"
CHUNK_SIZE = 1200         
CHUNK_OVERLAP = 200      
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
IS_DOCKER = os.path.exists('/.dockerenv')
QDRANT_URL = "http://nlu_vector_db:6333" if IS_DOCKER else "http://localhost:16333"

def load_markdown(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return path.read_text(encoding="utf-8")

def normalize_metadata(meta: Dict, source_path: Path) -> Dict:
    h1 = meta.get("Header1", "").strip()
    h2 = meta.get("Header2", "").strip()
    context_path = f"{h1} | {h2}".strip(" |")
    if not context_path:
        context_path = "Mục chung"

    return {
        "ContextPath": context_path,
        "Source": source_path.name,
    }

def hierarchical_chunking(text: str, source_path: Path) -> List[Document]:
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header1"), ("##", "Header2"), ("###", "Header3")],
        strip_headers=False,
    )
    coarse_docs = md_splitter.split_text(text)

    rc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    for d in coarse_docs:
        meta = normalize_metadata(d.metadata, source_path)
        tiny_docs = rc_splitter.create_documents([d.page_content], metadatas=[meta])
        for tiny in tiny_docs:
            context_path = meta["ContextPath"]
            source = meta["Source"]

            enriched_content = (
                f"passage: [Trích từ {source} | {context_path}]\n"
                f"Nội dung: {tiny.page_content}"
            )

            tiny.page_content = enriched_content
            tiny.metadata = meta
            chunks.append(tiny)
    return chunks

def run_ingestion():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder="./hf_cache")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    all_docs = []
    for fp in SOURCE_FILES:
        raw_text = load_markdown(fp)
        pieces = hierarchical_chunking(raw_text, fp)
        all_docs.extend(pieces)
    try:
        QdrantVectorStore.from_documents(
            documents=all_docs,
            embedding=embeddings,   
            sparse_embedding=sparse_embeddings, 
            retrieval_mode="hybrid",    
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
            force_recreate=True,
            batch_size=50,
            timeout=300
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    run_ingestion()