import os
import time
from typing import List, Dict

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =============================
# CẤU HÌNH
# =============================

DATA_FOLDER = os.path.join("data", "raw")
FILES = ["quyche.md", "sosinhvien.md"]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =============================
# 1️⃣ LOAD DOCUMENTS
# =============================

def load_documents() -> List[Document]:
    documents = []

    for file_name in FILES:
        file_path = os.path.join(DATA_FOLDER, file_name)

        if not os.path.exists(file_path):
            print(f"[ERROR] Không tìm thấy file: {file_path}")
            continue

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_name
            doc.metadata["full_path"] = file_path

        documents.extend(docs)
        print(f"[OK] Loaded {file_name}")

    print(f"Tổng số document gốc: {len(documents)}\n")
    return documents


# =============================
# 2️⃣ FLAT CHUNKING (Baseline)
# =============================

def flat_chunking(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    start = time.time()
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_type"] = "flat"

    print(f"[FLAT] Tổng chunk: {len(chunks)} | {time.time() - start:.2f}s\n")
    return chunks


# =============================
# 3️⃣ HIERARCHICAL CHUNKING (Custom)
# =============================

def hierarchical_chunking(documents: List[Document]) -> Dict:

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_map: Dict[str, str] = {}
    child_chunks: List[Document] = []

    start = time.time()

    for doc in documents:
        parents = header_splitter.split_text(doc.page_content)

        for idx, parent in enumerate(parents):

            section = (
                parent.metadata.get("Header 2")
                or parent.metadata.get("Header 1")
                or f"section_{idx}"
            )

            parent_id = f"{doc.metadata['source']}_{idx}"

            parent.metadata["section"] = section
            parent.metadata["parent_id"] = parent_id
            parent.metadata["source"] = doc.metadata["source"]

            # Lưu parent full text
            parent_map[parent_id] = parent.page_content

            # Split thành child
            childs = child_splitter.create_documents(
                [parent.page_content],
                metadatas=[parent.metadata],
            )

            for child in childs:
                child.metadata["chunk_type"] = "child"

            child_chunks.extend(childs)

    print(
        f"[HIERARCHICAL] Child chunks: {len(child_chunks)} | "
        f"Parents: {len(parent_map)} | "
        f"{time.time() - start:.2f}s\n"
    )

    return {
        "child_chunks": child_chunks,
        "parent_map": parent_map,
    }


# =============================
# 4️⃣ BUILD VECTOR STORE
# =============================

def build_vector_store(
    chunks: List[Document],
    collection_name: str,
) -> Chroma:

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    persist_dir = f"./chroma_{collection_name}"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    vectorstore.add_documents(chunks)

    print(
        f"[VECTOR STORE] {collection_name} | "
        f"Đã lưu {len(chunks)} chunks vào {persist_dir}\n"
    )

    return vectorstore


# =============================
# 5️⃣ RETRIEVAL
# =============================

def retrieve_flat(vectorstore: Chroma, query: str, k: int = 4) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)


def retrieve_hierarchical(
    vectorstore: Chroma,
    parent_map: Dict[str, str],
    query: str,
    k: int = 5,
) -> List[Document]:

    child_results = vectorstore.similarity_search(query, k=k)

    unique_parent_ids = set(
        doc.metadata.get("parent_id") for doc in child_results
    )

    final_docs = []

    for pid in unique_parent_ids:
        if pid in parent_map:
            final_docs.append(
                Document(
                    page_content=parent_map[pid],
                    metadata={"parent_id": pid},
                )
            )

    print(
        f"[HIER RETRIEVE] "
        f"{len(child_results)} child → "
        f"{len(unique_parent_ids)} parent"
    )

    return final_docs


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    print("=== RAG PIPELINE TEST ===\n")

    # LOAD
    docs = load_documents()
    if not docs:
        print("Không có dữ liệu.")
        exit()

    # ===== FLAT =====
    print("=== FLAT RAG ===")
    flat_chunks = flat_chunking(docs)
    flat_vs = build_vector_store(flat_chunks, "flat")

    # ===== HIERARCHICAL =====
    print("=== HIERARCHICAL RAG ===")
    hier_data = hierarchical_chunking(docs)
    hier_vs = build_vector_store(
        hier_data["child_chunks"],
        "hierarchical",
    )

    # TEST QUERY
    test_queries = [
        "quy định sinh viên bị cảnh cáo học vụ",
        "Em rớt môn A, muốn học môn B nhưng trùng lịch C, làm gì để kịp tốt nghiệp?",
    ]

    for query in test_queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        print("\n--- FLAT RESULTS ---")
        flat_res = retrieve_flat(flat_vs, query)
        for i, doc in enumerate(flat_res, 1):
            print(f"[{i}] {doc.page_content[:250]}...\n")

        print("\n--- HIERARCHICAL RESULTS ---")
        hier_res = retrieve_hierarchical(
            hier_vs,
            hier_data["parent_map"],
            query,
        )
        for i, doc in enumerate(hier_res, 1):
            print(f"[{i}] {doc.page_content[:400]}...\n")