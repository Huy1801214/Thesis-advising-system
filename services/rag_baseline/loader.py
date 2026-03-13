from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import os

class DocumentLoader:
    def __init__(self, data_folder: str, files: List[str]):
        self.data_folder = data_folder
        self.files = files

    def load(self) -> List[Document]:
        documents = []

        for file_name in self.files:
            file_path = os.path.join(self.data_folder, file_name)

            if not os.path.exists(file_path):
                print(f"[ERROR] Không tìm thấy file: {file_path}")
                continue

            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file_name
                doc.metadata["full_path"] = file_path

            documents.extend(docs)
            print(f"Loaded {file_name}")

        print(f"Tổng số document gốc: {len(documents)}\n")
        return documents