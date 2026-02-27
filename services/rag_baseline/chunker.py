from typing import List, Dict, Union
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

class Chunker:
    def __init__(self, mode="flat"):
        self.mode = mode

        self.flat_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: List[Document]) -> Union[List[Document], Dict]:

        if self.mode == "flat":
            return self._flat(documents)

        elif self.mode == "hierarchical":
            return self._hierarchical(documents)

        else:
            raise ValueError("Mode must be 'flat' or 'hierarchical'")

    def _flat(self, documents):
        chunks = self.flat_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_type"] = "flat"
            chunk.metadata["chunk_id"] = i
        return chunks

    def _hierarchical(self, documents):
        parent_map = {}
        child_chunks = []

        for doc in documents:
            parents = self.header_splitter.split_text(doc.page_content)

            for idx, parent in enumerate(parents):
                parent_id = f"{doc.metadata['source']}_{idx}"
                parent_map[parent_id] = parent.page_content

                childs = self.child_splitter.create_documents(
                    [parent.page_content],
                    metadatas=[{"parent_id": parent_id}]
                )

                for child in childs:
                    child.metadata["chunk_type"] = "child"

                child_chunks.extend(childs)

        return {
            "child_chunks": child_chunks,
            "parent_map": parent_map
        }