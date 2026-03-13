from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


class ChromaStore:

    def __init__(self, collection_name: str, embedding_model: str):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_dir = f"./chroma_{collection_name}"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(self, docs: List[Document]):
        self.vectorstore.add_documents(docs)

    def similarity_search(self, query: str, k=4):
        return self.vectorstore.similarity_search(query, k=k)