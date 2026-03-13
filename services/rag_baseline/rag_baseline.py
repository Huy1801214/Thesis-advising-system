import json
import os
from services.rag_baseline.chunker import Chunker
from services.rag_baseline.loader import DocumentLoader
from services.rag_baseline.store import ChromaStore
from services.rag_baseline.retriever import HierarchicalRetriever
from services.rag_baseline.llm import get_llm
from services.rag_baseline.answer import Answer

class RAGPipeline:
    def __init__(self, loader, chunker, store):
        self.loader = loader
        self.chunker = chunker
        self.store = store
        self.parent_map = {}

    def run(self):
        docs = self.loader.load()
        
        result = self.chunker.split(docs)

        if self.chunker.mode == "flat":
            self.store.add_documents(result)

        elif self.chunker.mode == "hierarchical":
            self.store.add_documents(result["child_chunks"])
            self.parent_map = result["parent_map"]

            with open("parent_map.json", "w", encoding="utf-8") as f:
                json.dump(self.parent_map, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    loader = DocumentLoader(data_folder="data/raw", files=["quyche.md", "sosinhvien.md"])

    store_flat = ChromaStore(
        collection_name="flat", 
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    pipeline_flat = RAGPipeline(loader, Chunker(mode="flat"), store_flat)
    pipeline_flat.run()

    store_hier = ChromaStore(
        collection_name="hierarchical", 
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    pipeline_hier = RAGPipeline(loader, Chunker(mode="hierarchical"), store_hier)
    pipeline_hier.run()
    retriever = HierarchicalRetriever(store_hier.vectorstore, pipeline_hier.parent_map)
