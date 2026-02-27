from services.rag_baseline.chunker import Chunker
from services.rag_baseline.loader import DocumentLoader
from services.rag_baseline.store import ChromaStore


class RAGPipeline:
    def __init__(self, loader, chunker, store):
        self.loader = loader
        self.chunker = chunker
        self.store = store

    def run(self):
        docs = self.loader.load()
        result = self.chunker.split(docs)

        if self.chunker.mode == "flat":
            self.store.add_documents(result)

        elif self.chunker.mode == "hierarchical":
            self.store.add_documents(result["child_chunks"])
            self.parent_map = result["parent_map"]

if __name__ == "__main__":

    print("=== RUN MODULAR RAG PIPELINE ===")

    loader = DocumentLoader(
        data_folder="data/raw",
        files=["quyche.md", "sosinhvien.md"]
    )

    chunker = Chunker(mode="flat")

    store = ChromaStore(
        collection_name="test_collection",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print("=== FLAT ===")
    pipeline = RAGPipeline(loader, Chunker(mode="flat"), store)
    pipeline.run()

    print("=== HIERARCHICAL ===")
    pipeline = RAGPipeline(loader, Chunker(mode="hierarchical"), store)
    pipeline.run()

    print("=== DONE ===")