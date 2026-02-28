from typing import Dict, List
from langchain_core.documents import Document


class HierarchicalRetriever:

    def __init__(self, vectorstore, parent_map: Dict[str, str]):
        self.vectorstore = vectorstore
        self.parent_map = parent_map

    def retrieve(self, query: str, k=5) -> List[Document]:
        child_results = self.vectorstore.similarity_search(query, k=k)

        unique_parent_ids = set(
            doc.metadata.get("parent_id") for doc in child_results
        )

        final_docs = []

        for pid in unique_parent_ids:
            if pid in self.parent_map:
                final_docs.append(
                    Document(
                        page_content=self.parent_map[pid],
                        metadata={"parent_id": pid},
                    )
                )

        return final_docs
    
    def get_context(self, query: str, k=5) -> str:
        docs = self.retrieve(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
