import os
import chromadb
from stores.sql_store import F1SQLStore

class F1VectorStore:
    """
    ChromaDB vector embedding storage and semantic retrieval layer for Agentic-GraphRAG v4.0.
    """
    def __init__(self, persist_directory: str = "agent_chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = "f1_v4_knowledge_base"
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        
        # Auto-populate with parquet/SQL data if empty
        if self.collection.count() == 0:
            self._ingest_gold_data()

    def _ingest_gold_data(self):
        """Ingests baseline gold parquet records into vector storage for semantic retrieval."""
        try:
            sql_store = F1SQLStore()
            df = sql_store.execute_query("SELECT * FROM dim_drivers LIMIT 50")
            if not df.empty:
                documents = []
                metadatas = []
                ids = []
                for idx, row in df.iterrows():
                    doc = f"Driver: {row.get('forename', '')} {row.get('surname', '')}, Nationality: {row.get('nationality', '')}, DOB: {row.get('dob', '')}"
                    documents.append(doc)
                    metadatas.append({"driverId": str(row.get("driverId", idx))})
                    ids.append(f"driver_{idx}")
                
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"Vector store auto-ingestion warning: {e}")

    def similarity_search(self, query: str, n_results: int = 15) -> list:
        """Retrieves top-N semantically similar text chunks for a given query."""
        try:
            if self.collection.count() == 0:
                return []
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, self.collection.count()))
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []