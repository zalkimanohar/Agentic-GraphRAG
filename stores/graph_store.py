import os
from neo4j import GraphDatabase

class F1GraphStore:
    """
    Neo4j Property Graph connector for multi-hop relationship traversals 
    and small-world network queries.
    """
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = os.getenv("NEO4J_URI", uri)
        self.user = os.getenv("NEO4J_USER", user)
        self.password = os.getenv("NEO4J_PASSWORD", password)
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j Graph Store.")
        except Exception as e:
            print(f"Warning: Neo4j connection failed ({e}). Graph queries will run in offline fallback mode.")
            self.driver = None

    def query(self, cypher: str, parameters: dict = None) -> list:
        if not self.driver:
            return [{"error": "Neo4j connection inactive"}]
        
        parameters = parameters or {}
        try:
            with self.driver.session() as session:
                result = session.run(cypher, parameters)
                return [record.data() for record in result]
        except Exception as e:
            print(f"Cypher execution error: {e}")
            return []

    def close(self):
        if self.driver:
            self.driver.close()