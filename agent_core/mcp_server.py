import os
from stores.sql_store import F1SQLStore
from stores.vector_store import F1VectorStore

class F1MCPServer:
    def __init__(self, gold_dir: str = "data/gold"):
        # Initialize your SQL store directly with explicit path fallback
        self.sql_store = F1SQLStore(data_dir=gold_dir)

    def execute_tool(self, tool_name: str, query_string: str):
        print(f"[MCP Server] Executing tool '{tool_name}' with query: {query_string}")
        
        if tool_name == "sql_query":
            try:
                df = self.sql_store.execute_query(query_string)
                if df is None or df.empty:
                    print("[MCP Server Warning]: SQL query returned 0 rows. Sending fallback rows.")
                    # Return mock/diagnostic rows so the UI never crashes or says 'no data'
                    return [{"status": "No rows returned by DuckDB", "query": query_string}]
                
                # Convert DataFrame rows to list of dictionaries / strings for the LLM
                return df.to_dict(orient="records")
            except Exception as e:
                print(f"[MCP Server SQL Error]: {e}")
                return [{"error": str(e)}]
                
        elif tool_name == "cypher_query":
            # Handle Neo4j graph queries here
            try:
                # Insert your Neo4j driver execution logic here
                return [{"driver_lineage": "Sample node connection data"}]
            except Exception as e:
                return [{"error": str(e)}]
                
        elif tool_name == "vector_search":
            # Handle ChromaDB vector search here
            return ["Sample historical rule change text chunk."]
            
        return ["Unknown tool requested."]