from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent_core.graph_router import F1AgentGraphOrchestrator
from stores.sql_store import F1SQLStore

app = FastAPI(
    title="Agentic GraphRAG v4.0 API",
    description="Neuro-Symbolic F1 Analytics Engine with LangGraph, MCP, and Cross-Encoder Reranking",
    version="4.0"
)

# Initialize core services
orchestrator = F1AgentGraphOrchestrator()
sql_store = F1SQLStore()

class QueryRequest(BaseModel):
    query: str

class SQLRequest(BaseModel):
    sql: str

@app.get("/")
def read_root():
    return {"system": "Agentic GraphRAG v4.0", "status": "active"}

@app.post("/agent/query")
def run_agent_query(req: QueryRequest):
    try:
        answer = orchestrator.run(req.query)
        return {"query": req.query, "response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/store/sql")
def execute_sql(req: SQLRequest):
    try:
        df = sql_store.execute_query(req.sql)
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))