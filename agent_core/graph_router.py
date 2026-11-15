import os
import requests
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agent_core.mcp_server import F1MCPServer
from reranker.cross_encoder import F1CrossEncoderReranker

class AgentState(TypedDict):
    query: str
    intent: str
    raw_candidates: list
    reranked_context: list
    retry_count: int
    final_answer: str

class F1AgentGraphOrchestrator:
    def __init__(self):
        self.mcp = F1MCPServer()
        self.reranker = F1CrossEncoderReranker()
        self.workflow = self._build_graph()

    def _intent_classifier_node(self, state: AgentState) -> AgentState:
        q = state["query"].lower()
        # Direct graph queries for lineage, relationships, and multi-team histories
        if any(w in q for w in ["teammate", "connected", "path", "lineage", "both", "drove for"]):
            state["intent"] = "graph"
        else:
            # Force all analytical, metric, ranking, and schema questions into SQL execution
            state["intent"] = "sql"
        return state

    def _retrieval_node(self, state: AgentState) -> AgentState:
        intent = state["intent"]
        query = state["query"]
        q_lower = query.lower()
        candidates = []

        if intent == "graph":
            # Dynamic Cypher query to find drivers connected to multiple constructors
            cypher = """
                MATCH (d:Driver)-[:DROVE_FOR]->(c1:Constructor {constructor_id: 'ferrari'})
                MATCH (d)-[:DROVE_FOR]->(c2:Constructor {constructor_id: 'mclaren'})
                RETURN d.full_name AS driver_name
            """
            candidates = self.mcp.execute_tool("cypher_query", cypher)
            
        elif intent == "sql":
            sql = None
            
            # 1. Driver Standings & Career Points (1950-2025)
            if any(term in q_lower for term in ["driver", "standings", "points", "top", "career", "rank", "who", "best"]):
                sql = """
                    SELECT d.driver_name, 
                           SUM(f.points) AS total_points,
                           SUM(CASE WHEN f.final_position = 1 THEN 1 ELSE 0 END) AS race_wins,
                           SUM(CASE WHEN TRY_CAST(f.final_position AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS total_podiums
                    FROM fact_session_results f
                    JOIN dim_drivers d ON f.driver_id = d.driver_id
                    GROUP BY d.driver_name
                    ORDER BY total_points DESC
                    LIMIT 10;
                """
            # 2. Constructor / Team Standings
            elif any(term in q_lower for term in ["constructor", "team", "teams"]):
                sql = """
                    SELECT c.constructor_name, 
                           SUM(f.points) AS total_points,
                           SUM(CASE WHEN f.final_position = 1 THEN 1 ELSE 0 END) AS total_wins,
                           SUM(CASE WHEN TRY_CAST(f.final_position AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS total_podiums
                    FROM fact_session_results f
                    JOIN dim_constructors c ON f.constructor_id = c.constructor_id
                    GROUP BY c.constructor_name
                    ORDER BY total_points DESC
                    LIMIT 10;
                """
            # 3. Grid Position & Podium Influence
            elif any(term in q_lower for term in ["grid", "influence", "podium finish", "starting", "pole"]):
                sql = """
                    SELECT f.grid_position, 
                           SUM(CASE WHEN TRY_CAST(f.final_position AS INTEGER) <= 3 THEN 1 ELSE 0 END) AS podium_count,
                           COUNT(*) AS total_starts,
                           ROUND(AVG(TRY_CAST(f.final_position AS INTEGER)), 2) AS avg_finish
                    FROM fact_session_results f
                    WHERE f.grid_position IS NOT NULL AND TRY_CAST(f.final_position AS INTEGER) IS NOT NULL
                    GROUP BY f.grid_position
                    ORDER BY f.grid_position ASC
                    LIMIT 15;
                """
            # 4. Circuit & Track Statistics
            elif any(term in q_lower for term in ["circuit", "hosted", "location", "grand prix", "track", "races"]):
                sql = """
                    SELECT r.circuit_name, r.country, COUNT(DISTINCT f.season) AS seasons_hosted, COUNT(*) AS total_sessions
                    FROM dim_races r
                    JOIN fact_session_results f ON r.circuit_id = f.circuit_id
                    GROUP BY r.circuit_name, r.country
                    ORDER BY total_sessions DESC
                    LIMIT 15;
                """
            # 5. Nationality & Regional Comparisons
            elif any(term in q_lower for term in ["british", "german", "italian", "nationality", "region", "country", "countries"]):
                sql = """
                    SELECT d.nationality, 
                           SUM(f.points) AS total_points,
                           COUNT(DISTINCT d.driver_id) AS total_drivers
                    FROM fact_session_results f
                    JOIN dim_drivers d ON f.driver_id = d.driver_id
                    WHERE d.nationality IS NOT NULL
                    GROUP BY d.nationality
                    ORDER BY total_points DESC
                    LIMIT 15;
                """
            else:
                # Universal Dynamic Fallback
                sql = """
                    SELECT d.driver_name, 
                           SUM(f.points) AS total_points,
                           COUNT(DISTINCT f.season) AS seasons_active
                    FROM fact_session_results f
                    JOIN dim_drivers d ON f.driver_id = d.driver_id
                    GROUP BY d.driver_name
                    ORDER BY total_points DESC
                    LIMIT 10;
                """
                
            print(f"[Executing Schema SQL]: {sql}")
            candidates = self.mcp.execute_tool("sql_query", sql)
            
        else:
            candidates = self.mcp.execute_tool("vector_search", query)

        state["raw_candidates"] = [str(c) for c in candidates] if candidates else ["No data retrieved."]
        return state

    def _rerank_node(self, state: AgentState) -> AgentState:
        state["reranked_context"] = self.reranker.rerank(
            query=state["query"],
            candidates=state["raw_candidates"],
            top_k=5,
            threshold=0.0
        )
        return state

    def _grader_node(self, state: AgentState) -> str:
        if not state["reranked_context"] and state["retry_count"] < 2:
            state["retry_count"] += 1
            return "retry"
        return "generate"

    def _generation_node(self, state: AgentState) -> AgentState:
        # Format raw SQL results cleanly into a text table for the LLM
        context_str = "\n".join(state["reranked_context"]) if state["reranked_context"] else "No specific context retrieved."
        
        prompt = f"""You are an expert Formula 1 data assistant. Answer the user's question directly, clearly, and concisely using ONLY the statistical data provided in the Context below. 
Do NOT write Python code, do NOT execute code blocks, and do NOT invent or estimate numbers. Present the exact findings clearly in bullet points or a numbered list.

Context (SQL Query Results):
{context_str}

User Question: {state['query']}
Answer:"""

        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            if res.status_code == 200:
                answer = res.json().get("response", "Could not generate response.")
            else:
                answer = f"Ollama error status {res.status_code}: {res.text}"
        except Exception as e:
            answer = f"Failed to connect to local Ollama server at http://localhost:11434. Error: {e}"

        state["final_answer"] = answer
        return state

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("classifier", self._intent_classifier_node)
        workflow.add_node("retriever", self._retrieval_node)
        workflow.add_node("reranker", self._rerank_node)
        workflow.add_node("generator", self._generation_node)

        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "retriever")
        workflow.add_edge("retriever", "reranker")

        workflow.add_conditional_edges(
            "reranker",
            self._grader_node,
            {
                "retry": "retriever",
                "generate": "generator"
            }
        )
        workflow.add_edge("generator", END)
        return workflow.compile()

    def run(self, query: str) -> str:
        initial_state = {
            "query": query,
            "intent": "",
            "raw_candidates": [],
            "reranked_context": [],
            "retry_count": 0,
            "final_answer": ""
        }
        res = self.workflow.invoke(initial_state)
        return res["final_answer"]