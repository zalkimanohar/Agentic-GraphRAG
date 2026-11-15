# 🏎️ Agentic GraphRAG v4.0 (LangGraph + MCP + Cross-Encoder)

A stateful, self-correcting **Neuro-Symbolic AI** platform built for high-precision Formula 1 analytics, multi-hop driver/constructor lineage traversal, deterministic SQL aggregations, and parallel multi-engine machine learning inference spanning historical data from **1950 to 2025**.

Upgrading from a linear pipeline (v3.0), the **v4.0 architecture** replaces hardcoded execution paths with a cyclic **LangGraph Orchestrator**, standardized **Model Context Protocol (MCP)** tool servers, a **BAAI/bge-reranker-v2-m3** cross-encoder, and a self-correcting **Relevance Grader** loop.

---

## 🏗️ Architectural Flow (v4.0 Upgrade)

```text
                       ┌──────────────────────────┐
                       │    User Prompt (RQ6)     │
                       └────────────┬─────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│           LANGGRAPH STATEFUL ORCHESTRATOR (Agentic Router)           │
│                                                                      │
│  1. Intent & Complexity Classifier                                   │
│     ├──► Path A: Lineage / Relationships  ──► [MCP Tool: Neo4j Cypher]│
│     ├──► Path B: Exact Math / Stats       ──► [MCP Tool: DuckDB SQL]  │
│     ├──► Path C: Predictions / Anomalies  ──► [MCP Tool: ML Registry] │
│     └──► Path D: Rules / History Text     ──► [MCP Tool: ChromaDB]    │
└──────────┬────────────────────────┬────────────────────────┬─────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  Neo4j Graph Store  │  │ Gold Parquet (SQL)  │  │ ChromaDB + Registry │
│ (Top-20 Subgraphs)  │  │(Deterministic Math) │  │(Top-30 Text Chunks) │
└──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                        (Raw Candidate Pool: ~50 Items)
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│            CROSS-ENCODER RERANKER (BAAI/bge-reranker-v2-m3)          │
│                                                                      │
│  • Jointly encodes (User Query + Candidate Chunk/Subgraph) pairs     │
│  • Scores true semantic relevance (0.0 to 1.0) & drops distractors   │
│  • Compresses ~50 raw candidates down to Top-5 highest-signal facts  │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│           LANGGRAPH RELEVANCE GRADER (Self-Correction Loop)          │
│                                                                      │
│  [Grader Node]: "Do these Top-5 facts fully answer the prompt?"      │
│     ├──► NO  (Score < Threshold & Loops < 2) ──► Rewrite Query ──┐   │
│     │                                                            │   │
│     └──► YES (Verified Evidence)                                 │   │
└───────────────────────────────────┬──────────────────────────────┼───┘
                                    │    ▲ (Loop back to Router)   │
                                    │    └─────────────────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │  Local LLM (Ollama Llama 3) │
                     │ Synthesizes Grounded Answer │
                     └─────────────────────────────┘
```

---

## 🧠 Core System Layers

### 1. Agentic Router & MCP Tool Layer (`LangGraph` + `MCP`)
* **Dynamic Intent Routing:** Inspects user queries and routes execution strictly to the optimal backend—avoiding heavy ML inference for simple historical queries or vector searches for exact mathematical aggregations.
* **Decoupled Backend Abstraction:** Uses standardized **Model Context Protocol (MCP)** interfaces (`cypher_query`, `sql_query`, `ml_predict`, `vector_search`), allowing underlying storage schemas (e.g., Medallion Gold Parquet partitions) to evolve without rewriting agent logic.

### 2. Hybrid Multi-Store Retrieval Layer
* **Path A — Neo4j Property Graph:** Executes multi-hop Cypher traversals across driver teammates, constructor lineages, and career paths.
* **Path B — DuckDB Analytical Engine:** Queries Medallion Gold star-schema Parquet datasets (`fact_session_results`, `dim_drivers`, `dim_constructors`, `dim_races`, `ref_nationality_region`) with Hive-partitioned multi-season discovery (1950–2025) for 100% deterministic math.
* **Path C — Parallel 6-Engine ML Registry:** Runs inference across serialized models (**XGBoost**, **LightGBM**, **Isolation Forest**, **Temporal Fusion Transformer (TFT)**, **TabNet**, and **GraphSAGE GNN**) for performance predictions and anomaly detection.
* **Path D — ChromaDB Vector Store:** Retrieves semantic context for sporting regulations, historical rule shifts, and race summaries.

### 3. Precision Reranking & Self-Correction Loop
* **Two-Stage Retrieval (`BAAI/bge-reranker-v2-m3`):** Jointly encodes `(Query, Candidate)` pairs from the ~50-item raw pool, scoring semantic relevance from `0.0` to `1.0` and compressing context down to the **Top-5 highest-signal facts**.
* **Corrective RAG Grader:** Evaluates whether the Top-5 facts satisfy the user prompt. If confidence falls below threshold, the orchestrator automatically rewrites the query and re-routes (capped at `2` retry loops to prevent latency spikes).

### 4. Local Grounded Generation (`Ollama Llama 3`)
* Synthesizes concise, hallucination-free natural language answers strictly grounded in verified SQL rows, graph subgraphs, ML outputs, and reranked text chunks.

---

## 📂 Project Directory Structure

```text
Agentic-GraphRAG-v4/
├── api/
│   ├── __init__.py
│   └── app.py                      # FastAPI server exposing agent endpoints
├── agent_core/
│   ├── __init__.py
│   ├── graph_router.py             # LangGraph stateful orchestrator & intent classifier
│   └── mcp_server.py               # Model Context Protocol (MCP) uniform tool servers
├── reranker/
│   ├── __init__.py
│   └── cross_encoder.py            # BAAI/bge-reranker-v2-m3 scoring & context compressor
├── stores/
│   ├── __init__.py
│   ├── vector_store.py             # ChromaDB vector embedding retrieval
│   ├── graph_store.py              # Neo4j property graph connector & Cypher generator
│   └── sql_store.py                # DuckDB / Gold Parquet deterministic SQL layer
├── ml_engine/
│   ├── __init__.py
│   ├── trainer.py                  # Parallel 6-engine ML registry (XGB, LGBM, Isolation Forest, etc.)
│   └── registry/                   # Serialized model weights & metadata JSON
├── ui/
│   └── app.py                      # Streamlit interactive chat & monitoring dashboard
├── requirements.txt                # Pinned python dependencies for v4.0
└── scripts/
    └── run_v4_pipeline.sh          # Execution script to start FastAPI, Streamlit, and verify stores
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python:** `3.11+` or `3.12` (`pyenv` or virtual environment recommended)
* **Local LLM:** [Ollama](https://ollama.com/) installed with `llama3` pulled:
  ```bash
  ollama pull llama3
  ```
* **Graph Database:** Active **Neo4j** instance populated with Formula 1 driver/constructor nodes and relationships.
* **Data Layer:** Medallion Gold Parquet files located under `data/gold/`.

### 2. Installation
```bash
git clone https://github.com/zalkimanohar/DataManagement.git
cd DataManagement
pip install -r requirements.txt
```

### 3. Running the Application
Start the full stack (FastAPI backend, store verification, and Streamlit UI) via the pipeline script:
```bash
bash scripts/run_v4_pipeline.sh
```

Or launch services individually:
* **FastAPI Server:**
  ```bash
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
  ```
* **Streamlit Dashboard:**
  ```bash
  streamlit run ui/app.py
  ```

---

## 🧪 Benchmark Evaluation Queries

| Path | Target Store | Example Query |
| :--- | :--- | :--- |
| **Path A** | Neo4j Cypher | *"Find all drivers who drove for both Ferrari and McLaren across their careers."* |
| **Path B** | DuckDB SQL | *"Who are the top 10 drivers ranked by total career points across all seasons from 1950 to 2025?"* |
| **Path C** | ML Registry | *"Predict podium probabilities and detect telemetry anomalies for the upcoming Grand Prix."* |
| **Path D** | ChromaDB | *"How did the official Formula 1 point-scoring system evolve between 1950 and the modern era?"* |