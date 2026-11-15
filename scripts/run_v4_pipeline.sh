#!/bin/bash
echo "--- Starting Agentic GraphRAG v4.0 Services ---"
# Check virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start FastAPI Backend in background
echo "Launching FastAPI backend..."
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload &

# Start Streamlit Frontend
echo "Launching Streamlit UI..."
streamlit run ui/app.py
