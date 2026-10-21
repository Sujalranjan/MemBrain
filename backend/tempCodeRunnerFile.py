from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
from dotenv import load_dotenv

# Load API Key securely
load_dotenv()
MEMBRAIN_API_KEY = os.getenv("MEMBRAIN_API_KEY")

if not MEMBRAIN_API_KEY:
    print("⚠️ WARNING: MEMBRAIN_API_KEY not found in .env file!")

# The exact Railway URL you found
ALPHANIMBLE_BASE_URL = "https://mem-brain-api-cutover-v4-production.up.railway.app/api/v1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

@app.get("/")
async def serve_ui():
    file_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(file_path)

# ─── 1. KNOWLEDGE INGESTION (POST) ───
@app.post("/api/capture")
async def capture_memory(request: Request):
    data = await request.json()
    
    headers = {
        "X-API-Key": MEMBRAIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # Map our frontend data to their exact schema
    # 🎯 Multi-Tenant Bonus: Adding the workspace ID to the tags
    combined_tags = data.get('tags', []) + ["workspace:engineering"]

    payload = {
        "content": f"Title: {data.get('title', 'Untitled')}\n\n{data.get('body', '')}", 
        "tags": combined_tags,
        "category": data.get('type', 'general')
    }
    
    try:
        print(f"Sending to Membrain: {payload}")
        response = requests.post(f"{ALPHANIMBLE_BASE_URL}/memories", json=payload, headers=headers, timeout=60)
        
        # It returns a 202 Accepted because it queues the job
        if response.status_code == 202:
            return {"status": "success", "alphanimble_response": response.json()}
        else:
            response.raise_for_status()
            
    except Exception as e:
        print(f"API Error during capture: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── 2. SEMANTIC SEARCH & Q&A (POST) ───
# We use POST here because our frontend is sending JSON data (the query)
@app.post("/api/search")
async def search_memory(request: Request):
    data = await request.json()
    query_text = data.get("query", "")
    
    headers = {
        "X-API-Key": MEMBRAIN_API_KEY,
        "Content-Type": "application/json"
    }
    
    # 🎯 We use 'both' to get the raw graph data for the UI AND the LLM summary for the Q&A!
    payload = {
        "query": query_text,
        "k": 6, 
        "keyword_filter": "workspace:engineering", # Enforcing the multi-tenant scope
        "response_format": "both" 
    }
    
    try:
        print(f"Searching Membrain for: {query_text}")
        response = requests.post(f"{ALPHANIMBLE_BASE_URL}/memories/search", json=payload, headers=headers)
        response.raise_for_status()
        
        raw_data = response.json()
        
        # We pass the entire raw response back to the React frontend.
        # The frontend will use raw_data["results"] for the graph/list,
        # and raw_data["interpreted"]["answer_summary"] for the Claude Q&A replacement!
        return raw_data
        
    except Exception as e:
        print(f"Search API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)