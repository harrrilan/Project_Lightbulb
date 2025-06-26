from fastapi import FastAPI
from pydantic import BaseModel
from .agent import graph, GraphConfig
from backend.utils.state import AgentState
from langchain_core.messages import HumanMessage
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    print("[DEBUG] backend/agent_api.py /chat endpoint called with:", req.message)
    
    # Initialize state for the backend agent
    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "permanent_knowledge": {},
        "retrieved_docs": [],
        "need_retrieval": False,
        "current_query": req.message,
        "summary": ""
    }
    
    # Configuration for the backend agent
    config = {
        "configurable": {
            "thread_id": "api_user",
            "model_name": "openai"
        }
    }
    
    print("[DEBUG] Invoking backend agent graph")
    result = graph.invoke(initial_state, config=config)
    
    # Extract the assistant's response
    assistant_reply = ""
    for msg in result["messages"]:
        if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
            assistant_reply = msg.content
            break
    
    print("[DEBUG] Top 5 chars of result:", assistant_reply[:5])
    return {"response": assistant_reply} 