from fastapi import FastAPI
from pydantic import BaseModel
from rag_app.llm_call import answer
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
    print("[DEBUG] /chat endpoint called with:", req.message)
    result = answer(req.message)
    print("[DEBUG] Top 5 items of result:", result[:5])
    return {"response": result}
