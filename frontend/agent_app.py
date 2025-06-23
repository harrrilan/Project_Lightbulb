import gradio as gr
import sys
import os
# Add the parent directory to Python path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agent import graph, GraphConfig
from backend.utils.state import AgentState
from langchain_core.messages import HumanMessage
import time

def chat_fn_streaming(message, history):
    """Gradio streaming wrapper - yields partial responses word by word."""
    # Debug prints (user rule)
    print("[DEBUG] agent_app.py streaming chat_fn called")
    print(f"[DEBUG] Top 5 chars of message: {message[:5]}")
    print(f"[DEBUG] Top 5 items of history: {str(history)[:5]}")

    # Initialize state for the backend agent
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "permanent_knowledge": {},
        "retrieved_docs": [],
        "need_retrieval": False,
        "current_query": message,
        "summary": ""
    }
    
    # Configuration for the backend agent
    config = {
        "configurable": {
            "thread_id": "gradio_user",
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
    
    print(f"[DEBUG] Full response length: {len(assistant_reply)}")
    
    # Simulate streaming by yielding words progressively
    words = assistant_reply.split()
    current_response = ""
    
    for i, word in enumerate(words):
        current_response += word + " "
        time.sleep(0.05)  # Adjust speed (50ms per word)
        yield current_response
    
    # Final yield with complete response
    yield assistant_reply

if __name__ == "__main__":
    iface = gr.ChatInterface(
        fn=chat_fn_streaming,  # Use streaming function
        title="Backend Agent RAG Chatbot",
        description="Ask questions about book characters and get psychological analysis!"
    )
    iface.launch() 