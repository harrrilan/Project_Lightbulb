import gradio as gr
from llm_call import build_graph, PersistentChatMemory

# Instantiate the LangGraph once so it can be reused across requests
memory = PersistentChatMemory("chat_state.json", k=6, threshold=10)
chat_graph = build_graph()

def chat_fn(message, history):
    """Gradio wrapper that sends the user message through the LangGraph chat pipeline."""
    # Debug prints (user rule)
    print("[DEBUG] chat_fn called")
    print(f"[DEBUG] Top 5 chars of message: {message[:5]}")
    print(f"[DEBUG] Top 5 items of history: {str(history)[:5]}")

    result = chat_graph.invoke({"memory": memory, "user_input": message})
    return result["assistant_reply"]

if __name__ == "__main__":
    iface = gr.ChatInterface(
        fn=chat_fn,
        title="RAG Chatbot",
        description="Ask questions and get answers based on retrieved context!"
    )
    iface.launch()