import gradio as gr
from llm_call import answer

def chat_fn(message, history):
    # Optionally, you can use history for multi-turn context
    return answer(message)

iface = gr.ChatInterface(
    fn=chat_fn,
    title="RAG Chatbot",
    description="Ask questions and get answers based on retrieved context!"
)

if __name__ == "__main__":
    iface.launch()