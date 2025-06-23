import json
import pathlib
from typing import Any, Dict, List
import os
from retrieval import retrieve

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# ---------------------------
# 1. Persistent Memory Class
# ---------------------------

class ChatState(BaseModel):
    """Defines the fields carried through the LangGraph execution."""
    memory: Any                 # PersistentChatMemory instance
    user_input: str             # The user's latest message
    assistant_reply: str = ""  # Populated after LLM call

class PersistentChatMemory:
    """Disk‑backed chat memory with automatic summarisation."""

    def __init__(self, file_path: str = "chat_state.json", k: int = 6, threshold: int = 10):
        self.path = pathlib.Path(file_path)
        self.k = k                      # keep this many recent messages verbatim
        self.threshold = threshold      # summarise when history > k + threshold
        self.summary: str = ""          # rolling summary of *older* messages
        self.chat_history: List[Dict[str, str]] = []  # every message ever sent
        self._load()

    # ----- basic persistence helpers -----
    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.chat_history = data.get("chat_history", [])
            self.summary = data.get("summary", "")
        else:
            self.chat_history, self.summary = [], ""

    def _save(self) -> None:
        self.path.write_text(
            json.dumps({"chat_history": self.chat_history, "summary": self.summary}, indent=2)
        )

    # ----- public API -----
    def add(self, role: str, content: str) -> None:
        self.chat_history.append({"role": role, "content": content})
        self._save()

    def get_context(self) -> List[Dict[str, str]]:
        ctx: List[Dict[str, str]] = []
        if self.summary:
            ctx.append({"role": "system", "content": f"Conversation summary so far:\n{self.summary}"})
        ctx.extend(self.chat_history[-self.k :])
        return ctx

    def maybe_summarise(self, summariser, force: bool = False) -> None:
        if len(self.chat_history) <= self.k + self.threshold and not force:
            return
        stale_msgs = self.chat_history[:-self.k]
        if not stale_msgs:
            return
        self.summary = summariser(stale_msgs, self.summary)
        self.chat_history = self.chat_history[-self.k :]
        self._save()

# ---------------------------
# 2. A very small summariser
# ---------------------------

def openai_summariser(msgs: List[Dict[str, str]], prev_summary: str) -> str:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    dialogue = "\n".join(f"{m['role'].title()}: {m['content']}" for m in msgs)
    prompt = (
        "You are the long‑term memory of a chatbot. Given the previous summary (if any) \n"
        "and additional dialogue, produce an UPDATED concise summary that captures \n"
        "important facts, decisions, and user preferences. Keep it brief.\n\n"
        f"Previous summary:\n{prev_summary}\n\nNew dialogue:\n{dialogue}\n\nUpdated summary:"
    )
    return llm([HumanMessage(content=prompt)]).content.strip()

# ---------------------------------
# 3. LangGraph nodes / state schema
# ---------------------------------

# Shared state keys: {"memory": PersistentChatMemory, "user_input": str, "assistant_reply": str}


def node_save_user(state: ChatState) -> ChatState:
    print("[DEBUG] node_save_user called")
    state.memory.add("user", state.user_input)
    return {}


def node_agent(state: ChatState) -> ChatState:
    """Core LLM call: now pulls extra context using your existing retriever."""
    print("[DEBUG] node_agent called")
    mem: PersistentChatMemory = state.memory
    query: str = state.user_input

    # Retrieve supplementary context -------------------------------------------------
    try:
        context = retrieve(query) 
    except Exception as e:
        print(f"[DEBUG] Retrieval failed ({e}); continuing with empty context.")
        context = ""

    # Truncate if over 4 000 characters ------------------------------------------------
    if len(context) > 4000:
        print(f"[DEBUG] Context too long ({len(context)} chars), truncating.")
        context = context[:4000]
    print(f"[DEBUG] Top 5 chars of context: {context[:5]}")

    # Build prompt block exactly as requested -----------------------------------------
    prompt_block = (
        "CONTEXT:\n"
        f"Context: {context}\n"
        f"Question: {query}\n"
        "Answer:"
    )

    # Assemble messages for the LLM ----------------------------------------------------
    messages = [
        SystemMessage(
            content=(
                "You are a literary analyst focused on psychological character analysis.\n"
                "You are given multiple excerpts from a novel. These passages may come from different chapters or points in time.\n"
                "Your goal is to infer and explain the main character's mindset, emotional state, patterns of behavior, and internal conflicts — based on what is revealed across all passages.\n"
                "Carefully examine each excerpt before answering. Look for consistent emotional cues, contradictions, and evolving beliefs. If the character expresses confusion, anger, affection, or guilt, explore the underlying reasons and how they connect across time.\n"
                "Make thoughtful, evidence-based inferences. Do not summarize — analyze.\n"
                "Do not say 'the context is vague' unless you have deeply considered every snippet. Reason step by step and support your claims with examples from the text. Keep your responses concise, around 250 tokens."
            )
        )
    ]

    # Add memory (summary + recent verbatim messages)
    messages += [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in mem.get_context()
    ]

    # Final user turn contains context + question
    messages.append(HumanMessage(content=prompt_block))

    # Hit the LLM ----------------------------------------------------------------------
    llm = ChatOpenAI(model_name="gpt-4o-2024-08-06", streaming=True)
    reply = llm(messages).content.strip()

    return {"assistant_reply": reply}


def node_save_assistant(state: ChatState) -> ChatState:
    print("[DEBUG] node_save_assistant called")
    mem: PersistentChatMemory = state.memory
    mem.add("assistant", state.assistant_reply)
    mem.maybe_summarise(openai_summariser)
    return {}

# ---------------------------
# 4. Build the graph
# ---------------------------

def build_graph():
    # Initialize the graph with the new schema-based constructor (no input_schema kwarg)
    sg = StateGraph(ChatState)
    sg.add_node("save_user", node_save_user)
    sg.add_node("agent", node_agent)
    sg.add_node("save_assistant", node_save_assistant)

    sg.add_edge("save_user", "agent")
    sg.add_edge("agent", "save_assistant")

    sg.set_entry_point("save_user")
    sg.set_finish_point("save_assistant")
    return sg.compile()

# ---------------------------
# 5. Simple REPL driver
# ---------------------------

def main():
    memory = PersistentChatMemory("chat_state.json", k=6, threshold=10)
    chat = build_graph()
    print("(Type 'quit' to exit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            break
        result = chat.invoke({"memory": memory, "user_input": user_input})
        print(f"Assistant: {result['assistant_reply']}\n")


if __name__ == "__main__":
    main()
