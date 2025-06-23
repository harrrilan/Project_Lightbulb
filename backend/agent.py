# agent.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from backend.utils.node import (
    check_permanent_knowledge,   # → sets state["need_retrieval"]
    retrieve_from_chroma,        # → queries Chroma DB into state["retrieved_docs"]
    generate_answer,             # → calls LLM with context & writes to state["messages"]
    update_permanent_knowledge,  # → summarizes / writes back to JSON file
)
from backend.utils.state import AgentState
from langgraph.checkpoint.memory import InMemorySaver

# ── 1) CONFIG ────────────────────────────────────────────────────────
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]
    # (could add temperature, similarity_threshold, etc.)

# ── 2) BUILD GRAPH ──────────────────────────────────────────────────
workflow = StateGraph(
    AgentState,            # your TypedDict with keys:
                           #   messages, permanent_knowledge, retrieved_docs, need_retrieval, etc.
    config_schema=GraphConfig
)

# ── 3) DEFINE NODES ─────────────────────────────────────────────────
# 3.1 Check if existing JSON "permanent_knowledge" is enough
workflow.add_node("check_knowledge", check_permanent_knowledge)

# 3.2 If not enough, pull extra context from Chroma vector DB
workflow.add_node("retrieve_docs", retrieve_from_chroma)

# 3.3 Build the final answer with LLM + (permanent_knowledge + retrieved_docs)
workflow.add_node("compose_answer", generate_answer)

# 3.4 After answering, summarize or append new facts into your JSON store
workflow.add_node("update_memory", update_permanent_knowledge)

# ── 4) ENTRY POINT ──────────────────────────────────────────────────
# Start every run by checking if we need retrieval
workflow.set_entry_point("check_knowledge")

# ── 5) ROUTING LOGIC ────────────────────────────────────────────────
# After `check_knowledge`, branch based on need_retrieval flag:
workflow.add_conditional_edges(
    "check_knowledge",
    # routing fn: returns "fetch" or "answer"
    lambda state: "fetch" if state["need_retrieval"] else "answer",
    {
        "fetch":   "retrieve_docs",     # go pull from vector DB
        "answer":  "compose_answer",    # skip straight to LLM
    },
)

# Once docs are fetched, always go on to compose the answer
workflow.add_edge("retrieve_docs", "compose_answer")

# After LLM finishes, update your JSON-backed knowledge store
workflow.add_edge("compose_answer", "update_memory")

# Finally, end the graph
workflow.add_edge("update_memory", END)

# ── 6) COMPILE & RUN ─────────────────────────────────────────────────
checkpointer = InMemorySaver()  # or DatabaseSaver for persistence
graph = workflow.compile(checkpointer=checkpointer)
# Now `graph.invoke(...)` will:
#   1. check permanent JSON
#   2. maybe retrieve from Chroma
#   3. call your LLM
#   4. update the JSON file

# When running the agent
config = {
    "configurable": {
        "thread_id": "user_123"  # Unique per conversation
    }
}
