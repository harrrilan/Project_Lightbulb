# state.py

from typing import Dict, List, Any, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    TypedDict defining the structure of the agent's state.
    
    Attributes:
        messages: List of LangChain message objects (HumanMessage, AIMessage, etc.)
        permanent_knowledge: Dictionary storing permanent knowledge about characters
        retrieved_docs: List of documents retrieved from vector store
        need_retrieval: Flag indicating if retrieval is needed
        current_query: Current user question being processed
        summary: Summary of the conversation
    """
    messages: List[BaseMessage]
    permanent_knowledge: Dict[str, Any]
    retrieved_docs: List[str]
    need_retrieval: bool
    current_query: str
    summary: str 