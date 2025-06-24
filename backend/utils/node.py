# node.py
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any
import chromadb
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage,
    BaseMessage
)
from langchain_openai import ChatOpenAI  # ✅ Use LangChain's OpenAI wrapper
import json
from pathlib import Path
from .state import AgentState  # Import AgentState from adjacent state.py
import openai
from datetime import datetime

# ✅ Initialize the LLM wrapper (handles role conversion & batching)
llm = ChatOpenAI(
    model="gpt-4o-2024-08-06",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3
)

# Keep the OpenAI client for embeddings (LangChain doesn't handle embeddings as cleanly)
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_permanent_knowledge(state: AgentState) -> AgentState:
    """
    Check if existing knowledge in JSON is sufficient to answer the query.
    This is a placeholder implementation, need vectordb or llm to determine if current knowledge is sufficient
    Sets state["need_retrieval"] based on analysis.
    """
    print("\n=== Executing: check_permanent_knowledge ===")
    
    # TODO: Define the path to your permanent knowledge JSON file
    KNOWLEDGE_FILE = Path("data/permanent_knowledge.json")
    
    try:
        # Ensure the data directory and file exist
        if not KNOWLEDGE_FILE.exists():
            KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)  # Ensure 'data/' exists
            with open(KNOWLEDGE_FILE, 'w') as f:
                json.dump({}, f)  # Create an empty JSON object
            state["permanent_knowledge"] = {}
        else:
            with open(KNOWLEDGE_FILE, 'r') as f:
                state["permanent_knowledge"] = json.load(f)
        
        # TODO: Implement logic to determine if current knowledge is sufficient
        
        # This is a placeholder implementation, need vectordb or llm to determine if current knowledge is sufficient
        current_query = state.get("current_query", "")
        relevant_knowledge_exists = any(
            current_query.lower() in str(value).lower()
            for value in state["permanent_knowledge"].values()
        )
        
        state["need_retrieval"] = not relevant_knowledge_exists
        
        print(f"Current query: {current_query}")
        print(f"Need retrieval: {state['need_retrieval']}")
        print(f"Top 5 permanent knowledge items: {list(state['permanent_knowledge'].items())[:5]}")
        
        return state
        
    except Exception as e:
        print(f"Error in check_permanent_knowledge: {e}")
        state["need_retrieval"] = True
        return state

def retrieve_from_chroma(state: AgentState) -> AgentState:
    """
    Query Chroma vector database for relevant documents based on the current query.
    """
    print("\n=== Executing: retrieve_from_chroma ===")
    
    try:
        # ✅ Use the same path as your working retrieval.py
        project_root = Path(__file__).resolve().parent.parent.parent
        client = chromadb.PersistentClient(path=str(project_root / "chroma_db"))
        collection = client.get_collection("metadata_collection")
        
        current_query = state.get("current_query", "")
        
        # ✅ Use embeddings like your retrieval.py does
        response = openai_client.embeddings.create(
            input=current_query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        
        # ✅ Query with embeddings (not query_texts)
        res = collection.query(query_embeddings=[query_embedding], n_results=5)
        print(f"[DEBUG] Raw Chroma results: {res}")
        state["retrieved_docs"] = res["documents"][0] if res.get("documents") else []
        
        print(f"Query: {current_query}")
        print(f"Top 5 retrieved docs: {state['retrieved_docs'][:5]}")
        
        return state
        
    except Exception as e:
        print(f"Error in retrieve_from_chroma: {e}")
        state["retrieved_docs"] = []
        return state

def generate_answer(state: AgentState) -> Dict[str, List[BaseMessage]]:
    """
    ✅ LangGraph node: build a prompt from state, call the LLM,
    and return ONLY the new AIMessage so add_messages can append it.
    """
    print("\n=== Executing: generate_answer ===")
    
    try:
        # ✅ Pull pieces out of shared state
        chat_history = state.get("messages", [])
        knowledge = state.get("permanent_knowledge", {})
        retrieved_docs = state.get("retrieved_docs", [])
        current_query = state.get("current_query", "")
        
        print(f"[DEBUG] Chat history length: {len(chat_history)}")
        print(f"[DEBUG] Retrieved docs count: {len(retrieved_docs)}")
        print(f"[DEBUG] Current query: {current_query[:100]}...")
        print(f"[DEBUG] Top 5 knowledge keys: {list(knowledge.keys())[:5]}")
        
        # ✅ Build the system persona + history + context + query
        prompt_msgs: List[BaseMessage] = [
            SystemMessage(
                content=(
                    "You are a literary analyst focused on psychological character analysis.\n"
                    "Analyze characters' mindset, emotional state, behavior patterns, and internal conflicts.\n"
                    "Use evidence from the provided context and conversation history."
                )
            )
        ]
        
        # Add chat history (this already contains the conversation flow)
        prompt_msgs.extend(chat_history)
        
        # Add context if available
        if retrieved_docs or knowledge:
            ctx = "\n".join(retrieved_docs) if retrieved_docs else ""
            knowledge_str = json.dumps(knowledge, indent=2) if knowledge else ""
            prompt_msgs.append(
                SystemMessage(content=f"Context:\n{ctx}\n\nKnowledge:\n{knowledge_str}")
            )
        
        # Add current query if it's not already in chat history
        if current_query:
            prompt_msgs.append(HumanMessage(content=current_query))
        
        print(f"[DEBUG] Total prompt messages: {len(prompt_msgs)}")
        print(f"[DEBUG] Message types: {[type(msg).__name__ for msg in prompt_msgs]}")
        
        # ✅ Invoke the model (returns an AIMessage directly)
        ai_reply: AIMessage = llm.invoke(prompt_msgs)
        
        print(f"[DEBUG] Generated response length: {len(ai_reply.content)}")
        print(f"[DEBUG] Response preview: {ai_reply.content[:100]}...")
        
        # ✅ Return only the new message; add_messages will append it
        return {"messages": [ai_reply]}

    except Exception as e:
        print(f"Error in generate_answer: {e}")
        # ✅ Return error as proper AIMessage
        error_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
        return {"messages": [error_response]}

def update_permanent_knowledge(state: AgentState) -> AgentState:
    """
    Update the permanent knowledge JSON file with LLM-summarized insights.
    """
    print("\n=== Executing: update_permanent_knowledge ===")
    KNOWLEDGE_FILE = Path("data/permanent_knowledge.json")
    try:
        # Ensure permanent_knowledge is a dict
        if "permanent_knowledge" not in state or not isinstance(state["permanent_knowledge"], dict):
            state["permanent_knowledge"] = {}

        # ✅ Extract the latest assistant message (should be the last one due to add_messages)
        messages = state.get("messages", [])
        if messages:
            # Get the most recent AIMessage
            latest_ai_message = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    latest_ai_message = msg
                    break
            
            if latest_ai_message:
                print(f"[DEBUG] Found latest AI message: {latest_ai_message.content[:100]}...")
                
                # 🧠 Use LLM to create concise knowledge summary
                summary_prompt = f"""
                Summarize this character analysis into 2-3 concise key insights about the character's psychology:
                
                Analysis: {latest_ai_message.content}
                
                Format as bullet points focusing on:
                - Core psychological traits
                - Key relationships or conflicts
                - Important behavioral patterns
                
                Keep it brief but insightful.
                """
                
                try:
                    # ✅ Use the LangChain wrapper for consistency
                    summary_response = llm.invoke([
                        SystemMessage(content="You are a literary analyst. Create concise character insights."),
                        HumanMessage(content=summary_prompt)
                    ])
                    
                    # Use timestamp as key, summarized insight as value
                    key = datetime.now().isoformat()
                    value = summary_response.content.strip()
                    state["permanent_knowledge"][key] = value
                    print(f"[DEBUG] Added summarized knowledge with key: {key}")
                    print(f"[DEBUG] Summary: {value[:100]}...")
                    
                except Exception as e:
                    print(f"[WARN] LLM summarization failed: {e}, storing original")
                    # Fallback: store original if LLM fails
                    key = datetime.now().isoformat()
                    value = latest_ai_message.content
                    state["permanent_knowledge"][key] = value

        # Save updated knowledge
        KNOWLEDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_FILE, 'w') as f:
            json.dump(state["permanent_knowledge"], f, indent=2)

        print(f"Updated knowledge file: {KNOWLEDGE_FILE}")
        print(f"Top 5 knowledge items: {list(state['permanent_knowledge'].items())[:5]}")
        return state

    except Exception as e:
        print(f"Error in update_permanent_knowledge: {e}")
        return state
