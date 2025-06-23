# node.py
import os
import chromadb
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List, Any
import chromadb
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # For message types
import json
from pathlib import Path
from .state import AgentState  # Import AgentState from adjacent state.py
import openai
from datetime import datetime

# Add this at the top after other imports
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_permanent_knowledge(state: AgentState) -> AgentState:
    """
    Check if existing knowledge in JSON is sufficient to answer the query.
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
        # This is a placeholder implementation
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
        client = chromadb.PersistentClient(path="chroma_db")  # Fixed path
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

def generate_answer(state: AgentState) -> AgentState:
    """
    Generate response using LLM with context from permanent knowledge and/or retrieved docs.
    """
    print("\n=== Executing: generate_answer ===")
    
    try:
        # Get existing messages (now proper LangChain messages)
        chat_history = state.get("messages", [])
        knowledge = state.get("permanent_knowledge", {})
        retrieved_docs = state.get("retrieved_docs", [])
        
        # Convert retrieved docs to context string
        context = "\n".join(retrieved_docs) if retrieved_docs else ""
        
        # Build system message for character analysis
        system_prompt = """
        You are a literary analyst focused on psychological character analysis.
        Analyze characters' mindset, emotional state, behavior patterns, and internal conflicts.
        Use evidence from the provided context and conversation history.
        """
        
        # Prepare messages for LLM
        messages = [SystemMessage(content=system_prompt)]
        
        # Add chat history (already LangChain messages)
        messages.extend(chat_history)
        
        # Add current context as system message if available
        if context or knowledge:
            context_msg = f"Context: {context}\nKnowledge: {knowledge}"
            messages.append(SystemMessage(content=context_msg))
        
        # Add current query
        current_query = state.get("current_query", "")
        if current_query:
            messages.append(HumanMessage(content=current_query))

        # ✅ Fix the message type mapping for OpenAI
        def get_openai_role(msg):
            if isinstance(msg, HumanMessage):
                return "user"
            elif isinstance(msg, AIMessage):
                return "assistant" 
            elif isinstance(msg, SystemMessage):
                return "system"
            else:
                return "system"  # fallback

        openai_messages = [
            {"role": get_openai_role(msg), "content": msg.content}
            for msg in messages
        ]
        
        print(f"[DEBUG] OpenAI messages: {[(msg['role'], msg['content'][:50]) for msg in openai_messages]}")
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=openai_messages
        )
        answer = response.choices[0].message.content

        # Add the response as AIMessage (proper LangChain format)
        new_messages = chat_history + [AIMessage(content=answer)]
        state["messages"] = new_messages

        print(f"Generated answer length: {len(answer)}")
        print(f"Total messages in history: {len(new_messages)}")
        print(f"Top 5 message types: {[type(msg).__name__ for msg in new_messages[:5]]}")
        
        return state

    except Exception as e:
        print(f"Error in generate_answer: {e}")
        # Ensure we still have a response as proper LangChain message
        chat_history = state.get("messages", [])
        error_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
        state["messages"] = chat_history + [error_response]
        return state

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

        # Extract the latest assistant message (now LangChain AIMessage)
        messages = state.get("messages", [])
        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, AIMessage):
                
                # 🧠 Use LLM to create concise knowledge summary
                summary_prompt = f"""
                Summarize this character analysis into 2-3 concise key insights about the character's psychology:
                
                Analysis: {latest_message.content}
                
                Format as bullet points focusing on:
                - Core psychological traits
                - Key relationships or conflicts
                - Important behavioral patterns
                
                Keep it brief but insightful.
                """
                
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a literary analyst. Create concise character insights."},
                            {"role": "user", "content": summary_prompt}
                        ],
                        temperature=0.3
                    )
                    
                    # Use timestamp as key, summarized insight as value
                    key = datetime.now().isoformat()
                    value = response.choices[0].message.content.strip()
                    state["permanent_knowledge"][key] = value
                    print(f"[DEBUG] Added summarized knowledge with key: {key}")
                    print(f"[DEBUG] Summary: {value[:100]}...")
                    
                except Exception as e:
                    print(f"[WARN] LLM summarization failed: {e}, storing original")
                    # Fallback: store original if LLM fails
                    key = datetime.now().isoformat()
                    value = latest_message.content
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
