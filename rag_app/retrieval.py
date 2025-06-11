import chromadb
from transformers import BertTokenizer, BertModel
import torch, json
import openai, os
from dotenv import load_dotenv

load_dotenv()  # Loads .env variables
openai.api_key = os.getenv("OPENAI_API_KEY")

client = chromadb.PersistentClient(path="chroma_db")
col    = client.get_collection("openai_embed_collection")

def embed(text: str):
    print("[DEBUG] Running embed() with OpenAI API")
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"  # or another embedding model if you prefer
    )
    embedding = response.data[0].embedding
    print(f"[DEBUG] Top 5 items of embedding: {embedding[:5]}")
    return embedding

def retrieve(query, k=3):
    print("[DEBUG] Running retrieve()")
    query_emb = embed(query)
    print(f"[DEBUG] Top 5 items of query_emb: {query_emb[:5]}")
    res = col.query(query_embeddings=[query_emb], n_results=k)
    print(f"[DEBUG] Top 5 items of res['documents'][0]: {res['documents'][0][:5]}")
    return "\n\n".join(res["documents"][0])
