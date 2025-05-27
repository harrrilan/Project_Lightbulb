import numpy as np
import chromadb
import json


# Load embeddings
embd = np.load('all_embeddings.npy').tolist()
print("First embedding vector:", embd[0])

# Load chunks
with open('all_chunks.json', 'r') as f:
    all_chunks = json.load(f)

# Create unique string IDs for each embedding
ids = [str(i) for i in range(len(embd))]

client = chromadb.PersistentClient(path="chroma_db")
#client.delete_collection(name="3_chuncks_collection")
collection = client.get_or_create_collection(name="3_chuncks_collection")

# Add embeddings to the collection
collection.add(
    embeddings=embd,
    ids=ids,
    documents=all_chunks

)

print(f"Added {len(embd)} embeddings to ChromaDB collection.")