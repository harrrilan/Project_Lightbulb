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
collection = client.get_or_create_collection(name="1_chunck_collection")

max_batch_size = 5461  # or use chromadb.config.settings.DEFAULT_MAX_BATCH_SIZE if available

for i in range(0, len(embd), max_batch_size):
    batch_embd = embd[i:i+max_batch_size]
    batch_ids = ids[i:i+max_batch_size]
    batch_docs = all_chunks[i:i+max_batch_size]
    collection.add(
        embeddings=batch_embd,
        ids=batch_ids,
        documents=batch_docs
    )

print(f"Added {len(embd)} embeddings to ChromaDB collection.")