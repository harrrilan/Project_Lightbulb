import numpy as np
import chromadb
import json


# Load embeddings
embd = np.load('all_embeddings.npy').tolist()
print("First embedding vector:", embd[0])

# Load chunks
with open('all_chunks.json', 'r') as f:
    all_chunks = json.load(f)

# Load IDs generated during embedding
with open('all_ids.json', 'r') as f:
    ids = json.load(f)

client = chromadb.PersistentClient(path="chroma_db")
#client.delete_collection(name="sentence_collection")
collection = client.get_or_create_collection(name="openai_embed_collection")

max_batch_size = 5461  # or use chromadb.config.settings.DEFAULT_MAX_BATCH_SIZE if available

for i in range(0, len(embd), max_batch_size):
    batch_embd = embd[i:i+max_batch_size]
    batch_ids = ids[i:i+max_batch_size]
    batch_docs = all_chunks[i:i+max_batch_size]
    print(f"DEBUG: Shape of batch_embd: {np.array(batch_embd).shape}")
    print(f"DEBUG: First embedding (top 5): {batch_embd[0][:5]}")
    collection.add(
        embeddings=batch_embd,
        ids=batch_ids,
        documents=batch_docs
    )

print(f"Added {len(embd)} embeddings to ChromaDB collection.")