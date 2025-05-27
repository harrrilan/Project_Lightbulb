import numpy as np
import chromadb

# Load embeddings
embd = np.load('all_embeddings.npy')
print("Shape:", embd.shape)
print("First embedding vector:", embd[0])
print(type(embd))
embd = embd.tolist()
print(type(embd))

# Create unique string IDs for each embedding
ids = [str(i) for i in range(len(embd))]

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="3_chuncks_collection")

# Add embeddings to the collection
collection.add(
    embeddings=embd,
    ids=ids
)

print(f"Added {len(embd)} embeddings to ChromaDB collection.")