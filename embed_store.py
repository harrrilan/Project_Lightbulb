import os
import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from text_preprocess_sentence import preprocess_sentences
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
print("2. OpenAI API key loaded successfully")


file_path = "text_chunk.txt"
documents = preprocess_sentences(file_path)
print(f"\n3. Text preprocessing complete:")
print(f"   - Total chunks: {len(documents)}")
print(f"   - First chunk: \"{documents[0]}\"")

# Initialize Chroma DB client
print("\n4. Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Define collection and embedding function
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-large")
collection = chroma_client.get_or_create_collection(name="test_embeddings", embedding_function=embedding_function)
print("5. ChromaDB initialized successfully")

# Add documents to collection
#=============Struggling here=============
print("\n6. Starting document embedding...")
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))],
    metadatas=[{"source": file_path}] * len(documents)
)

print(f"\n7. Complete! Embedded and stored {len(documents)} chunks.")