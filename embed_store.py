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


file_path = "text_rye.txt"
documents = preprocess_sentences(file_path)

# Initialize Chroma DB client
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="chromadb",
    persist_directory="./chroma_storage"  # local directory
))

# Define collection and embedding function
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-small")
collection = chroma_client.get_or_create_collection(name="test_embeddings", embedding_function=embedding_function)

# Add documents to collection
for i, doc in enumerate(documents):
    collection.add(
        documents=[doc],
        ids=[f"doc_{i}"],
        metadatas=[{"source": file_path}]
    )

print(f"Embedded and stored {len(documents)} chunks.")