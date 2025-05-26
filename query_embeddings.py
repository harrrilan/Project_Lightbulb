import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

# Initialize the client to access existing embeddings
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Set up the embedding function (needed for queries)
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-large"
)

# Get the existing collection
collection = chroma_client.get_collection(
    name="test_embeddings",
    embedding_function=embedding_function
)

# Example query function
def query_similar(query_text, n_results=5):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results

# Example usage
if __name__ == "__main__":
    results = query_similar("your query here")
    print(results) 