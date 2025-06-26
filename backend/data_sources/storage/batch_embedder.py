# test_embedding.py
print("[DEBUG] Converting JSON to embeddings and inserting to DB...")

from dotenv import load_dotenv
import os
import json
from sqlalchemy import create_engine, text
from openai import OpenAI

load_dotenv()

# Load your JSON data
print("[DEBUG] Loading JSON data...")
with open("literacy_analysis_combined.json", "r") as f:
    data = json.load(f)

chunks = data["chunks"]
metadatas = data["metadatas"] 
ids = data["ids"]

print(f"[DEBUG] Loaded {len(chunks)} chunks")

# Setup OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup database
connection_string = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:5432/postgres"
)
engine = create_engine(connection_string)

# Process each chunk
try:
    with engine.connect() as conn:
        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size] 
            batch_ids = ids[i:i+batch_size]
            
            print(f"[DEBUG] Processing batch {i//batch_size + 1}...")
            
            # Get embeddings for entire batch
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=batch_chunks  # ← Multiple chunks at once
            )
            
            # Insert all batch results
            for j, embedding_data in enumerate(response.data):
                conn.execute(text("""
                    INSERT INTO embeddings_table (content, metadata, node_id, embedding)
                    VALUES (:content, :metadata, :node_id, :embedding)
                """), {
                    "content": batch_chunks[j],
                    "metadata": json.dumps(batch_metadatas[j]),
                    "node_id": batch_ids[j],
                    "embedding": embedding_data.embedding
                })
            
        conn.commit()
        print(f"✅ DONE! Inserted {len(chunks)} embeddings to your database!")
        
except Exception as e:
    print(f"❌ FAILED: {e}")