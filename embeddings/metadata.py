'''
Embed with 150-300 tokens with overlaps
This embedding contains metadata, such as chapter, character, etc.
Using Chroma, but from Langchain
'''

import os
from dotenv import load_dotenv
import re
import json
import tiktoken
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# load text_rye
def load_text_rye():
    with open("/files/text_rye.txt", "r") as f:
        return f.read()
    

# Add this token counting function
def create_token_length_function():
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    def token_length(text):
        return len(encoding.encode(text))
    return token_length

def create_chapter_metadata_with_splitter(file_path, chunk_tokens=500, chunk_overlap_tokens=50):
    """
    Split text using RecursiveCharacterTextSplitter with TOKEN-based splitting
    """
    print(f"[DEBUG] Starting metadata creation for {file_path}")
    print(f"[DEBUG] Target tokens per chunk: {chunk_tokens}")
    
    # Read the full text
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    print(f"[DEBUG] Full text length: {len(full_text)} characters")
    
    # Create token length function
    token_length_func = create_token_length_function()
    total_tokens = token_length_func(full_text)
    print(f"[DEBUG] Full text tokens: {total_tokens}")
    
    # Find all chapter boundaries (standalone numbers on their own lines)
    chapter_pattern = r"(?m)^\s*(\d+)\s*$"
    chapter_matches = list(re.finditer(chapter_pattern, full_text))
    
    print(f"[DEBUG] Found {len(chapter_matches)} chapter markers")
    print(f"[DEBUG] Chapter markers at positions: {[m.start() for m in chapter_matches[:5]]}")
    
    # Create chapter boundaries with positions
    chapter_boundaries = []
    for i, match in enumerate(chapter_matches):
        chapter_num = int(match.group(1))
        chapter_start = match.start()
        chapter_boundaries.append((chapter_num, chapter_start))
    
    # Add end boundary
    chapter_boundaries.append((None, len(full_text)))
    
    print(f"[DEBUG] Chapter boundaries: {chapter_boundaries[:5]}")
    
    # Initialize the text splitter with TOKEN-based length function
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,           # Now in TOKENS, not characters
        chunk_overlap=chunk_overlap_tokens, # Now in TOKENS, not characters  
        length_function=token_length_func,  # Use TOKEN counting function
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the text and get chunk positions
    chunks = splitter.split_text(full_text)
    print(f"[DEBUG] Total chunks created: {len(chunks)}")
    
    # Calculate token lengths for each chunk
    chunk_token_lengths = [token_length_func(chunk) for chunk in chunks]
    print(f"[DEBUG] First 5 chunk token lengths: {chunk_token_lengths[:5]}")
    
    # Helper function to determine chapter number for a given position
    def chapter_number_of(position):
        """Find which chapter a given text position belongs to"""
        for i in range(len(chapter_boundaries) - 1):
            chapter_num, start_pos = chapter_boundaries[i]
            next_start_pos = chapter_boundaries[i + 1][1]
            
            if start_pos <= position < next_start_pos:
                return chapter_num
        return 1  # Default to chapter 1 if not found
    
    # Calculate chunk positions in the original text
    chunk_positions = []
    current_pos = 0
    
    for chunk in chunks:
        chunk_start = full_text.find(chunk, current_pos)
        if chunk_start == -1:
            chunk_start = current_pos
        
        chunk_positions.append(chunk_start)
        current_pos = chunk_start + len(chunk) - 100  # Rough character overlap estimate
    
    print(f"[DEBUG] First 5 chunk positions: {chunk_positions[:5]}")
    
    # Create metadata for each chunk
    metadatas = []
    for idx, chunk in enumerate(chunks):
        chunk_start_pos = chunk_positions[idx]
        chapter_num = chapter_number_of(chunk_start_pos)
        
        metadata = {
            "chunk_id": idx,
            "chapter": chapter_num,
            "start_position": chunk_start_pos,
            "chunk_length": chunk_token_lengths[idx],  # Now in TOKENS
        }
        metadatas.append(metadata)
    
    print(f"[DEBUG] Sample metadata created for first 5 chunks:")
    for i, meta in enumerate(metadatas[:5]):
        print(f"[DEBUG] Chunk {i}: {meta}")
    
    return chunks, metadatas

def embed_and_store_chunks(chunks, metadatas):
    """
    1. Create embeddings for chunks using Langchain
    2. Save embeddings as .npy file locally  
    3. Store embeddings + metadata to Chroma vector database
    4. Return the embeddings array
    """
    print(f"[DEBUG] Starting embedding process for {len(chunks)} chunks")
    
    # Initialize Langchain OpenAI embeddings
    embeddings_func = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    print(f"[DEBUG] Langchain OpenAI embedding initialized")

# Step 1: Create embeddings using Langchain
    print("[DEBUG] Creating embeddings with Langchain...")
    all_embeddings = []
    
    for i, chunk in enumerate(chunks):
        print(f"[DEBUG] Embedding chunk {i+1}/{len(chunks)}")
        embedding = embeddings_func.embed_query(chunk)
        all_embeddings.append(embedding)
        
        if i < 5:  # Debug first 5
            print(f"[DEBUG] Chunk {i} embedding length: {len(embedding)}")
            print(f"[DEBUG] First 5 values: {embedding[:5]}")
    
    print(f"[DEBUG] Created {len(all_embeddings)} embeddings")
    print(f"[DEBUG] Embedding shape: {len(all_embeddings)} x {len(all_embeddings[0])}")

    embeddings_array = np.array(all_embeddings)
    np.save("metadata_embeddings.npy", embeddings_array)
    print(f"[DEBUG] Saved embeddings to metadata_embeddings.npy")
    print(f"[DEBUG] Embeddings array shape: {embeddings_array.shape}")
    
    # Save chunks and metadata (like openai_embed.py style)
    with open("metadata_chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved chunks to metadata_chunks.json")
    
    with open("metadata_metadatas.json", "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved metadata to metadata_metadatas.json")
    
    # Create IDs like openai_embed.py
    all_ids = [f"chunk_{meta['chunk_id']}_ch_{meta['chapter']}" for meta in metadatas]
    with open("metadata_ids.json", "w") as f:
        json.dump(all_ids, f)
    print(f"[DEBUG] Saved IDs to metadata_ids.json")
    
    print(f"[DEBUG] Top 5 IDs: {all_ids[:5]}")
 
    # Step 3: Always store to Chroma vector database using Langchain
    print("[DEBUG] Storing to Chroma vector database...")
    
    # Create Chroma vector store
    persist_directory = "../chroma_db"
    vectorstore = Chroma(
        collection_name="metadata_collection",
        embedding_function=embeddings_func,
        persist_directory=persist_directory
    )
    
    # Prepare data for Chroma
    texts = chunks
    formatted_metadatas = []
    ids = []
    
    for i, metadata in enumerate(metadatas):
        # Format metadata for Chroma (all values must be strings/numbers)
        formatted_metadata = {
            "chunk_id": metadata["chunk_id"],
            "chapter": metadata["chapter"],
            "start_position": metadata["start_position"], 
            "chunk_length": metadata["chunk_length"],
            "source": "text_rye.txt"
        }
        formatted_metadatas.append(formatted_metadata)
        ids.append(f"chunk_{metadata['chunk_id']}_ch_{metadata['chapter']}")
    
    print(f"[DEBUG] Prepared {len(texts)} texts for Chroma")
    print(f"[DEBUG] Sample Chroma metadata: {formatted_metadatas[0]}")
    
    # Add to Chroma in batches using Langchain
    batch_size = 50
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = formatted_metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        batch_num = (i // batch_size) + 1
        print(f"[DEBUG] Processing Chroma batch {batch_num}/{total_batches}")
        
        vectorstore.add_texts(
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    # Persist the database
    vectorstore.persist()
    print(f"[DEBUG] Vector database persisted to {persist_directory}")
    
    print(f"[DEBUG] Top 5 embeddings (first 5 values each):")
    for i, emb in enumerate(all_embeddings[:5]):
        print(f"[DEBUG] Embedding {i}: {emb[:5]}")
    
    return all_embeddings

if __name__ == "__main__":
    file_path = "../files/text_rye.txt"
    
    # Create chunks and metadata with TOKEN-based splitting
    chunks, metadatas = create_chapter_metadata_with_splitter(
        file_path, 
        chunk_tokens=300,        # 300 TOKENS per chunk
        chunk_overlap_tokens=50  # 50 TOKENS overlap
    )
    
    # Save the results
    print(f"\n[INFO] Saving chunks and metadata to JSON files...")
    
    with open("chunks_with_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "chunks": chunks,
            "metadatas": metadatas
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Saved to chunks_with_metadata.json")

    print(f"\n[DEBUG] Creating embeddings and storing...")
    embeddings = embed_and_store_chunks(chunks, metadatas)
    
    print(f"\n[SUMMARY] Process complete!")
    print(f"[SUMMARY] Total chunks: {len(chunks)}")
    print(f"[SUMMARY] Total embeddings: {len(embeddings)}")
    print(f"[SUMMARY] Embedding dimensions: {len(embeddings[0])}")
    
    print(f"\n[SUMMARY] Files created:")
    print(f"[SUMMARY] - metadata_embeddings.npy ({len(embeddings)} embeddings)")
    print(f"[SUMMARY] - metadata_chunks.json ({len(chunks)} chunks)")
    print(f"[SUMMARY] - metadata_metadatas.json ({len(metadatas)} metadata entries)")
    print(f"[SUMMARY] - metadata_ids.json ({len(metadatas)} IDs)")
    print(f"[SUMMARY] - chroma_db/ (vector database)")
    
    print(f"\n[DEBUG] Top 5 metadata entries:")
    for i, meta in enumerate(metadatas[:5]):
        print(f"[DEBUG] Metadata {i}: {meta}")