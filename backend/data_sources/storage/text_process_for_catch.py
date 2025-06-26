'''
Embed with 150-300 tokens with overlaps
This embedding contains metadata, such as chapter, character, etc.

'''
import os
from dotenv import load_dotenv
import re
import json
import tiktoken
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# load text_rye
def load_text_rye():
    with open("../../../files/text_rye.txt", "r") as f:
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

def save_processed_chunks(chunks, metadatas, output_prefix="processed_text"):
    """
    Save processed chunks and metadata to files for embeddings.py to consume
    NO embedding or Chroma DB creation - just file output
    """
    print(f"[DEBUG] Saving processed text chunks and metadata...")
    
    # Save chunks and metadata separately
    chunks_file = f"{output_prefix}_chunks.json"
    metadata_file = f"{output_prefix}_metadata.json"
    
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved {len(chunks)} chunks to {chunks_file}")
    
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved {len(metadatas)} metadata entries to {metadata_file}")
    
    # Create IDs for embeddings.py to use
    all_ids = [f"chunk_{meta['chunk_id']}_ch_{meta['chapter']}" for meta in metadatas]
    ids_file = f"{output_prefix}_ids.json"
    with open(ids_file, "w") as f:
        json.dump(all_ids, f)
    print(f"[DEBUG] Saved {len(all_ids)} IDs to {ids_file}")
    
    # Save combined format for convenience
    combined_file = f"{output_prefix}_combined.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": chunks,
            "metadatas": metadatas,
            "ids": all_ids
        }, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] Saved combined data to {combined_file}")
    
    print(f"[DEBUG] Top 5 IDs: {all_ids[:5]}")
    
    return {
        "chunks_file": chunks_file,
        "metadata_file": metadata_file,
        "ids_file": ids_file,
        "combined_file": combined_file,
        "total_chunks": len(chunks)
    }

if __name__ == "__main__":
    file_path = "../../../files/text_rye.txt"
    
    # Create chunks and metadata with TOKEN-based splitting
    chunks, metadatas = create_chapter_metadata_with_splitter(
        file_path, 
        chunk_tokens=300,        # 300 TOKENS per chunk
        chunk_overlap_tokens=50  # 50 TOKENS overlap
    )
    
    # Save processed data for embeddings.py to consume
    print(f"\n[INFO] Saving processed text for embedding...")
    result = save_processed_chunks(chunks, metadatas, output_prefix="literacy_analysis")
    
    print(f"\n[SUMMARY] Text processing complete!")
    print(f"[SUMMARY] Total chunks: {len(chunks)}")
    print(f"[SUMMARY] Total metadata entries: {len(metadatas)}")
    
    print(f"\n[SUMMARY] Files created for embeddings.py:")
    print(f"[SUMMARY] - {result['chunks_file']} ({result['total_chunks']} chunks)")
    print(f"[SUMMARY] - {result['metadata_file']} ({len(metadatas)} metadata entries)")
    print(f"[SUMMARY] - {result['ids_file']} ({len(metadatas)} IDs)")
    print(f"[SUMMARY] - {result['combined_file']} (combined format)")
    
    print(f"\n[DEBUG] Top 5 metadata entries:")
    for i, meta in enumerate(metadatas[:5]):
        print(f"[DEBUG] Metadata {i}: {meta}")
    
    print(f"\n[INFO] Ready for embeddings.py to process these files!")