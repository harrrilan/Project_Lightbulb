import openai
import numpy as np
import json
import re
import tiktoken
from openai import OpenAI

def split_into_chapters(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    parts = re.split(r"(?m)^\s*\d+\s*$", text)

    chapters = [p.strip() for p in parts if p.strip()]
    return chapters

def get_openai_embedding(text, model="text-embedding-3-large"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

file_path = "text_rye.txt"
print(f"[INFO] Loading and splitting chapters from {file_path} ...")
chapters = split_into_chapters(file_path)
print(f"[INFO] Total chapters found: {len(chapters)}")

print(f"[DEBUG] Chapters found: {len(chapters)}")

# Print the number of tokens in each chapter

client = OpenAI()

def count_tokens(text: str, model: str = "text-embedding-3-large") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

chapter_token_counts = []
for i, c in enumerate(chapters, start=1):
    tokens = count_tokens(c)
    chapter_token_counts.append(tokens)
    print(f"[DEBUG] Chapter {i} starts: {repr(c[:100])}")
    print(f"[DEBUG] Chapter {i} token count: {tokens}")

chapter_token_counts = []
for i, c in enumerate(chapters):
    # Use OpenAI's tokenizer to count tokens
    try:
        tokens = count_tokens(c)
    except Exception as e:
        print(f"[DEBUG] Token count error for chapter {i+1}: {e}")
        tokens = -1
    chapter_token_counts.append(tokens)
    print(f"[DEBUG] Chapter {i+1} starts: {repr(c[:100])}")
    print(f"[DEBUG] Chapter {i+1} token count: {tokens}")

# =====================   for chapter 25  =====================
MAX_TOKENS = 8192
OVERLAP    = 500

if len(chapters) >= 25 and chapter_token_counts[24] > MAX_TOKENS:
    print(f"[WARN] Chapter 25 is too long: {chapter_token_counts[24]} tokens. Splitting with overlap.")
    chapter_25 = chapters[24]

    # 1) Load tiktoken for “text-embedding-3-large”
    import tiktoken
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")

    # 2) Encode the entire chapter into token IDs
    token_ids = encoding.encode(chapter_25)
    total_tokens = len(token_ids)

    # 3) Find the midpoint and split boundaries
    half_point   = total_tokens // 2
    half_overlap = OVERLAP // 2

    # ensure we don’t go negative or past the end
    start2 = max(0, half_point - half_overlap)
    end1   = min(total_tokens, half_point + half_overlap)

    # 4) Build each part’s token-ID slice
    part1_ids = token_ids[:end1]
    part2_ids = token_ids[start2:]

    # 5) Decode token IDs back into text
    part1_text = encoding.decode(part1_ids)
    part2_text = encoding.decode(part2_ids)

    print(f"[DEBUG] Chapter 25 total tokens: {total_tokens}")
    print(f"[DEBUG] Part 1 tokens: {len(part1_ids)}  (0 → {end1-1})")
    print(f"[DEBUG] Part 2 tokens: {len(part2_ids)}  ({start2} → {total_tokens-1})")

    # Replace chapter 25 with these two halves
    chapters = chapters[:24] + [part1_text, part2_text] + chapters[25:]
    print(f"[INFO] Chapters after splitting chapter 25: {len(chapters)}")


# =====================   for chapter 25 END  =====================

all_embeddings = []
all_chunks = []
print("[INFO] Starting embedding for each chapter ...")
for i, chapter in enumerate(chapters):
    print(f"[DEBUG] Embedding chapter {i+1}/{len(chapters)}")
    embedding = get_openai_embedding(chapter)
    all_embeddings.append(embedding)
    all_chunks.append(chapter)
    print(f"[DEBUG] Finished embedding chapter {i+1}")
    print(f"        First 5 embedding values: {embedding[:5]}")

# Show samples after all embeddings are created
print("[SAMPLE] First 5 all_embeddings (first 5 values of each):")
for idx, emb in enumerate(all_embeddings[:5]):
    print(f"  Embedding {idx+1}: {emb[:5]}")

print("[SAMPLE] First 5 all_chunks:")
for idx, chunk in enumerate(all_chunks[:5]):
    print(f"  Chunk {idx+1}: {repr(chunk[:100])} ...")

print("[INFO] Saving all_embeddings.npy ...")
np.save("all_embeddings.npy", np.array(all_embeddings))
print("[INFO] Saved all_embeddings.npy")

print("[INFO] Saving all_chunks.json ...")
with open("all_chunks.json", "w", encoding="utf-8") as fh:
    json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
print("[INFO] Saved all_chunks.json")

print("[INFO] Saving all_ids.json ...")
all_ids = [f"chapter_{i+1}" for i in range(len(all_chunks))]
with open("all_ids.json", "w") as fh:
    json.dump(all_ids, fh)
print("[INFO] Saved all_ids.json")

print("[SAMPLE] First 5 all_ids:")
for idx, cid in enumerate(all_ids[:5]):
    print(f"  ID {idx+1}: {cid}")

print(f"[SUMMARY] Embedding complete. Chapters: {len(all_chunks)}. Embeddings shape: {np.array(all_embeddings).shape}")