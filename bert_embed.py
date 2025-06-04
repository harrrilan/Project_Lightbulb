from transformers import AutoTokenizer, AutoModel
from text_preprocess_sentence import preprocess_sentences
import numpy as np
import json
import statistics
import torch
from visualization import plot_embeddings_3d_tsne

TARGET_TOKENS = 25      # aim per chunk
OVERLAP_TOKENS = 10      # keep this many tokens from the previous chunk
MAX_MODEL_TOKENS = 50

all_embeddings = []  # each = np.ndarray (hidden_size,)
all_chunks = []      # raw text of each chunk
token_counts = []    # number of tokens in each chunk


# Step 1: Preprocess sentences
file_path = "text_rye.txt"
print(f"[DEBUG] Preprocessing sentences from {file_path}...")
sentences = preprocess_sentences(file_path)

# Step 2: Print first few preprocessed sentences
print("[DEBUG] First 5 preprocessed sentences:")
for i, sent in enumerate(sentences[:5]):
    print(f"{i+1}: {sent}")

# Step 3: Load BERT model and tokenizer
print("[DEBUG] Loading BERT model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Embed in chunks
print("[DEBUG] Building and embedding chunks (~100 tokens, 50‑token overlap) ...")
start_idx = 0  # sentence index
chunk_id = 1

while start_idx < len(sentences):
    chunk_sentences = []
    token_so_far = 0

    # Accumulate sentences until we hit the target token count
    while start_idx + len(chunk_sentences) < len(sentences) and token_so_far < TARGET_TOKENS:
        next_sentence = sentences[start_idx + len(chunk_sentences)]
        chunk_sentences.append(next_sentence)
        token_so_far = len(
            tokenizer(" ".join(chunk_sentences), return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ][0]
        )

    chunk_text = " ".join(chunk_sentences)
    encoded_input = tokenizer(
        chunk_text, return_tensors="pt", truncation=True, max_length=MAX_MODEL_TOKENS
    )
    num_tokens = encoded_input["input_ids"].shape[1]
    print(f"[DEBUG] Chunk {chunk_id}: {num_tokens} tokens")

    # Embed (mean pooling)
    with torch.no_grad():
        output = model(**encoded_input)
        attention_mask = encoded_input['attention_mask']
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask)[0].cpu().numpy()
    # Store artefacts
    all_embeddings.append(embedding)
    all_chunks.append(chunk_text)
    token_counts.append(num_tokens)

    # Debug sample values
    if chunk_id <= 3:
        print(f"        First 5 embedding vals: {embedding[:5]}")

    # Advance the start index by (num_tokens - OVERLAP_TOKENS) worth of tokens
    #   – walk sentence‑by‑sentence until we've skipped that many tokens.
    tokens_to_skip = max(num_tokens - OVERLAP_TOKENS, 1)
    skipped = 0
    while start_idx < len(sentences) and skipped < tokens_to_skip:
        skipped += len(
            tokenizer(sentences[start_idx], add_special_tokens=False)["input_ids"]
        )
        start_idx += 1

    chunk_id += 1

print(f"[DEBUG] Total embeddings collected: {len(all_embeddings)}")

# ---------------------------------------------------------------------------
# Step 4: Persist embeddings & chunks
# ---------------------------------------------------------------------------
np.save("all_embeddings.npy", np.array(all_embeddings))
with open("all_chunks.json", "w", encoding="utf-8") as fh:
    json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
print("[DEBUG] Saved all_embeddings.npy and all_chunks.json")

# ---------------------------------------------------------------------------
# Step 5: Summary statistics for token counts
# ---------------------------------------------------------------------------
print("\n[SUMMARY] Token count statistics")
print(f"  # chunks : {len(token_counts)}")
print(f"  min      : {min(token_counts)}")
print(f"  max      : {max(token_counts)}")
print(f"  mean     : {statistics.mean(token_counts):.2f}")
print(f"  std      : {statistics.stdev(token_counts):.2f}")

all_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
with open("all_ids.json", "w") as fh:
    json.dump(all_ids, fh)

# ---------------------------------------------------------------------------
# Step 6: Visualize embeddings in 3D using t-SNE
# ---------------------------------------------------------------------------
plot_embeddings_3d_tsne("all_embeddings.npy", "all_chunks.json") 