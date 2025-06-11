import sys, chromadb, torch
from transformers import AutoTokenizer, AutoModel

# ── 1. open the same collection you built earlier ──────────────────────────────
client = chromadb.PersistentClient(path="chroma_db")          # same folder you used
col    = client.get_collection("sentence_collection")

print("Total vectors in collection:", col.count())
print("Peek:", col.peek()["ids"][:10], "…")                   # sanity check

# ── 2. load the **same** BERT weights you used for ingestion ───────────────────
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model     = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model.eval()                                                  # disable gradients

# ── 3. the missing helper: turn text → 768-d CLS vector ────────────────────────
def embed(text: str):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        output = model(**inputs)
        # Mean Pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        vec = sum_embeddings / sum_mask
    return vec[0].numpy().tolist()

# ── 4. get the query text (from CLI or fallback) ───────────────────────────────
while True:
    query_text = input("Enter your question (or type 'exit' to quit): ").strip()
    if not query_text or query_text.lower() == "exit":
        print("Goodbye!")
        break

    q_vec = embed(query_text)
    res = col.query(query_embeddings=[q_vec], n_results=10)

    print("==============================================\nQuestion:", query_text)
    for rank, (doc_id, dist) in enumerate(zip(res["ids"][0],
                                              res["distances"][0]), 1):
        print(f"{rank}. id={doc_id}  distance={dist:.4f}")
        chunk = res.get("documents", [["(no text saved)"]])[0][rank-1]
        print("   ", chunk if chunk else "(no documents)\n")