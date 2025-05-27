import sys, chromadb, torch
from transformers import BertTokenizer, BertModel

# ── 1. open the same collection you built earlier ──────────────────────────────
client = chromadb.PersistentClient(path="chroma_db")          # same folder you used
col    = client.get_collection("3_chuncks_collection")

print("Total vectors in collection:", col.count())
print("Peek:", col.peek()["ids"][:10], "…")                   # sanity check

# ── 2. load the **same** BERT weights you used for ingestion ───────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased")
model.eval()                                                  # disable gradients

# ── 3. the missing helper: turn text → 768-d CLS vector ────────────────────────
def embed(text: str):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        vec    = model(**inputs).last_hidden_state[0, 0]       # [CLS] token
    return vec.numpy().tolist()                                # plain Python list

# ── 4. get the query text (from CLI or fallback) ───────────────────────────────
query_text = " ".join(sys.argv[1:]) or "Test question"
q_vec      = embed(query_text)

# ── 5. search Chroma ───────────────────────────────────────────────────────────
res = col.query(query_embeddings=[q_vec], n_results=5)

print("\nQuestion:", query_text)
for rank, (doc_id, dist) in enumerate(zip(res["ids"][0],
                                          res["distances"][0]), 1):
    print(f"{rank}. id={doc_id}  distance={dist:.4f}")
    # If you saved chunk text, show it; otherwise it'll be None
    chunk = res.get("documents", [["(no text saved)"]])[0][rank-1]
    print("   ", (chunk[:200] + "…") if chunk else "(no documents)\n")
