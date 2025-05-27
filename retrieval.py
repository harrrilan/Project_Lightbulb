import chromadb
from transformers import BertTokenizer, BertModel
import torch, json

client = chromadb.PersistentClient(path="chroma_db")
col    = client.get_collection("3_chuncks_collection")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertModel.from_pretrained("bert-base-uncased").eval()

def embed(text: str):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        return model(**inputs).last_hidden_state[0, 0].numpy().tolist()

def retrieve(query, k=5):
    res = col.query(query_embeddings=[embed(query)], n_results=k)
    return "\n\n".join(res["documents"][0])
