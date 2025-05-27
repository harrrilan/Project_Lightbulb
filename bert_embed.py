from transformers import BertModel, BertTokenizer
from text_preprocess_sentence import preprocess_sentences
import numpy as np

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

# Step 4: Embed in chunks
chunk_size = 3  # <-- You can change this value as needed
print(f"[DEBUG] Embedding sentences in chunks of {chunk_size}...")

all_embeddings = []

for start_idx in range(0, len(sentences), chunk_size):
    chunk_sentences = sentences[start_idx:start_idx + chunk_size]
    chunk_text = ' '.join(chunk_sentences)
    # Tokenize and check token count
    encoded_input = tokenizer(chunk_text, return_tensors='pt')
    num_tokens = encoded_input['input_ids'].shape[1]
    print(f"[DEBUG] Chunk {start_idx//chunk_size + 1}: {num_tokens} tokens")

    # Embed
    output = model(**encoded_input)
    print(f"[DEBUG] Chunk {start_idx//chunk_size + 1} embedding completed.")

    # Optionally, print first few embedding values for debug
    embedding = output.last_hidden_state[0][0].detach().numpy()
    all_embeddings.append(embedding)

    print(f"[DEBUG] First 5 embedding values: {embedding[:5]}")
    print("-" * 40)

print(f"[DEBUG] Total embeddings collected: {len(all_embeddings)}")

# Save embeddings
np.save('all_embeddings.npy', np.array(all_embeddings))
print("[DEBUG] Embeddings saved to all_embeddings.npy")