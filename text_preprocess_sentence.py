import re
import nltk
import tiktoken
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


### Run this once to download the necessary data
'''
nltk.download('punkt', download_dir='/Users/harrilan/nltk_data')
print("Download complete! 1/4")
nltk.download('punkt_tab', download_dir='/Users/harrilan/nltk_data')
print("Download complete! 2/4")
nltk.download('stopwords', download_dir='/Users/harrilan/nltk_data')
print("Download complete! 3/4")
nltk.download('wordnet', download_dir='/Users/harrilan/nltk_data')
print("Download complete! 4/4")
'''

def num_tokens(text, model="text-embedding-3-large"):
    """Count the number of tokens in a text string"""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def clean_title(text):
    """Special handling for title and dedication"""
    text = text.title()
    text = re.sub(r'\bJ\.d\.|J\.D\.|J\.d', 'J.D.', text)
    
    parts = text.split('To My Mother')
    if len(parts) > 1:
        return f"{parts[0].strip()} | To My Mother"
    return text.strip()

def split_long_sentence(sentence, max_tokens=100):
    """Split long sentences at logical points based on token count"""
    if num_tokens(sentence) <= max_tokens:
        return [sentence]
    
    split_points = [
        ', and', ', but', ', what', 
        ', how', ', if', '; '
    ]
    
    for point in split_points:
        if point in sentence:
            parts = sentence.split(point, 1)
            return [parts[0]] + split_long_sentence(parts[1], max_tokens)
    
    return [sentence]

def preprocess_sentences(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    sentences = sent_tokenize(text)
    processed_chunks = []
    
    # Handle title
    if sentences:
        title = clean_title(sentences[0])
        processed_chunks.append(title)
        sentences = sentences[1:]
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence = re.sub(r"(\w)'(\w)", r"\1'\2", sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Split based on token count instead of character length
        if num_tokens(sentence) > 100:  # OpenAI recommends chunks < 100 tokens
            chunks = split_long_sentence(sentence)
            processed_chunks.extend(chunks)
        else:
            processed_chunks.append(sentence)
    
    # Final cleaning with token count information
    final_chunks = []
    for chunk in processed_chunks:
        chunk = chunk.strip()
        if chunk and num_tokens(chunk) > 5:  # Avoid very short chunks
            final_chunks.append(chunk)
    
    return final_chunks

if __name__ == "__main__":
    file_path = "text_chunk.txt"
    processed_chunks = preprocess_sentences(file_path)
    
    print(f"\nTotal chunks after preprocessing: {len(processed_chunks)}")
    print("\nFirst 5 chunks (to show formatting):")
    for i, chunk in enumerate(processed_chunks[:5]):
        tokens = num_tokens(chunk)
        print(f"\n{i+1}. Tokens: {tokens}")
        print(f"Content: \"{chunk}\"")
        print("-" * 80)