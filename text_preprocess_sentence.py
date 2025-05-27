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

def preprocess_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    sentences = sent_tokenize(text)
    
    token_by_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z\s\.,!?]', '', sentence)
        sentence = ' '.join(sentence.split())

        if len(sentence) > 0:
            token_by_sentences.append(sentence)
    
    return token_by_sentences

file_path = "text_rye.txt"
processed_sentences = preprocess_sentences(file_path)

if __name__ == "__main__":
    print(f"Total sentences after preprocessing: {len(processed_sentences)}")
    for i, sentence in enumerate(processed_sentences[:10]):
        print(f"\n{i+1}. {sentence}")