import re
import nltk
from nltk.tokenize import word_tokenize
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
def preprocess_text(file_path):
    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens_by_words = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens_by_words


if __name__ == "__main__":
    file_path = "text_rye.txt"
    
    processed_tokens = preprocess_text(file_path)
    print(f"Total tokens after preprocessing: {len(processed_tokens)}")
    print("\nFirst 100 tokens:")
    print(processed_tokens[:100])