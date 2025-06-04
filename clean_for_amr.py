import argparse
import json
import re
import sys
from nltk.tokenize import sent_tokenize

#python amr.py --input text_rye.txt --output preprocessed_for_amr.json

def clean_raw_text(raw_text: str) -> str:
    """
    Perform light cleaning on the raw text extracted from a PDF:
    1. Remove hyphenation at line breaks (e.g., "exam-\nple" → "example").
    2. Strip page numbers that appear alone on a line or at the start of a line.
    3. Collapse multiple newlines into a single space.
    4. Replace remaining newline characters with spaces.
    """

    # 1) Remove hyphenation at line breaks
    text = re.sub(r'-\s*\n\s*', '', raw_text)

    # 2) Remove standalone page numbers (lines that consist of only digits possibly surrounded by whitespace)
    text = re.sub(r'(?m)^\s*\d+\s*$\n?', '', text)

    # 3) Remove any page-number-like stray digits at the start of lines (e.g., "  12 Once upon a time...")
    text = re.sub(r'(?m)^\s*\d+\s+', '', text)

    # 4) Replace any remaining newline characters with a single space
    #    Also collapse multiple spaces into one
    text = re.sub(r'\s*\n\s*', ' ', text)
    text = re.sub(r'[ ]{2,}', ' ', text)

    return text.strip()


def segment_sentences(clean_text: str) -> list:
    """
    Use NLTK's PunktSentenceTokenizer to split the cleaned text into sentences.
    Returns a list of sentence strings.
    """
    # nltk.sent_tokenize expects a reasonably "flattened" text with no spurious newlines.
    sentences = sent_tokenize(clean_text)
    # Optionally, filter out empty or too-short segments
    filtered = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= 5:  # skip extremely short fragments
            filtered.append(sent)
    return filtered


def save_sentences_to_json(sentences: list, output_path: str):
    """
    Save the list of sentences to a JSON file, one sentence per list element.
    """
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(sentences, f_out, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(sentences)} sentences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess a raw text file into sentence-level JSON for AMR parsing."
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help="Path to the raw text file (e.g., text_rye.txt)"
    )
    parser.add_argument(
        '--output', '-o', default='sentences.json',
        help="Path to the output JSON file (default: sentences.json)"
    )
    args = parser.parse_args()

    # 1) Load raw text
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # 2) Clean the raw text
    cleaned = clean_raw_text(raw)
    if not cleaned:
        print("Warning: After cleaning, text is empty. Check the input file.")
        sys.exit(1)

    # 3) Segment into sentences
    sentences = segment_sentences(cleaned)
    if not sentences:
        print("Warning: No sentences extracted. Check the tokenizer or text content.")
        sys.exit(1)

    # 4) Save to JSON
    save_sentences_to_json(sentences, args.output)


if __name__ == '__main__':
    main()
