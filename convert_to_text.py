import fitz  # PyMuPDF
import os

def pdf_to_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    pdf_path = "text_rye.pdf"

    base_name = os.path.splitext(pdf_path)[0]
    output_path = f"{base_name}.txt"
    
    # Extract and save the text
    extracted_text = pdf_to_text(pdf_path)
    save_text_to_file(extracted_text, output_path)
    print(f"Text has been extracted and saved to {output_path}")