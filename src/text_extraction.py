
import os
import json
import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File Not Found: {pdf_path}")
    
    extracted_text = []

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                extracted_text.append(text)
            else:
                print(f"Warning: Page {page_number} has no extractable text")
            
    return "\n".join(text.replace('\n', '') for text in extracted_text)


def extract_all_pdfs(raw_dir: str, output_dir: str) -> None:
    """
    Loops over every PDF in raw_dir and saves each one's extracted text
    as its own .txt file in output_dir, named after the PDF.
    This is what lets you add more PDFs later without touching this file.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {raw_dir}")
        return

    for pdf_file in pdf_files:
        pdf_path = os.path.join(raw_dir, pdf_file)
        output_name = os.path.splitext(pdf_file)[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        try:
            text = extract_text_from_pdf(pdf_path)
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(text)
            print(f"Extraction complete. Text saved to {output_path}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")


if __name__ == "__main__":
    # Put ALL your source PDFs in ../data/raw/ (e.g. delhi_traffic_rules.pdf,
    # motor_vehicles_act.pdf, etc.) — every PDF found there gets processed.
    raw_dir = os.path.join("..", "data", "raw")
    output_dir = os.path.join("..", "data", "processed")

    extract_all_pdfs(raw_dir, output_dir)