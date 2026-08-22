# src/chunking.py

import os
import json

def load_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def chunk_text(text: str, source: str, chunk_size: int = 300, overlap: int = 50, start_chunk_id: int = 1) -> list:
    """
    Same chunking logic as before, but now tags every chunk with its
    `source` filename and accepts a starting chunk_id so IDs stay unique
    when chunks from multiple PDFs get merged together.
    """
    start, text_len = 0, len(text)
    chunks = []
    chunk_id = start_chunk_id
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            'chunk_id': chunk_id,
            'source': source,
            'text': chunk.strip()
        })
        chunk_id += 1
        start = end - overlap
    
    return chunks


def save_chunks(chunks: list, output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)
    print(f"File saved to {output_path}")


def chunk_all_texts(processed_dir: str, output_path: str) -> None:
    """
    Finds every .txt file in processed_dir (produced by text_extraction.py,
    one per source PDF), chunks each one, and merges everything into ONE
    master chunks.json file that embedding.py and retriever.py read from.
    """
    txt_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]

    if not txt_files:
        print(f"No .txt files found in {processed_dir}. Run text_extraction.py first.")
        return

    all_chunks = []
    next_id = 1

    for txt_file in txt_files:
        txt_path = os.path.join(processed_dir, txt_file)
        text = load_text(txt_path)
        chunks = chunk_text(text, source=txt_file, start_chunk_id=next_id)
        all_chunks.extend(chunks)
        next_id += len(chunks)
        print(f"  {txt_file}: {len(chunks)} chunks")

    save_chunks(all_chunks, output_path)


if __name__ == "__main__":
    processed_dir = os.path.join("..", "data", "processed")
    output_file = os.path.join(processed_dir, "all_chunks.json")

    try:
        chunk_all_texts(processed_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")