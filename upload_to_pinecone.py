# upload_to_pinecone.py

import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Loads the variables from your secret .env file
load_dotenv()

# 1. Config — safely reads from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "delhi-traffic-rules"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found! Please set it in your .env file.")

# 2. Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

print("Loading embedding model: paraphrase-multilingual-MiniLM-L12-v2 ...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 3. Load your existing chunks
chunks_path = os.path.join("data", "processed", "all_chunks.json")
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks. Generating embeddings...")

# 4. Extract text and embed in batches
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# 5. Prepare vectors for Pinecone
vectors = []
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk_{i}",
        "values": emb.tolist(),
        "metadata": {
            "text": chunk["text"],
            "source": chunk.get("source", "unknown")
        }
    })

# 6. Upsert to Pinecone in batches of 100
BATCH_SIZE = 100
print("Uploading vectors to Pinecone...")
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i : i + BATCH_SIZE]
    index.upsert(vectors=batch)
    print(f"Uploaded {min(i + BATCH_SIZE, len(vectors))} / {len(vectors)} chunks")

print("✅ Successfully uploaded all chunks to Pinecone!")