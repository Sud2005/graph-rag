import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb

print("📦 Loading chunks...")
with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"   Loaded {len(chunks)} chunks")

print("\n🧠 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast, free, runs locally

print("\n📊 Setting up ChromaDB...")
client = chromadb.PersistentClient(path="data/chroma_db")

# Delete existing collection if rebuilding
try:
    client.delete_collection("pubmed_papers")
except:
    pass

collection = client.create_collection(
    name="pubmed_papers",
    metadata={"hnsw:space": "cosine"}
)

print("\n⚡ Embedding and storing chunks (this takes a few minutes)...")
BATCH_SIZE = 64

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [c["text"] for c in batch]
    ids = [c["chunk_id"] for c in batch]
    metadatas = [{"paper_id": c["paper_id"], "chunk_index": c["chunk_index"]} for c in batch]

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

print(f"\n✅ Vector DB built!")
print(f"   Total chunks stored: {collection.count()}")
print(f"   Saved to: data/chroma_db/")