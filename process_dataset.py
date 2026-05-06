import json
import os
import re
from tqdm import tqdm

def clean_text(text):
    """Remove junk characters, excessive whitespace"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\[\d+\]', '', text)  # Remove citation numbers like [1]
    text = text.strip()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks by word count.
    500 words ~ 650 tokens — good size for RAG retrieval.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap so context isn't lost at boundaries
    return chunks

def main():
    print("📂 Loading raw papers...")
    with open("data/raw/pubmed_cancer_papers.json", "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"   Loaded {len(papers)} papers")

    all_chunks = []
    skipped = 0

    print("\n✂️  Cleaning and chunking...")
    for paper in tqdm(papers):
        text = clean_text(paper["text"])

        # Skip papers that are too short (likely just abstracts)
        if len(text.split()) < 300:
            skipped += 1
            continue

        chunks = chunk_text(text, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "paper_id": paper["id"],
                "chunk_id": f"{paper['id']}_chunk_{i}",
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

    print(f"\n📊 Stats:")
    print(f"   Papers processed: {len(papers) - skipped}")
    print(f"   Papers skipped (too short): {skipped}")
    print(f"   Total chunks created: {len(all_chunks)}")
    print(f"   Avg chunks per paper: {len(all_chunks) // max(1, len(papers) - skipped)}")

    # Save processed chunks
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Save a separate clean full-text version for GraphRAG ingestion
    clean_papers = []
    for paper in papers:
        text = clean_text(paper["text"])
        if len(text.split()) >= 300:
            clean_papers.append({
                "id": paper["id"],
                "text": text
            })

    clean_path = "data/processed/clean_papers.json"
    with open(clean_path, "w", encoding="utf-8") as f:
        json.dump(clean_papers, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done!")
    print(f"   Chunks saved to: {output_path}")
    print(f"   Clean papers saved to: {clean_path}")

if __name__ == "__main__":
    main()