"""
Advanced hierarchical chunking for biomedical papers.
Implements: Document -> Section -> Semantic chunk splitting.
Uses sentence similarity drops to find natural topic boundaries.
"""

import json
import os
import re
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Biomedical paper section patterns
SECTION_PATTERNS = [
    (r"(?i)\b(abstract)\b", "Abstract"),
    (r"(?i)\b(introduction)\b", "Introduction"),
    (r"(?i)\b(background)\b", "Background"),
    (r"(?i)\b(materials?\s*and\s*methods?|methods?|methodology)\b", "Methods"),
    (r"(?i)\b(results?\s*and\s*discussion|results?)\b", "Results"),
    (r"(?i)\b(discussion)\b", "Discussion"),
    (r"(?i)\b(conclusion|conclusions|concluding\s*remarks?)\b", "Conclusion"),
    (r"(?i)\b(references?|bibliography)\b", "References"),
    (r"(?i)\b(supplementary|supplemental|supporting\s*information)\b", "Supplementary"),
    (r"(?i)\b(acknowledgement|acknowledgment)\b", "Acknowledgements"),
]


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based rules tuned for biomedical text."""
    # Handle common abbreviations that shouldn't split
    text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Fig|Figs|Tab|Eq|Eqs|Vol|No|vs|etc|al|approx)\.", r"\1<PERIOD>", text)
    text = re.sub(r"(\d)\.", r"\1<PERIOD>", text)  # Numbers with dots
    text = re.sub(r"\be\.g\.", "e<PERIOD>g<PERIOD>", text)
    text = re.sub(r"\bi\.e\.", "i<PERIOD>e<PERIOD>", text)

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    # Restore periods
    sentences = [s.replace("<PERIOD>", ".") for s in sentences]

    # Filter out very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences


def detect_sections(text: str) -> List[Dict]:
    """
    Detect section boundaries in a biomedical paper.
    Returns a list of dicts: {section_name, start_idx, end_idx, text}.
    """
    lines = text.split("\n")
    sections = []
    current_section = "Unknown"
    current_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            current_lines.append(line)
            continue

        matched = False
        for pattern, section_name in SECTION_PATTERNS:
            # Section headers are typically short lines matching a pattern
            if len(stripped) < 80 and re.search(pattern, stripped):
                # Save current section before starting new one
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if len(section_text) > 50:
                        sections.append({
                            "section": current_section,
                            "text": section_text,
                            "start_line": current_start,
                            "end_line": i - 1
                        })
                current_section = section_name
                current_lines = []
                current_start = i
                matched = True
                break

        if not matched:
            current_lines.append(line)

    # Capture the last section
    if current_lines:
        section_text = "\n".join(current_lines).strip()
        if len(section_text) > 50:
            sections.append({
                "section": current_section,
                "text": section_text,
                "start_line": current_start,
                "end_line": len(lines) - 1
            })

    # If no sections were detected, treat the whole text as one section
    if not sections:
        sections.append({
            "section": "FullText",
            "text": text,
            "start_line": 0,
            "end_line": len(lines) - 1
        })

    return sections


def semantic_chunk_section(
    sentences: List[str],
    embedder: SentenceTransformer,
    similarity_threshold: float = 0.3,
    min_chunk_sentences: int = 3,
    max_chunk_sentences: int = 15,
    overlap: int = 2
) -> List[List[str]]:
    """
    Split a list of sentences into semantic chunks by detecting
    drops in cosine similarity between consecutive sentence embeddings.

    Args:
        sentences: List of sentences to chunk.
        embedder: SentenceTransformer model for embedding.
        similarity_threshold: Split where similarity drops below this.
        min_chunk_sentences: Minimum sentences per chunk.
        max_chunk_sentences: Maximum sentences per chunk.
        overlap: Number of overlapping sentences between chunks.

    Returns:
        List of sentence groups (each group is a chunk).
    """
    if len(sentences) <= min_chunk_sentences:
        return [sentences]

    # Embed all sentences
    embeddings = embedder.encode(sentences, show_progress_bar=False)

    # Compute pairwise cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[i + 1].reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    # Find split points where similarity drops below threshold
    split_points = []
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold and (i + 1) >= min_chunk_sentences:
            split_points.append(i + 1)

    # Also enforce max chunk size
    if not split_points:
        # No natural breaks found; split by max_chunk_sentences
        split_points = list(range(max_chunk_sentences, len(sentences), max_chunk_sentences))

    # Build chunks from split points
    chunks = []
    prev = 0
    for sp in split_points:
        chunk_sentences = sentences[prev:sp]
        if len(chunk_sentences) >= min_chunk_sentences:
            chunks.append(chunk_sentences)
            prev = max(0, sp - overlap)  # slide back for overlap

    # Add remaining sentences
    remaining = sentences[prev:]
    if remaining:
        if chunks and len(remaining) < min_chunk_sentences:
            # Merge small trailing chunk with the previous one
            chunks[-1].extend(remaining)
        else:
            chunks.append(remaining)

    return chunks


def process_paper_hierarchical(
    paper_id: str,
    text: str,
    embedder: SentenceTransformer,
    similarity_threshold: float = 0.3,
    overlap: int = 2
) -> List[Dict]:
    """
    Process a single paper into hierarchical chunks.
    Document -> Section -> Semantic chunks.

    Args:
        paper_id: The PMC ID of the paper.
        text: Full text of the paper.
        embedder: SentenceTransformer model.
        similarity_threshold: Cosine similarity threshold for splitting.
        overlap: Number of overlapping sentences between chunks.

    Returns:
        List of chunk dicts with metadata.
    """
    sections = detect_sections(text)
    all_chunks = []
    chunk_counter = 0

    for section_info in sections:
        section_name = section_info["section"]
        section_text = section_info["text"]

        # Skip references and acknowledgements
        if section_name in ("References", "Supplementary", "Acknowledgements"):
            continue

        sentences = split_into_sentences(section_text)
        if not sentences:
            continue

        # Perform semantic chunking within each section
        sentence_groups = semantic_chunk_section(
            sentences,
            embedder,
            similarity_threshold=similarity_threshold,
            overlap=overlap
        )

        for group in sentence_groups:
            chunk_text = " ".join(group)
            # Skip very short chunks
            if len(chunk_text.split()) < 30:
                continue

            all_chunks.append({
                "chunk_id": f"{paper_id}_hchunk_{chunk_counter}",
                "paper_id": str(paper_id),
                "section": section_name,
                "text": chunk_text,
                "num_sentences": len(group),
                "word_count": len(chunk_text.split()),
                "chunk_index": chunk_counter
            })
            chunk_counter += 1

    return all_chunks


def build_hierarchical_chunks(
    input_path: str = None,
    output_path: str = None,
    similarity_threshold: float = 0.3,
    overlap: int = 2
) -> List[Dict]:
    """
    Process all papers into hierarchical semantic chunks and save to disk.

    Args:
        input_path: Path to clean_papers.json.
        output_path: Path to save hierarchical_chunks.json.
        similarity_threshold: Cosine similarity threshold for chunking.
        overlap: Sentence overlap between chunks.

    Returns:
        List of all hierarchical chunks.
    """
    if input_path is None:
        input_path = os.path.join("data", "processed", "clean_papers.json")
    if output_path is None:
        output_path = os.path.join("data", "processed", "hierarchical_chunks.json")

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Loading papers from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    print(f"Processing {len(papers)} papers with hierarchical chunking...")
    all_chunks = []

    for i, paper in enumerate(papers):
        paper_chunks = process_paper_hierarchical(
            paper_id=paper["id"],
            text=paper["text"],
            embedder=embedder,
            similarity_threshold=similarity_threshold,
            overlap=overlap
        )
        all_chunks.extend(paper_chunks)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(papers)} papers, {len(all_chunks)} chunks so far")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Stats
    sections_found = {}
    for c in all_chunks:
        sec = c["section"]
        sections_found[sec] = sections_found.get(sec, 0) + 1

    print(f"\nHierarchical Chunking Complete:")
    print(f"  Total papers: {len(papers)}")
    print(f"  Total chunks: {len(all_chunks)}")
    print(f"  Avg chunks/paper: {len(all_chunks) / max(1, len(papers)):.1f}")
    print(f"  Avg words/chunk: {sum(c['word_count'] for c in all_chunks) / max(1, len(all_chunks)):.0f}")
    print(f"  Section distribution:")
    for sec, count in sorted(sections_found.items(), key=lambda x: -x[1]):
        print(f"    {sec}: {count}")
    print(f"  Saved to: {output_path}")

    return all_chunks


if __name__ == "__main__":
    build_hierarchical_chunks()
