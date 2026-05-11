"""
Advanced hybrid retrieval system.
Implements: BM25 + Vector search + HyDE + RRF fusion + Cross-encoder reranking + Context compression.
"""

import json
import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()


def count_tokens(text: str) -> int:
    """Approximate token count from text."""
    return len(text.split()) * 4 // 3


class HybridRetriever:
    """
    Advanced retrieval system combining BM25, vector search, HyDE,
    reciprocal rank fusion, cross-encoder reranking, and context compression.
    """

    def __init__(
        self,
        chroma_db_path: str = None,
        collection_name: str = "pubmed_advanced",
        chunks_path: str = None,
        embedder_name: str = "all-MiniLM-L6-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the hybrid retriever.

        Args:
            chroma_db_path: Path to ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection.
            chunks_path: Path to hierarchical_chunks.json.
            embedder_name: SentenceTransformer model name.
            reranker_name: Cross-encoder model name for reranking.
        """
        if chroma_db_path is None:
            chroma_db_path = os.path.join("data", "chroma_advanced_db")
        if chunks_path is None:
            chunks_path = os.path.join("data", "processed", "hierarchical_chunks.json")

        self.embedder_name = embedder_name
        self.reranker_name = reranker_name
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.chunks_path = chunks_path

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedder_name)

        print("Loading cross-encoder reranker...")
        self.reranker = CrossEncoder(reranker_name)

        # Load chunks
        print(f"Loading chunks from {chunks_path}...")
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Build BM25 index
        print("Building BM25 index...")
        self.chunk_texts = [c["text"] for c in self.chunks]
        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Build chunk ID -> index map
        self.chunk_id_map = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self._ensure_collection()

        # Initialize Gemini client for HyDE and compression
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.llm_client = genai.Client(api_key=api_key)
        else:
            self.llm_client = None
            print("WARNING: No GEMINI_API_KEY found. HyDE and compression disabled.")

        print(f"Retriever ready. {len(self.chunks)} chunks indexed.")

    def _ensure_collection(self):
        """Create or load the ChromaDB collection and index chunks if needed."""
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            existing_count = self.collection.count()
            if existing_count >= len(self.chunks):
                print(f"  ChromaDB collection already has {existing_count} chunks.")
                return
            else:
                print(f"  Collection has {existing_count} chunks but we have {len(self.chunks)}. Rebuilding...")
                self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            print("  Creating new ChromaDB collection...")

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Batch insert chunks
        batch_size = 200
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            metadatas = [
                {
                    "paper_id": c["paper_id"],
                    "section": c["section"],
                    "word_count": c["word_count"],
                    "chunk_index": c["chunk_index"]
                }
                for c in batch
            ]
            embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

            if (i + batch_size) % 1000 == 0 or i + batch_size >= len(self.chunks):
                print(f"  Indexed {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")

        print(f"  ChromaDB collection ready with {self.collection.count()} chunks.")

    def vector_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        section_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search ChromaDB for similar chunks using vector similarity.

        Args:
            query_embedding: The query embedding vector.
            top_k: Number of results to return.
            section_filter: Optional section name to filter by.

        Returns:
            List of result dicts with chunk_id, text, score, metadata.
        """
        where_filter = None
        if section_filter:
            where_filter = {"section": section_filter}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine distance -> similarity
                "metadata": results["metadatas"][0][i]
            })

        return output

    def bm25_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Search chunks using BM25 keyword matching.

        Args:
            query: The search query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk_id, text, score.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        output = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                output.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(scores[idx]),
                    "metadata": {
                        "paper_id": chunk["paper_id"],
                        "section": chunk["section"],
                        "word_count": chunk["word_count"]
                    }
                })

        return output

    def generate_hyde_document(self, question: str) -> str:
        """
        Generate a Hypothetical Document Embedding (HyDE).
        Creates a fake ideal answer that is then embedded and used for retrieval.

        Args:
            question: The user question.

        Returns:
            A hypothetical answer document.
        """
        if not self.llm_client:
            return question  # Fallback: just use the question itself

        prompt = f"""You are a biomedical research expert. Write a short, detailed paragraph
that would be the ideal answer to this question, as if it were found in a research paper.
Include specific gene names, mechanisms, and technical details.
Do NOT include phrases like "I think" or "In my opinion".

Question: {question}

Ideal research paper excerpt:"""

        try:
            for attempt in range(3):
                try:
                    response = self.llm_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    if "503" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                        if attempt < 2:
                            print(f"Gemini API unavailable during HyDE. Retrying in {2 ** attempt}s...")
                            time.sleep(2 ** attempt)
                            continue
                    raise
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return question

    def reciprocal_rank_fusion(
        self,
        result_lists: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.

        Args:
            result_lists: Multiple lists of ranked results.
            k: RRF constant (higher = more emphasis on lower-ranked results).

        Returns:
            Merged and re-scored results list.
        """
        rrf_scores = {}
        chunk_data = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                chunk_id = result["chunk_id"]
                rrf_score = 1.0 / (k + rank + 1)

                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_score
                else:
                    rrf_scores[chunk_id] = rrf_score
                    chunk_data[chunk_id] = result

        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        output = []
        for chunk_id in sorted_ids:
            item = chunk_data[chunk_id].copy()
            item["rrf_score"] = rrf_scores[chunk_id]
            output.append(item)

        return output

    def rerank(self, question: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank candidates using a cross-encoder model.

        Args:
            question: The query question.
            candidates: List of candidate chunks to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of results with cross-encoder scores.
        """
        if not candidates:
            return []

        pairs = [(question, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def compress_context(self, question: str, chunks: List[Dict]) -> str:
        """
        Use the LLM to compress/filter context by removing irrelevant sentences.

        Args:
            question: The user question.
            chunks: Retrieved and reranked chunks.

        Returns:
            Compressed context string with only relevant information.
        """
        if not self.llm_client or not chunks:
            return "\n\n---\n\n".join(c["text"] for c in chunks)

        full_context = "\n\n---\n\n".join(
            f"[Chunk {i+1} | Paper: {c.get('metadata', {}).get('paper_id', 'unknown')} | "
            f"Section: {c.get('metadata', {}).get('section', 'unknown')}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )

        prompt = f"""You are a context compression assistant. Given a question and retrieved research paper chunks,
extract ONLY the sentences that are directly relevant to answering the question.
Remove boilerplate, methodology details unless asked, and irrelevant content.
Preserve the chunk labels [Chunk N] for citation tracking.

Question: {question}

Retrieved Context:
{full_context}

Compressed relevant context (keep chunk labels):"""

        try:
            for attempt in range(3):
                try:
                    response = self.llm_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt
                    )
                    return response.text
                except Exception as e:
                    if "503" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                        if attempt < 2:
                            print(f"Gemini API unavailable during compression. Retrying in {2 ** attempt}s...")
                            time.sleep(2 ** attempt)
                            continue
                    raise
        except Exception as e:
            print(f"Context compression failed: {e}")
            return full_context

    def retrieve(
        self,
        question: str,
        top_k_initial: int = 20,
        top_k_final: int = 5,
        use_hyde: bool = True,
        use_compression: bool = True,
        section_filter: Optional[str] = None
    ) -> Dict:
        """
        Full hybrid retrieval pipeline:
        1. BM25 search
        2. Vector search (optionally with HyDE)
        3. Reciprocal Rank Fusion
        4. Cross-encoder reranking
        5. Context compression

        Args:
            question: The user question.
            top_k_initial: Initial candidates per search method.
            top_k_final: Final number of chunks after reranking.
            use_hyde: Whether to use HyDE for vector search.
            use_compression: Whether to compress context.
            section_filter: Optional section name to filter vector search.

        Returns:
            Dict with chunks, compressed_context, and timing info.
        """
        timings = {}

        # Step 1: BM25 search
        t0 = time.time()
        bm25_results = self.bm25_search(question, top_k=top_k_initial)
        timings["bm25"] = round(time.time() - t0, 3)

        # Step 2: Vector search with optional HyDE
        t0 = time.time()
        if use_hyde:
            hyde_doc = self.generate_hyde_document(question)
            query_embedding = self.embedder.encode([hyde_doc]).tolist()[0]
            timings["hyde_generation"] = round(time.time() - t0, 3)
        else:
            query_embedding = self.embedder.encode([question]).tolist()[0]

        t0 = time.time()
        vector_results = self.vector_search(
            query_embedding,
            top_k=top_k_initial,
            section_filter=section_filter
        )
        timings["vector_search"] = round(time.time() - t0, 3)

        # Step 3: Reciprocal Rank Fusion
        t0 = time.time()
        fused_results = self.reciprocal_rank_fusion([bm25_results, vector_results])
        timings["rrf_fusion"] = round(time.time() - t0, 3)

        # Step 4: Cross-encoder reranking
        t0 = time.time()
        reranked = self.rerank(question, fused_results[:30], top_k=top_k_final)
        timings["reranking"] = round(time.time() - t0, 3)

        # Step 5: Context compression
        compressed_context = ""
        if use_compression and reranked:
            t0 = time.time()
            compressed_context = self.compress_context(question, reranked)
            timings["compression"] = round(time.time() - t0, 3)
        else:
            compressed_context = "\n\n---\n\n".join(c["text"] for c in reranked)

        total_retrieval_time = sum(timings.values())

        return {
            "chunks": reranked,
            "compressed_context": compressed_context,
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "fused_count": len(fused_results),
            "final_count": len(reranked),
            "timings": timings,
            "total_retrieval_time": round(total_retrieval_time, 3)
        }


# --- Module-level singleton for reuse across requests ---
_retriever_instance: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Get or create the singleton retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # Quick test
    retriever = HybridRetriever()
    result = retriever.retrieve(
        "What is the role of BRCA1 gene mutations in breast cancer drug resistance?",
        top_k_initial=15,
        top_k_final=5
    )
    print(f"\nRetrieved {result['final_count']} chunks in {result['total_retrieval_time']}s")
    print(f"Timings: {result['timings']}")
    for i, chunk in enumerate(result["chunks"]):
        print(f"\n--- Chunk {i+1} (score: {chunk.get('rerank_score', 0):.4f}) ---")
        print(chunk["text"][:200] + "...")
