"""
Pipeline 2 Advanced: Full RAG with hybrid retrieval, reranking, compression, and citations.
Uses the HybridRetriever from retrieval.py for all retrieval operations.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"


def count_tokens(text: str) -> int:
    """Approximate token count from text."""
    return len(text.split()) * 4 // 3


def generate_answer_with_citations(
    question: str,
    compressed_context: str,
    chunks: List[Dict]
) -> Dict[str, Any]:
    """
    Generate a detailed answer with inline citations [1], [2], etc.

    Args:
        question: The user question.
        compressed_context: The compressed context from retrieval.
        chunks: The retrieved and reranked chunks with metadata.

    Returns:
        Dict with answer, citations, and token metrics.
    """
    # Build citation reference list
    citation_refs = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        paper_id = meta.get("paper_id", "unknown")
        section = meta.get("section", "unknown")
        score = chunk.get("rerank_score", chunk.get("rrf_score", 0.0))
        citation_refs.append(
            f"[{i+1}] Paper: {paper_id} | Section: {section} | Score: {score:.4f}"
        )

    citation_block = "\n".join(citation_refs)

    prompt = f"""You are a biomedical research assistant. Answer the following question
using ONLY the provided research context. Be specific, detailed, and accurate.

IMPORTANT: Include inline citations like [1], [2] etc. to reference the source chunks.
Every factual claim must have a citation. If the context does not contain enough
information to answer fully, say so explicitly.

AVAILABLE SOURCES:
{citation_block}

RESEARCH CONTEXT:
{compressed_context}

QUESTION: {question}

ANSWER (with inline citations [1], [2], etc.):"""

    start = time.time()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    gen_time = round(time.time() - start, 2)

    answer = response.text
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(answer)

    # Build structured citations
    citations = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        citations.append({
            "index": i + 1,
            "paper_id": meta.get("paper_id", "unknown"),
            "section": meta.get("section", "unknown"),
            "relevance_score": round(chunk.get("rerank_score", chunk.get("rrf_score", 0.0)), 4),
            "snippet": chunk["text"][:150] + "..."
        })

    return {
        "answer": answer,
        "citations": citations,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "generation_time": gen_time
    }


def query_advanced_rag(
    question: str,
    retriever,
    top_k_initial: int = 20,
    top_k_final: int = 5,
    use_hyde: bool = True,
    use_compression: bool = True,
    section_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the full advanced RAG pipeline.

    Args:
        question: The biomedical question.
        retriever: HybridRetriever instance.
        top_k_initial: Initial retrieval candidates.
        top_k_final: Final chunks after reranking.
        use_hyde: Whether to use HyDE.
        use_compression: Whether to compress context.
        section_filter: Optional section filter.

    Returns:
        Dict with answer, citations, metrics, and detailed timings.
    """
    total_start = time.time()

    # Step 1: Hybrid retrieval
    retrieval_result = retriever.retrieve(
        question=question,
        top_k_initial=top_k_initial,
        top_k_final=top_k_final,
        use_hyde=use_hyde,
        use_compression=use_compression,
        section_filter=section_filter
    )

    retrieval_time = retrieval_result["total_retrieval_time"]
    chunks = retrieval_result["chunks"]
    compressed_context = retrieval_result["compressed_context"]

    # Step 2: Generate answer with citations
    gen_result = generate_answer_with_citations(
        question, compressed_context, chunks
    )

    total_time = round(time.time() - total_start, 2)

    # Estimate compression tokens (from the compressed context generation)
    compression_tokens = count_tokens(compressed_context) if use_compression else 0

    # Total token accounting
    total_prompt_tokens = gen_result["prompt_tokens"]
    total_completion_tokens = gen_result["completion_tokens"]

    # Add HyDE tokens if used (rough estimate)
    if use_hyde:
        total_prompt_tokens += 100  # HyDE prompt
        total_completion_tokens += 150  # HyDE response

    # Add compression tokens if used
    if use_compression:
        total_prompt_tokens += compression_tokens
        total_completion_tokens += count_tokens(compressed_context)

    total_tokens = total_prompt_tokens + total_completion_tokens
    cost = (total_prompt_tokens * 0.075 + total_completion_tokens * 0.30) / 1_000_000

    return {
        "answer": gen_result["answer"],
        "citations": gen_result["citations"],
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "paper_id": c.get("metadata", {}).get("paper_id", "unknown"),
                "section": c.get("metadata", {}).get("section", "unknown"),
                "text": c["text"],
                "relevance_score": round(c.get("rerank_score", 0.0), 4)
            }
            for c in chunks
        ],
        "metrics": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": round(cost, 6),
            "chunks_retrieved": retrieval_result["fused_count"],
            "chunks_after_rerank": retrieval_result["final_count"]
        },
        "timings": {
            "retrieval_seconds": retrieval_time,
            "compression_seconds": retrieval_result["timings"].get("compression", 0.0),
            "generation_seconds": gen_result["generation_time"],
            "total_seconds": total_time,
            "detailed": retrieval_result["timings"]
        }
    }


def main():
    """Run the advanced RAG pipeline on test questions."""
    # Import retriever here to avoid circular imports during module loading
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.retrieval import HybridRetriever

    print("Pipeline 2 (Advanced): Hybrid RAG with Reranking\n")
    print("=" * 60)

    # Check if hierarchical chunks exist
    chunks_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "processed", "hierarchical_chunks.json"
    )
    if not os.path.exists(chunks_path):
        print("Hierarchical chunks not found. Building them first...")
        from backend.chunking import build_hierarchical_chunks
        input_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "processed", "clean_papers.json"
        )
        build_hierarchical_chunks(input_path=input_path, output_path=chunks_path)

    # Initialize retriever
    chroma_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "chroma_advanced_db"
    )
    retriever = HybridRetriever(
        chroma_db_path=chroma_path,
        chunks_path=chunks_path
    )

    # Load test questions
    questions_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "test_question.json"
    )
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        print(f"\nQuestion: {q['question']}")
        print("-" * 60)

        result = query_advanced_rag(
            question=q["question"],
            retriever=retriever,
            top_k_initial=20,
            top_k_final=5,
            use_hyde=True,
            use_compression=True
        )

        print(f"Answer (first 300 chars):\n{result['answer'][:300]}...")
        print(f"\nCitations:")
        for cit in result["citations"]:
            print(f"  [{cit['index']}] Paper {cit['paper_id']} | {cit['section']} | Score: {cit['relevance_score']}")
        print(f"\nMetrics:")
        print(f"   Total tokens:        {result['metrics']['total_tokens']:,}")
        print(f"   Chunks retrieved:    {result['metrics']['chunks_retrieved']}")
        print(f"   Chunks after rerank: {result['metrics']['chunks_after_rerank']}")
        print(f"   Cost:                ${result['metrics']['cost_usd']}")
        print(f"\nTimings:")
        print(f"   Retrieval:    {result['timings']['retrieval_seconds']}s")
        print(f"   Compression:  {result['timings']['compression_seconds']}s")
        print(f"   Generation:   {result['timings']['generation_seconds']}s")
        print(f"   Total:        {result['timings']['total_seconds']}s")

        results.append({
            "pipeline": "advanced_rag",
            "question_id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "metrics": result["metrics"],
            "timings": result["timings"]
        })

        time.sleep(2)  # Avoid rate limiting

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "pipeline2_advanced_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    avg_latency = sum(r["timings"]["total_seconds"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("PIPELINE 2 ADVANCED SUMMARY")
    print("=" * 60)
    print(f"   Questions answered:  {len(results)}")
    print(f"   Total tokens used:   {total_tokens:,}")
    print(f"   Total cost:          ${total_cost}")
    print(f"   Avg latency:         {avg_latency:.2f}s")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
