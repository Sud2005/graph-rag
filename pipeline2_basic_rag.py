import json
import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai

load_dotenv()

client_llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_collection("pubmed_papers")

TOP_K = 5  # number of chunks to retrieve


def count_tokens(text):
    return len(text.split()) * 4 // 3


def retrieve_chunks(question, top_k=TOP_K):
    """Embed the question and find most similar chunks"""
    query_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    return chunks


def query_basic_rag(question):
    """Retrieve relevant chunks then ask LLM"""

    # Step 1: Retrieve
    retrieved_chunks = retrieve_chunks(question)
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Step 2: Build prompt with retrieved context
    prompt = f"""You are a biomedical research assistant. Use ONLY the following research paper excerpts to answer the question. 
Be specific and cite information from the provided context.

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {question}

ANSWER (based on the provided context):"""

    start_time = time.time()

    response = client_llm.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    latency = round(time.time() - start_time, 2)
    answer = response.text

    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000

    return {
        "answer": answer,
        "retrieved_chunks": len(retrieved_chunks),
        "context_length": len(context.split()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": latency,
        "cost_usd": round(cost, 6)
    }


def main():
    print("Pipeline 2: Basic RAG\n")
    print("=" * 60)

    with open("test_question.json", "r") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        print(f"\nQuestion: {q['question']}")
        print("-" * 60)

        result = query_basic_rag(q["question"])

        print(f"Retrieved {result['retrieved_chunks']} chunks ({result['context_length']} words of context)")
        print(f"Answer (first 300 chars):\n{result['answer'][:300]}...")
        print(f"\nMetrics:")
        print(f"   Prompt tokens:     {result['prompt_tokens']:,}")
        print(f"   Completion tokens: {result['completion_tokens']:,}")
        print(f"   Total tokens:      {result['total_tokens']:,}")
        print(f"   Latency:           {result['latency_seconds']}s")
        print(f"   Cost:              ${result['cost_usd']}")

        results.append({
            "pipeline": "basic_rag",
            "question_id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "metrics": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_seconds": result["latency_seconds"],
                "cost_usd": result["cost_usd"],
                "chunks_retrieved": result["retrieved_chunks"]
            }
        })

        time.sleep(1)

    os.makedirs("results", exist_ok=True)
    with open("results/pipeline2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    avg_latency = sum(r["metrics"]["latency_seconds"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("PIPELINE 2 SUMMARY")
    print("=" * 60)
    print(f"   Questions answered:  {len(results)}")
    print(f"   Total tokens used:   {total_tokens:,}")
    print(f"   Total cost:          ${total_cost}")
    print(f"   Avg latency:         {avg_latency:.2f}s")
    print(f"   vs Pipeline 1:       {((7613 - total_tokens) / 7613 * 100):.1f}% token reduction")
    print(f"\nResults saved to results/pipeline2_results.json")


if __name__ == "__main__":
    main()