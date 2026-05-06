import json
import time
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def count_tokens(text):
    """Approximate token count"""
    return len(text.split()) * 4 // 3


def query_raw_llm(question):
    """Send question directly to LLM with no retrieval context"""

    prompt = f"""You are a biomedical research assistant. Answer the following question 
about cancer biology, genetics, and drug treatments based on your knowledge.
Be specific and detailed in your answer.

Question: {question}

Answer:"""

    start_time = time.time()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    answer = response.text

    # Token counting
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens

    # Cost estimate (Gemini 1.5 Flash pricing)
    # $0.075 per 1M input tokens, $0.30 per 1M output tokens
    cost = (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000

    return {
        "answer": answer,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": latency,
        "cost_usd": round(cost, 6)
    }


def main():
    print("Pipeline 1: Raw LLM\n")
    print("=" * 60)

    with open("test_question.json", "r") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        print(f"\nQuestion: {q['question']}")
        print("-" * 60)

        result = query_raw_llm(q["question"])

        print(f"Answer (first 300 chars):\n{result['answer'][:300]}...")
        print(f"\nMetrics:")
        print(f"   Prompt tokens:     {result['prompt_tokens']:,}")
        print(f"   Completion tokens: {result['completion_tokens']:,}")
        print(f"   Total tokens:      {result['total_tokens']:,}")
        print(f"   Latency:           {result['latency_seconds']}s")
        print(f"   Cost:              ${result['cost_usd']}")

        results.append({
            "pipeline": "raw_llm",
            "question_id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "metrics": {
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "total_tokens": result["total_tokens"],
                "latency_seconds": result["latency_seconds"],
                "cost_usd": result["cost_usd"]
            }
        })

        time.sleep(1)  # avoid rate limiting

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/pipeline1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    avg_latency = sum(r["metrics"]["latency_seconds"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("PIPELINE 1 SUMMARY")
    print("=" * 60)
    print(f"   Questions answered: {len(results)}")
    print(f"   Total tokens used:  {total_tokens:,}")
    print(f"   Total cost:         ${total_cost}")
    print(f"   Avg latency:        {avg_latency:.2f}s")
    print(f"\nResults saved to results/pipeline1_results.json")


if __name__ == "__main__":
    main()