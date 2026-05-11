"""
Pipeline 1 Upgrade: Agentic Raw LLM.
Multi-step reasoning loop: think -> answer -> reflect -> refine.
Includes self-critique, hallucination detection, and confidence scoring.
"""

import json
import os
import time
from typing import Dict, List, Any
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"


def count_tokens(text: str) -> int:
    """Approximate token count from text."""
    return len(text.split()) * 4 // 3


def llm_call(prompt: str, retries: int = 3) -> str:
    """Make a single LLM call and return the response text, with retries."""
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                if attempt < retries - 1:
                    print(f"Gemini API unavailable. Retrying in {2 ** attempt}s...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Gemini API failed after {retries} attempts. Falling back to Groq Llama 3.3 70B...")
                    try:
                        import os
                        from groq import Groq
                        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                        response = groq_client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0
                        )
                        return response.choices[0].message.content
                    except Exception as groq_e:
                        print(f"Groq fallback also failed: {groq_e}")
                        raise e
            raise


def step_think(question: str) -> Dict[str, Any]:
    """
    Step 1: Think - Break down the question and plan the answer.
    The LLM identifies key concepts, required knowledge, and potential pitfalls.
    """
    prompt = f"""You are a biomedical research expert. Before answering the following question,
think step by step about what knowledge is needed.

Question: {question}

Provide your analysis in this exact format:
KEY CONCEPTS: [list the main biomedical concepts involved]
KNOWLEDGE NEEDED: [what specific knowledge areas are required]
POTENTIAL PITFALLS: [common misconceptions or areas where hallucination is likely]
ANSWER PLAN: [brief outline of how to structure the answer]"""

    start = time.time()
    response = llm_call(prompt)
    elapsed = round(time.time() - start, 2)

    return {
        "step": "think",
        "prompt": prompt,
        "response": response,
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(response),
        "latency": elapsed
    }


def step_answer(question: str, thinking: str) -> Dict[str, Any]:
    """
    Step 2: Answer - Generate a detailed answer using the thinking plan.
    """
    prompt = f"""You are a biomedical research expert. Based on your analysis below,
provide a comprehensive, detailed answer to the question.
Be specific with gene names, mechanisms, drug names, and clinical details.
Only state facts you are confident about.

YOUR PRIOR ANALYSIS:
{thinking}

QUESTION: {question}

DETAILED ANSWER:"""

    start = time.time()
    response = llm_call(prompt)
    elapsed = round(time.time() - start, 2)

    return {
        "step": "answer",
        "prompt": prompt,
        "response": response,
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(response),
        "latency": elapsed
    }


def step_reflect(question: str, answer: str) -> Dict[str, Any]:
    """
    Step 3: Reflect - Self-critique the answer for accuracy and hallucinations.
    The LLM reviews its own answer and identifies potential issues.
    """
    prompt = f"""You are a strict biomedical fact-checker. Review the following answer
for a research question and identify any potential issues.

QUESTION: {question}

ANSWER TO REVIEW:
{answer}

Analyze the answer and provide:
ACCURACY CHECK: [Are the stated facts likely correct? Flag any suspicious claims]
HALLUCINATION RISK: [Which specific statements might be hallucinated or unverifiable?]
COMPLETENESS: [What important aspects are missing from the answer?]
CONFIDENCE LEVEL: [Rate overall confidence as a number from 0.0 to 1.0]
SUGGESTIONS: [Specific improvements to make the answer more accurate]"""

    start = time.time()
    response = llm_call(prompt)
    elapsed = round(time.time() - start, 2)

    return {
        "step": "reflect",
        "prompt": prompt,
        "response": response,
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(response),
        "latency": elapsed
    }


def step_refine(question: str, original_answer: str, reflection: str) -> Dict[str, Any]:
    """
    Step 4: Refine - Improve the answer based on self-critique feedback.
    """
    prompt = f"""You are a biomedical research expert. Refine your previous answer
based on the self-critique feedback below. Remove any potentially hallucinated claims,
add missing information, and improve accuracy.

QUESTION: {question}

ORIGINAL ANSWER:
{original_answer}

SELF-CRITIQUE FEEDBACK:
{reflection}

REFINED ANSWER (be precise, remove uncertain claims, keep only well-established facts):"""

    start = time.time()
    response = llm_call(prompt)
    elapsed = round(time.time() - start, 2)

    return {
        "step": "refine",
        "prompt": prompt,
        "response": response,
        "prompt_tokens": count_tokens(prompt),
        "completion_tokens": count_tokens(response),
        "latency": elapsed
    }


def extract_confidence(reflection_text: str) -> float:
    """
    Extract the confidence score from the reflection step output.
    Falls back to 0.5 if parsing fails.
    """
    import re
    patterns = [
        r"CONFIDENCE\s*LEVEL\s*:\s*\[?\s*(0?\.\d+|1\.0|1)\s*\]?",
        r"confidence\s*(?:level|score|rating)?\s*(?:is|:)\s*(0?\.\d+|1\.0|1)",
        r"(0\.\d+)\s*(?:out of 1|/1)",
    ]
    for pattern in patterns:
        match = re.search(pattern, reflection_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return 0.5


def query_agentic(question: str) -> Dict[str, Any]:
    """
    Run the full agentic pipeline: think -> answer -> reflect -> refine.
    Tracks token usage and latency for each step separately.

    Args:
        question: The biomedical question to answer.

    Returns:
        Dict with the final answer, per-step traces, and aggregate metrics.
    """
    total_start = time.time()
    trace = []

    # Step 1: Think
    think_result = step_think(question)
    trace.append(think_result)

    # Step 2: Answer (using thinking output)
    answer_result = step_answer(question, think_result["response"])
    trace.append(answer_result)

    # Step 3: Reflect (self-critique)
    reflect_result = step_reflect(question, answer_result["response"])
    trace.append(reflect_result)

    # Step 4: Refine (improve based on critique)
    refine_result = step_refine(
        question,
        answer_result["response"],
        reflect_result["response"]
    )
    trace.append(refine_result)

    # Aggregate metrics
    total_prompt_tokens = sum(s["prompt_tokens"] for s in trace)
    total_completion_tokens = sum(s["completion_tokens"] for s in trace)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_latency = round(time.time() - total_start, 2)

    # Gemini 2.5 Flash pricing estimate
    cost = (total_prompt_tokens * 0.075 + total_completion_tokens * 0.30) / 1_000_000

    # Extract confidence from reflection
    confidence = extract_confidence(reflect_result["response"])

    return {
        "answer": refine_result["response"],
        "confidence": confidence,
        "trace": trace,
        "metrics": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "latency_seconds": total_latency,
            "cost_usd": round(cost, 6),
            "steps": len(trace),
            "per_step_tokens": [s["prompt_tokens"] + s["completion_tokens"] for s in trace],
            "per_step_latency": [s["latency"] for s in trace]
        }
    }


def main():
    """Run the agentic pipeline on test questions and compare with simple Pipeline 1."""
    print("Pipeline 1 (Agentic): Multi-step Reasoning LLM\n")
    print("=" * 60)

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

        result = query_agentic(q["question"])

        print(f"Answer (first 300 chars):\n{result['answer'][:300]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nMetrics:")
        print(f"   Steps:            {result['metrics']['steps']}")
        print(f"   Total tokens:     {result['metrics']['total_tokens']:,}")
        print(f"   Per-step tokens:  {result['metrics']['per_step_tokens']}")
        print(f"   Total latency:    {result['metrics']['latency_seconds']}s")
        print(f"   Per-step latency: {result['metrics']['per_step_latency']}")
        print(f"   Cost:             ${result['metrics']['cost_usd']}")

        results.append({
            "pipeline": "agentic_llm",
            "question_id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "metrics": result["metrics"],
            "trace": [
                {"step": s["step"], "response_preview": s["response"][:200]}
                for s in result["trace"]
            ]
        })

        time.sleep(2)  # Avoid rate limiting

    # Save results
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "pipeline1_agentic_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    total_tokens = sum(r["metrics"]["total_tokens"] for r in results)
    total_cost = sum(r["metrics"]["cost_usd"] for r in results)
    avg_latency = sum(r["metrics"]["latency_seconds"] for r in results) / len(results)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    # Pipeline 1 simple baseline for comparison
    simple_tokens = 7907  # From earlier run

    print("\n" + "=" * 60)
    print("PIPELINE 1 AGENTIC SUMMARY")
    print("=" * 60)
    print(f"   Questions answered:  {len(results)}")
    print(f"   Total tokens used:   {total_tokens:,}")
    print(f"   Simple P1 tokens:    {simple_tokens:,}")
    print(f"   Token overhead:      {((total_tokens - simple_tokens) / simple_tokens * 100):.1f}%")
    print(f"   Total cost:          ${total_cost}")
    print(f"   Avg latency:         {avg_latency:.2f}s")
    print(f"   Avg confidence:      {avg_confidence:.2f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
