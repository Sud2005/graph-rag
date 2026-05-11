"""
Evaluation module for the GraphRAG benchmarking system.
Implements: RAGAS-style metrics, BERTScore, LLM-as-a-Judge,
Precision@K, Recall@K, and hallucination detection.
"""

import json
import os
import re
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


# ---------- RAGAS-style Metrics (LLM-based) ----------

def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    RAGAS Faithfulness: measures whether the answer is grounded in the context.
    Score 0-1 where 1 means every claim is supported by context.

    Args:
        answer: The generated answer.
        context: The retrieved context used to generate the answer.

    Returns:
        Faithfulness score between 0 and 1.
    """
    prompt = f"""You are an evaluation judge. Assess the FAITHFULNESS of the answer
with respect to the provided context. Faithfulness measures whether every claim
in the answer can be directly supported by the context.

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

Instructions:
1. List each factual claim in the answer.
2. For each claim, check if it is supported by the context.
3. Calculate: faithfulness = (supported claims) / (total claims)

Respond with ONLY a JSON object:
{{"claims_total": <int>, "claims_supported": <int>, "score": <float 0-1>}}"""

    try:
        response = llm_call(prompt)
        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return min(max(float(data.get("score", 0.0)), 0.0), 1.0)
    except Exception as e:
        print(f"Faithfulness evaluation error: {e}")
    return 0.0


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    RAGAS Answer Relevancy: measures how relevant the answer is to the question.
    Score 0-1 where 1 means perfectly relevant.

    Args:
        question: The original question.
        answer: The generated answer.

    Returns:
        Answer relevancy score between 0 and 1.
    """
    prompt = f"""You are an evaluation judge. Assess the ANSWER RELEVANCY.
This measures how well the answer addresses the specific question asked.
A relevant answer directly addresses all parts of the question.
An irrelevant answer is off-topic, too vague, or answers a different question.

QUESTION: {question}

ANSWER: {answer}

Score the relevancy from 0.0 to 1.0 where:
- 1.0 = Perfectly addresses every aspect of the question
- 0.7 = Addresses most aspects but misses some
- 0.5 = Partially relevant
- 0.2 = Mostly irrelevant
- 0.0 = Completely off topic

Respond with ONLY a JSON object:
{{"reasoning": "<brief reason>", "score": <float 0-1>}}"""

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return min(max(float(data.get("score", 0.0)), 0.0), 1.0)
    except Exception as e:
        print(f"Answer relevancy evaluation error: {e}")
    return 0.0


def evaluate_context_precision(question: str, context_chunks: List[str]) -> float:
    """
    RAGAS Context Precision: measures how many retrieved chunks are actually relevant.

    Args:
        question: The original question.
        context_chunks: List of retrieved chunk texts.

    Returns:
        Context precision score between 0 and 1.
    """
    if not context_chunks:
        return 0.0

    chunks_text = ""
    for i, chunk in enumerate(context_chunks):
        chunks_text += f"\n[Chunk {i+1}]: {chunk[:500]}\n"

    prompt = f"""You are an evaluation judge. Assess CONTEXT PRECISION.
For each retrieved chunk, determine if it is relevant to answering the question.

QUESTION: {question}

RETRIEVED CHUNKS:
{chunks_text}

For each chunk, output RELEVANT or NOT_RELEVANT.
Then calculate precision = relevant_chunks / total_chunks.

Respond with ONLY a JSON object:
{{"relevant_count": <int>, "total_count": <int>, "score": <float 0-1>}}"""

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return min(max(float(data.get("score", 0.0)), 0.0), 1.0)
    except Exception as e:
        print(f"Context precision evaluation error: {e}")
    return 0.0


def evaluate_context_recall(question: str, reference_answer: str, context: str) -> float:
    """
    RAGAS Context Recall: measures whether the context contains all information
    needed to produce the reference answer.

    Args:
        question: The original question.
        reference_answer: The ground truth reference answer.
        context: The retrieved context.

    Returns:
        Context recall score between 0 and 1.
    """
    prompt = f"""You are an evaluation judge. Assess CONTEXT RECALL.
Determine what fraction of the claims in the reference answer can be
attributed to the retrieved context.

QUESTION: {question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context[:3000]}

Instructions:
1. Break the reference answer into individual claims.
2. Check how many claims are supported by the context.
3. Calculate recall = supported_claims / total_claims.

Respond with ONLY a JSON object:
{{"claims_total": <int>, "claims_found": <int>, "score": <float 0-1>}}"""

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return min(max(float(data.get("score", 0.0)), 0.0), 1.0)
    except Exception as e:
        print(f"Context recall evaluation error: {e}")
    return 0.0


# ---------- BERTScore ----------

def compute_bertscore(candidate: str, reference: str) -> float:
    """
    Compute BERTScore F1 for semantic similarity between candidate and reference.
    Uses a lightweight approach with sentence-transformers to avoid heavy dependencies.

    Args:
        candidate: The generated answer.
        reference: The reference/ground truth answer.

    Returns:
        BERTScore F1 approximation between 0 and 1.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode both texts
        cand_sentences = [s.strip() for s in candidate.split(".") if len(s.strip()) > 10]
        ref_sentences = [s.strip() for s in reference.split(".") if len(s.strip()) > 10]

        if not cand_sentences or not ref_sentences:
            # Fallback to full-text similarity
            emb_cand = model.encode([candidate])
            emb_ref = model.encode([reference])
            return float(cosine_similarity(emb_cand, emb_ref)[0][0])

        cand_embs = model.encode(cand_sentences)
        ref_embs = model.encode(ref_sentences)

        sim_matrix = cosine_similarity(cand_embs, ref_embs)

        # Precision: for each candidate sentence, max similarity to any reference sentence
        precision = float(sim_matrix.max(axis=1).mean())
        # Recall: for each reference sentence, max similarity to any candidate sentence
        recall = float(sim_matrix.max(axis=0).mean())

        # F1
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)

    except Exception as e:
        print(f"BERTScore computation error: {e}")
        return 0.0


# ---------- LLM-as-a-Judge ----------

def llm_judge(question: str, answer: str, reference_answer: str, context: str = "") -> Dict[str, str]:
    """
    Use the LLM as a judge to grade an answer as PASS or FAIL.

    Args:
        question: The original question.
        answer: The generated answer to evaluate.
        reference_answer: The ground truth answer.
        context: The retrieved context (if any).

    Returns:
        Dict with 'verdict' (PASS/FAIL) and 'reason'.
    """
    context_section = ""
    if context:
        context_section = f"\nRETRIEVED CONTEXT (if RAG was used):\n{context[:2000]}\n"

    prompt = f"""You are a strict biomedical research evaluator. Grade the following
answer as PASS or FAIL based on these criteria:

1. ACCURACY: Does the answer contain correct biomedical facts?
2. COMPLETENESS: Does it cover the key points from the reference answer?
3. NO HALLUCINATION: Does it avoid making claims not supported by evidence?
4. RELEVANCE: Does it directly address the question asked?

QUESTION: {question}

REFERENCE ANSWER:
{reference_answer}
{context_section}
ANSWER TO EVALUATE:
{answer}

An answer should PASS if it is mostly accurate, covers the main points,
and does not contain major hallucinations. Minor omissions are acceptable.

Respond with ONLY a JSON object:
{{"verdict": "PASS" or "FAIL", "reason": "<concise explanation>"}}"""

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "verdict": data.get("verdict", "FAIL"),
                "reason": data.get("reason", "Could not parse judge response")
            }
    except Exception as e:
        print(f"LLM judge error: {e}")

    return {"verdict": "FAIL", "reason": "Evaluation failed due to an error"}


# ---------- Hallucination Detection ----------

def detect_hallucinations(answer: str, context: str) -> float:
    """
    Detect hallucinations in the answer by checking which claims
    are NOT supported by the context.

    Args:
        answer: The generated answer.
        context: The retrieved context.

    Returns:
        Hallucination score 0-1 (0 = no hallucination, 1 = fully hallucinated).
    """
    if not context:
        # No context means everything is potentially hallucinated (raw LLM)
        return 0.8  # High but not 1.0 since LLM knowledge may be correct

    prompt = f"""You are a hallucination detector for biomedical text.
Identify claims in the answer that are NOT supported by the provided context.

CONTEXT:
{context[:3000]}

ANSWER:
{answer}

Instructions:
1. List each factual claim in the answer.
2. Mark each as SUPPORTED or UNSUPPORTED based on the context.
3. Calculate hallucination_score = unsupported_claims / total_claims.
Note: General scientific knowledge that is widely established can be
considered supported even if not explicitly in context.

Respond with ONLY a JSON object:
{{"total_claims": <int>, "unsupported_claims": <int>, "hallucination_score": <float 0-1>}}"""

    try:
        response = llm_call(prompt)
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return min(max(float(data.get("hallucination_score", 0.5)), 0.0), 1.0)
    except Exception as e:
        print(f"Hallucination detection error: {e}")
    return 0.5


# ---------- Retrieval Quality Metrics ----------

def compute_precision_at_k(relevant_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int = 5) -> float:
    """
    Compute Precision@K for retrieval quality.

    Args:
        relevant_chunk_ids: Ground truth relevant chunk IDs.
        retrieved_chunk_ids: Retrieved chunk IDs in ranked order.
        k: Cutoff for evaluation.

    Returns:
        Precision@K score.
    """
    if not retrieved_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_chunk_ids))
    return relevant_in_top_k / len(top_k)


def compute_recall_at_k(relevant_chunk_ids: List[str], retrieved_chunk_ids: List[str], k: int = 5) -> float:
    """
    Compute Recall@K for retrieval quality.

    Args:
        relevant_chunk_ids: Ground truth relevant chunk IDs.
        retrieved_chunk_ids: Retrieved chunk IDs in ranked order.
        k: Cutoff for evaluation.

    Returns:
        Recall@K score.
    """
    if not relevant_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    relevant_in_top_k = len(set(top_k) & set(relevant_chunk_ids))
    return relevant_in_top_k / len(relevant_chunk_ids)


# ---------- Full Evaluation Pipeline ----------

def evaluate_single(
    pipeline_name: str,
    question: str,
    question_id: str,
    answer: str,
    reference_answer: str,
    context: str = "",
    context_chunks: Optional[List[str]] = None,
    retrieved_ids: Optional[List[str]] = None,
    relevant_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Run all evaluation metrics on a single pipeline answer.

    Args:
        pipeline_name: Name of the pipeline being evaluated.
        question: The original question.
        question_id: Question identifier.
        answer: The generated answer.
        reference_answer: Ground truth reference answer.
        context: Retrieved context string (empty for raw LLM).
        context_chunks: Individual chunk texts for context precision.
        retrieved_ids: Retrieved chunk IDs for P@K/R@K.
        relevant_ids: Ground truth relevant chunk IDs.

    Returns:
        Full evaluation result dict.
    """
    print(f"    Evaluating {pipeline_name} on {question_id}...")

    scores = {}

    # 1. Faithfulness (only meaningful if context exists)
    if context:
        scores["faithfulness"] = evaluate_faithfulness(answer, context)
        time.sleep(1)
    else:
        scores["faithfulness"] = 0.0  # No context = can't measure faithfulness

    # 2. Answer Relevancy
    scores["answer_relevancy"] = evaluate_answer_relevancy(question, answer)
    time.sleep(1)

    # 3. Context Precision
    if context_chunks:
        scores["context_precision"] = evaluate_context_precision(question, context_chunks)
        time.sleep(1)
    else:
        scores["context_precision"] = 0.0

    # 4. Context Recall
    if context:
        scores["context_recall"] = evaluate_context_recall(question, reference_answer, context)
        time.sleep(1)
    else:
        scores["context_recall"] = 0.0

    # 5. BERTScore F1
    scores["bertscore_f1"] = compute_bertscore(answer, reference_answer)

    # 6. Hallucination Detection
    scores["hallucination_score"] = detect_hallucinations(answer, context)
    time.sleep(1)

    # 7. LLM-as-a-Judge
    judge_result = llm_judge(question, answer, reference_answer, context)
    scores["judge_verdict"] = judge_result["verdict"]
    scores["judge_reason"] = judge_result["reason"]
    time.sleep(1)

    # 8. Precision@K and Recall@K
    if retrieved_ids and relevant_ids:
        scores["precision_at_k"] = compute_precision_at_k(relevant_ids, retrieved_ids)
        scores["recall_at_k"] = compute_recall_at_k(relevant_ids, retrieved_ids)
    else:
        scores["precision_at_k"] = 0.0
        scores["recall_at_k"] = 0.0

    return {
        "pipeline": pipeline_name,
        "question_id": question_id,
        "question": question,
        "answer": answer[:500],
        "reference_answer": reference_answer,
        "scores": scores
    }


def evaluate_all_pipelines(
    results_dir: str = None,
    questions_path: str = None
) -> Dict[str, Any]:
    """
    Run evaluation on all available pipeline results.

    Args:
        results_dir: Directory containing pipeline result JSON files.
        questions_path: Path to the test questions file.

    Returns:
        Dict with per-question evaluations and aggregate summaries.
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    if questions_path is None:
        questions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_question.json"
        )

    # Load reference questions
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    ref_map = {q["id"]: q for q in questions}

    # Discover and load pipeline results
    pipeline_files = {
        "raw_llm": "pipeline1_results.json",
        "agentic_llm": "pipeline1_agentic_results.json",
        "basic_rag": "pipeline2_results.json",
        "advanced_rag": "pipeline2_advanced_results.json",
    }

    all_evaluations = []
    pipeline_summaries = {}

    for pipeline_name, filename in pipeline_files.items():
        filepath = os.path.join(results_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Skipping {pipeline_name}: {filename} not found")
            continue

        print(f"\nEvaluating {pipeline_name} ({filename})...")
        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        pipeline_evals = []
        for result in results:
            q_id = result.get("question_id", "unknown")
            ref = ref_map.get(q_id, {})
            ref_answer = ref.get("reference_answer", "")
            question_text = result.get("question", ref.get("question", ""))

            # Build context from chunks if available
            context = ""
            context_chunks = []
            if "answer" in result:
                # For RAG pipelines, try to reconstruct context
                if "citations" in result:
                    context_chunks = [c.get("snippet", "") for c in result.get("citations", [])]
                    context = "\n\n".join(context_chunks)

            eval_result = evaluate_single(
                pipeline_name=pipeline_name,
                question=question_text,
                question_id=q_id,
                answer=result.get("answer", ""),
                reference_answer=ref_answer,
                context=context,
                context_chunks=context_chunks if context_chunks else None
            )

            pipeline_evals.append(eval_result)
            all_evaluations.append(eval_result)

        # Compute aggregate summary for this pipeline
        if pipeline_evals:
            n = len(pipeline_evals)
            summary = {
                "pipeline": pipeline_name,
                "num_questions": n,
                "avg_faithfulness": round(sum(e["scores"]["faithfulness"] for e in pipeline_evals) / n, 4),
                "avg_answer_relevancy": round(sum(e["scores"]["answer_relevancy"] for e in pipeline_evals) / n, 4),
                "avg_context_precision": round(sum(e["scores"]["context_precision"] for e in pipeline_evals) / n, 4),
                "avg_context_recall": round(sum(e["scores"]["context_recall"] for e in pipeline_evals) / n, 4),
                "avg_bertscore": round(sum(e["scores"]["bertscore_f1"] for e in pipeline_evals) / n, 4),
                "avg_hallucination": round(sum(e["scores"]["hallucination_score"] for e in pipeline_evals) / n, 4),
                "pass_rate": round(sum(1 for e in pipeline_evals if e["scores"]["judge_verdict"] == "PASS") / n, 4),
            }
            pipeline_summaries[pipeline_name] = summary
            print(f"  {pipeline_name} summary: pass_rate={summary['pass_rate']}, "
                  f"avg_bertscore={summary['avg_bertscore']}, "
                  f"avg_faithfulness={summary['avg_faithfulness']}")

    # Save evaluation results
    output = {
        "evaluations": all_evaluations,
        "summaries": pipeline_summaries
    }

    output_path = os.path.join(results_dir, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to {output_path}")
    return output


if __name__ == "__main__":
    evaluate_all_pipelines()
