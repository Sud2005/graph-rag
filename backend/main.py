"""
FastAPI backend for the GraphRAG benchmarking system.
Provides endpoints for all pipelines, comparison, evaluation, and metrics.
"""

import asyncio
import json
import os
import sys
import time
import traceback
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.models import (
    PipelineRequest, PipelineResponse, PipelineMetrics,
    TimingBreakdown, Citation, ChunkResult, ComparisonResult,
    EvaluationResult, EvaluationScores, MetricsSummary, PipelineName
)

# Global retriever (initialized on startup)
_retriever = None
_startup_complete = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    global _retriever, _startup_complete

    print("Starting GraphRAG Benchmark API...")

    # Check for hierarchical chunks and build if missing
    chunks_path = os.path.join(PROJECT_ROOT, "data", "processed", "hierarchical_chunks.json")
    if not os.path.exists(chunks_path):
        print("Hierarchical chunks not found. Building them (this may take a few minutes)...")
        from backend.chunking import build_hierarchical_chunks
        input_path = os.path.join(PROJECT_ROOT, "data", "processed", "clean_papers.json")
        build_hierarchical_chunks(input_path=input_path, output_path=chunks_path)

    # Initialize the retriever
    try:
        from backend.retrieval import HybridRetriever
        chroma_path = os.path.join(PROJECT_ROOT, "data", "chroma_advanced_db")
        _retriever = HybridRetriever(
            chroma_db_path=chroma_path,
            chunks_path=chunks_path
        )
        _startup_complete = True
        print("API ready.")
    except Exception as e:
        print(f"WARNING: Failed to initialize retriever: {e}")
        print("Advanced RAG pipeline will be unavailable.")
        _startup_complete = True

    yield

    # Cleanup
    print("Shutting down API...")


app = FastAPI(
    title="GraphRAG Benchmark API",
    description="Compare Raw LLM, Basic RAG, Advanced RAG, and GraphRAG pipelines",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Helper Functions ----------

def load_test_questions():
    """Load the test questions with reference answers."""
    path = os.path.join(PROJECT_ROOT, "test_question.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_tokens(text: str) -> int:
    """Approximate token count."""
    return len(text.split()) * 4 // 3


async def run_raw_llm(question: str) -> PipelineResponse:
    """Run the simple raw LLM pipeline (Pipeline 1 basic)."""
    from google import genai as genai_module
    from dotenv import load_dotenv as ld
    ld()

    llm_client = genai_module.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""You are a biomedical research assistant. Answer the following question
about cancer biology, genetics, and drug treatments based on your knowledge.
Be specific and detailed in your answer.

Question: {question}

Answer:"""

    start = time.time()
    response = await asyncio.to_thread(
        llm_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt
    )
    latency = round(time.time() - start, 2)

    answer = response.text
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000

    return PipelineResponse(
        pipeline="raw_llm",
        question=question,
        answer=answer,
        confidence=0.0,
        citations=[],
        chunks=[],
        metrics=PipelineMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=round(cost, 6)
        ),
        timings=TimingBreakdown(
            generation_seconds=latency,
            total_seconds=latency
        )
    )


async def run_agentic_llm(question: str) -> PipelineResponse:
    """Run the agentic LLM pipeline (Pipeline 1 upgraded)."""
    from backend.pipeline1_agent import query_agentic

    result = await asyncio.to_thread(query_agentic, question)

    return PipelineResponse(
        pipeline="agentic_llm",
        question=question,
        answer=result["answer"],
        confidence=result["confidence"],
        citations=[],
        chunks=[],
        metrics=PipelineMetrics(
            prompt_tokens=result["metrics"]["prompt_tokens"],
            completion_tokens=result["metrics"]["completion_tokens"],
            total_tokens=result["metrics"]["total_tokens"],
            cost_usd=result["metrics"]["cost_usd"]
        ),
        timings=TimingBreakdown(
            generation_seconds=result["metrics"]["latency_seconds"],
            total_seconds=result["metrics"]["latency_seconds"],
            agentic_steps_seconds=result["metrics"]["per_step_latency"]
        ),
        agentic_trace=[
            {"step": s["step"], "preview": s["response"][:300]}
            for s in result["trace"]
        ]
    )


async def run_basic_rag(question: str) -> PipelineResponse:
    """Run the basic RAG pipeline (Pipeline 2 basic)."""
    from sentence_transformers import SentenceTransformer
    import chromadb
    from google import genai as genai_module

    llm_client = genai_module.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_path = os.path.join(PROJECT_ROOT, "data", "chroma_db")
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = chroma_client.get_collection("pubmed_papers")
    except Exception:
        raise HTTPException(status_code=500, detail="Basic RAG ChromaDB collection not found. Run pipeline2_basic_rag.py first.")

    # Retrieve
    start = time.time()
    query_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieval_time = round(time.time() - start, 2)

    chunks_texts = results["documents"][0]
    context = "\n\n---\n\n".join(chunks_texts)

    prompt = f"""You are a biomedical research assistant. Use ONLY the following research paper excerpts to answer the question.
Be specific and cite information from the provided context.

CONTEXT FROM RESEARCH PAPERS:
{context}

QUESTION: {question}

ANSWER (based on the provided context):"""

    gen_start = time.time()
    response = await asyncio.to_thread(
        llm_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt
    )
    gen_time = round(time.time() - gen_start, 2)

    answer = response.text
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(answer)
    total_tokens = prompt_tokens + completion_tokens
    cost = (prompt_tokens * 0.075 + completion_tokens * 0.30) / 1_000_000

    return PipelineResponse(
        pipeline="basic_rag",
        question=question,
        answer=answer,
        confidence=0.0,
        citations=[],
        chunks=[
            ChunkResult(
                chunk_id=results["ids"][0][i],
                paper_id="unknown",
                section="unknown",
                text=chunks_texts[i][:200],
                relevance_score=0.0
            )
            for i in range(len(chunks_texts))
        ],
        metrics=PipelineMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=round(cost, 6),
            chunks_retrieved=len(chunks_texts)
        ),
        timings=TimingBreakdown(
            retrieval_seconds=retrieval_time,
            generation_seconds=gen_time,
            total_seconds=round(retrieval_time + gen_time, 2)
        )
    )


async def run_advanced_rag(question: str) -> PipelineResponse:
    """Run the advanced RAG pipeline (Pipeline 2 upgraded)."""
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Advanced RAG retriever not initialized.")

    from backend.pipeline2_advanced_rag import query_advanced_rag

    result = await asyncio.to_thread(
        query_advanced_rag,
        question,
        _retriever,
        20, 5, True, True, None
    )

    citations = [
        Citation(
            index=c["index"],
            paper_id=c["paper_id"],
            section=c["section"],
            relevance_score=c["relevance_score"],
            snippet=c["snippet"]
        )
        for c in result.get("citations", [])
    ]

    chunks = [
        ChunkResult(
            chunk_id=c["chunk_id"],
            paper_id=c["paper_id"],
            section=c["section"],
            text=c["text"][:200],
            relevance_score=c["relevance_score"]
        )
        for c in result.get("chunks", [])
    ]

    return PipelineResponse(
        pipeline="advanced_rag",
        question=question,
        answer=result["answer"],
        confidence=0.0,
        citations=citations,
        chunks=chunks,
        metrics=PipelineMetrics(
            prompt_tokens=result["metrics"]["prompt_tokens"],
            completion_tokens=result["metrics"]["completion_tokens"],
            total_tokens=result["metrics"]["total_tokens"],
            cost_usd=result["metrics"]["cost_usd"],
            chunks_retrieved=result["metrics"]["chunks_retrieved"],
            chunks_after_rerank=result["metrics"]["chunks_after_rerank"]
        ),
        timings=TimingBreakdown(
            retrieval_seconds=result["timings"]["retrieval_seconds"],
            compression_seconds=result["timings"]["compression_seconds"],
            generation_seconds=result["timings"]["generation_seconds"],
            total_seconds=result["timings"]["total_seconds"]
        )
    )


async def run_graphrag(question: str) -> PipelineResponse:
    """
    Placeholder for GraphRAG pipeline (Pipeline 3 - TigerGraph).
    Returns a stub response indicating it is handled externally.
    """
    return PipelineResponse(
        pipeline="graphrag",
        question=question,
        answer="GraphRAG pipeline is handled via TigerGraph Savanna. Please run it separately.",
        confidence=0.0,
        citations=[],
        chunks=[],
        metrics=PipelineMetrics(),
        timings=TimingBreakdown()
    )


PIPELINE_RUNNERS = {
    "raw_llm": run_raw_llm,
    "agentic_llm": run_agentic_llm,
    "basic_rag": run_basic_rag,
    "advanced_rag": run_advanced_rag,
    "graphrag": run_graphrag,
}


# ---------- API Endpoints ----------

@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "service": "GraphRAG Benchmark API",
        "status": "ready" if _startup_complete else "initializing",
        "pipelines": list(PIPELINE_RUNNERS.keys())
    }


@app.post("/query/{pipeline}", response_model=PipelineResponse)
async def query_pipeline(pipeline: str, request: PipelineRequest):
    """
    Run a specific pipeline on a question.

    Args:
        pipeline: Pipeline name (raw_llm, agentic_llm, basic_rag, advanced_rag, graphrag).
        request: Request body with question and options.

    Returns:
        PipelineResponse with answer, metrics, and timings.
    """
    if pipeline not in PIPELINE_RUNNERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pipeline: {pipeline}. Available: {list(PIPELINE_RUNNERS.keys())}"
        )

    try:
        start = time.time()
        result = await PIPELINE_RUNNERS[pipeline](request.question)
        print(f"[{pipeline}] Answered in {time.time() - start:.2f}s")
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline {pipeline} failed: {str(e)}")


class CompareRequest(BaseModel):
    """Request body for comparing multiple pipelines."""
    question: str
    pipelines: Optional[list] = None  # If None, runs all available


@app.post("/compare", response_model=ComparisonResult)
async def compare_pipelines(request: CompareRequest):
    """
    Run multiple pipelines on the same question simultaneously.
    Uses asyncio.gather for parallel execution.

    Args:
        request: Question and optional list of pipelines to compare.

    Returns:
        ComparisonResult with all pipeline responses.
    """
    pipelines_to_run = request.pipelines or ["raw_llm", "basic_rag", "advanced_rag"]
    pipelines_to_run = [p for p in pipelines_to_run if p in PIPELINE_RUNNERS]

    if not pipelines_to_run:
        raise HTTPException(status_code=400, detail="No valid pipelines specified.")

    # Run all pipelines concurrently
    tasks = [PIPELINE_RUNNERS[p](request.question) for p in pipelines_to_run]

    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for pipeline_name, result in zip(pipelines_to_run, results_list):
        if isinstance(result, Exception):
            results[pipeline_name] = PipelineResponse(
                pipeline=pipeline_name,
                question=request.question,
                answer=f"Error: {str(result)}",
                metrics=PipelineMetrics(),
                timings=TimingBreakdown()
            )
        else:
            results[pipeline_name] = result

    return ComparisonResult(question=request.question, results=results)


class EvaluateRequest(BaseModel):
    """Request body for running evaluation."""
    pipelines: Optional[list] = None


@app.post("/evaluate")
async def evaluate_pipelines(request: EvaluateRequest):
    """
    Run RAGAS + BERTScore + LLM-as-Judge evaluation on saved pipeline results.

    Returns:
        Evaluation results with per-question scores and aggregate summaries.
    """
    try:
        from backend.evaluation import evaluate_all_pipelines

        results_dir = os.path.join(PROJECT_ROOT, "results")
        questions_path = os.path.join(PROJECT_ROOT, "test_question.json")

        result = await asyncio.to_thread(
            evaluate_all_pipelines,
            results_dir,
            questions_path
        )
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Return aggregated benchmark stats from saved evaluation results.

    Returns:
        Dict mapping pipeline names to MetricsSummary objects.
    """
    eval_path = os.path.join(PROJECT_ROOT, "results", "evaluation_results.json")

    if not os.path.exists(eval_path):
        # Try to compute summaries from pipeline results directly
        summaries = {}
        results_dir = os.path.join(PROJECT_ROOT, "results")

        pipeline_files = {
            "raw_llm": "pipeline1_results.json",
            "agentic_llm": "pipeline1_agentic_results.json",
            "basic_rag": "pipeline2_results.json",
            "advanced_rag": "pipeline2_advanced_results.json",
        }

        for name, filename in pipeline_files.items():
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                n = len(data)
                if n > 0:
                    summaries[name] = {
                        "pipeline": name,
                        "num_questions": n,
                        "avg_tokens": round(sum(r.get("metrics", {}).get("total_tokens", 0) for r in data) / n, 1),
                        "avg_latency": round(sum(
                            r.get("metrics", {}).get("latency_seconds", 0) or
                            r.get("timings", {}).get("total_seconds", 0)
                            for r in data
                        ) / n, 2),
                        "avg_cost": round(sum(r.get("metrics", {}).get("cost_usd", 0) for r in data) / n, 6),
                    }

        return {"summaries": summaries, "source": "pipeline_results"}

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    return {
        "summaries": eval_data.get("summaries", {}),
        "source": "evaluation_results"
    }


@app.get("/stream/{pipeline}")
async def stream_pipeline(pipeline: str, q: str = Query(..., description="The question to answer")):
    """
    SSE streaming endpoint that streams answer tokens as they arrive.
    Falls back to chunked delivery of the complete answer.

    Args:
        pipeline: Pipeline name.
        q: The question to answer.

    Returns:
        Server-Sent Events stream of answer tokens.
    """
    if pipeline not in PIPELINE_RUNNERS:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {pipeline}")

    async def event_generator():
        """Generate SSE events with the pipeline response."""
        try:
            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'pipeline': pipeline})}\n\n"

            # Run the pipeline
            result = await PIPELINE_RUNNERS[pipeline](q)

            # Stream the answer in chunks to simulate streaming
            answer = result.answer
            chunk_size = 20  # characters per chunk
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming effect

            # Send metrics
            yield f"data: {json.dumps({'type': 'metrics', 'data': result.metrics.model_dump()})}\n\n"

            # Send timings
            yield f"data: {json.dumps({'type': 'timings', 'data': result.timings.model_dump()})}\n\n"

            # Send citations
            citations_data = [c.model_dump() for c in result.citations]
            yield f"data: {json.dumps({'type': 'citations', 'data': citations_data})}\n\n"

            # Send done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
