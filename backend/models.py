"""
Pydantic models for the GraphRAG benchmarking system.
Defines request/response schemas for all pipeline endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PipelineName(str, Enum):
    """Enum for available pipeline names."""
    RAW_LLM = "raw_llm"
    BASIC_RAG = "basic_rag"
    ADVANCED_RAG = "advanced_rag"
    GRAPHRAG = "graphrag"
    AGENTIC_LLM = "agentic_llm"


class PipelineRequest(BaseModel):
    """Request body for querying a pipeline."""
    question: str = Field(..., description="The question to answer")
    pipeline_name: PipelineName = Field(..., description="Which pipeline to use")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional config overrides (e.g., top_k, section_filter)"
    )


class ChunkResult(BaseModel):
    """A single retrieved chunk with metadata."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    paper_id: str = Field(..., description="Source paper PMC ID")
    section: str = Field(default="unknown", description="Paper section (Abstract, Methods, etc.)")
    text: str = Field(..., description="The chunk text content")
    relevance_score: float = Field(default=0.0, description="Relevance score from retrieval/reranking")


class TimingBreakdown(BaseModel):
    """Detailed timing for each phase of a pipeline."""
    retrieval_seconds: float = Field(default=0.0, description="Time spent on retrieval")
    compression_seconds: float = Field(default=0.0, description="Time spent on context compression")
    generation_seconds: float = Field(default=0.0, description="Time spent on LLM generation")
    total_seconds: float = Field(default=0.0, description="Total end-to-end time")
    agentic_steps_seconds: Optional[List[float]] = Field(
        default=None,
        description="Time for each agentic reasoning step (Pipeline 1 only)"
    )


class PipelineMetrics(BaseModel):
    """Token usage and cost metrics for a pipeline run."""
    prompt_tokens: int = Field(default=0, description="Number of prompt/input tokens")
    completion_tokens: int = Field(default=0, description="Number of completion/output tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")
    cost_usd: float = Field(default=0.0, description="Estimated cost in USD")
    chunks_retrieved: int = Field(default=0, description="Number of chunks retrieved")
    chunks_after_rerank: int = Field(default=0, description="Chunks remaining after reranking")


class Citation(BaseModel):
    """A citation reference linking answer text to source chunks."""
    index: int = Field(..., description="Citation number [1], [2], etc.")
    paper_id: str = Field(..., description="Source paper PMC ID")
    section: str = Field(default="unknown", description="Paper section")
    relevance_score: float = Field(default=0.0, description="Relevance score")
    snippet: str = Field(default="", description="Short excerpt from the source chunk")


class PipelineResponse(BaseModel):
    """Full response from a pipeline query."""
    pipeline: str = Field(..., description="Pipeline name that generated this response")
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The generated answer")
    confidence: float = Field(default=0.0, description="Confidence score (0-1)")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    chunks: List[ChunkResult] = Field(default_factory=list, description="Retrieved chunks")
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    timings: TimingBreakdown = Field(default_factory=TimingBreakdown)
    agentic_trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Step-by-step reasoning trace (agentic pipeline only)"
    )


class EvaluationScores(BaseModel):
    """Evaluation metrics for a single answer."""
    faithfulness: float = Field(default=0.0, description="RAGAS faithfulness score (0-1)")
    answer_relevancy: float = Field(default=0.0, description="RAGAS answer relevancy (0-1)")
    context_precision: float = Field(default=0.0, description="RAGAS context precision (0-1)")
    context_recall: float = Field(default=0.0, description="RAGAS context recall (0-1)")
    bertscore_f1: float = Field(default=0.0, description="BERTScore F1 vs reference answer")
    hallucination_score: float = Field(default=0.0, description="Hallucination detection score (0=no hallucination, 1=full hallucination)")
    judge_verdict: str = Field(default="PENDING", description="LLM-as-Judge verdict: PASS or FAIL")
    judge_reason: str = Field(default="", description="LLM-as-Judge reasoning")
    precision_at_k: float = Field(default=0.0, description="Precision@K for retrieval")
    recall_at_k: float = Field(default=0.0, description="Recall@K for retrieval")


class EvaluationResult(BaseModel):
    """Full evaluation result for a pipeline answer."""
    pipeline: str
    question_id: str
    question: str
    answer: str
    reference_answer: str
    scores: EvaluationScores = Field(default_factory=EvaluationScores)


class ComparisonResult(BaseModel):
    """Side-by-side comparison of all pipelines on a single question."""
    question: str
    results: Dict[str, PipelineResponse] = Field(
        default_factory=dict,
        description="Pipeline name -> response mapping"
    )


class MetricsSummary(BaseModel):
    """Aggregated benchmark statistics across all questions."""
    pipeline: str
    num_questions: int = 0
    avg_tokens: float = 0.0
    avg_latency: float = 0.0
    avg_cost: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevancy: float = 0.0
    avg_bertscore: float = 0.0
    avg_hallucination: float = 0.0
    pass_rate: float = 0.0
