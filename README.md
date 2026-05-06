# GraphRAG Benchmark

A production-grade benchmarking system comparing **Raw LLM**, **Basic RAG**, **Advanced RAG**, and **GraphRAG** pipelines for answering biomedical research questions. Built with Gemini 2.5 Flash, ChromaDB, and React.

## Pipelines

| Pipeline | Description | Retrieval | Features |
|---|---|---|---|
| **Raw LLM** | Direct LLM query, no context | None | Baseline for comparison |
| **Agentic LLM** | Multi-step reasoning (think/answer/reflect/refine) | None | Self-critique, confidence scoring |
| **Basic RAG** | Simple vector search + LLM | ChromaDB | Fixed-size chunks |
| **Advanced RAG** | Hybrid retrieval + reranking + compression | BM25 + Vector + HyDE + Cross-encoder | Hierarchical chunks, citations, context compression |
| **GraphRAG** | Knowledge graph-based retrieval | TigerGraph Savanna | Entity relationships (external) |

## Quick Start

### 1. Backend
```bash
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install google-genai python-dotenv fastapi uvicorn sentence-transformers chromadb rank-bm25 scikit-learn

# Add your API key to .env
echo GEMINI_API_KEY=your_key_here > .env

# Start the API server
python -m backend.main
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 to access the dashboard.

## Project Structure

```
graphrag-hackathon/
├── backend/                       # FastAPI backend
│   ├── main.py                    # API server with all endpoints
│   ├── models.py                  # Pydantic request/response schemas
│   ├── chunking.py                # Hierarchical semantic chunking
│   ├── retrieval.py               # Hybrid BM25+Vector+HyDE+Reranking
│   ├── pipeline1_agent.py         # Agentic multi-step LLM
│   ├── pipeline2_advanced_rag.py  # Advanced RAG with citations
│   └── evaluation.py              # RAGAS + BERTScore + LLM Judge
├── frontend/                      # React + Tailwind dashboard
├── data/                          # Papers and vector DBs
├── results/                       # Pipeline outputs
├── pipeline1_raw_llm.py           # Simple LLM baseline
├── pipeline2_basic_rag.py         # Basic RAG baseline
├── test_question.json             # Test questions with references
└── setup.md                       # Detailed setup guide
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/query/{pipeline}` | POST | Run a single pipeline |
| `/compare` | POST | Run all pipelines simultaneously |
| `/evaluate` | POST | Run RAGAS + BERTScore evaluation |
| `/metrics/summary` | GET | Aggregated benchmark stats |
| `/stream/{pipeline}?q=...` | GET | SSE streaming |

## Tech Stack

- **LLM**: Gemini 2.5 Flash (google-genai SDK)
- **Embeddings**: all-MiniLM-L6-v2 (sentence-transformers)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Vector DB**: ChromaDB
- **Backend**: FastAPI + Uvicorn
- **Frontend**: React + Tailwind CSS + Recharts
- **Graph DB**: TigerGraph Savanna (Pipeline 3)

## Dataset

63 PubMed open-access papers on cancer genomics covering BRCA1, EGFR, TP53, immunotherapy, and tumor microenvironment. ~2.5M tokens of biomedical research text.
