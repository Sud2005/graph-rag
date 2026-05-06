# GraphRAG Benchmark - Setup Guide

## Prerequisites
- Python 3.10+ (tested with 3.12)
- Node.js 18+ (for the React frontend)
- A valid Gemini API key ([Get one here](https://ai.google.dev/gemini-api/docs/api-key))

---

## 1. Backend Setup

### Create and activate virtual environment
```bash
cd C:\Users\sudha\PycharmProjects\graphrag-hackathon
python -m venv venv

# Windows
.\venv\Scripts\activate
```

### Install Python dependencies
```bash
pip install google-genai python-dotenv fastapi uvicorn
pip install sentence-transformers chromadb rank-bm25
pip install scikit-learn numpy tqdm pydantic
```

### Configure environment variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

---

## 2. Data Preparation (if not already done)

### Download PubMed papers
```bash
python download_dataset.py
```

### Process and chunk papers (basic)
```bash
python process_dataset.py
```

### Build hierarchical chunks (advanced)
```bash
python -m backend.chunking
```
This creates `data/processed/hierarchical_chunks.json` with semantic, section-aware chunks.

---

## 3. Running Individual Pipelines

### Pipeline 1: Raw LLM (simple baseline)
```bash
python pipeline1_raw_llm.py
```

### Pipeline 1: Agentic LLM (upgraded with multi-step reasoning)
```bash
python -m backend.pipeline1_agent
```

### Pipeline 2: Basic RAG
```bash
python pipeline2_basic_rag.py
```

### Pipeline 2: Advanced RAG (hybrid retrieval + reranking)
```bash
python -m backend.pipeline2_advanced_rag
```

### Run Evaluation (RAGAS + BERTScore + LLM-as-Judge)
```bash
python -m backend.evaluation
```

---

## 4. Running the Full Stack (API + Frontend)

### Start the FastAPI backend
```bash
python -m backend.main
```
The API will be available at http://localhost:8000.
API docs at http://localhost:8000/docs.

### Start the React frontend (in a separate terminal)
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at http://localhost:5173.
It proxies API calls to the backend automatically.

---

## 5. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/query/{pipeline}` | POST | Run a single pipeline |
| `/compare` | POST | Run all pipelines on the same question |
| `/evaluate` | POST | Run RAGAS + BERTScore evaluation |
| `/metrics/summary` | GET | Get aggregated benchmark stats |
| `/stream/{pipeline}?q=...` | GET | SSE streaming endpoint |

### Example: Query a pipeline
```bash
curl -X POST http://localhost:8000/query/raw_llm \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the role of BRCA1 in cancer?", "pipeline_name": "raw_llm"}'
```

### Example: Compare all pipelines
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"question": "How do targeted therapies work against EGFR mutations?"}'
```

---

## 6. Project Architecture

```
graphrag-hackathon/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── models.py                  # Pydantic schemas
│   ├── chunking.py                # Hierarchical semantic chunking
│   ├── retrieval.py               # Hybrid BM25+Vector+HyDE+Reranking
│   ├── pipeline1_agent.py         # Agentic multi-step LLM
│   ├── pipeline2_advanced_rag.py  # Advanced RAG with citations
│   └── evaluation.py              # RAGAS + BERTScore + LLM Judge
├── frontend/
│   └── src/
│       ├── App.jsx                # Main dashboard
│       └── components/
│           ├── PipelineColumn.jsx # Pipeline result card
│           └── MetricsChart.jsx   # Comparison charts
├── data/
│   ├── raw/                       # Downloaded papers
│   ├── processed/                 # Chunks and clean papers
│   ├── chroma_db/                 # Basic vector DB
│   └── chroma_advanced_db/        # Advanced vector DB
├── results/                       # Pipeline outputs and evaluations
├── pipeline1_raw_llm.py           # Original simple LLM baseline
├── pipeline2_basic_rag.py         # Original basic RAG
├── test_question.json             # Test questions with reference answers
└── .env                           # API keys
```

---

## 7. Key Design Decisions

- **Hierarchical Chunking**: Papers are split into sections (Abstract, Methods, Results, etc.) then semantically chunked using sentence embedding similarity drops. This preserves context better than fixed-size chunks.

- **Hybrid Retrieval (RRF)**: BM25 captures keyword matches while vector search captures semantic similarity. Reciprocal Rank Fusion merges both rankings without needing score normalization.

- **HyDE**: Generates a hypothetical ideal answer and uses its embedding for retrieval. This bridges the vocabulary gap between questions and paper text.

- **Cross-Encoder Reranking**: The bi-encoder retrieval is fast but imprecise. A cross-encoder (ms-marco-MiniLM) re-scores the top candidates with much higher accuracy.

- **Context Compression**: Strips irrelevant sentences from retrieved chunks before feeding them to the LLM, reducing token costs and hallucination risk.

- **Agentic Pipeline**: The 4-step reasoning loop (think → answer → reflect → refine) trades higher token cost for improved answer quality and self-corrected hallucinations.
