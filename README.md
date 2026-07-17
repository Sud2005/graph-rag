1) initShard
   ✓ shard created at /data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test (dim=384), points_count=0

2) upsertMemory — storing 2 fake vectors
   ✓ upserted 11111111-1111-1111-1111-111111111111, points_count now=1
   ✓ upserted 22222222-2222-2222-2222-222222222222, points_count now=2

3) searchMemories — querying with vector close to A
   ✓ search returned 2 hit(s):
     - id=11111111-1111-1111-1111-111111111111  score=0.9879951477050781  payload={'handle': 'fake memory A'}
     - id=22222222-2222-2222-2222-222222222222  score=0.6173088550567627  payload={'handle': 'fake memory B'}

✅ Full init → upsert → search loop works via Python/Chaquopy.






 
 
 
 initShard
   ✓ shard created at /data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test (dim=384), points_count=0

2) upsertMemory — storing 2 fake vectors

❌ PlatformException: TypeError: 'Collections$SingletonList' object is not iterable

com.chaquo.python.PyException: TypeError: 'Collections$SingletonList' object is not iterable
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-b9b722656810a95d70e571e3271195c28545127149b82340774d43e4d10f3e66:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-b9b722656810a95d70e571e3271195c28545127149b82340774d43e4d10f3e66:1)
	at q0.a.b(r8-map-id-b9b722656810a95d70e571e3271195c28545127149b82340774d43e4d10f3e66:580)
	at b2.b.h(r8-map-id-b9b722656810a95d70e571e3271195c28545127149b82340774d43e4d10f3e66:26)
	at q1.c.run(r8-map-id-b9b722656810a95d70e571e3271195c28545127149b82340774d43e4d10f3e66:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)






	


tShard
   ✓ shard created at /data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test (dim=384), points_count=0

2) upsertMemory — storing 2 fake vectors

❌ PlatformException: TypeError: argument 'points': 'Collections$SingletonList' object is not an instance of 'Sequence'

com.chaquo.python.PyException: TypeError: argument 'points': 'Collections$SingletonList' object is not an instance of 'Sequence'
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-a2a374e083dd53b169443c49d8df15792893337ebf090ffd541d1dcd5a74db7d:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-a2a374e083dd53b169443c49d8df15792893337ebf090ffd541d1dcd5a74db7d:1)
	at o0.c.b(r8-map-id-a2a374e083dd53b169443c49d8df15792893337ebf090ffd541d1dcd5a74db7d:580)
	at b2.b.h(r8-map-id-a2a374e083dd53b169443c49d8df15792893337ebf090ffd541d1dcd5a74db7d:26)
	at q1.c.run(r8-map-id-a2a374e083dd53b169443c49d8df15792893337ebf090ffd541d1dcd5a74db7d:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)


	


) initShard
   ✓ shard created at /data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test (dim=384), points_count=0

2) upsertMemory — storing 2 fake vectors

❌ PlatformException: TypeError: 'ArrayList' object is not iterable

com.chaquo.python.PyException: TypeError: 'ArrayList' object is not iterable
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-367b713d71e7e0a3ac6a4c12bb9f7c3fa6cac7042ca7cf3972693c70713edc2c:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-367b713d71e7e0a3ac6a4c12bb9f7c3fa6cac7042ca7cf3972693c70713edc2c:1)
	at q0.a.b(r8-map-id-367b713d71e7e0a3ac6a4c12bb9f7c3fa6cac7042ca7cf3972693c70713edc2c:561)
	at b2.b.h(r8-map-id-367b713d71e7e0a3ac6a4c12bb9f7c3fa6cac7042ca7cf3972693c70713edc2c:26)
	at q1.c.run(r8-map-id-367b713d71e7e0a3ac6a4c12bb9f7c3fa6cac7042ca7cf3972693c70713edc2c:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)
) initShard
   ✓ shard created at /data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test (dim=384), points_count=0

2) upsertMemory — storing 2 fake vectors

❌ PlatformException: TypeError: argument 'vector': failed to extract enum Helper ('Single | MultiDense | Named')
- variant Single (Single): TypeError: failed to extract field Helper::Single.0, caused by TypeError: 'ArrayList' object is not an instance of 'Sequence'
- variant MultiDense (MultiDense): TypeError: failed to extract field Helper::MultiDense.0, caused by TypeError: 'ArrayList' object is not an instance of 'Sequence'
- variant Named (Named): TypeError: failed to extract field Helper::Named.0, caused by TypeError: 'ArrayList' object is not an instance of 'dict'

com.chaquo.python.PyException: TypeError: argument 'vector': failed to extract enum Helper ('Single | MultiDense | Named')
- variant Single (Single): TypeError: failed to extract field Helper::Single.0, caused by TypeError: 'ArrayList' object is not an instance of 'Sequence'
- variant MultiDense (MultiDense): TypeError: failed to extract field Helper::MultiDense.0, caused by TypeError: 'ArrayList' object is not an instance of 'Sequence'
- variant Named (Named): TypeError: failed to extract field Helper::Named.0, caused by TypeError: 'ArrayList' object is not an instance of 'dict'
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:1)
	at o0.c.b(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:547)
	at b2.b.h(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:26)
	at q1.c.run(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)

initShard

❌ PlatformException: Exception: Service runtime error: cannot create edge shard: path already contains segment data

com.chaquo.python.PyException: Exception: Service runtime error: cannot create edge shard: path already contains segment data
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:1)
	at o0.c.b(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:426)
	at b2.b.h(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:26)
	at q1.c.run(r8-map-id-f423047df784ca52f41f6ebe5afa6b415371aab4f16a4eb8acd15321bfc17a1b:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)
❌ PlatformException: Exception: Service runtime error: failed to create WAL directory: failed to create directory `/data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test/wal`: No such file or directory (os error 2)

com.chaquo.python.PyException: Exception: Service runtime error: failed to create WAL directory: failed to create directory `/data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_python_test/wal`: No such file or directory (os error 2)
	at <python>.chaquopy_java.call(chaquopy_java.pyx:352)
	at <python>.chaquopy_java.Java_com_chaquo_python_PyObject_callAttrThrowsNative(chaquopy_java.pyx:324)
	at com.chaquo.python.PyObject.callAttrThrowsNative(Native Method)
	at com.chaquo.python.PyObject.callAttrThrows(r8-map-id-3809493d95322eab3f73b67b8e6840614b2056f8f2f32adf47ba5acdbbc8c662:1)
	at com.chaquo.python.PyObject.callAttr(r8-map-id-3809493d95322eab3f73b67b8e6840614b2056f8f2f32adf47ba5acdbbc8c662:1)
	at q0.a.b(r8-map-id-3809493d95322eab3f73b67b8e6840614b2056f8f2f32adf47ba5acdbbc8c662:129)
	at b2.b.h(r8-map-id-3809493d95322eab3f73b67b8e6840614b2056f8f2f32adf47ba5acdbbc8c662:26)
	at q1.c.run(r8-map-id-3809493d95322eab3f73b67b8e6840614b2056f8f2f32adf47ba5acdbbc8c662:128)
	at android.os.Handler.handleCallback(Handler.java:942)
	at android.os.Handler.dispatchMessage(Handler.java:99)
	at android.os.Looper.loopOnce(Looper.java:211)
	at android.os.Looper.loop(Looper.java:300)
	at android.app.ActivityThread.main(ActivityThread.java:8503)
	at java.lang.reflect.Method.invoke(Native Method)
	at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:561)
	at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:954)


1) init_shard("/data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_test")
   ✓ shard initialized

2) embed_text — real semantic embeddings

❌ FAILED: failed to load embedding model: Failed to retrieve model.onnx

#0      SimpleDecoder.decode (package:flutter_rust_bridge/src/codec/base.dart:32)
#1      SseCodec._decode (package:flutter_rust_bridge/src/codec/sse.dart:45)
#2      SseCodec.decodeObject (package:flutter_rust_bridge/src/codec/sse.dart:35)
<asynchronous suspension>
#3      _RustTestScreenState._runTest (package:thought_graph_app/main.dart:99)
<asynchronous suspension>


) init_shard("/data/user/0/com.example.thought_graph_app/app_flutter/qdrant_edge_test")
   ✓ shard initialized

2) embed_text — real semantic embeddings

❌ FAILED: PanicException(Cache directory cannot be found)

#0      SimpleDecoder.decode (package:flutter_rust_bridge/src/codec/base.dart:35)
#1      SseCodec._decode (package:flutter_rust_bridge/src/codec/sse.dart:45)
#2      SseCodec.decodeWireSyncType (package:flutter_rust_bridge/src/codec/sse.dart:40)
#3      BaseHandler.executeSync (package:flutter_rust_bridge/src/main_components/handler.dart:34)
#4      RustLibApiImpl.crateApiSimpleEmbedText (package:thought_graph_app/src/rust/frb_generated.dart:110)
#5      embedText (package:thought_graph_app/src/rust/api/simple.dart:43)
#6      _RustTestScreenState._runTest (package:thought_graph_app/main.dart:94)
<asynchronous suspension>
# GraphRAG Benchmark

A production-grade benchmarking system comparing **Raw LLM**, **Basic RAG**, **Advanced RAG**, and **GraphRAG** pipelines for answering biomedical research questions. Built with Gemini 2.5 Flash, ChromaDB, and React.
#to implement some sort of graph rag system 
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
