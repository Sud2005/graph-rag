# Graph RAG Hackathon

This repository contains scripts and data pipelines for the Graph RAG Hackathon project. The goal of this project is to compare raw LLM responses with Graph Retrieval-Augmented Generation (Graph RAG) approaches for answering complex domain-specific questions (e.g., biomedical and cancer biology).

## Project Structure

- `download_dataset.py`: Script to fetch the necessary datasets for processing.
- `process_dataset.py`: Script to clean and preprocess the downloaded data, preparing it for the Graph RAG pipeline.
- `pipeline1_raw_llm.py`: A baseline script that queries the Gemini LLM directly with questions (without any retrieval context) to measure raw LLM performance, token usage, latency, and cost.
- `test_question.json`: A set of test questions and their reference answers used for evaluating the pipelines.
- `graphrag/`: Directory intended to contain the core logic for the Graph RAG implementation.
- `data/`: Directory for storing raw and processed datasets (ignored by git).
- `results/`: Directory for storing the output metrics and answers from the pipelines (ignored by git).

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sud2005/graph-rag.git
   cd graph-rag
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install google-genai python-dotenv
   # Install any other necessary packages such as pandas, networkx, etc.
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

## Running the Pipelines

### Baseline: Raw LLM
To run the baseline pipeline that evaluates how the LLM answers the test questions without retrieval context:
```bash
python pipeline1_raw_llm.py
```
This will output the answers, metrics, and save a summary in the `results/` folder.

## Future Work
- Implementation of `pipeline2` focusing on Graph RAG.
- Integration of knowledge graph construction and retrieval algorithms.
