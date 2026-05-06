import { useState, useCallback } from 'react';
import PipelineColumn from './components/PipelineColumn';
import MetricsChart from './components/MetricsChart';

const API_BASE = '/api';

/**
 * Main application component for the GraphRAG Benchmark Dashboard.
 * Three-column layout comparing pipeline results with charts below.
 */
export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState({});
  const [results, setResults] = useState({});
  const [errors, setErrors] = useState({});
  const [isRunning, setIsRunning] = useState(false);

  const PIPELINES = [
    { name: 'raw_llm', label: 'Pipeline 1: Raw LLM', color: 'red' },
    { name: 'basic_rag', label: 'Pipeline 2: Basic RAG', color: 'yellow' },
    { name: 'advanced_rag', label: 'Pipeline 3: Advanced RAG', color: 'green' },
  ];

  const SAMPLE_QUESTIONS = [
    'What is the role of BRCA1 gene mutations in breast cancer drug resistance?',
    'How do targeted therapies for lung cancer work against EGFR mutations?',
    'What biomarkers predict response to immunotherapy in colorectal cancer?',
    'What is the relationship between TP53 mutations and chemotherapy resistance in leukemia?',
    'How does tumor microenvironment affect drug delivery in solid tumors?',
  ];

  /**
   * Run a single pipeline query via the REST API.
   */
  const runPipeline = useCallback(async (pipelineName, q) => {
    try {
      setLoading(prev => ({ ...prev, [pipelineName]: true }));
      setErrors(prev => ({ ...prev, [pipelineName]: null }));

      const response = await fetch(`${API_BASE}/query/${pipelineName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: q,
          pipeline_name: pipelineName,
          options: {},
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setResults(prev => ({ ...prev, [pipelineName]: data }));
    } catch (err) {
      setErrors(prev => ({ ...prev, [pipelineName]: err.message }));
    } finally {
      setLoading(prev => ({ ...prev, [pipelineName]: false }));
    }
  }, []);

  /**
   * Run all pipelines simultaneously on the same question.
   */
  const runAllPipelines = useCallback(async () => {
    if (!question.trim()) return;

    setIsRunning(true);
    setResults({});
    setErrors({});

    const promises = PIPELINES.map(p => runPipeline(p.name, question));
    await Promise.allSettled(promises);

    setIsRunning(false);
  }, [question, runPipeline]);

  /**
   * Handle pressing Enter in the question input.
   */
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !isRunning && question.trim()) {
      runAllPipelines();
    }
  };

  return (
    <div className="min-h-screen bg-[#13111f] text-text-primary font-sans">
      {/* Header */}
      <header className="border-b border-border bg-surface/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-purple-500 flex items-center justify-center">
              <span className="text-white font-bold text-sm">G</span>
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-primary to-purple-400 bg-clip-text text-transparent">
                GraphRAG Benchmark
              </h1>
              <p className="text-xs text-text-secondary">Compare LLM, RAG, and GraphRAG pipelines</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary bg-surface-light px-2 py-1 rounded-md border border-border">
              Gemini 2.5 Flash
            </span>
            <span className="text-xs text-text-secondary bg-surface-light px-2 py-1 rounded-md border border-border">
              63 Papers
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Query Input Section */}
        <div className="mb-8">
          <div className="flex gap-3">
            <input
              id="question-input"
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a biomedical research question..."
              disabled={isRunning}
              className="flex-1 bg-surface border border-border rounded-xl px-5 py-3 text-text-primary 
                         placeholder-text-secondary text-sm focus:outline-none focus:ring-2 
                         focus:ring-primary/50 focus:border-primary transition-all disabled:opacity-50"
            />
            <button
              id="run-button"
              onClick={runAllPipelines}
              disabled={isRunning || !question.trim()}
              className="bg-gradient-to-r from-primary to-purple-500 text-white font-semibold px-8 py-3 
                         rounded-xl hover:from-primary-dark hover:to-purple-600 transition-all 
                         disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-primary/20
                         active:scale-[0.98]"
            >
              {isRunning ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Running...
                </span>
              ) : (
                'Compare All'
              )}
            </button>
          </div>

          {/* Sample questions */}
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-xs text-text-secondary">Try:</span>
            {SAMPLE_QUESTIONS.map((sq, i) => (
              <button
                key={i}
                onClick={() => setQuestion(sq)}
                className="text-xs text-primary/80 hover:text-primary bg-surface-light hover:bg-surface-lighter 
                           px-3 py-1 rounded-full border border-border/50 transition-all truncate max-w-xs"
              >
                {sq.length > 50 ? sq.slice(0, 50) + '...' : sq}
              </button>
            ))}
          </div>
        </div>

        {/* Pipeline Results - Three Columns */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {PIPELINES.map((pipeline) => (
            <PipelineColumn
              key={pipeline.name}
              name={pipeline.name}
              label={pipeline.label}
              color={pipeline.color}
              result={results[pipeline.name]}
              loading={loading[pipeline.name]}
              error={errors[pipeline.name]}
            />
          ))}
        </div>

        {/* Comparison Charts */}
        {Object.keys(results).length > 0 && (
          <MetricsChart results={results} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-12 py-6 text-center">
        <p className="text-xs text-text-secondary">
          GraphRAG Benchmark System &middot; Built for Hackathon 2026 &middot; 
          Powered by Gemini + ChromaDB + TigerGraph
        </p>
      </footer>
    </div>
  );
}
