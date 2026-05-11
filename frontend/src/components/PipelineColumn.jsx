import { useState } from 'react';

/**
 * PipelineColumn displays a single pipeline's result in a vertical card.
 * Shows streaming answer text, metrics badges, citations, and accuracy score.
 */
export default function PipelineColumn({ name, label, color, result, loading, error }) {
  const [showCitations, setShowCitations] = useState(false);

  const colorClasses = {
    red: {
      border: 'border-pipeline-1',
      bg: 'bg-pipeline-1',
      text: 'text-pipeline-1',
      bgLight: 'bg-pipeline-1/10',
      ring: 'ring-pipeline-1/30',
    },
    yellow: {
      border: 'border-pipeline-2',
      bg: 'bg-pipeline-2',
      text: 'text-pipeline-2',
      bgLight: 'bg-pipeline-2/10',
      ring: 'ring-pipeline-2/30',
    },
    green: {
      border: 'border-pipeline-3',
      bg: 'bg-pipeline-3',
      text: 'text-pipeline-3',
      bgLight: 'bg-pipeline-3/10',
      ring: 'ring-pipeline-3/30',
    },
  };

  const c = colorClasses[color] || colorClasses.red;

  /**
   * Render a small metric badge with label and value.
   */
  const MetricBadge = ({ label, value, unit }) => (
    <div className={`flex flex-col items-center px-3 py-2 rounded-lg ${c.bgLight} ring-1 ${c.ring}`}>
      <span className="text-[10px] uppercase tracking-wider text-text-secondary font-medium">{label}</span>
      <span className={`text-sm font-bold ${c.text}`}>
        {value}{unit && <span className="text-xs font-normal ml-0.5">{unit}</span>}
      </span>
    </div>
  );

  /**
   * Render the loading skeleton animation.
   */
  const Skeleton = () => (
    <div className="space-y-3 animate-pulse">
      <div className="h-3 bg-surface-lighter rounded w-full" />
      <div className="h-3 bg-surface-lighter rounded w-5/6" />
      <div className="h-3 bg-surface-lighter rounded w-4/6" />
      <div className="h-3 bg-surface-lighter rounded w-full" />
      <div className="h-3 bg-surface-lighter rounded w-3/6" />
      <div className="flex gap-2 mt-4">
        <div className="h-12 bg-surface-lighter rounded flex-1" />
        <div className="h-12 bg-surface-lighter rounded flex-1" />
        <div className="h-12 bg-surface-lighter rounded flex-1" />
      </div>
    </div>
  );

  return (
    <div className={`flex flex-col rounded-xl border ${c.border} bg-surface overflow-hidden shadow-lg shadow-black/20`}>
      {/* Header */}
      <div className={`px-4 py-3 ${c.bg} flex items-center justify-between`}>
        <h2 className="text-sm font-bold text-white tracking-wide uppercase">{label}</h2>
        {result && (
          <span className="text-xs text-white/80 font-mono">
            {result.timings?.total_seconds?.toFixed(1) || '—'}s
          </span>
        )}
      </div>

      <div className="p-4 flex-1 flex flex-col gap-4">
        {/* Error state */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Loading state */}
        {loading && <Skeleton />}

        {/* Answer */}
        {result && !loading && (
          <>
            <div className="bg-surface-light rounded-lg p-3 max-h-64 overflow-y-auto scrollbar-thin">
              <p className="text-text-primary text-sm leading-relaxed whitespace-pre-wrap">
                {result.answer || 'No answer generated.'}
              </p>
            </div>

            {/* Metrics badges */}
            <div className="grid grid-cols-3 gap-2">
              <MetricBadge
                label="Tokens"
                value={result.metrics?.total_tokens?.toLocaleString() || '0'}
              />
              <MetricBadge
                label="Latency"
                value={result.timings?.total_seconds?.toFixed(1) || '0'}
                unit="s"
              />
              <MetricBadge
                label="Cost"
                value={`$${result.metrics?.cost_usd?.toFixed(4) || '0'}`}
              />
            </div>

            {/* Confidence bar (if available) */}
            {result.confidence > 0 && (
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-text-secondary">Confidence</span>
                  <span className={c.text}>{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-surface-lighter rounded-full overflow-hidden">
                  <div
                    className={`h-full ${c.bg} rounded-full transition-all duration-700`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>
            )}

            {/* Citations */}
            {result.citations && result.citations.length > 0 && (
              <div>
                <button
                  onClick={() => setShowCitations(!showCitations)}
                  className={`text-xs ${c.text} hover:underline flex items-center gap-1`}
                >
                  {showCitations ? '▼' : '▶'} {result.citations.length} citations
                </button>
                {showCitations && (
                  <div className="mt-2 space-y-2 max-h-40 overflow-y-auto">
                    {result.citations.map((cit, i) => (
                      <div
                        key={i}
                        className="bg-surface-lighter rounded p-2 text-xs border border-border/50"
                      >
                        <div className="flex justify-between items-center mb-1">
                          <span className={`font-bold ${c.text}`}>[{cit.index}]</span>
                          <span className="text-text-secondary">
                            {cit.section} | Score: {cit.relevance_score?.toFixed(3)}
                          </span>
                        </div>
                        <p className="text-text-secondary">{cit.snippet.replace(/<\/?[^>]+(>|$)/g, "")}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Chunks retrieved count */}
            {result.metrics?.chunks_retrieved > 0 && (
              <div className="text-xs text-text-secondary">
                Retrieved: {result.metrics.chunks_retrieved} chunks
                {result.metrics.chunks_after_rerank > 0 &&
                  ` → ${result.metrics.chunks_after_rerank} after reranking`}
              </div>
            )}
          </>
        )}

        {/* Empty state */}
        {!result && !loading && !error && (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-text-secondary text-sm italic">Run a query to see results</p>
          </div>
        )}
      </div>
    </div>
  );
}
