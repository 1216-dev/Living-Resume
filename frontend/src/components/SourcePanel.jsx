const CITATION_COLORS = {
  resume: { bg: 'var(--accent-teal-dim)', color: 'var(--accent-teal)' },
  linkedin: { bg: 'var(--accent-blue-dim)', color: 'var(--accent-blue)' },
  github: { bg: 'var(--accent-green-dim)', color: 'var(--accent-green)' },
  knowledge_graph: { bg: 'var(--accent-purple-dim)', color: 'var(--accent-purple)' },
  graph_community: { bg: 'var(--accent-orange-dim)', color: 'var(--accent-orange)' },
  interview: { bg: 'var(--accent-teal-dim)', color: 'var(--accent-teal)' },
}

function getCitationStyle(label) {
  const key = Object.keys(CITATION_COLORS).find(k => label?.toLowerCase().includes(k))
  return CITATION_COLORS[key] || { bg: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }
}

function ScoreBadge({ score }) {
  const pct = score < 1 ? Math.round(score * 100) : Math.round(score * 100) / 100
  const color = score >= 0.7 ? 'var(--accent-green)' : score >= 0.4 ? 'var(--accent-yellow)' : 'var(--accent-orange)'
  return (
    <span
      className="score-badge"
      style={{ background: `${color}22`, color }}
    >
      {score < 1 ? `${Math.round(score * 100)}%` : score.toFixed(3)}
    </span>
  )
}

function SourceCard({ citation, index }) {
  const style = getCitationStyle(citation)
  return (
    <div className="source-card">
      <div className="source-card-header">
        <span className="source-label" style={{ color: style.color }}>{citation}</span>
        <span className="stat-chip" style={{ background: style.bg, color: style.color, fontSize: 10 }}>
          #{index + 1}
        </span>
      </div>
    </div>
  )
}

export default function SourcePanel({ retrieval, cacheStats }) {
  const citations = retrieval?.citations || []
  const sourcesUsed = retrieval?.sources_used || []
  const queryEntities = retrieval?.query_entities || []
  const confidence = retrieval?.confidence
  const cache = cacheStats || {}

  return (
    <>
      {/* Sources */}
      <div>
        <div className="card-title">Sources</div>
        {citations.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--text-dim)', textAlign: 'center', padding: '20px 0' }}>
            Ask a question to see sources
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {citations.map((c, i) => (
              <SourceCard key={i} citation={c} index={i} />
            ))}
          </div>
        )}
      </div>

      {/* Graph context */}
      {queryEntities.length > 0 && (
        <div>
          <div className="card-title">Graph Context</div>
          <div className="card" style={{ padding: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 6 }}>Entities matched:</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
              {queryEntities.map((e, i) => (
                <span key={i} className="stat-chip chip-purple" style={{ fontSize: 11 }}>{e}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Retrieval path */}
      {sourcesUsed.length > 0 && (
        <div>
          <div className="card-title">Retrieval Path</div>
          <div className="retrieval-path">
            {sourcesUsed.join(' → ')}
            {'\n'}
            {sourcesUsed.length} source{sourcesUsed.length !== 1 ? 's' : ''} fused via RRF
          </div>
        </div>
      )}

      {/* Token Usage / KV Cache stats */}
      <div>
        <div className="card-title">Token Usage</div>
        <div className="kv-stats-grid">
          <div className="kv-stat">
            <div className="kv-stat-label">Input tokens</div>
            <div className="kv-stat-value">{(cache.input_tokens_total || 0).toLocaleString()}</div>
          </div>
          <div className="kv-stat">
            <div className="kv-stat-label">Cache hits</div>
            <div className="kv-stat-value kv-saved">{(cache.cache_read_tokens || 0).toLocaleString()}</div>
          </div>
          <div className="kv-stat">
            <div className="kv-stat-label">Output tokens</div>
            <div className="kv-stat-value">{(cache.output_tokens_total || 0).toLocaleString()}</div>
          </div>
          <div className="kv-stat">
            <div className="kv-stat-label">Saved</div>
            <div className="kv-stat-value kv-saved">~{cache.net_savings_pct || 0}%</div>
          </div>
        </div>

        {/* Hit rate bar */}
        {cache.total_calls > 0 && (
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 11, color: 'var(--text-dim)', marginBottom: 4 }}>
              Cache hit rate: {cache.hit_rate_pct || 0}% ({cache.cache_hits || 0}/{cache.total_calls || 0} calls)
            </div>
            <div style={{ height: 4, background: 'var(--bg-input)', borderRadius: 10, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: `${cache.hit_rate_pct || 0}%`,
                  background: 'linear-gradient(90deg, var(--accent-orange), var(--accent-yellow))',
                  borderRadius: 10,
                  transition: 'width 0.4s ease',
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Confidence */}
      {confidence != null && (
        <div>
          <div className="card-title">Confidence</div>
          <div className="card" style={{ padding: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{
                flex: 1,
                height: 8,
                background: 'var(--bg-input)',
                borderRadius: 10,
                overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%',
                  width: `${Math.round(confidence * 100)}%`,
                  background: confidence >= 0.7
                    ? 'linear-gradient(90deg, var(--accent-green), var(--accent-teal))'
                    : confidence >= 0.4
                      ? 'linear-gradient(90deg, var(--accent-yellow), var(--accent-orange))'
                      : 'var(--accent-orange)',
                  borderRadius: 10,
                  transition: 'width 0.4s ease',
                }} />
              </div>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                {Math.round(confidence * 100)}%
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
