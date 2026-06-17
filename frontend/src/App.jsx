import { useState, useEffect, useCallback } from 'react'
import ChatPanel from './components/ChatPanel.jsx'
import SourcePanel from './components/SourcePanel.jsx'
import GraphView from './components/GraphView.jsx'
import IngestPanel from './components/IngestPanel.jsx'
import InterviewPanel from './components/InterviewPanel.jsx'

const API = '/api'

const TAB_COLORS = {
  chat: 'active',
  ingest: 'active-orange',
  interview: 'active-teal',
  graph: 'active',
}

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [personName, setPersonName] = useState('Devshree')
  const [sessionId] = useState(() => `session_${Date.now()}`)
  const [stats, setStats] = useState(null)
  const [lastRetrieval, setLastRetrieval] = useState(null)

  // Fetch system stats every 10s
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/stats`)
      if (res.ok) setStats(await res.json())
    } catch (_) {}
  }, [])

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 10000)
    return () => clearInterval(interval)
  }, [fetchStats])

  const graphStats = stats?.graph || {}
  const cacheStats = stats?.cache || {}
  const vectorStats = stats?.vector_store || {}

  return (
    <div className="app-layout">
      {/* ── Top Bar ── */}
      <header className="app-topbar">
        <div className="app-logo">
          <div className="app-logo-dot" />
          Living Resume
        </div>

        {/* Tab Bar */}
        <nav className="tab-bar">
          {[
            { id: 'chat', label: '💬 Chat' },
            { id: 'ingest', label: '📥 Ingest' },
            { id: 'interview', label: '🎙 Interview' },
            { id: 'graph', label: '🕸 Graph' },
          ].map(tab => (
            <button
              key={tab.id}
              id={`tab-${tab.id}`}
              className={`tab-btn ${activeTab === tab.id ? TAB_COLORS[tab.id] : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Person name input in topbar */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>Profile:</span>
          <input
            id="person-name-input"
            className="text-input"
            style={{ width: 140, padding: '5px 10px', fontSize: 13 }}
            value={personName}
            onChange={e => setPersonName(e.target.value)}
            placeholder="Person name"
          />
        </div>
      </header>

      {/* ── Left Sidebar ── */}
      <aside className="sidebar-left">
        <div className="profile-card">
          <div className="avatar-ring">
            {personName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()}
          </div>
          <div className="profile-name">{personName}</div>
          <div className="profile-title">Living Resume · AI Profile</div>
          <div className="profile-stats">
            <div className="stat-row">
              <span>Sources</span>
              <span className="stat-chip chip-teal">
                {vectorStats.sources?.length || 0} ingested
              </span>
            </div>
            <div className="stat-row">
              <span>Graph</span>
              <span className="stat-chip chip-purple">
                {graphStats.total_nodes || 0} nodes
              </span>
            </div>
            <div className="stat-row">
              <span>Communities</span>
              <span className="stat-chip chip-orange">
                {graphStats.communities || 0} clusters
              </span>
            </div>
            <div className="stat-row">
              <span>Chunks</span>
              <span className="stat-chip chip-blue">
                {vectorStats.total_chunks || 0}
              </span>
            </div>
          </div>
        </div>

        {/* Quick Ask */}
        <div className="quick-asks">
          <div className="section-title">Ask About</div>
          {[
            'What ML work did they do?',
            'What companies have they worked for?',
            'What AWS services have they used?',
            'Tell me about their projects',
            'What are their leadership experiences?',
            'Describe their career arc',
          ].map(q => (
            <button
              key={q}
              className="quick-chip"
              onClick={() => {
                setActiveTab('chat')
                window.__quickAsk?.(q)
              }}
            >
              {q}
            </button>
          ))}
        </div>

        {/* System Status */}
        <div className="status-chips">
          <div className="section-title">System</div>
          <div className={`status-chip ${cacheStats.cache_hits > 0 ? 'status-green' : 'status-orange'}`}>
            KV cache: {cacheStats.hit_rate_pct || 0}% hits
          </div>
          <div className="status-chip status-purple">
            Graph: {graphStats.total_nodes || 0} nodes
          </div>
          <div className="status-chip status-green">
            Sources: {vectorStats.sources?.length || 0} ingested
          </div>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="main-content">
        {activeTab === 'chat' && (
          <ChatPanel
            personName={personName}
            sessionId={sessionId}
            onRetrieval={setLastRetrieval}
          />
        )}
        {activeTab === 'ingest' && (
          <IngestPanel 
            personName={personName} 
            onIngestComplete={fetchStats}
            onNameDetected={setPersonName}
          />
        )}
        {activeTab === 'interview' && (
          <InterviewPanel personName={personName} sessionId={sessionId} onComplete={fetchStats} />
        )}
        {activeTab === 'graph' && (
          <GraphView />
        )}
      </main>

      {/* ── Right Sidebar ── */}
      <aside className="sidebar-right">
        <SourcePanel retrieval={lastRetrieval} cacheStats={cacheStats} />
      </aside>
    </div>
  )
}
