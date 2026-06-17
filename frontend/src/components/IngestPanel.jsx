import { useState, useRef, useCallback } from 'react'

const API = '/api'

export default function IngestPanel({ personName, onIngestComplete, onNameDetected }) {
  const [log, setLog] = useState([])
  const [loading, setLoading] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [crawlSite, setCrawlSite] = useState(false)
  const [maxPages, setMaxPages] = useState(5)
  const [linkedinUrl, setLinkedinUrl] = useState('')
  const [githubUser, setGithubUser] = useState('')
  const [tinyfishQuery, setTinyfishQuery] = useState(personName || '')
  const [dragOver, setDragOver] = useState(false)
  const [resetKb, setResetKb] = useState(false)
  const fileRef = useRef(null)

  const addLog = useCallback((msg, type = 'info') => {
    setLog(prev => [...prev, { msg, type, ts: Date.now() }])
  }, [])

  // ── File Upload ────────────────────────────────────────────────────────────
  const handleFile = useCallback(async (file) => {
    if (!file) return
    setLoading(true)
    addLog(`📄 Uploading: ${file.name}…`, 'info')

    const form = new FormData()
    form.append('file', file)
    form.append('person_name', personName)
    form.append('reset_kb', resetKb)

    try {
      const res = await fetch(`${API}/ingest/file`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.status === 'success') {
        const { chunks_ingested } = data.ingestion
        const { entities_added, relationships_added, graph_nodes } = data.extraction || {}
        addLog(`✅ Ingested ${chunks_ingested} chunks from ${file.name}`, 'success')
        
        if (data.person_name && data.person_name !== personName) {
          addLog(`✨ Extracted name: ${data.person_name}`, 'success')
          onNameDetected?.(data.person_name)
        }

        if (entities_added) addLog(`🔗 Extracted ${entities_added} entities, ${relationships_added} relations → graph now ${graph_nodes} nodes`, 'success')
        onIngestComplete?.()
      } else {
        addLog(`❌ Error: ${data.detail || JSON.stringify(data)}`, 'error')
      }
    } catch (e) {
      addLog(`❌ Upload failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [personName, resetKb, addLog, onIngestComplete, onNameDetected])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  // ── URL Crawl ──────────────────────────────────────────────────────────────
  const handleCrawl = useCallback(async () => {
    if (!urlInput.trim()) return
    setLoading(true)
    addLog(`🕷 Crawling: ${urlInput}${crawlSite ? ` (site, max ${maxPages} pages)` : ''}…`, 'info')

    try {
      const res = await fetch(`${API}/ingest/url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInput, person_name: personName, crawl_site: crawlSite, max_pages: maxPages, reset_kb: resetKb }),
      })
      const data = await res.json()
      addLog(`✅ Crawled ${data.pages_crawled} pages, ingested ${data.pages_ingested}`, 'success')
      const ext = data.extraction || {}
      if (ext.entities_added) addLog(`🔗 +${ext.entities_added} entities, graph now ${ext.graph_nodes} nodes`, 'success')
      setUrlInput('')
      onIngestComplete?.()
    } catch (e) {
      addLog(`❌ Crawl failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [urlInput, crawlSite, maxPages, personName, resetKb, addLog, onIngestComplete])

  // ── LinkedIn ───────────────────────────────────────────────────────────────
  const handleLinkedIn = useCallback(async () => {
    if (!linkedinUrl.trim()) return
    setLoading(true)
    addLog(`🔗 Fetching LinkedIn profile…`, 'info')

    try {
      const res = await fetch(`${API}/ingest/linkedin`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ linkedin_url: linkedinUrl, person_name: personName, reset_kb: resetKb }),
      })
      const data = await res.json()
      if (data.status === 'success') {
        if (data.note) addLog(`ℹ️ ${data.note}`, 'info')
        addLog(`✅ LinkedIn ingested: ${data.ingestion.chunks_ingested} chunks`, 'success')
        onIngestComplete?.()
      } else if (data.status === 'fallback') {
        addLog(`⚠️ ${data.message}`, 'error')
        addLog(`💡 ${data.suggestion}`, 'info')
      }
      setLinkedinUrl('')
    } catch (e) {
      addLog(`❌ LinkedIn failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [linkedinUrl, personName, resetKb, addLog, onIngestComplete])

  // ── GitHub ─────────────────────────────────────────────────────────────────
  const handleGitHub = useCallback(async () => {
    if (!githubUser.trim()) return
    setLoading(true)
    addLog(`🐙 Fetching GitHub: ${githubUser}…`, 'info')

    try {
      const res = await fetch(`${API}/ingest/github`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: githubUser, person_name: personName, reset_kb: resetKb }),
      })
      const data = await res.json()
      addLog(`✅ GitHub: ${data.repos_found} repos ingested as ${data.ingestion.chunks_ingested} chunks`, 'success')
      const ext = data.extraction || {}
      if (ext.entities_added) addLog(`🔗 +${ext.entities_added} entities extracted`, 'success')
      setGithubUser('')
      onIngestComplete?.()
    } catch (e) {
      addLog(`❌ GitHub failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [githubUser, personName, resetKb, addLog, onIngestComplete])

  const handleTinyfish = useCallback(async () => {
    if (!tinyfishQuery.trim()) return
    setLoading(true)
    addLog(`🐟 Searching web footprint for: ${tinyfishQuery}…`, 'info')

    try {
      const res = await fetch(`${API}/ingest/tinyfish`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: tinyfishQuery, person_name: personName, reset_kb: resetKb }),
      })
      const data = await res.json()
      addLog(`✅ Tinyfish Search ingested as ${data.ingestion.chunks_ingested} chunks`, 'success')
      const ext = data.extraction || {}
      if (ext.entities_added) addLog(`🔗 +${ext.entities_added} entities extracted`, 'success')
      onIngestComplete?.()
    } catch (e) {
      addLog(`❌ Tinyfish failed: ${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }, [tinyfishQuery, personName, resetKb, addLog, onIngestComplete])

  return (
    <div className="ingest-layout">
      <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>
        📥 Ingest Knowledge
      </div>
      <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
        Add sources to {personName}'s knowledge base. All sources feed the same GraphRAG + hybrid retrieval pipeline.
      </div>
      
      <div className="toggle-row" style={{ marginTop: 8, padding: '12px 16px', background: 'rgba(239,68,68,0.05)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 8 }}>
        <div
          className={`toggle ${resetKb ? 'on' : ''}`}
          onClick={() => setResetKb(p => !p)}
        />
        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Wipe existing Knowledge Base before ingestion</span>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)', marginLeft: 8 }}>(Use this to start fresh for a different person)</span>
      </div>

      <div className="ingest-grid">
        {/* ── File Upload ── */}
        <div className="ingest-section">
          <h3>📄 Document Upload</h3>
          <p>PDF, DOCX, or TXT. Resume, LinkedIn export, transcripts, notes.</p>
          <div
            id="file-dropzone"
            className={`dropzone ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={e => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
          >
            <div className="dropzone-icon">📂</div>
            <div>Drop file here or click to browse</div>
            <div style={{ fontSize: 11, marginTop: 4, color: 'var(--text-dim)' }}>PDF · DOCX · TXT</div>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.docx,.txt"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
          />
        </div>

        {/* ── Website Crawl ── */}
        <div className="ingest-section">
          <h3>🕷 Website Crawl</h3>
          <p>Crawl a portfolio or personal website. Handles JS-rendered SPAs via Crawl4AI.</p>
          <div className="input-group">
            <label className="input-label">URL</label>
            <input
              id="crawl-url-input"
              className="text-input"
              placeholder="https://yourwebsite.com"
              value={urlInput}
              onChange={e => setUrlInput(e.target.value)}
            />
          </div>
          <div className="toggle-row">
            <div
              id="crawl-site-toggle"
              className={`toggle ${crawlSite ? 'on' : ''}`}
              onClick={() => setCrawlSite(p => !p)}
            />
            <span>Crawl entire site</span>
            {crawlSite && (
              <input
                className="text-input"
                type="number"
                min={1} max={20}
                value={maxPages}
                onChange={e => setMaxPages(+e.target.value)}
                style={{ width: 60 }}
              />
            )}
            {crawlSite && <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>pages max</span>}
          </div>
          <button
            id="crawl-btn"
            className="primary-btn btn-blue"
            onClick={handleCrawl}
            disabled={loading || !urlInput.trim()}
          >
            {loading ? '⏳ Crawling…' : '🕷 Crawl & Ingest'}
          </button>
        </div>

        {/* ── LinkedIn MCP ── */}
        <div className="ingest-section">
          <h3>💼 LinkedIn</h3>
          <p>Fetch live LinkedIn profile via Proxycurl API. Or upload the PDF export via the Document tab above.</p>
          <div className="input-group">
            <label className="input-label">LinkedIn Profile URL</label>
            <input
              id="linkedin-url-input"
              className="text-input"
              placeholder="https://linkedin.com/in/username"
              value={linkedinUrl}
              onChange={e => setLinkedinUrl(e.target.value)}
            />
          </div>
          <button
            id="linkedin-btn"
            className="primary-btn btn-blue"
            onClick={handleLinkedIn}
            disabled={loading || !linkedinUrl.trim()}
          >
            {loading ? '⏳ Fetching…' : '💼 Fetch LinkedIn'}
          </button>
          <div style={{ fontSize: 11, color: 'var(--text-dim)', lineHeight: 1.5 }}>
            Fetches profile via Proxycurl API if configured. Automatically falls back to Tinyfish Web Search to grab your public LinkedIn snippets!
          </div>
        </div>

        {/* ── GitHub ── */}
        <div className="ingest-section">
          <h3>🐙 GitHub</h3>
          <p>Pull repos, descriptions, topics, and contribution summary from GitHub public API.</p>
          <div className="input-group">
            <label className="input-label">GitHub Username</label>
            <input
              id="github-user-input"
              className="text-input"
              placeholder="octocat"
              value={githubUser}
              onChange={e => setGithubUser(e.target.value)}
            />
          </div>
          <button
            id="github-btn"
            className="primary-btn btn-teal"
            onClick={handleGitHub}
            disabled={loading || !githubUser.trim()}
          >
            {loading ? '⏳ Fetching…' : '🐙 Fetch GitHub'}
          </button>
        </div>

        {/* ── Tinyfish Web Search ── */}
        <div className="ingest-section">
          <h3>🐟 Web Footprint</h3>
          <p>Search the web using Tinyfish API and ingest digital footprint context.</p>
          <div className="input-group">
            <label className="input-label">Search Query</label>
            <input
              className="text-input"
              placeholder="Your name or company..."
              value={tinyfishQuery}
              onChange={e => setTinyfishQuery(e.target.value)}
            />
          </div>
          <button
            className="primary-btn btn-blue"
            onClick={handleTinyfish}
            disabled={loading || !tinyfishQuery.trim()}
          >
            {loading ? '⏳ Searching…' : '🐟 Search & Ingest'}
          </button>
        </div>
      </div>

      {/* ── Ingest Log ── */}
      {log.length > 0 && (
        <div>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.8px' }}>
            Ingest Log
          </div>
          <div className="ingest-log" id="ingest-log">
            {log.map((entry, i) => (
              <div key={i} className={`log-${entry.type}`}>
                {entry.msg}
              </div>
            ))}
          </div>
          <button
            style={{
              marginTop: 8,
              background: 'transparent',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--text-dim)',
              fontSize: 11,
              padding: '4px 10px',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
            }}
            onClick={() => setLog([])}
          >
            Clear log
          </button>
        </div>
      )}
    </div>
  )
}
