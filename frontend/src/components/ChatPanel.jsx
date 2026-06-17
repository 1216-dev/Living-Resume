import { useState, useRef, useEffect, useCallback } from 'react'

const API = '/api'

// ── Source Badge Colors ──────────────────────────────────────────────────────

const SOURCE_STYLES = {
  'Resume':          { bg: 'rgba(45, 212, 191, 0.12)', color: '#2dd4bf', icon: '📄' },
  'LinkedIn':        { bg: 'rgba(96, 165, 250, 0.12)', color: '#60a5fa', icon: '🔗' },
  'GitHub':          { bg: 'rgba(74, 222, 128, 0.12)', color: '#4ade80', icon: '⌨️' },
  'Website':         { bg: 'rgba(251, 146, 60, 0.12)', color: '#fb923c', icon: '🌐' },
  'Interview':       { bg: 'rgba(124, 106, 247, 0.12)', color: '#7c6af7', icon: '🎙' },
  'Knowledge Graph': { bg: 'rgba(250, 204, 21, 0.10)', color: '#facc15', icon: '🕸' },
}

function getSourceStyle(label) {
  return SOURCE_STYLES[label] || { bg: 'rgba(255,255,255,0.06)', color: '#94a3b8', icon: '📎' }
}

// ── Profile Header ───────────────────────────────────────────────────────────

function ProfileHeader({ personName, profileData }) {
  const initials = personName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()
  const currentRole = profileData?.current_role || 'Professional'
  const currentCompany = profileData?.current_company || ''
  const topSkills = profileData?.top_skills?.slice(0, 4) || []

  const roleLine = currentCompany ? `${currentRole} · ${currentCompany}` : currentRole

  return (
    <div className="bio-profile-header">
      <div className="bio-avatar-ring">
        <span className="bio-avatar-initials">{initials}</span>
        <div className="bio-avatar-pulse" />
      </div>
      <div className="bio-profile-info">
        <div className="bio-profile-name">{personName}</div>
        <div className="bio-profile-role">{roleLine}</div>
        {topSkills.length > 0 && (
          <div className="bio-profile-skills">
            {topSkills.map(s => (
              <span key={s} className="bio-skill-pill">{s}</span>
            ))}
          </div>
        )}
      </div>
      <div className="bio-profile-badge">
        <span className="bio-ai-dot" />
        AI Biographer
      </div>
    </div>
  )
}

// ── Source Badges ────────────────────────────────────────────────────────────

function SourceBadges({ citations }) {
  if (!citations || citations.length === 0) return null
  return (
    <div className="bio-sources-row">
      <span className="bio-sources-label">Sources</span>
      {citations.map((c, i) => {
        const style = getSourceStyle(c)
        return (
          <span
            key={i}
            className="bio-source-badge"
            style={{ background: style.bg, color: style.color }}
          >
            {style.icon} {c}
          </span>
        )
      })}
    </div>
  )
}

// ── Follow-Up Questions ──────────────────────────────────────────────────────

function FollowUpChips({ questions, onAsk }) {
  if (!questions || questions.length === 0) return null
  return (
    <div className="bio-followups">
      <div className="bio-followups-label">Continue exploring</div>
      <div className="bio-followups-chips">
        {questions.map((q, i) => (
          <button
            key={i}
            className="bio-followup-chip"
            onClick={() => onAsk(q)}
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Chat Message ─────────────────────────────────────────────────────────────

function ChatMessage({ msg, onAsk }) {
  if (msg.role === 'user') {
    return (
      <div className="bio-msg-wrapper bio-msg-user fade-in">
        <div className="bio-bubble-user">{msg.content}</div>
      </div>
    )
  }

  const isStreaming = msg.streaming
  const isQuotaError = msg.isQuotaError

  return (
    <div className="bio-msg-wrapper bio-msg-ai fade-in">
      <div className={`bio-ai-icon ${isQuotaError ? 'bio-ai-icon-error' : ''}`}>
        {isQuotaError ? '⚠' : '✦'}
      </div>
      <div className="bio-msg-body">
        <div className={`bio-bubble-ai ${isStreaming ? 'streaming' : ''} ${isQuotaError ? 'bio-bubble-error' : ''}`}>
          {isStreaming && msg.content === '' ? (
            <div className="typing-indicator">
              <div className="typing-dot" /><div className="typing-dot" /><div className="typing-dot" />
            </div>
          ) : (
            <span className={isStreaming ? 'streaming-cursor' : ''}>{msg.content}</span>
          )}
          {isQuotaError && !isStreaming && (
            <div className="bio-error-hint">↻ The service should be available again shortly. Try asking again.</div>
          )}
        </div>

        {!isStreaming && !isQuotaError && (
          <>
            <SourceBadges citations={msg.citations} />
            <FollowUpChips questions={msg.followUps} onAsk={onAsk} />
          </>
        )}
      </div>
    </div>
  )
}

// ── Empty State ──────────────────────────────────────────────────────────────

const STARTER_QUESTIONS = [
  "Tell me about their background and career journey",
  "What are their strongest technical skills?",
  "What notable projects have they worked on?",
  "What experience do they have in AI & machine learning?",
  "Which companies have they worked for?",
  "What are their biggest achievements?",
]

function EmptyState({ personName, onAsk }) {
  return (
    <div className="bio-empty-state">
      <div className="bio-empty-icon">✦</div>
      <div className="bio-empty-title">Ask me anything about {personName}</div>
      <div className="bio-empty-sub">
        I have access to their resume, LinkedIn, GitHub, and more. Start a conversation below.
      </div>
      <div className="bio-starter-grid">
        {STARTER_QUESTIONS.map(q => (
          <button key={q} className="bio-starter-card" onClick={() => onAsk(q)}>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Main ChatPanel ────────────────────────────────────────────────────────────

export default function ChatPanel({ personName, sessionId }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [profileData, setProfileData] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Load profile summary for the header
  useEffect(() => {
    fetch(`${API}/profile?person_name=${encodeURIComponent(personName)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setProfileData(data) })
      .catch(() => {})
  }, [personName])

  // Expose quick-ask hook
  useEffect(() => {
    window.__quickAsk = (q) => { setInput(q); inputRef.current?.focus() }
    return () => { delete window.__quickAsk }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || isLoading) return
    setIsLoading(true)
    setInput('')

    const userMsg = { role: 'user', content: query }
    const aiMsgId = Date.now()
    const aiMsg = { id: aiMsgId, role: 'ai', content: '', streaming: true, citations: [], followUps: [], isQuotaError: false }

    setMessages(prev => [...prev, userMsg, aiMsg])

    try {
      const url = new URL('/api/qa/stream', window.location.origin)
      url.searchParams.set('query', query)
      url.searchParams.set('person_name', personName)
      url.searchParams.set('session_id', sessionId)

      const es = new EventSource(url.toString())
      let fullText = ''
      let retrievalCitations = []

      es.onmessage = (e) => {
        if (!e.data || e.data === '[DONE]') return
        try {
          const data = JSON.parse(e.data)

          if (data.type === 'retrieval') {
            retrievalCitations = data.citations ?? []
          } else if (data.type === 'token') {
            const isQuotaError = data.is_quota_error === true
            fullText += data.text || ''
            setMessages(prev => prev.map(m =>
              m.id === aiMsgId ? { ...m, content: fullText, isQuotaError } : m
            ))
          } else if (data.type === 'done') {
            es.close()
            const followUps = data.follow_ups ?? []
            const isQuotaError = data.is_quota_error === true
            setMessages(prev => prev.map(m =>
              m.id === aiMsgId
                ? { ...m, streaming: false, citations: isQuotaError ? [] : retrievalCitations, followUps, isQuotaError }
                : m
            ))
            setIsLoading(false)
          }
        } catch (_) {}
      }

      es.onerror = () => {
        es.close()
        setMessages(prev => prev.map(m =>
          m.id === aiMsgId
            ? { ...m, streaming: false, content: fullText || '[Connection error — check backend is running]' }
            : m
        ))
        setIsLoading(false)
      }
    } catch (err) {
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, streaming: false, content: `[Error: ${err.message}]` }
          : m
      ))
      setIsLoading(false)
    }
  }, [personName, sessionId, isLoading])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  return (
    <div className="bio-chat-layout">
      {/* Profile Header */}
      <ProfileHeader personName={personName} profileData={profileData} />

      {/* Messages */}
      <div className="bio-messages-area">
        {messages.length === 0 ? (
          <EmptyState personName={personName} onAsk={(q) => sendMessage(q)} />
        ) : (
          messages.map((msg, i) => (
            <ChatMessage key={msg.id || i} msg={msg} onAsk={(q) => sendMessage(q)} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bio-input-area">
        <div className="bio-input-wrap">
          <textarea
            id="chat-input"
            ref={inputRef}
            className="bio-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`Ask anything about ${personName}...`}
            rows={1}
            disabled={isLoading}
          />
          <button
            id="chat-send-btn"
            className="bio-send-btn"
            onClick={() => sendMessage(input)}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? (
              <span className="bio-send-spinner" />
            ) : (
              <span>↑</span>
            )}
          </button>
        </div>
        <div className="bio-input-hint">
          Press Enter to send · Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
