import { useState, useRef, useEffect, useCallback } from 'react'

const API = '/api'

const ALL_TOPICS = [
  'career_journey', 'work_experience', 'education', 'projects',
  'technical_skills', 'leadership', 'research_and_publications',
  'achievements', 'failures_and_lessons', 'personal_interests', 'future_goals',
]

const DEFAULT_LABELS = {
  career_journey:            '🚀 Career Journey',
  work_experience:           '💼 Work Experience',
  education:                 '🎓 Education',
  projects:                  '🔧 Projects',
  technical_skills:          '⚙️ Technical Skills',
  leadership:                '👥 Leadership',
  research_and_publications: '📄 Research',
  achievements:              '🏆 Achievements',
  failures_and_lessons:      '💡 Lessons Learned',
  personal_interests:        '🎯 Personal Interests',
  future_goals:              '🌟 Future Goals',
}

const ENTITY_COLORS = {
  COMPANY:     { bg: 'rgba(124,106,247,0.12)', border: 'rgba(124,106,247,0.3)', color: '#a78bfa' },
  PROJECT:     { bg: 'rgba(20,184,166,0.12)',  border: 'rgba(20,184,166,0.3)',  color: '#2dd4bf' },
  TECHNOLOGY:  { bg: 'rgba(59,130,246,0.12)',  border: 'rgba(59,130,246,0.3)',  color: '#60a5fa' },
  FRAMEWORK:   { bg: 'rgba(59,130,246,0.12)',  border: 'rgba(59,130,246,0.3)',  color: '#60a5fa' },
  SKILL:       { bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.3)',  color: '#fbbf24' },
  TOOL:        { bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.3)',  color: '#fbbf24' },
  ACHIEVEMENT: { bg: 'rgba(16,185,129,0.12)',  border: 'rgba(16,185,129,0.3)',  color: '#34d399' },
  CHALLENGE:   { bg: 'rgba(251,146,60,0.12)',  border: 'rgba(251,146,60,0.3)',  color: '#fb923c' },
  DEGREE:      { bg: 'rgba(236,72,153,0.12)',  border: 'rgba(236,72,153,0.3)',  color: '#f472b6' },
  PERSON:      { bg: 'rgba(148,163,184,0.12)', border: 'rgba(148,163,184,0.3)', color: '#94a3b8' },
  ROLE:        { bg: 'rgba(124,106,247,0.08)', border: 'rgba(124,106,247,0.2)', color: '#c4b5fd' },
  MOTIVATION:  { bg: 'rgba(16,185,129,0.08)',  border: 'rgba(16,185,129,0.2)',  color: '#6ee7b7' },
  VALUE:       { bg: 'rgba(16,185,129,0.08)',  border: 'rgba(16,185,129,0.2)',  color: '#6ee7b7' },
  INTEREST:    { bg: 'rgba(251,191,36,0.08)',  border: 'rgba(251,191,36,0.2)',  color: '#fde68a' },
  DEFAULT:     { bg: 'rgba(148,163,184,0.1)',  border: 'rgba(148,163,184,0.25)', color: '#94a3b8' },
}

function EntityBadge({ entity }) {
  const style = ENTITY_COLORS[entity.type] || ENTITY_COLORS.DEFAULT
  const isTrigger = entity.is_trigger
  return (
    <div
      className={`iv-entity-badge ${isTrigger ? 'iv-entity-trigger' : ''}`}
      style={{ background: style.bg, border: `1px solid ${style.border}`, color: style.color }}
      title={entity.context || entity.type}
    >
      {isTrigger && <span className="iv-trigger-dot" />}
      <span className="iv-entity-type">{entity.type}</span>
      <span className="iv-entity-name">{entity.name}</span>
    </div>
  )
}

function TopicPill({ topic, label, status }) {
  // status: 'covered' | 'current' | 'remaining'
  return (
    <div className={`iv-topic-pill iv-topic-${status}`}>
      {label}
      {status === 'covered' && <span className="iv-topic-check">✓</span>}
      {status === 'current' && <span className="iv-topic-dot" />}
    </div>
  )
}

function InterviewMessage({ msg, personName }) {
  const isAI = msg.role === 'ai'
  return (
    <div className={`iv-msg-wrapper ${isAI ? 'iv-msg-ai' : 'iv-msg-user'} fade-in`}>
      {isAI ? (
        <div className="iv-avatar-col">
          <div className="iv-ai-avatar">🎙</div>
        </div>
      ) : (
        <div className="iv-avatar-col">
          <div className="iv-user-avatar">{personName?.[0]?.toUpperCase() || 'U'}</div>
        </div>
      )}
      <div className="iv-msg-body-col">
        <div className="iv-msg-label">{isAI ? 'Interviewer' : personName || 'You'}</div>
        <div className={`iv-bubble ${isAI ? 'iv-bubble-ai' : 'iv-bubble-user'}`}>
          {msg.content}
        </div>
        {/* Entity badges extracted from this answer */}
        {!isAI && msg.entities && msg.entities.length > 0 && (
          <div className="iv-entities-row">
            {msg.entities.map((e, i) => (
              <EntityBadge key={i} entity={e} />
            ))}
          </div>
        )}
        {/* Follow-up indicator */}
        {isAI && msg.isFollowup && (
          <div className="iv-followup-tag">↳ follow-up</div>
        )}
      </div>
    </div>
  )
}

function TypingBubble() {
  return (
    <div className="iv-msg-wrapper iv-msg-ai fade-in">
      <div className="iv-avatar-col">
        <div className="iv-ai-avatar iv-avatar-pulse">🎙</div>
      </div>
      <div className="iv-msg-body-col">
        <div className="iv-msg-label">Interviewer</div>
        <div className="iv-bubble iv-bubble-ai">
          <div className="typing-indicator">
            <div className="typing-dot" />
            <div className="typing-dot" />
            <div className="typing-dot" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default function InterviewPanel({ personName, sessionId, onComplete }) {
  const [messages, setMessages]         = useState([])
  const [input, setInput]               = useState('')
  const [phase, setPhase]               = useState('intro')   // intro | active | complete
  const [isLoading, setIsLoading]       = useState(false)
  const [progress, setProgress]         = useState({ pct: 0, covered: [], remaining: ALL_TOPICS, topicLabels: DEFAULT_LABELS })
  const [totalEntities, setTotalEntities] = useState(0)
  const [completionMsg, setCompletionMsg] = useState('')
  const [currentTopic, setCurrentTopic]   = useState('')
  const [isAudioEnabled, setIsAudioEnabled] = useState(true)
  const [isRecording, setIsRecording]       = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef       = useRef(null)
  const recognitionRef = useRef(null)

  const speakQuestion = useCallback((text) => {
    if (!isAudioEnabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    // Try to find a good English voice
    const preferredVoice = voices.find(v => 
      v.name.includes('Google US English') || 
      v.name.includes('Samantha') || 
      (v.lang.startsWith('en') && v.localService === false)
    ) || voices.find(v => v.lang.startsWith('en'));
    
    if (preferredVoice) utterance.voice = preferredVoice;
    utterance.rate = 1.05; // Slightly faster for natural pacing
    window.speechSynthesis.speak(utterance);
  }, [isAudioEnabled]);

  // Load voices async and cancel speech on unmount
  useEffect(() => {
    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }

    // Initialize Speech Recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        
        // Append final parts to the input immediately
        if (finalTranscript) {
          setInput(prev => {
            const sep = prev && !prev.endsWith(' ') ? ' ' : '';
            return prev + sep + finalTranscript;
          });
        }
        
        // For interim results, we could display them, but it's simpler to just 
        // rely on final transcripts since we're using a standard textarea.
      };

      recognition.onerror = (event) => {
        console.error("Speech recognition error:", event.error);
        setIsRecording(false);
      };

      recognition.onend = () => {
        // Automatically stop recording UI if the browser ends it
        setIsRecording(false);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      window.speechSynthesis?.cancel();
      recognitionRef.current?.stop();
    };
  }, []);

  const toggleRecording = useCallback(() => {
    if (!recognitionRef.current) return;
    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (err) {
        console.error("Could not start speech recognition", err);
      }
    }
  }, [isRecording]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const startInterview = useCallback(async () => {
    setIsLoading(true)
    try {
      const res = await fetch(`${API}/interview/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_name: personName, session_id: sessionId }),
      })
      const data = await res.json()
      setPhase('active')
      setProgress({
        pct:         data.progress_pct || 0,
        covered:     data.covered_topics || [],
        remaining:   data.remaining_topics || ALL_TOPICS,
        topicLabels: data.topic_labels || DEFAULT_LABELS,
      })
      setCurrentTopic(data.current_topic || '')
      setMessages([{ role: 'ai', content: data.question, isFollowup: false }])
      if (data.question) {
        speakQuestion(data.question)
      }
    } catch (e) {
      console.error('Interview start failed:', e)
    } finally {
      setIsLoading(false)
      setTimeout(() => inputRef.current?.focus(), 200)
    }
  }, [personName, sessionId])

  const resetInterview = useCallback(async () => {
    try {
      await fetch(`${API}/interview/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_name: personName, session_id: sessionId }),
      })
    } catch (_) {}
    setMessages([])
    setInput('')
    setPhase('intro')
    setProgress({ pct: 0, covered: [], remaining: ALL_TOPICS, topicLabels: DEFAULT_LABELS })
    setTotalEntities(0)
    setCurrentTopic('')
    window.speechSynthesis?.cancel();
  }, [personName, sessionId])

  const submitAnswer = useCallback(async () => {
    if (!input.trim() || isLoading) return
    const answer = input.trim()
    setInput('')
    setIsLoading(true)

    if (isRecording && recognitionRef.current) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }

    // Optimistically add user message
    setMessages(prev => [...prev, { role: 'user', content: answer, entities: [] }])

    try {
      const res = await fetch(`${API}/interview/answer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_name: personName, session_id: sessionId, answer }),
      })
      const data = await res.json()

      // Update user message with extracted entities
      const entities = data.extracted_entities || []
      setMessages(prev => {
        const updated = [...prev]
        // Find last user message and attach entities
        for (let i = updated.length - 1; i >= 0; i--) {
          if (updated[i].role === 'user') {
            updated[i] = { ...updated[i], entities }
            break
          }
        }
        return updated
      })

      setTotalEntities(prev => prev + entities.length)

      if (data.is_complete) {
        setPhase('complete')
        setCompletionMsg(data.message || 'Interview complete!')
        setProgress(prev => ({ ...prev, pct: 100, covered: data.covered_topics || prev.covered }))
        onComplete?.()
      } else {
        setProgress({
          pct:         data.progress_pct || 0,
          covered:     data.covered_topics || [],
          remaining:   data.remaining_topics || [],
          topicLabels: data.topic_labels || DEFAULT_LABELS,
        })
        setCurrentTopic(data.current_topic || '')
        setMessages(prev => [
          ...prev,
          { role: 'ai', content: data.question, isFollowup: data.is_followup || false },
        ])
        if (data.question) {
          speakQuestion(data.question)
        }
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: 'ai', content: 'Something went wrong. Please try again.', isFollowup: false }])
    } finally {
      setIsLoading(false)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [input, isLoading, personName, sessionId, onComplete])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitAnswer()
    }
  }

  const labels = progress.topicLabels || DEFAULT_LABELS

  // ── Completion screen ───────────────────────────────────────────────────────
  if (phase === 'complete') {
    return (
      <div className="iv-container">
        <div className="iv-complete-screen fade-in">
          <div className="iv-complete-orb">🎙</div>
          <div className="iv-complete-title">Interview Complete!</div>
          <div className="iv-complete-sub">{completionMsg}</div>
          <div className="iv-complete-stats">
            <div className="iv-stat-card">
              <div className="iv-stat-num">{progress.covered.length}</div>
              <div className="iv-stat-lbl">Topics Covered</div>
            </div>
            <div className="iv-stat-card">
              <div className="iv-stat-num">{messages.filter(m => m.role === 'user').length}</div>
              <div className="iv-stat-lbl">Questions Answered</div>
            </div>
            <div className="iv-stat-card">
              <div className="iv-stat-num">{totalEntities}</div>
              <div className="iv-stat-lbl">Entities Extracted</div>
            </div>
          </div>
          <div className="iv-complete-topics">
            {progress.covered.map(t => (
              <span key={t} className="iv-topic-pill iv-topic-covered">
                {labels[t] || t}<span className="iv-topic-check">✓</span>
              </span>
            ))}
          </div>
          <p className="iv-complete-hint">
            Switch to the <strong>Chat</strong> tab to ask questions about {personName}.
          </p>
          <button className="iv-reset-btn" onClick={resetInterview}>↺ Start New Interview</button>
        </div>
      </div>
    )
  }

  // ── Intro screen ────────────────────────────────────────────────────────────
  if (phase === 'intro') {
    return (
      <div className="iv-container">
        <div className="iv-intro fade-in">
          <div className="iv-intro-orb">🎙</div>
          <h2 className="iv-intro-title">AI Biography Interviewer</h2>
          <p className="iv-intro-desc">
            An AI that interviews like a podcast host, recruiter, and career coach —
            asking one thoughtful question at a time to build a rich digital biography of{' '}
            <strong>{personName}</strong>.
          </p>
          <div className="iv-intro-features">
            <div className="iv-intro-feature">
              <span>🎯</span>
              <span>Adapts based on your answers</span>
            </div>
            <div className="iv-intro-feature">
              <span>🔍</span>
              <span>Digs deeper on interesting threads</span>
            </div>
            <div className="iv-intro-feature">
              <span>🧠</span>
              <span>Extracts entities into the knowledge graph in real-time</span>
            </div>
            <div className="iv-intro-feature">
              <span>📖</span>
              <span>Covers 11 biography topic areas</span>
            </div>
          </div>
          <button
            id="start-interview-btn"
            className="iv-start-btn"
            onClick={startInterview}
            disabled={isLoading}
          >
            {isLoading ? '⏳ Starting…' : '▶ Begin Interview'}
          </button>
        </div>
      </div>
    )
  }

  // ── Active interview ────────────────────────────────────────────────────────
  const allTopicsSorted = [...ALL_TOPICS]

  return (
    <div className="iv-container">

      {/* ── Header bar */}
      <div className="iv-header">
        <div className="iv-header-left">
          <div className="iv-header-avatar">🎙</div>
          <div>
            <div className="iv-header-title">AI Interviewer</div>
            <div className="iv-header-sub">
              {currentTopic ? `Exploring: ${labels[currentTopic] || currentTopic}` : 'Active Session'}
            </div>
          </div>
        </div>
        <div className="iv-header-right">
          <button 
            className="iv-restart-btn" 
            onClick={() => {
              window.speechSynthesis?.cancel()
              setIsAudioEnabled(!isAudioEnabled)
            }} 
            title={isAudioEnabled ? "Mute AI voice" : "Enable AI voice"}
          >
            {isAudioEnabled ? '🔊' : '🔇'}
          </button>
          <div className="iv-progress-ring-wrap">
            <svg className="iv-progress-ring" width="48" height="48" viewBox="0 0 48 48">
              <circle cx="24" cy="24" r="20" fill="none" stroke="rgba(124,106,247,0.12)" strokeWidth="4" />
              <circle
                cx="24" cy="24" r="20"
                fill="none"
                stroke="#7c6af7"
                strokeWidth="4"
                strokeDasharray={`${2 * Math.PI * 20}`}
                strokeDashoffset={`${2 * Math.PI * 20 * (1 - progress.pct / 100)}`}
                strokeLinecap="round"
                transform="rotate(-90 24 24)"
                style={{ transition: 'stroke-dashoffset 0.6s ease' }}
              />
            </svg>
            <span className="iv-progress-pct">{progress.pct}%</span>
          </div>
          <div className="iv-header-stats">
            <div className="iv-hstat"><span>{messages.filter(m => m.role === 'user').length}</span> answers</div>
            <div className="iv-hstat"><span>{totalEntities}</span> entities</div>
          </div>
          <button className="iv-restart-btn" onClick={resetInterview} title="Restart interview">↺</button>
        </div>
      </div>

      {/* ── Topic track */}
      <div className="iv-topic-track">
        {allTopicsSorted.map(t => {
          const isCovered  = progress.covered.includes(t)
          const isCurrent  = t === currentTopic
          const status     = isCovered ? 'covered' : isCurrent ? 'current' : 'remaining'
          return <TopicPill key={t} topic={t} label={labels[t] || t} status={status} />
        })}
      </div>

      {/* ── Transcript */}
      <div className="iv-transcript">
        {messages.map((msg, i) => (
          <InterviewMessage key={i} msg={msg} personName={personName} />
        ))}
        {isLoading && <TypingBubble />}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Input */}
      <div className="iv-input-area">
        <button
          className={`iv-mic-btn ${isRecording ? 'recording' : ''}`}
          onClick={toggleRecording}
          disabled={isLoading || !recognitionRef.current}
          title={!recognitionRef.current ? "Speech recognition not supported in your browser" : isRecording ? "Stop listening" : "Dictate your answer"}
        >
          {isRecording ? '🛑' : '🎙️'}
        </button>
        <textarea
          id="interview-answer-input"
          ref={inputRef}
          className="iv-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Share your answer… (Enter to submit, Shift+Enter for new line)"
          disabled={isLoading}
          rows={3}
        />
        <button
          id="interview-submit-btn"
          className="iv-submit-btn"
          onClick={submitAnswer}
          disabled={isLoading || !input.trim()}
        >
          {isLoading ? (
            <span className="iv-btn-loading">⏳</span>
          ) : (
            <>Send <span className="iv-btn-arrow">→</span></>
          )}
        </button>
      </div>

    </div>
  )
}
