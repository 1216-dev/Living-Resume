import { useEffect, useRef, useState, useCallback } from 'react'

const API = '/api'

const NODE_STYLES = {
  PERSON:   { color: '#7c6af7', size: 28, shape: 'dot' },
  COMPANY:  { color: '#2dd4bf', size: 22, shape: 'dot' },
  SKILL:    { color: '#fb923c', size: 16, shape: 'dot' },
  TOOL:     { color: '#f97316', size: 16, shape: 'dot' },
  PROJECT:  { color: '#60a5fa', size: 20, shape: 'dot' },
  DEGREE:   { color: '#facc15', size: 18, shape: 'dot' },
  LOCATION: { color: '#a78bfa', size: 14, shape: 'dot' },
  ROLE:     { color: '#f472b6', size: 18, shape: 'dot' },
  ACHIEVEMENT: { color: '#fde047', size: 16, shape: 'star' },
  RESPONSIBILITY: { color: '#9ca3af', size: 14, shape: 'dot' },
  FRAMEWORK: { color: '#10b981', size: 16, shape: 'square' },
  PUBLICATION: { color: '#8b5cf6', size: 16, shape: 'triangle' },
  DATASET: { color: '#06b6d4', size: 16, shape: 'database' },
  TECHNOLOGY: { color: '#f43f5e', size: 16, shape: 'hexagon' },
  UNKNOWN:  { color: '#475569', size: 12, shape: 'dot' },
  CATEGORY: { color: '#ef4444', size: 30, shape: 'diamond' },
  HUB:      { color: '#ec4899', size: 18, shape: 'hexagon' },
}

const COMMUNITY_COLORS = [
  '#7c6af7', '#2dd4bf', '#fb923c', '#60a5fa',
  '#4ade80', '#facc15', '#f472b6', '#a78bfa',
]

function getCommunityColor(communityId) {
  if (communityId == null) return '#475569'
  return COMMUNITY_COLORS[communityId % COMMUNITY_COLORS.length]
}

export default function GraphView() {
  const containerRef = useRef(null)
  const networkRef = useRef(null)
  const nodesRef = useRef(null)
  const edgesRef = useRef(null)
  
  const [graphData, setGraphData] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  
  // State machine for contextual progression
  const [activePath, setActivePath] = useState([])

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Virtual Graph refs
  const vNodesRef = useRef([])
  const vEdgesRef = useRef([])

  const fetchGraph = useCallback(async () => {
    try {
      setLoading(true)
      const res = await fetch(`${API}/graph`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setGraphData(data)
      setActivePath([]) // reset path
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchGraph() }, [fetchGraph])

  // Process raw graph into Virtual Graph & initialize Network
  useEffect(() => {
    if (!graphData || !containerRef.current) return

    const { nodes: rawNodes, edges: rawEdges } = graphData.graph || { nodes: [], edges: [] }

    // --- BUILD VIRTUAL GRAPH ---
    const virtualNodes = []
    const virtualEdges = []
    const nodeMap = new Map()

    const addVNode = (id, label, type, orig = null) => {
      if (!nodeMap.has(id)) {
        const n = { id, label, type, orig }
        virtualNodes.push(n)
        nodeMap.set(id, n)
      }
    }

    const ROOT_NODES = ['root_work', 'root_edu', 'root_proj', 'root_skills']
    addVNode('root_work', 'Work Experience', 'CATEGORY')
    addVNode('root_edu', 'Education', 'CATEGORY')
    addVNode('root_proj', 'Projects', 'CATEGORY')
    addVNode('root_skills', 'Skills Overview', 'CATEGORY')

    rawNodes.forEach(n => {
      addVNode(n.id, n.label, n.type, n)
      
      // Attach to roots and generate diamond hubs
      if (n.type === 'COMPANY') {
        virtualEdges.push({ from: 'root_work', to: n.id, label: 'includes' })
      } 
      else if (n.type === 'DEGREE' || n.type === 'UNIVERSITY') {
        virtualEdges.push({ from: 'root_edu', to: n.id, label: 'includes' })
      } 
      else if (n.type === 'SKILL' || n.type === 'TOOL' || n.type === 'TECHNOLOGY') {
        virtualEdges.push({ from: 'root_skills', to: n.id, label: 'includes' })
      } 
      else if (n.type === 'PROJECT') {
        virtualEdges.push({ from: 'root_proj', to: n.id, label: 'includes' })
      }
    })

    rawEdges.forEach(e => {
      const fromNode = nodeMap.get(e.from)
      const toNode = nodeMap.get(e.to)
      if (!fromNode || !toNode) return
      
      // Direct natural edge mapping
      virtualEdges.push({ from: e.from, to: e.to, label: e.relation })
    })

    vNodesRef.current = virtualNodes
    vEdgesRef.current = virtualEdges

    // --- VIS.JS INITIALIZATION ---
    import('vis-network/standalone/esm/vis-network.min.js').then(({ Network, DataSet }) => {
      
      const nodes = new DataSet(
        virtualNodes.map(n => {
          const style = NODE_STYLES[n.type] || NODE_STYLES.UNKNOWN
          const commColor = n.orig ? getCommunityColor(n.orig.community) : style.color
          return {
            id: n.id,
            label: n.label,
            title: n.orig ? `${n.type}: ${n.label}\n${n.orig.context || ''}` : n.label,
            color: {
              background: style.color,
              border: commColor,
              highlight: { background: '#fff', border: commColor },
              hover: { background: style.color, border: '#fff' },
            },
            size: style.size,
            shape: style.shape,
            font: { color: '#e2e8f0', size: 11, face: 'Inter' },
            borderWidth: (n.orig && n.orig.community != null) ? 2.5 : 2,
            shadow: { enabled: true, color: `${style.color}44`, size: 12 },
            hidden: true // managed by visibility effect later
          }
        })
      )

      const edges = new DataSet(
        virtualEdges.map((e, i) => ({
          id: i,
          from: e.from,
          to: e.to,
          label: e.label,
          color: { color: 'rgba(255,255,255,0.12)', highlight: 'rgba(124,106,247,0.7)' },
          font: { 
            color: '#0f172a', 
            size: 10, 
            face: 'JetBrains Mono',
            background: 'rgba(255, 255, 255, 0.95)',
            strokeWidth: 0,
            vadjust: 0
          },
          arrows: { to: { enabled: true, scaleFactor: 0.5 } },
          width: 1,
          smooth: { type: 'curvedCW', roundness: 0.1 },
          hidden: true
        }))
      )

      nodesRef.current = nodes
      edgesRef.current = edges

      const options = {
        physics: {
          enabled: true,
          barnesHut: {
            gravitationalConstant: -35000,
            centralGravity: 0.05,
            springLength: 300,
            springConstant: 0.02,
            damping: 0.09,
          },
          stabilization: { iterations: 200 },
        },
        interaction: {
          hover: true,
          tooltipDelay: 200,
          navigationButtons: false,
          keyboard: true,
          zoomView: true,
        },
        layout: { randomSeed: 42 },
      }

      if (networkRef.current) networkRef.current.destroy()

      const network = new Network(containerRef.current, { nodes, edges }, options)
      networkRef.current = network

      network.on('click', (params) => {
        if (params.nodes.length > 0) {
          const nodeId = params.nodes[0]
          
          setActivePath(prev => {
            const idx = prev.indexOf(nodeId)
            if (idx >= 0) {
              return prev.slice(0, idx + 1)
            } else {
              return [...prev, nodeId]
            }
          })

          network.focus(nodeId, {
            scale: 1.2,
            animation: { duration: 600, easingFunction: 'easeInOutQuad' }
          })

          const vNode = virtualNodes.find(n => n.id === nodeId)
          if (vNode && vNode.orig) {
            setSelectedNode(vNode.orig)
          } else {
            setSelectedNode(null)
          }
        } else {
          // Clicked background -> just deselect node, do NOT collapse path or move camera
          setSelectedNode(null)
        }
      })

      // Turn off physics entirely after initial load stabilization
      network.once('stabilizationIterationsDone', () => {
        network.setOptions({ physics: false })
      })

      // Initial render trigger
      setActivePath([])

    }).catch(err => console.error('vis-network error:', err))

    return () => {
      networkRef.current?.destroy()
      networkRef.current = null
      nodesRef.current = null
      edgesRef.current = null
    }
  }, [graphData])

  // Visibility engine (Spotlight / Blur effect)
  useEffect(() => {
    if (!vNodesRef.current.length || !nodesRef.current || !edgesRef.current) return

    const focusedNodeIds = new Set()
    
    // 1. Root nodes are ALWAYS focused
    const ROOT_NODES = ['root_work', 'root_edu', 'root_proj', 'root_skills']
    ROOT_NODES.forEach(id => focusedNodeIds.add(id))

    // 2. Nodes in the active path are ALWAYS focused
    activePath.forEach(id => focusedNodeIds.add(id))

    // 3. Direct children of the LEAF node are focused
    const leafNodeId = activePath.length > 0 ? activePath[activePath.length - 1] : null
    
    if (leafNodeId) {
      vEdgesRef.current.forEach(e => {
        if (e.from === leafNodeId) focusedNodeIds.add(e.to)
        if (e.to === leafNodeId) focusedNodeIds.add(e.from)
      })
    }

    const nodesUpdate = vNodesRef.current.map(n => {
      const isFocused = focusedNodeIds.has(n.id)
      const isPath = activePath.includes(n.id)
      
      return {
        id: n.id,
        hidden: false, // Never hide, keep physics stable
        opacity: isFocused ? 1 : 0.15, // Blur unfocused nodes
        borderWidth: isPath ? 4 : (isFocused ? 2 : 1),
        font: {
          color: isFocused ? '#e2e8f0' : 'transparent',
        }
      }
    })
    nodesRef.current.update(nodesUpdate)

    const edgesUpdate = vEdgesRef.current.map((e, i) => {
      const fromFocused = focusedNodeIds.has(e.from)
      const toFocused = focusedNodeIds.has(e.to)
      const isHighlighted = fromFocused && toFocused;
      
      return {
        id: i,
        hidden: false,
        color: { 
          opacity: isHighlighted ? 0.7 : 0.05 
        },
        font: {
          color: isHighlighted ? '#0f172a' : 'transparent',
          background: isHighlighted ? 'rgba(255, 255, 255, 0.95)' : 'transparent',
        }
      }
    })
    edgesRef.current.update(edgesUpdate)

  }, [activePath])

  const stats = graphData?.stats || {}

  return (
    <div className="graph-container">
      {loading && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--bg-primary)', zIndex: 20,
          flexDirection: 'column', gap: 12, color: 'var(--text-secondary)',
        }}>
          <div style={{ fontSize: 32 }}>🕸</div>
          <div style={{ fontSize: 14 }}>Loading progressive graph…</div>
        </div>
      )}

      {error && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'var(--bg-primary)', zIndex: 20,
          flexDirection: 'column', gap: 12,
        }}>
          <div style={{ fontSize: 32 }}>⚠️</div>
          <div style={{ fontSize: 14, color: 'var(--accent-orange)' }}>
            Graph unavailable
          </div>
          <button className="primary-btn btn-purple" onClick={fetchGraph}>Retry</button>
        </div>
      )}

      {/* Path Breadcrumbs overlay */}
      {!loading && !error && (
        <div style={{
          position: 'absolute', top: 16, left: 16, zIndex: 10,
          display: 'flex', gap: 8, alignItems: 'center',
          background: 'rgba(13,15,20,0.8)', padding: '8px 12px',
          borderRadius: 8, border: '1px solid var(--border)',
        }}>
          <div 
            style={{ color: activePath.length === 0 ? '#fff' : '#64748b', cursor: 'pointer', fontSize: 12 }}
            onClick={() => {
              setActivePath([])
              if (networkRef.current) {
                networkRef.current.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } })
              }
            }}
          >
            Overview
          </div>
          {activePath.map((id, idx) => {
            const vNode = vNodesRef.current.find(n => n.id === id)
            if (!vNode) return null
            return (
              <div key={id} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ color: '#475569' }}>/</span>
                <div 
                  style={{ color: idx === activePath.length - 1 ? '#fff' : '#64748b', cursor: 'pointer', fontSize: 12 }}
                  onClick={() => {
                    setActivePath(activePath.slice(0, idx + 1))
                    if (networkRef.current) {
                      networkRef.current.focus(id, { scale: 1.2, animation: { duration: 600, easingFunction: 'easeInOutQuad' } })
                    }
                  }}
                >
                  {vNode.label}
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div ref={containerRef} id="graph-canvas" />

      {/* Selected node info */}
      {selectedNode && (
        <div style={{
          position: 'absolute', top: 60, right: 16,
          background: 'rgba(13,15,20,0.95)',
          backdropFilter: 'blur(12px)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          padding: '14px 16px',
          zIndex: 10,
          maxWidth: 280,
          maxHeight: '80vh',
          overflowY: 'auto',
          boxShadow: '0 10px 25px rgba(0,0,0,0.5)'
        }}>
          <div style={{ fontSize: 12, color: 'var(--text-dim)', marginBottom: 4 }}>{selectedNode.type}</div>
          <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
            {selectedNode.label}
          </div>
          {selectedNode.context && (
           <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
             {selectedNode.context}
           </div>
          )}

          {selectedNode.type === 'PROJECT' && (() => {
            const projId = selectedNode.id;
            const relatedEdges = vEdgesRef.current.filter(e => e.from === projId || e.to === projId);
            
            const skills = [];
            const tools = [];
            const achievements = [];
            const companies = [];
            const roles = [];

            relatedEdges.forEach(e => {
              const otherNodeId = e.from === projId ? e.to : e.from;
              const otherNode = vNodesRef.current.find(n => n.id === otherNodeId);
              if (!otherNode) return;

              if (otherNode.type === 'SKILL') skills.push(otherNode.label);
              if (otherNode.type === 'TOOL' || otherNode.type === 'FRAMEWORK' || otherNode.type === 'TECHNOLOGY') tools.push(otherNode.label);
              if (otherNode.type === 'ACHIEVEMENT' || otherNode.type === 'METRICS') achievements.push(otherNode.label);
              if (otherNode.type === 'COMPANY') companies.push(otherNode.label);
              if (otherNode.type === 'ROLE') roles.push(otherNode.label);
            });

            return (
              <div style={{ marginTop: 12, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 10 }}>
                {companies.length > 0 && <div style={{ fontSize: 12, marginBottom: 4 }}><strong style={{color: '#2dd4bf'}}>Company:</strong> {companies.join(', ')}</div>}
                {roles.length > 0 && <div style={{ fontSize: 12, marginBottom: 4 }}><strong style={{color: '#f472b6'}}>Role:</strong> {roles.join(', ')}</div>}
                {tools.length > 0 && <div style={{ fontSize: 12, marginBottom: 4 }}><strong style={{color: '#f43f5e'}}>Tech & Tools:</strong> {tools.join(', ')}</div>}
                {skills.length > 0 && <div style={{ fontSize: 12, marginBottom: 4 }}><strong style={{color: '#fb923c'}}>Skills:</strong> {skills.join(', ')}</div>}
                {achievements.length > 0 && (
                  <div style={{ fontSize: 12, marginTop: 8 }}>
                    <strong style={{color: '#fde047'}}>Key Achievements:</strong>
                    <ul style={{ paddingLeft: 16, marginTop: 4, marginBottom: 0, color: 'var(--text-secondary)' }}>
                      {achievements.map((a, i) => <li key={i} style={{marginBottom: 2}}>{a}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* Bottom stats bar */}
      {!loading && !error && (
        <div className="graph-stats-bar">
          <div className="graph-stat"><span>{stats.total_nodes || 0}</span> nodes</div>
          <div className="graph-stat"><span>{stats.total_edges || 0}</span> edges</div>
          <button
            onClick={fetchGraph}
            style={{
              background: 'var(--accent-purple-dim)',
              color: 'var(--accent-purple)',
              border: '1px solid rgba(124,106,247,0.3)',
              borderRadius: 20,
              padding: '2px 10px',
              fontSize: 11,
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
            }}
          >
            Refresh Data
          </button>
        </div>
      )}
    </div>
  )
}
