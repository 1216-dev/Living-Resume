import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';

const graphData = {
  nodes: [
    { id: "Kaushal", group: "Person", level: 0 },
    { id: "Stony Brook University", group: "Org", level: 1 },
    { id: "Campus Life Centers", group: "Org", level: 1 },
    { id: "MS Data Science", group: "Education", level: 2 },
    { id: "Grad AV Coordinator", group: "Role", level: 2 },
    { id: "CSE 523 - RAG System", group: "Project", level: 3 },
    { id: "Troubleshot Yamaha TF5 & Staging", group: "Project", level: 3 },
    { id: "Python", group: "Skill", level: 4 },
    { id: "SQL", group: "Skill", level: 4 },
    { id: "Yamaha TF5", group: "Skill", level: 4 }
  ],
  links: [
    { source: "Kaushal", target: "Stony Brook University", type: "STUDIED_AT" },
    { source: "Kaushal", target: "Campus Life Centers", type: "WORKED_AT" },
    { source: "Stony Brook University", target: "MS Data Science", type: "PURSUED" },
    { source: "MS Data Science", target: "CSE 523 - RAG System", type: "INCLUDES" },
    { source: "Campus Life Centers", target: "Grad AV Coordinator", type: "ROLE" },
    { source: "Grad AV Coordinator", target: "Troubleshot Yamaha TF5 & Staging", type: "ACHIEVED" },
    { source: "CSE 523 - RAG System", target: "Python", type: "LEVERAGED" },
    { source: "CSE 523 - RAG System", target: "SQL", type: "LEVERAGED" },
    { source: "Troubleshot Yamaha TF5 & Staging", target: "Yamaha TF5", type: "USED" },
    { source: "Kaushal", target: "Python", type: "SKILLED_IN" } // Cross-link example
  ]
};

const COLOR_MAP = {
  Person: "#ff5722",
  Org: "#3f51b5",
  Education: "#9c27b0",
  Role: "#009688",
  Project: "#ffc107",
  Skill: "#e91e63"
};

export default function LivingResumeGraph() {
  const fgRef = useRef();
  
  // Initialize expandedNodes with levels 0, 1, and 2
  const initialExpanded = useMemo(() => {
    return new Set(graphData.nodes.filter(n => n.level <= 2).map(n => n.id));
  }, []);

  const [expandedNodes, setExpandedNodes] = useState(initialExpanded);
  const [focusedNode, setFocusedNode] = useState(null);

  const handleNodeClick = useCallback((node) => {
    setFocusedNode(node.id);
    
    // Find all children (targets of links where clicked node is source)
    // Note: react-force-graph converts link source/target to node objects after initialization
    const childrenIds = graphData.links
      .filter(l => (typeof l.source === 'object' ? l.source.id : l.source) === node.id)
      .map(l => typeof l.target === 'object' ? l.target.id : l.target);
      
    if (childrenIds.length > 0) {
      setExpandedNodes(prev => {
        const next = new Set(prev);
        childrenIds.forEach(id => next.add(id));
        return next;
      });
    }

    // Optional cinematic zoom to node
    if (fgRef.current) {
      fgRef.current.centerAt(node.x, node.y, 1000);
      fgRef.current.zoom(1.5, 1000);
    }
  }, []);

  const handleResetFocus = () => {
    setFocusedNode(null);
    if (fgRef.current) {
      fgRef.current.zoomToFit(800, 50);
    }
  };

  // 1. Derive visible graph data based on expanded nodes
  const visibleGraphData = useMemo(() => {
    const nodes = graphData.nodes.filter(n => expandedNodes.has(n.id));
    const links = graphData.links.filter(l => {
      const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
      const targetId = typeof l.target === 'object' ? l.target.id : l.target;
      return expandedNodes.has(sourceId) && expandedNodes.has(targetId);
    });
    return { nodes, links };
  }, [expandedNodes]);

  // 2. Compute focus cluster (focusedNode + direct neighbors)
  const focusCluster = useMemo(() => {
    if (!focusedNode) return new Set();
    const cluster = new Set([focusedNode]);
    visibleGraphData.links.forEach(l => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      if (s === focusedNode) cluster.add(t);
      if (t === focusedNode) cluster.add(s);
    });
    return cluster;
  }, [focusedNode, visibleGraphData]);

  // Custom Node Rendering (with Cinematic Blur)
  const paintNode = useCallback((node, ctx, globalScale) => {
    const isFocused = !focusedNode || focusCluster.has(node.id);
    const label = node.id;
    const fontSize = 14 / globalScale;
    const radius = 6;
    
    ctx.save();
    
    // Apply Cinematic Blur & Opacity
    if (!isFocused) {
      ctx.globalAlpha = 0.15;
      ctx.filter = 'blur(2px)';
    } else {
      ctx.globalAlpha = 1.0;
      ctx.filter = 'none';
      
      // Add glow effect for focused node specifically
      if (node.id === focusedNode) {
        ctx.shadowColor = COLOR_MAP[node.group] || "#fff";
        ctx.shadowBlur = 15;
      }
    }

    // Draw Node Circle
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
    ctx.fillStyle = COLOR_MAP[node.group] || "#999";
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();
    
    ctx.shadowColor = 'transparent'; // reset shadow for text

    // Draw Node Label
    ctx.font = `${fontSize}px Inter, Sans-Serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = isFocused ? '#f8fafc' : '#94a3b8';
    ctx.fillText(label, node.x, node.y + radius + (fontSize * 1.2));
    
    ctx.restore();
  }, [focusedNode, focusCluster]);

  // Custom Link Rendering (with Cinematic Blur)
  const paintLink = useCallback((link, ctx, globalScale) => {
    const s = typeof link.source === 'object' ? link.source.id : link.source;
    const t = typeof link.target === 'object' ? link.target.id : link.target;
    // Link is focused if it connects to the focusedNode
    const isFocused = !focusedNode || (s === focusedNode || t === focusedNode);
    
    ctx.save();
    
    if (!isFocused) {
      ctx.globalAlpha = 0.15;
      ctx.filter = 'blur(2px)';
    } else {
      ctx.globalAlpha = 0.8;
      ctx.filter = 'none';
    }
    
    ctx.beginPath();
    ctx.moveTo(link.source.x, link.source.y);
    ctx.lineTo(link.target.x, link.target.y);
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2 / Math.sqrt(globalScale);
    ctx.stroke();

    // Render Link Label (Relationship type)
    if (isFocused) {
      const midX = (link.source.x + link.target.x) / 2;
      const midY = (link.source.y + link.target.y) / 2;
      const fontSize = 10 / globalScale;
      
      ctx.font = `italic ${fontSize}px Inter, Sans-Serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(link.type, midX, midY - (5 / globalScale));
    }
    
    ctx.restore();
  }, [focusedNode]);

  // Run initial zoom-to-fit once the engine has stabilized slightly
  useEffect(() => {
    const timer = setTimeout(() => {
      if (fgRef.current) fgRef.current.zoomToFit(600, 50);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh', background: '#020617', fontFamily: 'Inter, sans-serif' }}>
      
      {/* HUD / Controls */}
      {focusedNode && (
        <div style={{ position: 'absolute', top: 24, left: 24, zIndex: 10 }}>
          <button 
            onClick={handleResetFocus}
            style={{
              padding: '12px 20px',
              background: 'rgba(255, 255, 255, 0.1)',
              color: '#fff',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '8px',
              backdropFilter: 'blur(10px)',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '600',
              boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
              transition: 'all 0.2s ease-in-out'
            }}
            onMouseOver={e => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
            onMouseOut={e => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
          >
            ← Reset Focus
          </button>
        </div>
      )}

      {/* Force Graph */}
      <ForceGraph2D
        ref={fgRef}
        graphData={visibleGraphData}
        nodeCanvasObject={paintNode}
        linkCanvasObject={paintLink}
        onNodeClick={handleNodeClick}
        d3VelocityDecay={0.2}
        cooldownTicks={150}
      />
    </div>
  );
}
