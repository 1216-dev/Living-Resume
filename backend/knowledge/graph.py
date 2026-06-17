"""
knowledge/graph.py
───────────────────
Builds and queries a knowledge graph of entities extracted from all chunks.
Uses NetworkX (no infrastructure required — swap to Neo4j later).

Entity types: PERSON, COMPANY, SKILL, PROJECT, DEGREE, LOCATION, TOOL
Edges: worked_at, used_skill, built_project, studied_at, located_in, used_tool

GraphRAG:
  - Local: find 2-hop neighbours of query entities
  - Global: community summaries via Louvain / greedy modularity clustering
  - Communities answer "career arc" questions that single-chunk retrieval misses
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

import networkx as nx

from backend.config import GRAPH_DB_PATH, ENTITY_TYPES, GRAPH_COMMUNITY_MIN_SIZE


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self._community_summaries: Dict[str, str] = {}
        self._community_map: Dict[str, int] = {}  # node_id → community_id

    # ── Building ──────────────────────────────────────────────────────────────

    def add_entities_from_extraction(self, extraction: Dict[str, Any], source: str) -> None:
        """
        Add entities + relationships extracted by LLM.
        Expected format:
          {
            "entities": [{"type": "COMPANY", "name": "ISRO", "context": "..."}],
            "relationships": [{"from": "Devshree", "relation": "worked_at", "to": "ISRO"}]
          }
        """
        for ent in extraction.get("entities", []):
            node_id = _normalize(ent["name"])
            if not node_id:
                continue
            if not self.g.has_node(node_id):
                self.g.add_node(node_id, **{
                    "type": ent.get("type", "UNKNOWN"),
                    "name": ent["name"],
                    "context": ent.get("context", ""),
                    "sources": [source],
                })
            else:
                existing = self.g.nodes[node_id]
                if source not in existing.get("sources", []):
                    existing["sources"] = existing.get("sources", []) + [source]
                if not existing.get("context") and ent.get("context"):
                    existing["context"] = ent["context"]

        for rel in extraction.get("relationships", []):
            src = _normalize(rel.get("from", ""))
            tgt = _normalize(rel.get("to", ""))
            relation = rel.get("relation", "related_to")
            if src and tgt:
                if not self.g.has_node(src):
                    self.g.add_node(src, name=rel["from"], type="UNKNOWN", sources=[source])
                if not self.g.has_node(tgt):
                    self.g.add_node(tgt, name=rel["to"], type="UNKNOWN", sources=[source])
                self.g.add_edge(src, tgt, relation=relation, source=source)

        # Invalidate community cache when graph changes
        self._community_summaries = {}
        self._community_map = {}

    def add_fact(self, subject: str, relation: str, obj: str,
                 subject_type: str = "UNKNOWN", obj_type: str = "UNKNOWN",
                 source: str = "interview") -> None:
        """Convenience method to add a single triple."""
        self.add_entities_from_extraction({
            "entities": [
                {"type": subject_type, "name": subject},
                {"type": obj_type, "name": obj},
            ],
            "relationships": [
                {"from": subject, "relation": relation, "to": obj}
            ]
        }, source=source)

    # ── GraphRAG Communities ──────────────────────────────────────────────────

    def compute_communities(self) -> None:
        """
        Detect communities using greedy modularity (NetworkX built-in).
        Stores community ID on each node.
        No external dependencies needed.
        """
        if self.g.number_of_nodes() < 3:
            return
        try:
            import networkx.algorithms.community as nx_comm
            undirected = self.g.to_undirected()
            communities = list(nx_comm.greedy_modularity_communities(undirected))
            self._community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    self._community_map[node] = i
                    if self.g.has_node(node):
                        self.g.nodes[node]["community"] = i
        except Exception as e:
            print(f"[Graph] Community detection failed: {e}")

    def community_summaries(self) -> Dict[int, str]:
        """
        Generate text summaries for each community cluster.
        These are the GraphRAG "global" summaries — used for broad career-arc questions.
        """
        if not self._community_map:
            self.compute_communities()

        summaries: Dict[int, str] = {}
        # Group nodes by community
        by_community: Dict[int, List[str]] = defaultdict(list)
        for node_id, comm_id in self._community_map.items():
            by_community[comm_id].append(node_id)

        for comm_id, nodes in by_community.items():
            if len(nodes) < GRAPH_COMMUNITY_MIN_SIZE:
                continue
            node_data = [self.g.nodes[n] for n in nodes if self.g.has_node(n)]

            companies = [d["name"] for d in node_data if d.get("type") == "COMPANY"]
            skills = [d["name"] for d in node_data if d.get("type") in ("SKILL", "TOOL")]
            projects = [d["name"] for d in node_data if d.get("type") == "PROJECT"]
            degrees = [d["name"] for d in node_data if d.get("type") == "DEGREE"]

            parts = [f"[Community {comm_id}]"]
            if companies:
                parts.append(f"Organizations: {', '.join(companies[:6])}")
            if skills:
                parts.append(f"Skills/Tools: {', '.join(skills[:10])}")
            if projects:
                parts.append(f"Projects: {', '.join(projects[:5])}")
            if degrees:
                parts.append(f"Education: {', '.join(degrees[:3])}")
            parts.append(f"({len(nodes)} connected entities)")

            summaries[comm_id] = " | ".join(parts)

        return summaries

    def get_community_for_entity(self, entity_name: str) -> Optional[int]:
        """Return community ID for the given entity name, or None."""
        node_id = _normalize(entity_name)
        return self._community_map.get(node_id)

    def community_context_for_query(self, query_entities: List[str]) -> str:
        """
        Return community summaries relevant to the query entities.
        This is the GraphRAG global retrieval path.
        """
        if not self._community_map:
            self.compute_communities()

        relevant_communities: Set[int] = set()
        for name in query_entities:
            comm = self.get_community_for_entity(name)
            if comm is not None:
                relevant_communities.add(comm)

        if not relevant_communities:
            return ""

        summaries = self.community_summaries()
        lines = ["[GraphRAG Community Context]"]
        for comm_id in relevant_communities:
            if comm_id in summaries:
                lines.append(f"  {summaries[comm_id]}")
        return "\n".join(lines)

    def global_summary(self) -> str:
        """
        GraphRAG global summary: full overview of person's career arc.
        Includes community cluster descriptions.
        """
        companies = [d["name"] for _, d in self.g.nodes(data=True) if d.get("type") == "COMPANY"]
        skills = [d["name"] for _, d in self.g.nodes(data=True) if d.get("type") in ("SKILL", "TOOL", "FRAMEWORK", "TECHNOLOGY", "DATASET")]
        projects = [d["name"] for _, d in self.g.nodes(data=True) if d.get("type") == "PROJECT"]

        lines = ["[Knowledge Graph Global Summary]"]
        if companies:
            lines.append(f"  Organizations: {', '.join(companies[:10])}")
        if skills:
            lines.append(f"  Skills & Tools: {', '.join(skills[:15])}")
        if projects:
            lines.append(f"  Projects: {', '.join(projects[:8])}")
        lines.append(f"  Total nodes: {self.g.number_of_nodes()}, edges: {self.g.number_of_edges()}")

        # Append community summaries
        summaries = self.community_summaries()
        if summaries:
            lines.append("  Communities detected:")
            for comm_id, summary in list(summaries.items())[:5]:
                lines.append(f"    {summary}")

        return "\n".join(lines)

    # ── Local Querying ────────────────────────────────────────────────────────

    def find_neighbours(self, entity_names: List[str], hops: int = 2) -> List[Dict[str, Any]]:
        """
        Return up to `hops` neighbourhood around any matched entity.
        Used to augment QA context with graph structure.
        """
        start_nodes: Set[str] = set()
        for name in entity_names:
            nid = _normalize(name)
            if self.g.has_node(nid):
                start_nodes.add(nid)
            else:
                for n, data in self.g.nodes(data=True):
                    if name.lower() in data.get("name", "").lower():
                        start_nodes.add(n)

        if not start_nodes:
            return []

        visited: Set[str] = set()
        frontier = start_nodes
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                visited.add(node)
                for nbr in list(self.g.successors(node)) + list(self.g.predecessors(node)):
                    if nbr not in visited:
                        next_frontier.add(nbr)
            frontier = next_frontier

        results = []
        for n in visited:
            data = self.g.nodes[n]
            edges_out = [
                {
                    "to": self.g.nodes[t].get("name", t),
                    "relation": self.g.edges[n, t].get("relation", ""),
                }
                for t in self.g.successors(n)
                if t in visited
            ]
            results.append({
                "name": data.get("name", n),
                "type": data.get("type", "UNKNOWN"),
                "context": data.get("context", ""),
                "sources": data.get("sources", []),
                "edges": edges_out,
                "community": data.get("community"),
            })
        return results

    def neighbours_as_text(self, entity_names: List[str], hops: int = 2) -> str:
        """Return graph neighbourhood as human-readable text for LLM context."""
        nodes = self.find_neighbours(entity_names, hops)
        if not nodes:
            return ""
        lines = ["[Graph Context]"]
        for n in nodes:
            line = f"  {n['type']}: {n['name']}"
            if n.get("context"):
                line += f" — {n['context'][:100]}"
            if n.get("edges"):
                for e in n["edges"][:3]:
                    line += f"\n    → {e['relation']} → {e['to']}"
            lines.append(line)
        return "\n".join(lines)

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Full-text search over node names and contexts."""
        q = query.lower()
        results = []
        for n, data in self.g.nodes(data=True):
            if entity_type and data.get("type") != entity_type:
                continue
            name_match = q in data.get("name", "").lower()
            ctx_match = q in data.get("context", "").lower()
            if name_match or ctx_match:
                results.append({
                    "name": data.get("name", n),
                    "type": data.get("type", "UNKNOWN"),
                    "context": data.get("context", ""),
                    "sources": data.get("sources", []),
                    "score": 1.0 if name_match else 0.6,
                    "community": data.get("community"),
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        data = nx.node_link_data(self.g)
        Path(GRAPH_DB_PATH).write_text(json.dumps(data, indent=2))

    def load(self) -> bool:
        p = Path(GRAPH_DB_PATH)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        self.g = nx.node_link_graph(data)
        # Rebuild community map from node attributes
        for node, d in self.g.nodes(data=True):
            if "community" in d:
                self._community_map[node] = d["community"]
        return True

    # ── Export for frontend ───────────────────────────────────────────────────

    def to_frontend_json(self) -> Dict[str, Any]:
        """Export graph as {nodes, edges} for vis.js rendering."""
        if not self._community_map:
            self.compute_communities()

        nodes = []
        for n, data in self.g.nodes(data=True):
            nodes.append({
                "id": n,
                "label": data.get("name", n),
                "type": data.get("type", "UNKNOWN"),
                "context": data.get("context", "")[:120],
                "community": data.get("community"),
                "sources": data.get("sources", []),
            })
        edges = []
        for u, v, data in self.g.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "relation": data.get("relation", ""),
            })
        return {"nodes": nodes, "edges": edges}

    @property
    def stats(self) -> Dict[str, Any]:
        type_counts: Dict[str, int] = defaultdict(int)
        for _, d in self.g.nodes(data=True):
            type_counts[d.get("type", "UNKNOWN")] += 1
        comm_summaries = self.community_summaries()
        return {
            "total_nodes": self.g.number_of_nodes(),
            "total_edges": self.g.number_of_edges(),
            "communities": len(comm_summaries),
            **dict(type_counts),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    """Normalize entity name to a stable node ID."""
    if not name:
        return ""
    return re.sub(r"\s+", "_", name.strip().lower())


# Module-level singleton
_graph = KnowledgeGraph()


def get_graph() -> KnowledgeGraph:
    return _graph


def load_graph_on_startup() -> None:
    loaded = _graph.load()
    if loaded:
        print(f"[Graph] Loaded {_graph.stats['total_nodes']} nodes, "
              f"{_graph.stats.get('communities', 0)} communities from disk.")
    else:
        print("[Graph] Starting with empty graph.")
