"""Market correlation graph built from semantic similarity and price dynamics.

Constructs a weighted graph where:
- Nodes = prediction markets
- Edges = semantic/price correlations between markets
- Edge weights = strength of correlation (KL/JS divergence or cosine similarity)

This enables:
1. Cross-market signal propagation (Fish agent communication)
2. Arbitrage detection (negation pairs, correlated event mispricings)
3. Network centrality analysis (systemically important markets)
4. Contagion modeling (how does an event in one market ripple?)

References:
- Baaijens et al. (2025): Semantic similarity > price correlation for GNNs
- Saguillo et al. (2025): $40M arbitrage via semantic matching on Polymarket
- Fanshawe et al. (2026): THGNN for correlation forecasting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine, jensenshannon
from scipy.stats import entropy


@dataclass
class MarketNode:
    """A node in the market graph."""

    market_id: str
    question: str
    category: str = ""
    tags: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    current_price: float = 0.5
    volume: float = 0.0


@dataclass
class MarketEdge:
    """An edge (correlation) between two markets."""

    source_id: str
    target_id: str
    weight: float  # Correlation strength (0 to 1)
    edge_type: str  # "semantic" | "price" | "negation" | "causal"
    metadata: dict[str, Any] = field(default_factory=dict)


class MarketGraph:
    """Weighted graph of prediction market correlations.

    Usage:
        graph = MarketGraph()

        # Add markets
        graph.add_market(MarketNode(id="m1", question="Will X?", embedding=emb1))
        graph.add_market(MarketNode(id="m2", question="Will Y?", embedding=emb2))

        # Build edges from semantic similarity
        graph.build_semantic_edges(threshold=0.7)

        # Detect negation pairs (arbitrage opportunities)
        pairs = graph.detect_negation_pairs()

        # Get neighbors of a market (for cross-market signals)
        neighbors = graph.get_neighbors("m1")

        # Compute centrality (systemically important markets)
        centrality = graph.compute_centrality()

        # Export for visualization
        fig = graph.to_plotly()
    """

    def __init__(self) -> None:
        self._graph = nx.Graph()
        self._markets: dict[str, MarketNode] = {}

    def add_market(self, market: MarketNode) -> None:
        """Add a market as a node in the graph."""
        self._markets[market.market_id] = market
        self._graph.add_node(
            market.market_id,
            question=market.question,
            category=market.category,
            price=market.current_price,
            volume=market.volume,
        )

    def remove_market(self, market_id: str) -> None:
        """Remove a market and all its edges."""
        self._markets.pop(market_id, None)
        if self._graph.has_node(market_id):
            self._graph.remove_node(market_id)

    def build_semantic_edges(self, threshold: float = 0.7) -> int:
        """Build edges based on embedding cosine similarity.

        Markets whose description embeddings have cosine similarity above
        the threshold get linked. This captures topical relatedness.
        """
        markets_with_embeddings = [
            m for m in self._markets.values() if m.embedding is not None
        ]
        edge_count = 0

        for i, m1 in enumerate(markets_with_embeddings):
            for m2 in markets_with_embeddings[i + 1:]:
                sim = 1 - cosine(m1.embedding, m2.embedding)
                if sim >= threshold:
                    self._add_edge(
                        m1.market_id, m2.market_id,
                        weight=sim,
                        edge_type="semantic",
                    )
                    edge_count += 1

        logger.info(f"Built {edge_count} semantic edges (threshold={threshold})")
        return edge_count

    def build_price_correlation_edges(
        self,
        price_history: dict[str, list[float]],
        threshold: float = 0.5,
    ) -> int:
        """Build edges based on historical price correlation.

        Args:
            price_history: market_id → list of historical prices (time series)
            threshold: Minimum absolute Pearson correlation to create an edge
        """
        market_ids = [mid for mid in price_history if mid in self._markets]
        edge_count = 0

        for i, m1 in enumerate(market_ids):
            for m2 in market_ids[i + 1:]:
                p1 = np.array(price_history[m1])
                p2 = np.array(price_history[m2])

                # Align lengths
                min_len = min(len(p1), len(p2))
                if min_len < 10:
                    continue

                corr = np.corrcoef(p1[:min_len], p2[:min_len])[0, 1]
                if abs(corr) >= threshold:
                    self._add_edge(
                        m1, m2,
                        weight=abs(corr),
                        edge_type="price",
                        metadata={"correlation": float(corr)},
                    )
                    edge_count += 1

        logger.info(f"Built {edge_count} price correlation edges (threshold={threshold})")
        return edge_count

    def detect_negation_pairs(self) -> list[tuple[str, str, float]]:
        """Detect negation pairs — markets that should be anti-correlated.

        These represent intra-market arbitrage opportunities:
        If market A = "Will X happen?" and market B = "Will X NOT happen?",
        then P(A) + P(B) should ≈ 1.0. Deviations are exploitable.
        """
        pairs = []
        market_list = list(self._markets.values())

        for i, m1 in enumerate(market_list):
            for m2 in market_list[i + 1:]:
                # Check if questions are negations of each other
                if self._is_negation_pair(m1.question, m2.question):
                    sum_prices = m1.current_price + m2.current_price
                    deviation = abs(sum_prices - 1.0)
                    if deviation > 0.02:  # More than 2% deviation
                        pairs.append((m1.market_id, m2.market_id, deviation))
                        self._add_edge(
                            m1.market_id, m2.market_id,
                            weight=deviation,
                            edge_type="negation",
                            metadata={"sum_prices": sum_prices, "deviation": deviation},
                        )

        logger.info(f"Detected {len(pairs)} negation pairs with >2% deviation")
        return pairs

    def compute_divergence_matrix(self) -> dict[tuple[str, str], float]:
        """Compute Jensen-Shannon divergence between all market pairs.

        Used to detect cross-market inefficiencies (per PolySwarm methodology).
        """
        divergences = {}
        market_ids = list(self._markets.keys())

        for i, m1_id in enumerate(market_ids):
            m1 = self._markets[m1_id]
            p1 = np.array([m1.current_price, 1 - m1.current_price])

            for m2_id in market_ids[i + 1:]:
                m2 = self._markets[m2_id]
                p2 = np.array([m2.current_price, 1 - m2.current_price])

                js_div = float(jensenshannon(p1, p2))
                divergences[(m1_id, m2_id)] = js_div

        return divergences

    def get_neighbors(self, market_id: str, max_hops: int = 1) -> list[str]:
        """Get neighboring markets within max_hops in the graph."""
        if not self._graph.has_node(market_id):
            return []

        if max_hops == 1:
            return list(self._graph.neighbors(market_id))

        # BFS for multi-hop
        visited = set()
        queue = [(market_id, 0)]
        neighbors = []

        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_hops:
                continue
            visited.add(node)
            if node != market_id:
                neighbors.append(node)
            if depth < max_hops:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return neighbors

    def compute_centrality(self, metric: str = "betweenness") -> dict[str, float]:
        """Compute centrality scores for all markets.

        High-centrality markets are "systemically important" — events affecting
        them likely ripple through many other markets.
        """
        if self._graph.number_of_nodes() == 0:
            return {}

        if metric == "betweenness":
            return dict(nx.betweenness_centrality(self._graph, weight="weight"))
        elif metric == "eigenvector":
            try:
                return dict(nx.eigenvector_centrality(self._graph, weight="weight"))
            except nx.PowerIterationFailedConvergence:
                return dict(nx.degree_centrality(self._graph))
        elif metric == "pagerank":
            return dict(nx.pagerank(self._graph, weight="weight"))
        elif metric == "degree":
            return dict(nx.degree_centrality(self._graph))
        else:
            raise ValueError(f"Unknown centrality metric: {metric}")

    def get_communities(self) -> list[set[str]]:
        """Detect market communities using Louvain modularity optimization."""
        if self._graph.number_of_nodes() < 2:
            return [set(self._graph.nodes())]

        try:
            from networkx.algorithms.community import louvain_communities
            return [set(c) for c in louvain_communities(self._graph, weight="weight")]
        except ImportError:
            # Fallback to connected components
            return [set(c) for c in nx.connected_components(self._graph)]

    def to_adjacency_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Export graph as adjacency matrix for GNN input."""
        nodes = sorted(self._graph.nodes())
        adj = nx.to_numpy_array(self._graph, nodelist=nodes, weight="weight")
        return adj, nodes

    def _add_edge(
        self,
        source: str,
        target: str,
        weight: float,
        edge_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._graph.add_edge(
            source, target,
            weight=weight,
            edge_type=edge_type,
            **(metadata or {}),
        )

    def _is_negation_pair(self, q1: str, q2: str) -> bool:
        """Heuristic: detect if two questions are negations of each other."""
        q1_lower = q1.lower().strip("?. ")
        q2_lower = q2.lower().strip("?. ")

        negation_patterns = [
            ("will ", "will not "), ("will ", "won't "),
            ("does ", "does not "), ("does ", "doesn't "),
            ("is ", "is not "), ("is ", "isn't "),
            ("can ", "cannot "), ("can ", "can't "),
        ]
        for pos, neg in negation_patterns:
            if q1_lower.startswith(pos) and q2_lower.startswith(neg):
                rest1 = q1_lower[len(pos):]
                rest2 = q2_lower[len(neg):]
                if rest1 == rest2:
                    return True
            if q2_lower.startswith(pos) and q1_lower.startswith(neg):
                rest1 = q2_lower[len(pos):]
                rest2 = q1_lower[len(neg):]
                if rest1 == rest2:
                    return True

        return False

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def density(self) -> float:
        return nx.density(self._graph) if self.num_nodes > 1 else 0.0
