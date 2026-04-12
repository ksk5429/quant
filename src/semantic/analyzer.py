"""Semantic market analyzer — embedding-based similarity and LLM analysis.

Uses sentence transformers for market question embeddings and
LLMs for deeper semantic analysis of market relationships.

Architecture:
1. Embed market questions using sentence-transformers
2. Compute cosine similarity matrix for all market pairs
3. Build semantic correlation graph from high-similarity pairs
4. Use LLM for nuanced relationship analysis (causal, temporal, conditional)

References:
- Baaijens et al. (2025): Cosine-similarity-based graphs outperform price-correlation
- Wang & Wei (2025): Event-aware semantic factors with IC > 0.05
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from scipy.spatial.distance import cdist


class SemanticAnalyzer:
    """Analyzes semantic relationships between prediction markets.

    Usage:
        analyzer = SemanticAnalyzer()

        # Embed market questions
        embeddings = analyzer.embed_texts([
            "Will Bitcoin exceed $100k by Dec 2026?",
            "Will Ethereum price surpass $10k by 2026?",
            "Will the Fed cut rates in 2026?",
        ])

        # Compute similarity matrix
        sim_matrix = analyzer.similarity_matrix(embeddings)

        # Find related markets
        related = analyzer.find_related(
            query_embedding=embeddings[0],
            corpus_embeddings=embeddings,
            top_k=5,
        )
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._device = device
        logger.info(f"SemanticAnalyzer initialized: model={model_name}")

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
            logger.info(f"Loaded sentence transformer: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using random embeddings for testing."
            )
            self._model = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into dense vectors.

        Returns: np.ndarray of shape (n_texts, embedding_dim)
        """
        self._load_model()

        if self._model is not None:
            embeddings = self._model.encode(texts, show_progress_bar=False)
            return np.array(embeddings)
        else:
            # Fallback: deterministic pseudo-embeddings for testing
            rng = np.random.RandomState(42)
            return rng.randn(len(texts), 384).astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed_texts([text])[0]

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity matrix.

        Returns: np.ndarray of shape (n, n), values in [-1, 1]
        """
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        return normalized @ normalized.T

    def find_related(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        corpus_ids: list[str] | None = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Find the most semantically similar items to a query.

        Returns list of {id, similarity} dicts, sorted by similarity descending.
        """
        # Compute cosine distances
        distances = cdist(
            query_embedding.reshape(1, -1),
            corpus_embeddings,
            metric="cosine",
        )[0]

        # Convert distance to similarity
        similarities = 1 - distances

        # Sort by similarity, descending
        indices = np.argsort(-similarities)

        results = []
        for idx in indices[:top_k + 1]:  # +1 because query might be in corpus
            sim = float(similarities[idx])
            if sim < threshold:
                break
            if sim > 0.999:  # Skip self-match
                continue

            entry: dict[str, Any] = {"index": int(idx), "similarity": round(sim, 4)}
            if corpus_ids:
                entry["id"] = corpus_ids[idx]
            results.append(entry)

        return results[:top_k]

    def cluster_markets(
        self,
        embeddings: np.ndarray,
        n_clusters: int | None = None,
    ) -> np.ndarray:
        """Cluster market embeddings into topic groups.

        If n_clusters is None, uses silhouette score to find optimal k.
        Returns cluster labels array.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = len(embeddings)
        if n_samples < 3:
            return np.zeros(n_samples, dtype=int)

        if n_clusters is None:
            # Find optimal k via silhouette score
            best_k = 2
            best_score = -1

            for k in range(2, min(n_samples, 15)):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

            n_clusters = best_k

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        logger.info(f"Clustered {n_samples} markets into {n_clusters} groups")
        return labels
