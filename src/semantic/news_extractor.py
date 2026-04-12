"""News extraction and processing module.

Uses trafilatura for high-accuracy article extraction (0.958 F1)
and sentence-transformers for semantic embedding and similarity.

Pipeline:
  URL/RSS → trafilatura extract → clean text → embed → similarity graph

This feeds the Researcher Fish with real-time news context.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import trafilatura
    from trafilatura.feeds import find_feed_urls
    from trafilatura.sitemaps import sitemap_search
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    logger.warning("trafilatura not installed. pip install trafilatura")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    logger.warning("sentence-transformers not installed. pip install sentence-transformers")


@dataclass
class Article:
    """Extracted and processed news article."""
    url: str
    title: str
    text: str
    date: str = ""
    author: str = ""
    source: str = ""
    word_count: int = 0
    content_hash: str = ""
    extracted_at: str = ""

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.text.split())
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        if not self.extracted_at:
            self.extracted_at = datetime.now().isoformat()


class NewsExtractor:
    """Extract and process news articles for market context.

    Uses trafilatura for best-in-class article extraction.
    Deduplicates by content hash. Caches extracted articles.
    """

    def __init__(self, cache_dir: str | Path = "data/news_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._seen_hashes: set[str] = set()

    def extract_url(self, url: str) -> Article | None:
        """Extract article text from a URL."""
        if not HAS_TRAFILATURA:
            logger.error("trafilatura required for news extraction")
            return None

        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None

            result = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
                output_format="json",
            )

            if not result:
                return None

            data = json.loads(result)
            text = data.get("text", "")
            if not text or len(text) < 50:
                return None

            article = Article(
                url=url,
                title=data.get("title", ""),
                text=text,
                date=data.get("date", ""),
                author=data.get("author", ""),
                source=data.get("source", ""),
            )

            # Dedup
            if article.content_hash in self._seen_hashes:
                return None
            self._seen_hashes.add(article.content_hash)

            return article

        except Exception as e:
            logger.warning(f"Failed to extract {url}: {e}")
            return None

    def extract_batch(self, urls: list[str], max_articles: int = 20) -> list[Article]:
        """Extract articles from multiple URLs. Deduplicates."""
        articles = []
        for url in urls:
            if len(articles) >= max_articles:
                break
            article = self.extract_url(url)
            if article:
                articles.append(article)
        logger.info(f"Extracted {len(articles)}/{len(urls)} articles")
        return articles

    def discover_feeds(self, base_url: str) -> list[str]:
        """Discover RSS/Atom feed URLs from a website."""
        if not HAS_TRAFILATURA:
            return []
        try:
            feeds = find_feed_urls(base_url)
            return list(feeds) if feeds else []
        except Exception as e:
            logger.warning(f"Feed discovery failed for {base_url}: {e}")
            return []

    def save_articles(self, articles: list[Article], filename: str = "articles.json") -> Path:
        """Save extracted articles to cache."""
        path = self.cache_dir / filename
        data = [
            {
                "url": a.url,
                "title": a.title,
                "text": a.text[:5000],
                "date": a.date,
                "source": a.source,
                "word_count": a.word_count,
                "content_hash": a.content_hash,
                "extracted_at": a.extracted_at,
            }
            for a in articles
        ]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


class SemanticMatcher:
    """Semantic similarity matching using sentence-transformers.

    Embeds articles and market questions, then computes cosine similarity
    to find the most relevant news for each market.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if not HAS_SBERT:
            raise ImportError("sentence-transformers required. pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}
        logger.info(f"SemanticMatcher initialized: {model_name}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)

    def find_relevant_articles(
        self,
        question: str,
        articles: list[Article],
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[Article, float]]:
        """Find the most relevant articles for a market question.

        Returns list of (article, similarity_score) tuples, sorted
        by relevance. Uses cosine similarity on sentence embeddings.
        """
        if not articles:
            return []

        # Embed question and article texts
        q_emb = self.embed([question])
        a_texts = [f"{a.title}. {a.text[:500]}" for a in articles]
        a_embs = self.embed(a_texts)

        # Cosine similarity (embeddings are normalized)
        similarities = (a_embs @ q_emb.T).flatten()

        # Sort and filter
        ranked = sorted(
            zip(articles, similarities),
            key=lambda x: -x[1],
        )
        filtered = [(a, float(s)) for a, s in ranked if s >= min_similarity]

        return filtered[:top_k]

    def compute_market_similarity(
        self,
        questions: list[str],
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between market questions.

        Returns NxN similarity matrix. Used for cross-market
        correlation detection and arbitrage identification.
        """
        embeddings = self.embed(questions)
        similarity_matrix = embeddings @ embeddings.T
        return similarity_matrix
