"""Real-time news retrieval for Fish context enrichment.

Scrapes top news articles relevant to each market question using
trafilatura, embeds them with sentence-transformers, and injects
the most relevant summaries into Fish prompts so they can reason
about events beyond the LLM training cutoff.

This addresses the #1 failure mode from retrodiction: 5 catastrophic
misses on surprise events the LLM didn't know about.

Usage:
    news = NewsContext()
    context = await news.get_context("Will Iran ceasefire happen?")
    # context.summary = "3 relevant articles found..."
    # Injected into Fish prompts as RECENT_NEWS_CONTEXT block
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


@dataclass
class NewsArticle:
    """A retrieved news article."""
    url: str
    title: str
    text: str
    relevance_score: float = 0.0


@dataclass
class NewsContextResult:
    """Aggregated news context for a market question."""
    question: str
    articles: list[NewsArticle]
    summary: str  # formatted for Fish prompt injection
    n_articles: int
    elapsed_s: float
    source: str = "web_search"

    def to_prompt_block(self) -> str:
        """Format as a prompt block for Fish injection."""
        if not self.articles:
            return ""
        return f"""RECENT NEWS CONTEXT (retrieved {self.n_articles} articles):
{self.summary}

NOTE: This news may contain information beyond the LLM training data.
Consider it as additional evidence when forming your estimate."""


class NewsContext:
    """Retrieves and ranks news articles relevant to market questions.

    Pipeline:
    1. Search for articles related to the question (via web search or RSS)
    2. Extract text with trafilatura
    3. Embed question + articles with sentence-transformers
    4. Rank by cosine similarity
    5. Return top-K as formatted context for Fish prompts
    """

    def __init__(
        self,
        top_k: int = 3,
        min_similarity: float = 0.3,
        max_article_length: int = 500,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.max_article_length = max_article_length
        self._embedder = None
        self._embedding_model = embedding_model
        self._cache: dict[str, NewsContextResult] = {}

    def _get_embedder(self):
        if self._embedder is None and HAS_SBERT:
            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    async def get_context(
        self,
        question: str,
        description: str = "",
    ) -> NewsContextResult:
        """Get news context for a market question.

        Uses web search to find articles, extracts text with trafilatura,
        ranks by semantic similarity to the question.
        """
        t0 = time.monotonic()

        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()[:12]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Search for relevant URLs
        urls = await self._search_urls(question)

        if not urls:
            result = NewsContextResult(
                question=question, articles=[], summary="No relevant news found.",
                n_articles=0, elapsed_s=round(time.monotonic() - t0, 1),
            )
            self._cache[cache_key] = result
            return result

        # Extract articles
        articles = await self._extract_articles(urls)

        if not articles:
            result = NewsContextResult(
                question=question, articles=[], summary="Article extraction failed.",
                n_articles=0, elapsed_s=round(time.monotonic() - t0, 1),
            )
            self._cache[cache_key] = result
            return result

        # Rank by relevance
        ranked = self._rank_by_relevance(question, articles)
        top = ranked[:self.top_k]

        # Build summary
        summary_parts = []
        for i, article in enumerate(top, 1):
            truncated = article.text[:self.max_article_length]
            summary_parts.append(
                f"[{i}] {article.title}\n"
                f"    Relevance: {article.relevance_score:.0%}\n"
                f"    {truncated}"
            )

        summary = "\n\n".join(summary_parts)

        result = NewsContextResult(
            question=question,
            articles=top,
            summary=summary,
            n_articles=len(top),
            elapsed_s=round(time.monotonic() - t0, 1),
        )
        self._cache[cache_key] = result

        logger.info(
            f"News context: {len(top)} articles for '{question[:40]}...' "
            f"({result.elapsed_s:.1f}s)"
        )
        return result

    async def _search_urls(self, question: str, max_urls: int = 10) -> list[str]:
        """Search for relevant news URLs.

        Uses trafilatura's feed discovery and Google News RSS as sources.
        Falls back to constructing search URLs from question keywords.
        """
        urls = []

        # Strategy 1: Google News RSS search
        import urllib.parse
        query = urllib.parse.quote_plus(question[:100])
        google_news_rss = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

        try:
            if HAS_TRAFILATURA:
                downloaded = trafilatura.fetch_url(google_news_rss)
                if downloaded:
                    # Parse RSS for links
                    import re
                    links = re.findall(r'<link>(https?://[^<]+)</link>', downloaded)
                    urls.extend(links[:max_urls])
        except Exception as e:
            logger.debug(f"Google News RSS failed: {e}")

        # Strategy 2: Construct search URLs from keywords
        if not urls:
            keywords = [w for w in question.split() if len(w) > 3][:5]
            search_query = "+".join(keywords)
            # Use DuckDuckGo lite (no JS required)
            urls.append(f"https://lite.duckduckgo.com/lite/?q={search_query}")

        return urls[:max_urls]

    async def _extract_articles(self, urls: list[str]) -> list[NewsArticle]:
        """Extract article text from URLs using trafilatura."""
        if not HAS_TRAFILATURA:
            return []

        articles = []
        for url in urls[:10]:  # cap at 10
            try:
                downloaded = trafilatura.fetch_url(url)
                if not downloaded:
                    continue

                result = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    favor_precision=True,
                )
                if not result or len(result) < 50:
                    continue

                # Try to get title
                metadata = trafilatura.extract(
                    downloaded,
                    output_format="json",
                    include_comments=False,
                )
                title = ""
                if metadata:
                    import json
                    try:
                        meta = json.loads(metadata)
                        title = meta.get("title", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

                articles.append(NewsArticle(
                    url=url,
                    title=title or url[:60],
                    text=result[:2000],
                ))

            except Exception as e:
                logger.debug(f"Extraction failed for {url[:50]}: {e}")
                continue

        return articles

    def _rank_by_relevance(
        self, question: str, articles: list[NewsArticle],
    ) -> list[NewsArticle]:
        """Rank articles by semantic similarity to the question."""
        embedder = self._get_embedder()
        if embedder is None or not articles:
            # No embedder — return in original order with default scores
            for a in articles:
                a.relevance_score = 0.5
            return articles

        # Embed question and articles
        q_emb = embedder.encode([question], normalize_embeddings=True)
        a_texts = [f"{a.title}. {a.text[:300]}" for a in articles]
        a_embs = embedder.encode(a_texts, normalize_embeddings=True)

        # Cosine similarity
        similarities = (a_embs @ q_emb.T).flatten()

        for article, sim in zip(articles, similarities):
            article.relevance_score = round(float(sim), 4)

        # Sort by relevance, filter by threshold
        ranked = sorted(articles, key=lambda a: -a.relevance_score)
        return [a for a in ranked if a.relevance_score >= self.min_similarity]
