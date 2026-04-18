"""
Paper Discovery Service
Searches arXiv and Semantic Scholar APIs for research papers.
Both APIs are free and require no authentication (Semantic Scholar key is optional).
"""
import asyncio
import hashlib
from typing import List, Optional
from datetime import datetime

import arxiv
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import Settings
from backend.models.schemas import (
    PaperSearchRequest, PaperSearchResponse,
    PaperMetadata, PaperSource,
)


class PaperDiscoveryService:
    SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
    SS_FIELDS = "paperId,title,authors,abstract,year,externalIds,openAccessPdf,citationCount,publicationTypes"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._headers = {}
        if settings.semantic_scholar_api_key:
            self._headers["x-api-key"] = settings.semantic_scholar_api_key

    # ── Public ─────────────────────────────────────────────────────────────

    async def search(self, request: PaperSearchRequest) -> PaperSearchResponse:
        tasks = []

        if request.source in (PaperSource.ARXIV, PaperSource.BOTH):
            tasks.append(self._search_arxiv(request))

        if request.source in (PaperSource.SEMANTIC_SCHOLAR, PaperSource.BOTH):
            tasks.append(self._search_semantic_scholar(request))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        papers: List[PaperMetadata] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[PaperDiscovery] source error: {r}")
                continue
            papers.extend(r)

        # Deduplicate by title similarity (simple exact-title dedup)
        seen_titles = set()
        unique_papers = []
        for p in papers:
            key = p.title.lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_papers.append(p)

        unique_papers = unique_papers[: request.max_results]

        return PaperSearchResponse(
            query=request.query,
            total_found=len(unique_papers),
            papers=unique_papers,
        )

    # ── arXiv ──────────────────────────────────────────────────────────────

    async def _search_arxiv(self, request: PaperSearchRequest) -> List[PaperMetadata]:
        logger.debug(f"[arXiv] searching: '{request.query}'")
        loop = asyncio.get_event_loop()

        def _sync_search():
            client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=3)
            search = arxiv.Search(
                query=request.query,
                max_results=request.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in client.results(search):
                # Filter by year if requested
                pub_year = result.published.year if result.published else None
                if request.year_from and pub_year and pub_year < request.year_from:
                    continue
                if request.year_to and pub_year and pub_year > request.year_to:
                    continue

                paper_id = f"arxiv_{result.entry_id.split('/')[-1]}"
                papers.append(
                    PaperMetadata(
                        paper_id=paper_id,
                        title=result.title,
                        authors=[a.name for a in result.authors],
                        abstract=result.summary,
                        published_date=result.published.strftime("%Y-%m-%d") if result.published else None,
                        url=result.entry_id,
                        pdf_url=result.pdf_url,
                        source="arxiv",
                        categories=result.categories,
                    )
                )
            return papers

        try:
            papers = await loop.run_in_executor(None, _sync_search)
            logger.info(f"[arXiv] found {len(papers)} papers")
            return papers
        except Exception as e:
            logger.error(f"[arXiv] search failed: {e}")
            return []

    # ── Semantic Scholar ───────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _search_semantic_scholar(self, request: PaperSearchRequest) -> List[PaperMetadata]:
        logger.debug(f"[SemanticScholar] searching: '{request.query}'")

        params = {
            "query": request.query,
            "limit": request.max_results,
            "fields": self.SS_FIELDS,
        }
        if request.year_from and request.year_to:
            params["year"] = f"{request.year_from}-{request.year_to}"
        elif request.year_from:
            params["year"] = f"{request.year_from}-"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_BASE}/paper/search",
                    params=params,
                    headers=self._headers,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"[SemanticScholar] HTTP {e.response.status_code}: {e.response.text}")
                return []
            except Exception as e:
                logger.error(f"[SemanticScholar] request failed: {e}")
                return []

        papers = []
        for item in data.get("data", []):
            pdf_url = None
            if item.get("openAccessPdf"):
                pdf_url = item["openAccessPdf"].get("url")

            ss_id = item.get("paperId", "")
            paper_id = f"ss_{ss_id}"

            papers.append(
                PaperMetadata(
                    paper_id=paper_id,
                    title=item.get("title", "Untitled"),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    abstract=item.get("abstract") or "",
                    published_date=str(item["year"]) if item.get("year") else None,
                    url=f"https://www.semanticscholar.org/paper/{ss_id}",
                    pdf_url=pdf_url,
                    source="semantic_scholar",
                    categories=[],
                    citation_count=item.get("citationCount"),
                )
            )

        logger.info(f"[SemanticScholar] found {len(papers)} papers")
        return papers

    async def search_by_author(self, author_name: str, max_results: int = 10) -> dict:
        """Search Semantic Scholar for an author and return their papers."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: find author ID
            try:
                r = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_BASE}/author/search",
                    params={"query": author_name, "limit": 5,
                            "fields": "authorId,name,paperCount,citationCount,hIndex"},
                    headers=self._headers,
                )
                r.raise_for_status()
                authors = r.json().get("data", [])
            except Exception as e:
                logger.error(f"[AuthorSearch] author lookup failed: {e}")
                return {"author": None, "papers": []}

            if not authors:
                return {"author": None, "papers": []}

            top = authors[0]
            author_id = top["authorId"]

            # Step 2: fetch their papers
            try:
                r2 = await client.get(
                    f"{self.SEMANTIC_SCHOLAR_BASE}/author/{author_id}/papers",
                    params={"limit": max_results,
                            "fields": "title,year,abstract,citationCount,openAccessPdf,externalIds"},
                    headers=self._headers,
                )
                r2.raise_for_status()
                papers_raw = r2.json().get("data", [])
            except Exception as e:
                logger.error(f"[AuthorSearch] papers fetch failed: {e}")
                papers_raw = []

        papers = []
        for p in papers_raw:
            arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
            papers.append({
                "title": p.get("title", "Untitled"),
                "year": p.get("year"),
                "citations": p.get("citationCount", 0),
                "abstract": (p.get("abstract") or "")[:300],
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "pdf_url": (p.get("openAccessPdf") or {}).get("url", ""),
            })

        return {
            "author": {
                "name": top.get("name"),
                "paper_count": top.get("paperCount", 0),
                "citation_count": top.get("citationCount", 0),
                "h_index": top.get("hIndex", 0),
            },
            "papers": sorted(papers, key=lambda x: x["citations"], reverse=True),
        }

    # ── Utility ────────────────────────────────────────────────────────────

    @staticmethod
    def make_paper_id(title: str, source: str) -> str:
        """Generate a stable paper ID from title + source."""
        raw = f"{source}_{title.lower().strip()}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
