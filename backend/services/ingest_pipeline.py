"""
arXiv Auto-Ingest Pipeline
Searches arXiv for papers matching a query, fetches their abstract/full text,
embeds them, and stores in ChromaDB + DynamoDB.

This is the main entry point for populating the vector database with research papers.
"""
import asyncio
import time
from typing import List, Optional

import httpx
from loguru import logger

from backend.config import Settings
from backend.models.schemas import (
    PaperSearchRequest, PaperMetadata, PaperSource,
    IngestRequest, IngestResponse,
)
from backend.services.paper_discovery import PaperDiscoveryService
from backend.services.pdf_ingestion import PDFIngestionService
from backend.services.mlflow_service import MLflowService


class IngestPipeline:
    """
    Orchestrates: search → fetch text → chunk → embed → store
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._discovery = PaperDiscoveryService(settings)
        self._ingestion = PDFIngestionService(settings)
        self._mlflow = MLflowService(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    # ── Main pipeline ──────────────────────────────────────────────────────

    async def search_and_ingest(
        self,
        query: str,
        max_papers: int = 5,
        source: PaperSource = PaperSource.ARXIV,
        year_from: Optional[int] = None,
    ) -> List[IngestResponse]:
        """
        Full pipeline: search → fetch full text → ingest.
        Returns a list of IngestResponse for each paper processed.
        """
        logger.info(f"[IngestPipeline] search_and_ingest: query='{query}' max={max_papers}")
        t0 = time.perf_counter()

        # 1. Search for papers
        search_req = PaperSearchRequest(
            query=query,
            max_results=max_papers,
            source=source,
            year_from=year_from,
        )
        search_result = await self._discovery.search(search_req)
        logger.info(f"[IngestPipeline] found {len(search_result.papers)} papers")

        # 2. Ingest each paper concurrently (abstract + full text if available)
        tasks = [self._ingest_paper(paper) for paper in search_result.papers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        success = 0
        for paper, resp in zip(search_result.papers, responses):
            if isinstance(resp, Exception):
                logger.warning(f"[IngestPipeline] failed to ingest {paper.paper_id}: {resp}")
                results.append(IngestResponse(
                    paper_id=paper.paper_id,
                    status="error",
                    chunks_created=0,
                    embeddings_stored=0,
                    message=str(resp),
                ))
            else:
                results.append(resp)
                if resp.status == "success":
                    success += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"[IngestPipeline] done: {success}/{len(results)} papers ingested in {elapsed_ms:.1f}ms")
        return results

    async def ingest_paper_by_id(self, arxiv_id: str) -> IngestResponse:
        """Ingest a single arXiv paper by its ID (e.g. '2310.06825')."""
        import arxiv
        loop = asyncio.get_event_loop()

        def _fetch():
            client = arxiv.Client()
            search = arxiv.Search(id_list=[arxiv_id])
            return next(client.results(search), None)

        result = await loop.run_in_executor(None, _fetch)
        if result is None:
            return IngestResponse(
                paper_id=f"arxiv_{arxiv_id}",
                status="error",
                chunks_created=0,
                embeddings_stored=0,
                message=f"arXiv paper {arxiv_id} not found.",
            )

        paper = PaperMetadata(
            paper_id=f"arxiv_{arxiv_id}",
            title=result.title,
            authors=[a.name for a in result.authors],
            abstract=result.summary,
            published_date=result.published.strftime("%Y-%m-%d") if result.published else None,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            source="arxiv",
            categories=result.categories,
        )
        return await self._ingest_paper(paper)

    # ── Internal ───────────────────────────────────────────────────────────

    async def _ingest_paper(self, paper: PaperMetadata) -> IngestResponse:
        """
        Ingests a single paper. Uses abstract as text if PDF fetch fails.
        """
        # Try to get full text from PDF URL
        full_text = None
        if paper.pdf_url:
            full_text = await self._try_fetch_pdf_text(paper.pdf_url, paper.paper_id)

        # Fall back to abstract
        if not full_text:
            logger.debug(f"[IngestPipeline] using abstract for {paper.paper_id}")
            full_text = self._build_abstract_doc(paper)

        req = IngestRequest(
            paper_id=paper.paper_id,
            title=paper.title,
            content=full_text,
        )
        return await self._ingestion.ingest_text(req)

    async def _try_fetch_pdf_text(self, pdf_url: str, paper_id: str) -> Optional[str]:
        """Download PDF and extract text. Returns None on any failure."""
        import tempfile, os
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(pdf_url)
                resp.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            try:
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None, self._ingestion._extract_text_from_pdf, tmp_path
                )
                return text if text.strip() else None
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.debug(f"[IngestPipeline] PDF fetch failed for {paper_id}: {e}")
            return None

    @staticmethod
    def _build_abstract_doc(paper: PaperMetadata) -> str:
        """Build a rich text document from paper metadata when PDF is unavailable."""
        authors = ", ".join(paper.authors[:5])
        if len(paper.authors) > 5:
            authors += f" et al."
        return (
            f"Title: {paper.title}\n\n"
            f"Authors: {authors}\n\n"
            f"Published: {paper.published_date or 'N/A'}\n\n"
            f"Categories: {', '.join(paper.categories)}\n\n"
            f"Abstract:\n{paper.abstract}\n"
        )
