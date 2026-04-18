"""
CLI script to search arXiv and ingest papers into the vector database.

Usage:
    python scripts/ingest_arxiv.py --query "transformer attention" --max 5
    python scripts/ingest_arxiv.py --id 2310.06825
"""
import sys
import os
import asyncio
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from backend.config import get_settings
from backend.services.ingest_pipeline import IngestPipeline


async def run(args):
    settings = get_settings()
    pipeline = IngestPipeline(settings)

    if args.id:
        logger.info(f"Ingesting arXiv paper: {args.id}")
        result = await pipeline.ingest_paper_by_id(args.id)
        print(f"\n{'='*50}")
        print(f"Paper ID  : {result.paper_id}")
        print(f"Status    : {result.status}")
        print(f"Chunks    : {result.chunks_created}")
        print(f"Embeddings: {result.embeddings_stored}")
        print(f"Message   : {result.message}")
    else:
        logger.info(f"Searching and ingesting: '{args.query}' (max {args.max} papers)")
        results = await pipeline.search_and_ingest(
            query=args.query,
            max_papers=args.max,
            year_from=args.year_from,
        )
        print(f"\n{'='*50}")
        print(f"{'Paper ID':<30} {'Status':<10} {'Chunks':>8} {'Embeddings':>12}")
        print("-" * 65)
        for r in results:
            print(f"{r.paper_id:<30} {r.status:<10} {r.chunks_created:>8} {r.embeddings_stored:>12}")
        success = sum(1 for r in results if r.status == "success")
        print(f"\n{success}/{len(results)} papers ingested successfully.")


def main():
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into ResearchCopilot")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--id",    "-i", type=str, help="Specific arXiv paper ID")
    parser.add_argument("--max",   "-m", type=int, default=5, help="Max papers to ingest")
    parser.add_argument("--year-from", type=int, default=None, help="Filter papers from this year")
    args = parser.parse_args()

    if not args.query and not args.id:
        parser.error("Provide either --query or --id")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
