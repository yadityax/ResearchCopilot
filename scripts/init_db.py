"""
Database Initialization Script
Run once before starting the backend to ensure all collections and tables exist.

Usage:
    python scripts/init_db.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
from backend.config import get_settings
from backend.services.vector_db import VectorDBService
from backend.services.dynamo_db import DynamoDBService


def init_chromadb(settings):
    logger.info("Initializing ChromaDB collections...")
    vdb = VectorDBService(settings)

    collections = [
        settings.chroma_collection_papers,
        settings.chroma_collection_memory,
    ]
    for name in collections:
        col = vdb._get_collection(name)
        logger.info(f"  ChromaDB collection '{name}': {col.count()} documents")

    logger.success("ChromaDB initialized.")


def init_dynamodb(settings):
    logger.info("Initializing DynamoDB tables...")
    dynamo = DynamoDBService(settings)

    # Trigger lazy table creation
    dynamo._get_papers_table()
    logger.info(f"  DynamoDB table '{settings.dynamodb_papers_table}': OK")

    dynamo._get_sessions_table()
    logger.info(f"  DynamoDB table '{settings.dynamodb_sessions_table}': OK")

    logger.success("DynamoDB initialized.")


def main():
    settings = get_settings()
    logger.info(f"=== ResearchCopilot DB Init ({settings.app_env}) ===")

    try:
        init_chromadb(settings)
    except Exception as e:
        logger.error(f"ChromaDB init failed: {e}")

    try:
        init_dynamodb(settings)
    except Exception as e:
        logger.error(f"DynamoDB init failed: {e}")

    logger.info("=== Init complete ===")


if __name__ == "__main__":
    main()
