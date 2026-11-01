"""
Embedding Service
Uses pretrained sentence-transformers/all-MiniLM-L6-v2 (384-dim, L2-normalised).

The EncoderMLP / SinusoidalMLP projection (backend/models/mlp_models.py) is intentionally NOT
applied. Untrained, those layers were re-initialised randomly in every process, so each pod and
each restart embedded the same text into a different space (cosine ~0.5 between two processes),
and on the eval set they lowered Recall@5 on average (see eval/results/report.md). They stay in
the codebase as a hook for contrastive fine-tuning with saved, versioned weights.
"""
import asyncio
import time
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.config import Settings
from backend.models.schemas import EmbeddingQueryRequest, EmbeddingQueryResponse, RetrievedChunk
from backend.services.mlflow_service import MLflowService


class EmbeddingService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mlflow = MLflowService(settings.mlflow_tracking_uri, settings.mlflow_experiment_name)

    # ── Lazy init ──────────────────────────────────────────────────────────

    def _load_models(self):
        if self._model is None:
            logger.info(f"[EmbeddingService] Loading {self.settings.embedding_model}")
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self._device,
                token=False,   # don't use cached/expired HF token; model is public
            )
            logger.info(f"[EmbeddingService] Model loaded on {self._device}")

    # ── Public ─────────────────────────────────────────────────────────────

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts → 384-dim normalised vectors."""
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        result = await loop.run_in_executor(None, self._embed_sync, texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._mlflow.log_embedding_batch(len(texts), elapsed_ms)
        return result

    async def embed_single(self, text: str) -> List[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def query(self, request: EmbeddingQueryRequest) -> EmbeddingQueryResponse:
        from backend.services.vector_db import VectorDBService
        vector_db = VectorDBService(self.settings)

        query_embedding = await self.embed_single(request.query)
        results = await vector_db.query_similar(
            collection_name=request.collection,
            query_embedding=query_embedding,
            top_k=request.top_k,
            filter_paper_ids=request.filter_paper_ids,
        )
        return EmbeddingQueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        self._load_models()

        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self._device,
                normalize_embeddings=True,
            )  # shape: (N, 384)

        return embeddings.cpu().numpy().tolist()
