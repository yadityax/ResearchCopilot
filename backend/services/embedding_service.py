"""
Embedding Service
Uses pretrained sentence-transformers/all-MiniLM-L6-v2 (384-dim).
Passes embeddings through a lightweight MLP projection (Encoder MLP + Sinusoidal MLP)
as described in the ResearchCopilot architecture.
"""
import asyncio
import time
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from backend.config import Settings
from backend.models.schemas import EmbeddingQueryRequest, EmbeddingQueryResponse, RetrievedChunk
from backend.models.mlp_models import EncoderMLP, SinusoidalMLP
from backend.services.mlflow_service import MLflowService


class EmbeddingService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: SentenceTransformer | None = None
        self._encoder_mlp: EncoderMLP | None = None
        self._sinusoidal_mlp: SinusoidalMLP | None = None
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
            dim = self.settings.embedding_dim
            self._encoder_mlp = EncoderMLP(input_dim=dim, hidden_dim=dim, output_dim=dim).to(self._device)
            self._sinusoidal_mlp = SinusoidalMLP(dim=dim).to(self._device)
            self._encoder_mlp.eval()
            self._sinusoidal_mlp.eval()
            logger.info(f"[EmbeddingService] Models loaded on {self._device}")

    # ── Public ─────────────────────────────────────────────────────────────

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts → 384-dim projected vectors."""
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

        # Stage 1: pretrained sentence-transformers (384-dim)
        with torch.no_grad():
            raw = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self._device,
                normalize_embeddings=True,
            )  # shape: (N, 384)

            # Stage 2: Encoder MLP projection
            projected = self._encoder_mlp(raw)  # (N, 384)

            # Stage 3: Sinusoidal MLP
            final = self._sinusoidal_mlp(projected)  # (N, 384)

            # L2 normalize final embeddings
            final = torch.nn.functional.normalize(final, p=2, dim=1)

        return final.cpu().numpy().tolist()
