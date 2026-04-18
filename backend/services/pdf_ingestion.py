"""
PDF Ingestion Service
Extracts text from PDFs, splits into overlapping chunks,
and stores embeddings + metadata into ChromaDB / DynamoDB.
"""
import asyncio
import hashlib
import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber
try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except ImportError:
    _PYMUPDF = False
from loguru import logger

from backend.config import Settings
from backend.models.schemas import IngestRequest, IngestResponse


class PDFIngestionService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.chunk_size = settings.chunk_size       # tokens ~ characters/4
        self.chunk_overlap = settings.chunk_overlap
        self._current_title = ""  # set in _process before _split_into_chunks

        # Lazy-loaded services to avoid circular imports
        self._embedding_svc = None
        self._vector_db = None
        self._dynamo_db = None

    # ── Lazy loaders ───────────────────────────────────────────────────────

    def _get_embedding_svc(self):
        if self._embedding_svc is None:
            from backend.services.embedding_service import EmbeddingService
            self._embedding_svc = EmbeddingService(self.settings)
        return self._embedding_svc

    def _get_vector_db(self):
        if self._vector_db is None:
            from backend.services.vector_db import VectorDBService
            self._vector_db = VectorDBService(self.settings)
        return self._vector_db

    def _get_dynamo_db(self):
        if self._dynamo_db is None:
            from backend.services.dynamo_db import DynamoDBService
            self._dynamo_db = DynamoDBService(self.settings)
        return self._dynamo_db

    # ── Public ─────────────────────────────────────────────────────────────

    async def ingest_text(self, request: IngestRequest) -> IngestResponse:
        """Ingest from raw text string (already extracted)."""
        text = request.content or ""
        if not text.strip():
            return IngestResponse(
                paper_id=request.paper_id,
                status="error",
                chunks_created=0,
                embeddings_stored=0,
                message="Empty content provided.",
            )
        return await self._process(request.paper_id, request.title, text)

    async def ingest_pdf(self, paper_id: str, title: str, pdf_path: str) -> IngestResponse:
        """Extract text from a PDF file then ingest."""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._extract_text_from_pdf, pdf_path)

        if not text.strip():
            return IngestResponse(
                paper_id=paper_id,
                status="error",
                chunks_created=0,
                embeddings_stored=0,
                message="Could not extract text from PDF.",
            )
        return await self._process(paper_id, title, text)

    # ── Core Pipeline ──────────────────────────────────────────────────────

    async def _process(self, paper_id: str, title: str, text: str) -> IngestResponse:
        logger.info(f"[PDFIngestion] processing paper_id={paper_id}, text_len={len(text)}")

        # 0. Skip if already ingested
        try:
            vector_db = self._get_vector_db()
            col = vector_db._get_collection(self.settings.chroma_collection_papers)
            existing = col.get(where={"paper_id": {"$eq": paper_id}}, limit=1, include=[])
            if existing and existing.get("ids"):
                logger.info(f"[PDFIngestion] paper_id={paper_id} already in KB — skipping")
                try:
                    from backend.services.prom_metrics import INGEST_TOTAL
                    INGEST_TOTAL.labels(status="already_exists").inc()
                except Exception:
                    pass
                return IngestResponse(
                    paper_id=paper_id,
                    status="already_exists",
                    chunks_created=0,
                    embeddings_stored=0,
                    message="Paper already in knowledge base.",
                )
        except Exception as e:
            logger.warning(f"[PDFIngestion] duplicate check failed (non-fatal): {e}")

        # 1. Clean text
        text = self._clean_text(text)

        # 2. Split into chunks (title stored on self so chunker can embed it in metadata)
        self._current_title = title
        chunks = self._split_into_chunks(text, paper_id)
        logger.debug(f"[PDFIngestion] created {len(chunks)} chunks")

        if not chunks:
            return IngestResponse(
                paper_id=paper_id,
                status="error",
                chunks_created=0,
                embeddings_stored=0,
                message="Chunking produced no output.",
            )

        # 3. Embed all chunks
        embedding_svc = self._get_embedding_svc()
        texts = [c["text"] for c in chunks]
        embeddings = await embedding_svc.embed_batch(texts)

        # 4. Store in ChromaDB
        vector_db = self._get_vector_db()
        stored = await vector_db.store_chunks(
            collection_name=self.settings.chroma_collection_papers,
            chunks=chunks,
            embeddings=embeddings,
        )

        # 5. Store metadata in DynamoDB
        dynamo = self._get_dynamo_db()
        await dynamo.put_paper(
            paper_id=paper_id,
            title=title,
            chunk_count=len(chunks),
        )

        logger.info(f"[PDFIngestion] stored {stored} embeddings for paper_id={paper_id}")
        try:
            from backend.services.prom_metrics import INGEST_TOTAL, INGEST_CHUNKS
            INGEST_TOTAL.labels(status="success").inc()
            INGEST_CHUNKS.inc(len(chunks))
        except Exception:
            pass
        return IngestResponse(
            paper_id=paper_id,
            status="success",
            chunks_created=len(chunks),
            embeddings_stored=stored,
            message=f"Ingested {len(chunks)} chunks successfully.",
        )

    # ── Text Extraction ────────────────────────────────────────────────────

    # Section headers commonly found in research papers
    _SECTION_RE = re.compile(
        r'^(abstract|introduction|related work|background|methodology|methods|'
        r'experiments?|results?|discussion|conclusion|references|appendix)',
        re.IGNORECASE | re.MULTILINE,
    )
    # Heuristics for LaTeX-style math: \command, $...$, display math markers
    _MATH_RE = re.compile(
        r'(\\\w+\{|\$[^$\n]{1,80}\$|\\frac|\\sum|\\int|\\alpha|\\beta|\\gamma|'
        r'\\delta|\\sigma|\\theta|\\lambda|\\mu|\\pi|\\nabla|\\partial|'
        r'\\mathbb|\\mathcal|\\text\{|\\begin\{equation|\\leq|\\geq|\\rightarrow)'
    )

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF with section + equation detection."""
        if _PYMUPDF:
            return self._extract_with_pymupdf(pdf_path)
        return self._extract_with_pdfplumber(pdf_path)

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """PyMuPDF extraction: preserves sections and flags math-heavy blocks."""
        pages_text = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
                page_lines = []
                for block in sorted(blocks, key=lambda b: (b[1], b[0])):  # top-to-bottom
                    text = block[4].strip()
                    if not text:
                        continue
                    # Tag section headers
                    if self._SECTION_RE.match(text) and len(text) < 80:
                        page_lines.append(f"\n## {text.strip()}\n")
                    else:
                        # Tag blocks with heavy math content
                        math_hits = len(self._MATH_RE.findall(text))
                        if math_hits >= 2:
                            page_lines.append(f"[EQUATION_BLOCK]\n{text}\n[/EQUATION_BLOCK]")
                        else:
                            page_lines.append(text)
                if page_lines:
                    pages_text.append(f"[Page {page_num+1}]\n" + "\n".join(page_lines))
            doc.close()
        except Exception as e:
            logger.error(f"[PDFIngestion] PyMuPDF error: {e}, falling back to pdfplumber")
            return self._extract_with_pdfplumber(pdf_path)
        return "\n\n".join(pages_text)

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Fallback: pdfplumber plain text extraction."""
        pages_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append(f"[Page {i+1}]\n{text}")
        except Exception as e:
            logger.error(f"[PDFIngestion] pdfplumber error: {e}")
        return "\n\n".join(pages_text)

    # ── Chunking ───────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str, paper_id: str) -> List[dict]:
        """
        Sliding-window character chunking with overlap.
        Returns list of chunk dicts ready for ChromaDB storage.
        """
        # Approximate char budget (1 token ≈ 4 chars)
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4

        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + char_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - char_size // 5
                match = None
                for pattern in [r'\.\s+', r'\n\n', r'\n']:
                    m = re.search(pattern, text[search_start:end])
                    if m:
                        end = search_start + m.end()
                        break

            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # skip tiny trailing chunks
                chunk_id = self._chunk_id(paper_id, idx)
                chunks.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "text": chunk_text,
                    "chunk_index": idx,
                    "metadata": {
                        "paper_id": paper_id,
                        "paper_title": self._current_title,  # set before call
                        "chunk_index": idx,
                        "char_start": start,
                        "char_end": end,
                    },
                })
                idx += 1

            start = end - char_overlap
            if start >= len(text):
                break

        return chunks

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove excessive whitespace, null bytes, ligature artifacts."""
        text = text.replace("\x00", "")
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    @staticmethod
    def _chunk_id(paper_id: str, idx: int) -> str:
        return hashlib.md5(f"{paper_id}_{idx}".encode()).hexdigest()[:16]
