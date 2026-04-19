"""
LLM Service — qwen3.5:35b via Ollama (local GPU).
Features:
  - rewrite_query: cleans/expands user input before search & retrieval
  - generate_long_answer: multi-pass inference loop for complete answers
  - Falls back to a template answer if Ollama is unavailable.
"""
import re
import asyncio
from typing import Optional

import httpx
from loguru import logger

from backend.config import Settings


class LLMService:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    # ── Low-level generate ──────────────────────────────────────────────────

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        num_predict: int = 1024,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt + " /no_think",
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": num_predict,
                "num_ctx": 8192,
            },
        }
        if system:
            payload["system"] = system

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                text = data.get("response", "").strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                return text
        except httpx.ConnectError:
            logger.warning("[LLMService] Ollama not reachable, using fallback")
            return self._fallback_response(prompt)
        except Exception as e:
            logger.error(f"[LLMService] generation error: {e}")
            return self._fallback_response(prompt)

    # ── Query Rewriting ─────────────────────────────────────────────────────

    async def rewrite_query(self, raw_query: str, memory_context: str = "") -> dict:
        """
        Takes the user's raw query and returns THREE distinct forms:
          - search_query:    broad keywords for arXiv paper discovery
          - retrieval_query: content-specific terms to hit the right ChromaDB chunks
                             (describes WHAT the answer looks like, not just the topic)
          - answer_question: a well-formed question for the LLM RAG prompt
        """
        prompt = f"""You are a research query assistant. Given the user's raw input, output FOUR lines exactly:
SEARCH: <3-8 broad keywords for arXiv paper search>
RETRIEVE: <specific mathematical/technical terms that would appear INSIDE the relevant paper sections>
QUESTION: <a clear, complete, well-formed research question>
PAPER: <ONLY a well-known published paper acronym/name if the user explicitly refers to a specific paper (e.g. "BERT paper", "LoRA paper", "DDPM paper"). Use NONE for general topic questions.>

Rules for PAPER field:
- "lora paper equations" → PAPER: LoRA
- "explain bert" → PAPER: BERT
- "ddpm loss function" or "diffusion paper loss" or "diffusion papers loss" → PAPER: DDPM
- "how does attention work" → PAPER: NONE  (general topic, not a specific paper)
- "explain diffusion models" → PAPER: DDPM  (DDPM is the canonical diffusion paper)
- "resnet architecture" → PAPER: ResNet
- Never invent paper names. If unsure, use NONE.

Examples:
Raw: "berts all equation"
SEARCH: BERT pre-training deep bidirectional transformers
RETRIEVE: BERT attention formula softmax QKV matrix layer normalization feed-forward equation
QUESTION: What are all the mathematical equations used in the BERT model?
PAPER: BERT

Raw: "explain lora paper equations"
SEARCH: LoRA low-rank adaptation large language models Hu 2021
RETRIEVE: LoRA W0 BA low-rank decomposition delta W weight update rank r scaling alpha pretrained weights frozen
QUESTION: What are all the mathematical equations in the original LoRA paper, including weight update, rank decomposition, and scaling factor?
PAPER: LoRA

Raw: "explain the diffusion papers loss function"
SEARCH: DDPM denoising diffusion probabilistic models loss function Ho 2020
RETRIEVE: DDPM loss function variational lower bound noise prediction epsilon simple objective L_simple L_vlb forward process reverse process
QUESTION: What is the loss function in the DDPM diffusion paper, including the variational lower bound, simplified objective, and noise prediction?
PAPER: DDPM

Raw: "how does gpt work"
SEARCH: GPT language model transformer autoregressive generation
RETRIEVE: GPT transformer decoder autoregressive next token prediction causal attention language model
QUESTION: How does GPT work including its architecture, attention, and training objective?
PAPER: NONE

Raw: "explain cross attention"
SEARCH: cross-attention mechanism transformer encoder decoder
RETRIEVE: cross-attention query key value encoder decoder attention weight softmax scaled dot product
QUESTION: What is cross-attention in transformers and how does it differ from self-attention?
PAPER: NONE

Context rule: If the raw query is short or ambiguous (e.g. "give equation", "explain more", "what about loss?"),
use the conversation history below to infer what topic the user is referring to, and expand the query accordingly.
If the raw query is self-contained, ignore the history.

Conversation history (most recent last):
{memory_context if memory_context else "(none)"}

Now process this:
Raw: "{raw_query}"
SEARCH:"""

        try:
            raw = await self.generate(prompt, num_predict=300)
            lines = raw.strip().splitlines()

            search_query = raw_query
            retrieval_query = raw_query
            answer_question = raw_query
            paper_name = None

            for line in lines:
                line = line.strip()
                if line.upper().startswith("SEARCH:"):
                    val = line[7:].strip()
                    if val:
                        search_query = val
                elif line.upper().startswith("RETRIEVE:"):
                    val = line[9:].strip()
                    if val:
                        retrieval_query = val
                elif line.upper().startswith("QUESTION:"):
                    val = line[9:].strip()
                    if val:
                        answer_question = val
                elif line.upper().startswith("PAPER:"):
                    val = line[6:].strip()
                    if val and val.upper() != "NONE":
                        paper_name = val

            if retrieval_query == raw_query and search_query != raw_query:
                retrieval_query = search_query

            missing = [
                field for field, val in [
                    ("SEARCH", search_query), ("RETRIEVE", retrieval_query), ("QUESTION", answer_question)
                ] if val == raw_query
            ]
            if missing:
                logger.warning(
                    f"[LLMService] rewrite_query: LLM output missing fields {missing}, "
                    f"falling back to raw query for those fields.\nRaw LLM output: {raw!r}"
                )

            logger.info(
                f"[LLMService] rewrite_query: '{raw_query}'\n"
                f"  search='{search_query}'\n"
                f"  retrieve='{retrieval_query}'\n"
                f"  question='{answer_question}'\n"
                f"  paper='{paper_name}'"
            )
            return {
                "search_query": search_query,
                "retrieval_query": retrieval_query,
                "answer_question": answer_question,
                "paper_name": paper_name,
            }

        except Exception as e:
            logger.warning(f"[LLMService] rewrite_query failed: {e} — using raw query")
            return {
                "search_query": raw_query,
                "retrieval_query": raw_query,
                "answer_question": raw_query,
                "paper_name": None,
            }

    # ── Multi-pass Long Answer ──────────────────────────────────────────────

    async def generate_long_answer(
        self,
        query: str,
        context_chunks: list,
        memory_context: str = "",
        max_passes: int = 3,
    ) -> str:
        """
        Generates a complete answer using multi-pass inference if the first
        response hits the token limit. Each pass continues from where the
        previous one stopped.

        Pass 1: full context, generate answer
        Pass 2+: if answer is truncated, continue with "continue from: <tail>"
        """
        # Pass 1 — full RAG answer
        prompt = self.build_rag_prompt(query, context_chunks, memory_context)
        answer = await self.generate(prompt, num_predict=1024)

        if not answer or answer == self._fallback_response(""):
            return answer

        # Continuation passes
        for pass_num in range(2, max_passes + 1):
            if self._is_complete(answer):
                logger.debug(f"[LLMService] answer complete after pass {pass_num - 1}")
                break

            logger.info(f"[LLMService] answer truncated — starting pass {pass_num}")
            tail = answer[-600:].strip()

            continuation_prompt = f"""You are ResearchCopilot. You were answering the question:
"{query}"

Your previous response ended mid-way. Continue EXACTLY from where it was cut off.
Do NOT repeat or summarize what was already said. Just continue the answer naturally.

Previous answer ended with:
"...{tail}"

Continue:"""

            continuation = await self.generate(continuation_prompt, num_predict=1024)
            if not continuation or len(continuation.strip()) < 30:
                break

            answer = answer.rstrip() + " " + continuation.strip()

        return answer

    # ── Prompt Builders ─────────────────────────────────────────────────────

    def build_rag_prompt(self, query: str, context_chunks: list, memory_context: str = "") -> str:
        context_str = "\n\n---\n\n".join(
            [f"[Source {i+1}: {c.paper_title or 'Paper'}]\n{c.text}"
             for i, c in enumerate(context_chunks)]
        )

        memory_section = ""
        if memory_context:
            memory_section = f"\n\nConversation history:\n{memory_context}\n"

        return f"""You are ResearchCopilot, an expert AI research assistant.

Use the provided research paper excerpts as your primary context and cite them with [Source N].
If the context does not contain specific equations, proofs, or foundational definitions that are standard academic knowledge, supplement using your own knowledge — but clearly state when you are doing so (e.g., "From standard GAN theory:").
Never refuse to explain equations or math just because they are absent from the context.

EQUATION RULES — follow exactly:
- Inline equations (within a sentence): $...$ — example: "the loss $L$ is minimized"
- Display equations (standalone block): put $$ on its own line before and after:
  $$
  <your equation here>
  $$
- NEVER output a display equation as raw LaTeX without $$ delimiters.
- NEVER write an equation twice (no plain-text duplicate after the LaTeX).
- Each equation appears exactly once, in LaTeX form.
{memory_section}
Research context:
{context_str}

Question: {query}

Answer:"""

    def build_report_prompt(self, topic: str, context_chunks: list, max_length: int = 800) -> str:
        context_str = "\n\n---\n\n".join(
            [f"[Source {i+1}: {c.paper_title or 'Paper'}]\n{c.text}"
             for i, c in enumerate(context_chunks)]
        )

        return f"""You are ResearchCopilot. Generate a comprehensive, detailed research report on the topic below.
Use ONLY the provided paper excerpts as sources. Format the report in Markdown with these sections:
## Summary, ## Key Findings, ## Mathematical Foundations, ## Methodology, ## Conclusion, ## References.

EQUATION RULES — mandatory:
- Include ALL relevant mathematical equations from the source material.
- Inline equations (within a sentence): use $...$ — example: "the weight $W_i$ scales"
- Display equations (standalone, on their own line): wrap with $$ on SEPARATE lines:
  $$
    \\frac{{\\partial L}}{{\\partial x}} = \\sum_i w_i
  $$
- NEVER output a display equation as raw LaTeX without $$ delimiters.
- NEVER write equations in plain text or duplicate them — LaTeX only, once each.
- NEVER use placeholder tokens like %%%LATEX_BLOCK_1%%% — always write the actual LaTeX.
- Always put a space before and after every inline equation: "position $i$ depends" not "position$i$depends".
- Derive and explain each equation clearly with variable definitions.

Target length: {max_length} words. Be thorough and detailed — do not cut short.

Topic: {topic}

Research context:
{context_str}

Report:"""

    async def generate_long_report(
        self,
        topic: str,
        context_chunks: list,
        max_length: int = 2000,
        max_passes: int = 50,
    ) -> str:
        """
        Multi-pass report generation. Continues until the report is complete
        or max_passes is reached. Each pass adds ~700-800 words.
        """
        prompt = self.build_report_prompt(topic, context_chunks, max_length)
        report = await self.generate(prompt, num_predict=1024)

        if not report:
            return report

        passes_used = 1

        for pass_num in range(2, max_passes + 1):
            if self._is_complete(report):
                logger.debug(f"[LLMService] report complete after pass {pass_num - 1}")
                break

            # Estimate current word count
            word_count = len(report.split())
            if word_count >= max_length:
                logger.debug(f"[LLMService] report reached target {word_count} words")
                break

            logger.info(f"[LLMService] report pass {pass_num} — {word_count} words so far")
            tail = report[-800:].strip()

            continuation_prompt = f"""You are ResearchCopilot writing a research report on: "{topic}"

Your report was cut off. Continue EXACTLY from where it ended.
Do NOT repeat anything already written. Do NOT restart sections.
Keep writing the report naturally, including all equations in LaTeX.

STRICT RULES:
- NEVER use placeholder tokens like %%%LATEX_BLOCK_1%%% or %%%...%%% — always write the actual LaTeX equation inline.
- Always put a space before and after every inline equation: "position $i$ depends" not "position$i$depends".
- Display equations MUST use $$ on separate lines: $$\n\\equation\n$$ — NEVER raw LaTeX on its own line without delimiters.

Report so far ended with:
"...{tail}"

Continue the report:"""

            continuation = await self.generate(continuation_prompt, num_predict=1024)
            if not continuation or len(continuation.strip()) < 30:
                break

            report = report.rstrip() + "\n\n" + continuation.strip()
            passes_used = pass_num

        logger.info(f"[LLMService] final report: {len(report.split())} words over {passes_used} passes")
        return report

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _is_complete(self, text: str) -> bool:
        """Heuristic: answer is complete if it ends with a sentence-ending token."""
        stripped = text.strip()
        if len(stripped) < 100:
            return False
        return stripped[-1] in ".!?)" or stripped.endswith(("```", "**", "---"))

    async def check_health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    def _fallback_response(self, prompt: str) -> str:
        return (
            f"Ollama LLM service is currently unavailable. "
            f"Please ensure Ollama is running and the '{self.model}' model is pulled "
            f"(`ollama pull {self.model}`). Your question was received but could not be answered."
        )
