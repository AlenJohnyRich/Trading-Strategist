"""
Extend orchestrator to include DeepSeek as an additional source.
The orchestrator will call DeepSeek if configured and include its results among other sources.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any

from TradingAI.ai.embeddings import EmbeddingService
from TradingAI.ingest.pdf_image_ingest import Ingestor
from TradingAI.memory.vector_store import VectorStore
from TradingAI.ai.deepseek import DeepSeekClient

logger = logging.getLogger('TradingAI.orchestrator')


class Summarizer:
    def __init__(self, provider: str = 'openai'):
        self.provider = provider
        try:
            import openai
            self.openai = openai
        except Exception:
            self.openai = None

    def compose(self, answers: list, query: str, context: dict | None = None) -> Dict:
        # Minimal local summarizer: concatenate answers and include DeepSeek highlights if present
        parts = []
        for i, a in enumerate(answers):
            try:
                src = a.get('source', f'source_{i}')
                ans = a.get('answer') or a.get('text') or str(a)
                parts.append(f"[{src}] {ans}")
            except Exception:
                parts.append(str(a))
        summary = '\n\n'.join(parts)
        return {'summary': summary}


class PipelineOrchestrator:
    def __init__(self, embed_provider: str = 'local', embed_model: str = 'all-MiniLM-L6-v2'):
        self.ingestor = Ingestor()
        self.embedder = EmbeddingService(provider=embed_provider, model_name=embed_model)
        self.vs = VectorStore(embedding_function=self.embedder.embed_text)
        self.summarizer = Summarizer()
        self.deepseek = DeepSeekClient()

    async def run_transformer_pipeline(self, query: str, ctx: Dict[str,Any]) -> Dict:
        qvec = self.embedder.embed_text(query)
        hits = self.vs.query(qvec, top_k=5)
        answer = ' '.join([h.get('meta', {}).get('snippet','') or h.get('id','') for h in hits])
        return {'source': 'transformer', 'answer': answer, 'hits': hits}

    async def run_db_pipeline(self, query: str, ctx: Dict[str,Any]) -> Dict:
        return {'source': 'db', 'answer': 'db results (stub)'}

    async def run_web_pipeline(self, query: str, ctx: Dict[str,Any]) -> Dict:
        return {'source': 'web', 'answer': 'web results (stub)'}

    async def run_memory_pipeline(self, query: str, ctx: Dict[str,Any]) -> Dict:
        qvec = self.embedder.embed_text(query)
        hits = self.vs.query(qvec, top_k=5)
        return {'source': 'memory', 'answer': ' '.join([h.get('meta', {}).get('snippet','') or h.get('id','') for h in hits]), 'hits': hits}

    async def run_deepseek_pipeline(self, query: str, ctx: Dict[str,Any]) -> Dict:
        # Use DeepSeek if available; return top results
        res = []
        try:
            res = self.deepseek.search(query, top_k=5)
        except Exception:
            res = []
        answer = ''
        if res:
            # concatenate titles/snippets if present
            snippets = []
            for r in res:
                if isinstance(r, dict):
                    snippets.append(r.get('snippet') or r.get('title') or str(r))
                else:
                    snippets.append(str(r))
            answer = ' '.join(snippets)
        return {'source': 'deepseek', 'answer': answer or 'no deepseek results', 'hits': res}

    async def run_all_sources(self, query: str, ctx: Dict[str,Any] | None = None) -> Dict[str,Any]:
        ctx = ctx or {}
        tasks = [
            asyncio.create_task(self.run_transformer_pipeline(query, ctx)),
            asyncio.create_task(self.run_db_pipeline(query, ctx)),
            asyncio.create_task(self.run_web_pipeline(query, ctx)),
            asyncio.create_task(self.run_memory_pipeline(query, ctx)),
            asyncio.create_task(self.run_deepseek_pipeline(query, ctx)),
        ]
        done, pending = await asyncio.wait(tasks, timeout=25)
        answers = []
        for t in tasks:
            if t in done:
                try:
                    answers.append(t.result())
                except Exception as exc:
                    answers.append({'error': str(exc)})
            else:
                t.cancel()
                answers.append({'error': 'timeout'})
        final = self.summarizer.compose(answers, query, ctx)
        return {'answers': answers, 'summary': final}
