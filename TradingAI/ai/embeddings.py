"""
Embedding service using local sentence-transformers (default) with OpenAI fallback.
"""
from __future__ import annotations

import os
import logging
from typing import List, Optional

logger = logging.getLogger('TradingAI.embeddings')

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None


class EmbeddingService:
    def __init__(self, provider: str = 'local', model_name: str = 'all-MiniLM-L6-v2'):
        self.provider = provider
        self.model_name = model_name
        self.model = None
        if provider == 'local':
            if SentenceTransformer is None:
                raise RuntimeError('sentence-transformers not installed')
            self.model = SentenceTransformer(model_name)
        elif provider == 'openai':
            if openai is None:
                raise RuntimeError('openai not installed')
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise RuntimeError('OPENAI_API_KEY not set')
            openai.api_key = self.api_key
        else:
            raise ValueError('Unknown provider')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.provider == 'local':
            emb = self.model.encode(texts, show_progress_bar=False)
            return emb.tolist() if hasattr(emb, 'tolist') else [list(e) for e in emb]
        else:
            # OpenAI embeddings (batching naive)
            res = []
            for t in texts:
                r = openai.Embedding.create(input=t, model='text-embedding-3-small')
                res.append(r['data'][0]['embedding'])
            return res

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
