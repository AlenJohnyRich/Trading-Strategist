"""
Vector store wrapper using Chroma (preferred) with filesystem fallback.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

logger = logging.getLogger('TradingAI.vector')

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None


class VectorStore:
    def __init__(self, persist_dir: str = './vectordb', embedding_function=None):
        self.persist_dir = persist_dir
        self.embedding_function = embedding_function
        self.client = None
        self.col = None
        if chromadb is not None:
            try:
                self.client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=self.persist_dir))
                self.col = self.client.get_or_create_collection(name='tradingai', metadata={})
            except Exception:
                self.client = None
        if self.client is None:
            logger.warning('Chroma not available, vector store will be filesystem-backed (not efficient)')
            # simple filesystem fallback will be handled by higher-level code

    def upsert_chunks(self, ids: List[str], metadatas: List[Dict], embeddings: List[List[float]]):
        if self.col:
            self.col.upsert(ids=ids, metadatas=metadatas, embeddings=embeddings)
        else:
            # fallback: write to disk files
            import os, json
            d = self.persist_dir
            os.makedirs(d, exist_ok=True)
            for i, eid in enumerate(ids):
                path = f"{d}/{eid}.json"
                with open(path, 'w', encoding='utf-8') as fh:
                    json.dump({'meta': metadatas[i], 'emb': embeddings[i]}, fh)

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        if self.col:
            res = self.col.query(query_embeddings=[query_embedding], n_results=top_k)
            out = []
            for i in range(len(res['ids'][0])):
                out.append({'id': res['ids'][0][i], 'meta': res['metadatas'][0][i], 'dist': None})
            return out
        else:
            # naive filesystem linear search (very slow)
            import os, json
            best = []
            for f in os.listdir(self.persist_dir):
                if not f.endswith('.json'): continue
                obj = json.load(open(f, 'r', encoding='utf-8'))
                best.append({'id': f[:-5], 'meta': obj['meta'], 'emb': obj['emb']})
            # no real distance calc here
            return best[:top_k]

    def upsert_hidden(self, ids: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        # store hidden activations similarly to chunks
        self.upsert_chunks(ids, metadatas or [{}]*len(ids), embeddings)
