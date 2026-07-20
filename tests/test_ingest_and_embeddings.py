"""
Tests for ingestion and embeddings pipeline.
"""
from TradingAI.ingest.pdf_image_ingest import Ingestor
from TradingAI.ai.embeddings import EmbeddingService


def test_ingest_text(tmp_path):
    p = tmp_path / 'sample.txt'
    p.write_text('This is a test document. It contains several words for embedding.', encoding='utf-8')
    ing = Ingestor(base_store=str(tmp_path / 'store'))
    meta = ing.ingest_file(str(p))
    assert 'id' in meta
    chunks = ing.get_chunks(meta['id'])
    assert len(chunks) > 0


def test_embeddings():
    es = EmbeddingService(provider='local')
    v = es.embed_text('hello world')
    assert hasattr(v, '__len__')
