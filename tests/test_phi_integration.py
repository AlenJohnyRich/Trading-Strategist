"""
Unit tests for Phi hashed embedding integration with tokenizer outputs.
"""
from TradingAI.ai.tokenizers import get_tokenizer
from TradingAI.ai.phi_embedding import PhiHashedEmbedding
import torch


def test_phi_embedding_forward():
    tok = get_tokenizer('phi')
    s = "I love chicken."
    ids = tok.encode(s)
    # create tensor shape (1, seq_len)
    t = torch.tensor([ids], dtype=torch.long)
    emb = PhiHashedEmbedding(num_buckets=2**10, embed_dim=16)
    out = emb(t)
    assert out.shape[0] == 1
    assert out.shape[1] == len(ids)
    assert out.shape[2] == 16
