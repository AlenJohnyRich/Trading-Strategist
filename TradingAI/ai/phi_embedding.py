"""
Learned hashed embedding table for Phi IDs.
Maps hashed integer IDs into a compact trainable embedding space with a
configurable number of buckets. Provides persistence helpers.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import json
from typing import Optional


class PhiHashedEmbedding(nn.Module):
    def __init__(self, num_buckets: int = 2**18, embed_dim: int = 128, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_buckets = num_buckets
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_buckets, embed_dim, padding_idx=padding_idx)

    def forward(self, ids_tensor: torch.LongTensor) -> torch.Tensor:
        # ids_tensor: (batch, seq_len)
        # reduce large IDs into buckets via modulo
        buckets = ids_tensor % self.num_buckets
        return self.embedding(buckets)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {'num_buckets': self.num_buckets, 'embed_dim': self.embed_dim}
        torch.save({'state': state, 'weight': self.embedding.weight.detach().cpu()}, path)

    @staticmethod
    def load(path: str) -> 'PhiHashedEmbedding':
        d = torch.load(path, map_location='cpu')
        st = d['state']
        emb = PhiHashedEmbedding(num_buckets=st['num_buckets'], embed_dim=st['embed_dim'])
        emb.embedding.weight.data.copy_(d['weight'])
        return emb
