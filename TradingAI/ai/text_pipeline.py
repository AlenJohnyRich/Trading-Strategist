"""
Text processing and tokenization pipeline used by the AIExpert.
Supports tokenization by word and letter, scrambling/unscrambling heuristics,
storage and retrieval of sentences, generation of random phrases, and building
simple embeddings to feed into transformer training.
"""
from __future__ import annotations

import re
import random
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
import pathlib

from TradingAI.db.connectors import FilesystemStore
from TradingAI.utils.encryption import EncryptionHelper


WORD_RE = re.compile(r"\w+|[^	\w\s]+")


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def tokenize_letters(text: str) -> List[str]:
    return list(text)


def normalize_sentence(text: str) -> str:
    # lowercase, strip extra whitespace, remove surrounding punctuation
    s = text.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def scramble_words(sentence: str) -> str:
    words = tokenize_words(sentence)
    if len(words) <= 1:
        return sentence
    random.shuffle(words)
    return ' '.join(words)


def random_phrase_from_sentences(sentences: List[str], n_words: int = 5) -> str:
    # pick random words from stored sentences and combine
    pool = []
    for s in sentences:
        pool.extend(tokenize_words(s))
    if not pool:
        return ''
    return ' '.join(random.choices(pool, k=n_words))


class SentenceStore:
    """Stores sentences encrypted on disk and records metadata.
    Uses FilesystemStore to store blob files; maintains an index mapping.
    """

    def __init__(self, base_path: str = './sentence_store', encryption_key: Optional[bytes] = None):
        self.base = pathlib.Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        self.store = FilesystemStore(str(self.base))
        self.enc = EncryptionHelper(encryption_key)
        self.index_path = self.base / 'index.json'
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding='utf-8'))
        else:
            self._index = {}

    def save_sentence(self, sid: str, sentence: str, source: str = 'user', validated_by: Optional[str] = None) -> str:
        norm = normalize_sentence(sentence)
        encrypted = self.enc.encrypt_text(norm)
        path = self.base / f'sent_{sid}.bin'
        path.write_bytes(encrypted)
        self._index[sid] = {'path': str(path), 'source': source, 'validated_by': validated_by}
        self.index_path.write_text(json.dumps(self._index), encoding='utf-8')
        return str(path)

    def load_sentence(self, sid: str) -> Optional[Dict]:
        meta = self._index.get(sid)
        if not meta:
            return None
        token = pathlib.Path(meta['path']).read_bytes()
        try:
            txt = self.enc.decrypt_text(token)
        except Exception:
            txt = token.decode('utf-8', errors='ignore')
        return {'id': sid, 'text': txt, 'meta': meta}

    def all_sentences(self) -> List[Dict]:
        out = []
        for sid in list(self._index.keys()):
            s = self.load_sentence(sid)
            if s:
                out.append(s)
        return out

    def get_text_list(self) -> List[str]:
        return [s['text'] for s in self.all_sentences()]


# Simple embedding builder (not a neural embedder) to create numeric vectors
# for the transformer pipeline when more advanced embeddings are not available.
# Uses character-level hashing to fixed-size vector (deterministic).

def simple_hash_embedding(text: str, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text):
        idx = (ord(ch) + i) % dim
        vec[idx] += 1.0
    # normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_feature_matrix(sentences: List[str], dim: int = 64) -> np.ndarray:
    if not sentences:
        return np.zeros((0, dim), dtype=np.float32)
    mat = np.vstack([simple_hash_embedding(s, dim=dim) for s in sentences])
    return mat
