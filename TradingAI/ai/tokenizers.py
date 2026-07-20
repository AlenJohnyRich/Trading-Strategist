"""
Tokenizer interface and several implementations.
Provides pluggable tokenizers for words, letters, sentencepiece/BPE and a placeholder
PhiTokenizer that implements "phi engine"-style heuristics when a specialized model
is not available. All tokenizers expose encode/decode and token->letter mappings so
they can be used to construct hidden-layer inputs for models.
"""
from __future__ import annotations

import re
import logging
from typing import List, Sequence, Optional, Dict, Any

logger = logging.getLogger('TradingAI.tokenizers')

WORD_RE = re.compile(r"\w+|[^	\w\s]+")


class TokenizerInterface:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError

    def tokens(self, text: str) -> List[str]:
        raise NotImplementedError


class WordTokenizer(TokenizerInterface):
    def tokens(self, text: str) -> List[str]:
        return WORD_RE.findall(text)

    def encode(self, text: str) -> List[int]:
        # simple hash-based ids (deterministic) — for prototyping only
        toks = self.tokens(text)
        return [abs(hash(t)) % 2**20 for t in toks]

    def decode(self, ids: Sequence[int]) -> str:
        # Not reversible: return placeholder
        return ' '.join([f'<tok_{i}>' for i in ids])


class LetterTokenizer(TokenizerInterface):
    def tokens(self, text: str) -> List[str]:
        return list(text)

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return ''.join([chr(i) for i in ids])


# Optional sentencepiece / BPE tokenizer using HuggingFace tokenizers or sentencepiece
class HFTokenizer(TokenizerInterface):
    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
n        except Exception:  # pragma: no cover
            raise RuntimeError('transformers package required for HFTokenizer')
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def tokens(self, text: str) -> List[str]:
        return self._tok.tokenize(text)

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True)


class PhiTokenizer(TokenizerInterface):
    """
    Placeholder 'Phi' tokenizer. If a real Phi engine/tokenizer model is available
    (e.g., a specific SentencePiece or BPE model), configure it by passing an HF model
    or a sentencepiece model path to `load_phi_model`. Otherwise this implementation
    uses a hybrid heuristic: word tokens plus positional phi-scaling ids to aid the
    hidden-layer mapping.
    """
    def __init__(self, phi_model_path: Optional[str] = None):
        self.phi_model_path = phi_model_path
        self._use_hf = False
        self._hf = None
        if phi_model_path:
            try:
                from transformers import AutoTokenizer
                self._hf = AutoTokenizer.from_pretrained(phi_model_path)
                self._use_hf = True
            except Exception:
                logger.warning('Failed to load phi model tokenizer at %s; falling back to heuristic', phi_model_path)
                self._use_hf = False

    def tokens(self, text: str) -> List[str]:
        if self._use_hf and self._hf:
            return self._hf.tokenize(text)
        # heuristic: words with position markers
        words = WORD_RE.findall(text)
        return [f"{w}|pos{idx}" for idx, w in enumerate(words)]

    def encode(self, text: str) -> List[int]:
        if self._use_hf and self._hf:
            return self._hf.encode(text)
        toks = self.tokens(text)
        # phi-like mapping: hash(word) XOR position-based prime multiplier
        ids = []
        for t in toks:
            word, _, pos = t.partition('|pos')
            p = int(pos) if pos.isdigit() else 0
            phi_id = (abs(hash(word)) % (2**20)) ^ ((p+1) * 2654435761 & 0xFFFFFFFF)
            ids.append(phi_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # not reversible
        return ' '.join([f'<phi_{i}>' for i in ids])


# Factory

def get_tokenizer(kind: str = 'word', **kwargs) -> TokenizerInterface:
    kind = (kind or 'word').lower()
    if kind == 'word':
        return WordTokenizer()
    if kind == 'letter':
        return LetterTokenizer()
    if kind == 'phi':
        return PhiTokenizer(kwargs.get('phi_model_path'))
    if kind == 'hf':
        return HFTokenizer(kwargs.get('model_name'))
    raise ValueError(f'Unknown tokenizer kind: {kind}')
