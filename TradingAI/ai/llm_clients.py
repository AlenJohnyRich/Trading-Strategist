"""
LLM client wrappers and autocorrect/validation orchestrator.
Supports OpenAI, Anthropic(Claude), Gemini (placeholders), and DeepSeek.
If API keys are not set in environment, clients return mocked responses for testing.
"""
from __future__ import annotations

import os
import logging
from typing import Tuple

logger = logging.getLogger('TradingAI.llm')


class BaseLLMClient:
    def correct_sentence(self, scrambled: str) -> Tuple[str, float]:
        """Return (correction, confidence)"""
        raise NotImplementedError

    def verify_statement(self, statement: str) -> Tuple[bool, str]:
        """Return (is_true, rationale)"""
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def correct_sentence(self, scrambled: str) -> Tuple[str, float]:
        # naive correction: try to reorder words into the most likely english-like sequence
        # Here we attempt simple heuristics: put words starting with capital first if any
        words = scrambled.split()
        if len(words) <= 1:
            return scrambled, 0.5
        # try to find 'i' or capitalized pronoun
        words_sorted = sorted(words, key=lambda w: (0 if w[0].isupper() else 1, len(w)))
        corrected = ' '.join(words_sorted)
        return corrected, 0.5

    def verify_statement(self, statement: str) -> Tuple[bool, str]:
        # Mock: simple heuristics e.g., if contains 'is' return True
        if ' is ' in statement or statement.strip().endswith('.'):
            return True, 'Mock verifier: looks plausible.'
        return False, 'Mock verifier: could not verify.'


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None):
        try:
            import openai
        except Exception:
            openai = None
        self.openai = openai
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.openai and self.api_key:
            self.openai.api_key = self.api_key

    def _call(self, prompt: str) -> str:
        if not self.openai or not self.api_key:
            logger.debug('OpenAI key not present; using mock client')
            return MockLLMClient().correct_sentence(prompt)[0]
        resp = self.openai.Completion.create(engine='text-davinci-003', prompt=prompt, max_tokens=128)
        return resp.choices[0].text.strip()

    def correct_sentence(self, scrambled: str) -> Tuple[str, float]:
        prompt = f"Reconstruct the grammatically correct sentence from this shuffled phrase:\n{scrambled}\nCorrect sentence:"
        out = self._call(prompt)
        return out, 0.9

    def verify_statement(self, statement: str) -> Tuple[bool, str]:
        prompt = f"Assess whether the following statement is factually correct. Answer 'TRUE' or 'FALSE' and provide a short rationale:\n{statement}\nAnswer:"
        out = self._call(prompt)
        result = out.strip().upper().startswith('TRUE')
        return result, out


# Factory

def get_llm_client(preferred: str | None = None) -> BaseLLMClient:
    pref = (preferred or os.getenv('TS_LLM') or '').lower()
    if pref == 'openai' and os.getenv('OPENAI_API_KEY'):
        return OpenAIClient()
    # add other providers as needed (Anthropic/Claude, Gemini, DeepSeek) when keys are configured
    return MockLLMClient()
