"""
DeepSeek API integration wrapper.
Provides a simple client for calling DeepSeek's search and answer endpoints.
Environment variables:
 - DEEPSEEK_API_KEY : your DeepSeek API key
 - DEEPSEEK_API_URL : optional base URL for DeepSeek API (default is placeholder)

This module is defensive: if requests or API key is not available it will disable itself
and provide helpful messages rather than raising on import.
"""
from __future__ import annotations

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('TradingAI.deepseek')

try:
    import requests
except Exception:
    requests = None


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.api_url = api_url or os.getenv('DEEPSEEK_API_URL') or 'https://api.deepseek.example/v1'
        self.enabled = bool(self.api_key) and (requests is not None)
        if not self.enabled:
            logger.info('DeepSeekClient disabled: requests installed=%s, api_key present=%s', requests is not None, bool(self.api_key))

    def _headers(self) -> Dict[str, str]:
        return {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform a DeepSeek search and return a list of results (dicts).
        Returns an empty list if client is disabled.
        """
        if not self.enabled:
            return []
        try:
            url = f"{self.api_url}/search"
            payload = {'query': query, 'top_k': top_k}
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # expect data['results'] or fallback
            return data.get('results') if isinstance(data, dict) else []
        except Exception as exc:
            logger.debug('DeepSeek search error: %s', exc)
            return []

    def answer(self, question: str, context: Optional[str] = None, max_tokens: int = 256) -> Dict[str, Any]:
        """Ask DeepSeek to answer a question using optional context. Returns dict or {}."""
        if not self.enabled:
            return {}
        try:
            url = f"{self.api_url}/answer"
            payload = {'question': question, 'context': context, 'max_tokens': max_tokens}
            resp = requests.post(url, json=payload, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except Exception as exc:
            logger.debug('DeepSeek answer error: %s', exc)
            return {}
