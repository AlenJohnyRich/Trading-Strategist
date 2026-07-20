"""
Lightweight encryption helper. Uses cryptography.Fernet when available, falls back to base64 encoding with warning.
Provides encrypt_text/decrypt_text to store small secrets or validated sentences.
"""
from __future__ import annotations

import base64
import os
import logging
from typing import Optional

logger = logging.getLogger('TradingAI.encryption')

try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:  # pragma: no cover
    Fernet = None
    InvalidToken = Exception


class EncryptionHelper:
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = os.getenv('ENCRYPTION_KEY')
            if key:
                if isinstance(key, str):
                    key = key.encode()
        if Fernet is None:
            if key is None:
                logger.warning('cryptography not installed and no ENCRYPTION_KEY provided — using base64 fallback (not secure)')
            self._use_fernet = False
            self.key = key
        else:
            self._use_fernet = True
            if key is None:
                # generate ephemeral key (not ideal for persistent storage)
                key = Fernet.generate_key()
            self.fernet = Fernet(key)
            self.key = key

    def encrypt_text(self, plaintext: str) -> bytes:
        if self._use_fernet:
            return self.fernet.encrypt(plaintext.encode('utf-8'))
        # fallback: base64 encode
        return base64.b64encode(plaintext.encode('utf-8'))

    def decrypt_text(self, token: bytes) -> str:
        if self._use_fernet:
            try:
                return self.fernet.decrypt(token).decode('utf-8')
            except InvalidToken as exc:
                raise ValueError('Invalid encryption token') from exc
        # fallback base64
        return base64.b64decode(token).decode('utf-8')
