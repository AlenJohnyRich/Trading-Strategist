"""
PDF / Image / Video ingest utilities.
Supports PDF text extraction (PyMuPDF), image OCR (pytesseract), and placeholder for video transcript extraction.
Chunks extracted text for embedding and stores metadata paths.
"""
from __future__ import annotations

import os
import pathlib
import uuid
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('TradingAI.ingest')

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        out.append(' '.join(chunk))
        i += chunk_size - overlap
    return out


class Ingestor:
    def __init__(self, base_store: str = './data_store'):
        self.base = pathlib.Path(base_store)
        self.base.mkdir(parents=True, exist_ok=True)

    def _extract_pdf(self, path: str) -> str:
        if fitz is None:
            raise RuntimeError('PyMuPDF (fitz) is required for PDF ingestion')
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text())
        return '\n'.join(texts)

    def _extract_image(self, path: str) -> str:
        if Image is None or pytesseract is None:
            raise RuntimeError('Pillow and pytesseract are required for image OCR')
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text

    def _extract_video(self, path: str) -> str:
        # Placeholder: in production use speech-to-text (whisper or cloud STT)
        # For now, raise to indicate not implemented
        raise NotImplementedError('Video transcript extraction not implemented in this build')

    def ingest_file(self, path: str, source: str = 'user', chunk_size: int = 512) -> Dict:
        p = pathlib.Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        suffix = p.suffix.lower()
        text = ''
        if suffix in ('.pdf',):
            text = self._extract_pdf(str(p))
        elif suffix in ('.png', '.jpg', '.jpeg', '.tiff'):
            text = self._extract_image(str(p))
        elif suffix in ('.mp4', '.wav', '.m4a'):
            text = self._extract_video(str(p))
        else:
            # fallback: read as text
            text = p.read_text(encoding='utf-8', errors='ignore')

        chunks = chunk_text(text, chunk_size=chunk_size)
        id = uuid.uuid4().hex
        # save extracted chunks to store
        out_dir = self.base / id
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(chunks):
            (out_dir / f'chunk_{i}.txt').write_text(c, encoding='utf-8')
        meta = {'id': id, 'source': source, 'file': str(p), 'chunks': len(chunks)}
        (out_dir / 'meta.json').write_text(str(meta), encoding='utf-8')
        logger.info('Ingested %s -> %s chunks=%s', path, id, len(chunks))
        return meta

    def get_chunks(self, ingest_id: str) -> List[Dict]:
        out_dir = self.base / ingest_id
        if not out_dir.exists():
            return []
        chunks = []
        for f in sorted(out_dir.glob('chunk_*.txt')):
            chunks.append({'id': f.stem, 'text': f.read_text(encoding='utf-8')})
        return chunks
