"""
Memory bank module for long-term and specialized memories.
This is a compact but extendable implementation - in real production this would
include persistence, eviction policies, and vector stores (Chroma, Redis, etc.).
"""
from __future__ import annotations

import json
import pathlib
import threading
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    id: str
    type: str
    payload: Dict[str, Any]


class MemoryBank:
    """Simple in-memory memory bank with optional persistence."""

    def __init__(self, path: str | None = None, autosave: bool = True):
        self._lock = threading.RLock()
        self._store: Dict[str, MemoryItem] = {}
        self.path = pathlib.Path(path) if path else None
        self.autosave = autosave
        if self.path and self.path.exists():
            self._load()

    def add(self, item: MemoryItem) -> None:
        with self._lock:
            self._store[item.id] = item
            if self.autosave:
                self._save()

    def get(self, id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._store.get(id)

    def query(self, type: str) -> List[MemoryItem]:
        with self._lock:
            return [it for it in self._store.values() if it.type == type]

    def delete(self, id: str) -> None:
        with self._lock:
            if id in self._store:
                del self._store[id]
                if self.autosave:
                    self._save()

    def _save(self):
        if not self.path:
            return
        data = {k: asdict(v) for k, v in self._store.items()}
        self.path.write_text(json.dumps(data, default=str), encoding='utf-8')

    def _load(self):
        raw = self.path.read_text(encoding='utf-8')
        data = json.loads(raw)
        for k, v in data.items():
            self._store[k] = MemoryItem(id=v['id'], type=v['type'], payload=v['payload'])


# Specialized memories can be thin wrappers around MemoryBank
class TradeMemory:
    def __init__(self, bank: MemoryBank):
        self.bank = bank

    def record_trade(self, trade_id: str, info: dict):
        self.bank.add(MemoryItem(id=trade_id, type='trade', payload=info))


class PatternMemory:
    def __init__(self, bank: MemoryBank):
        self.bank = bank

    def record_pattern(self, pattern_id: str, info: dict):
        self.bank.add(MemoryItem(id=pattern_id, type='pattern', payload=info))


if __name__ == '__main__':
    mb = MemoryBank(path='./memory_store.json', autosave=False)
    mb.add(MemoryItem(id='t1', type='trade', payload={'pnl': 10}))
    print('Trade memories:', mb.query('trade'))
