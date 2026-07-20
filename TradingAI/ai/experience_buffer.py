"""
Experience buffer and replay utilities for RL and supervised learning artifacts.
Supports on-disk persistence for audit and replay, plus simple in-memory sampling.
"""
from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional


class ExperienceBuffer:
    def __init__(self, capacity: int = 100000, persist_dir: Optional[str] = None):
        self.capacity = capacity
        self.buffer: List[Dict[str, Any]] = []
        self.persist_dir = persist_dir
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)

    def add(self, item: Dict[str, Any]):
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        if self.persist_dir:
            # append to file for durability
            with open(os.path.join(self.persist_dir, 'buffer.jsonl'), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(item) + '\n')

    def sample(self, n: int) -> List[Dict[str, Any]]:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def size(self) -> int:
        return len(self.buffer)

    def load_from_file(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    self.add(json.loads(line))
                except Exception:
                    pass

    def clear(self):
        self.buffer = []
