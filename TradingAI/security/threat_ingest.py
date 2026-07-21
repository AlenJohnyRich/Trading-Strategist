"""
Threat intelligence ingest utilities

This module provides parsers/ingestors for local TTP/CVE/indicator files. It
intentionally avoids making external HTTP calls — instead it expects local files
or pre-fetched feeds. For integration with online feeds, wrap these functions in
a secure ingest pipeline that runs in an isolated environment and validates inputs.
"""
from __future__ import annotations

import json
from typing import List, Dict, Any
from TradingAI.security.cyber_kb import CyberKnowledgeBase


class ThreatIngestor:
    def __init__(self, kb: CyberKnowledgeBase):
        self.kb = kb

    def parse_cve_file(self, path: str) -> int:
        """Parse a local CVE JSON file and add entries to KB. Returns count added."""
        return self.kb.load_cve_json(path)

    def parse_stix_json(self, path: str) -> int:
        """Parse a local STIX-style JSON and extract indicators and CVE-like artifacts.
        This is a best-effort extractor for offline STIX bundles.
        """
        added = 0
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            objects = data.get('objects') or data.get('items') or []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if obj.get('type') == 'vulnerability' or 'cve' in obj.get('name','').lower():
                    # convert to minimal cve record
                    rec = {'id': obj.get('id') or obj.get('name'), 'summary': obj.get('description',''), 'product': obj.get('affected_products', [])}
                    try:
                        self.kb.add_cve(rec)
                        added += 1
                    except Exception:
                        pass
        return added

    def load_simple_indicators(self, path: str) -> List[Dict[str, Any]]:
        """Load a newline-delimited JSON list of indicators. Returns parsed list.
        Not added to KB by default; caller may add or index them.
        """
        out = []
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
