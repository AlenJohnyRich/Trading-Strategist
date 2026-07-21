"""
Cybersecurity Knowledge Base

Provides a lightweight in-memory store for CVE-like records, indicators, and
threat intelligence artifacts. Designed to be offline-first (no network calls)
so it is safe to include in the repository and to run in sandboxed environments.

Usage:
  kb = CyberKnowledgeBase()
  kb.add_cve({'id':'CVE-2026-0001', 'summary':'Example vuln', 'cvss':7.5, 'product':['example/pkg']})
  kb.query_by_cve('CVE-2026-0001')
  kb.search_keywords('remote code execution')

"""
from __future__ import annotations

import json
from typing import Dict, Any, List, Optional, Iterable
import re


class CyberKnowledgeBase:
    def __init__(self):
        # store CVE records by id
        self._cves: Dict[str, Dict[str, Any]] = {}
        # simple inverted index: keyword -> set of CVE ids
        self._index: Dict[str, set] = {}

    def add_cve(self, record: Dict[str, Any]) -> None:
        cid = record.get('id') or record.get('cve') or record.get('CVE')
        if not cid:
            raise ValueError('CVE record must have an id field')
        cid = str(cid).upper()
        self._cves[cid] = record
        # index words from summary and product names
        text = ' '.join([str(record.get('summary','')), ' '.join(record.get('product',[]))])
        tokens = set(re.findall(r"\w+", text.lower()))
        for t in tokens:
            self._index.setdefault(t, set()).add(cid)

    def load_cve_json(self, path: str) -> int:
        """Load local JSON file with a list of CVE-like dicts. Returns number added."""
        added = 0
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                # try to find items
                items = data.get('CVE_Items') or data.get('items') or data.get('results') or []
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for it in items:
                # support both raw cve objects and simplified dicts
                if isinstance(it, dict) and ('id' in it or 'cve' in it or 'CVE' in it):
                    self.add_cve(it)
                    added += 1
        return added

    def query_by_cve(self, cve_id: str) -> Optional[Dict[str, Any]]:
        return self._cves.get(cve_id.upper())

    def search_keywords(self, query: str, top: int = 20) -> List[Dict[str, Any]]:
        toks = set(re.findall(r"\w+", query.lower()))
        hits: Dict[str, int] = {}
        for t in toks:
            for cid in self._index.get(t, []):
                hits[cid] = hits.get(cid, 0) + 1
        # sort by score
        sorted_ids = sorted(hits.items(), key=lambda kv: kv[1], reverse=True)
        return [self._cves[cid] for cid, _ in sorted_ids[:top]]

    def list_all_cves(self) -> Iterable[Dict[str, Any]]:
        return list(self._cves.values())

    def export_index(self) -> Dict[str, Any]:
        return {'cves': list(self._cves.keys()), 'index_terms': list(self._index.keys())}
