# Cybersecurity features documentation

This document explains the new cybersecurity knowledge and analysis features
added to the Trading-Strategist repo under TradingAI/security/.

Files added
-----------
- TradingAI/security/cyber_kb.py        : in-memory cybersecurity knowledge base (CVE-like records)
- TradingAI/security/threat_ingest.py   : offline parsers for CVE and STIX JSON files
- TradingAI/security/cyber_scanner.py   : wrapper around static scanners (bandit/semgrep/safety)
- TradingAI/security/cyber_policies.py  : simple policy engine and summary utilities

Design goals
------------
- Offline-first: the ingestors and KB do not make network calls. Feeds must be
  fetched by an operator and placed into the sandbox for ingestion.
- Auditability: every detected finding, ingestion event, and policy evaluation
  can be logged and stored in the system's audit trail.
- Safe-by-default: if external tools (bandit/semgrep/safety) are not installed,
  the scanner returns informative guidance rather than failing.

Usage examples
--------------
1) Ingest a local CVE JSON file into the KB:

```python
from TradingAI.security.cyber_kb import CyberKnowledgeBase
from TradingAI.security.threat_ingest import ThreatIngestor
kb = CyberKnowledgeBase()
ing = ThreatIngestor(kb)
ing.parse_cve_file('local-cve-feed.json')
print('CVE count:', len(list(kb.list_all_cves())))
```

2) Run static analysis on a code directory and evaluate policies:

```python
from TradingAI.security.cyber_scanner import CyberScanner
from TradingAI.security.cyber_policies import PolicyEngine
scanner = CyberScanner()
findings = scanner.analyze_code('TradingAI/')
engine = PolicyEngine()
report = engine.evaluate_findings(findings)
print(engine.summarize_report(report))
```

Operational notes
-----------------
- For production ingestion from threat intel feeds (CVE, NVD, STIX), run the
  ingest pipeline in a restricted environment, validate inputs and store
  artifacts in the encrypted file store before indexing into the KB.
- Integrate scanner outputs into the self-code approval pipeline so generated
  code is rejected if critical policy violations are detected.
- Maintain a regular CVE feed update job that fetches and stores the latest
  NVD/CVE JSON in the secure ingest location; the pipeline can then run the
  ThreatIngestor against those files to keep the KB current.
