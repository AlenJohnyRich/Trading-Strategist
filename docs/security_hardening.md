# Security hardening and governance summary

This document summarizes the enforced security and governance rules for the AI system.

- Generated code NEVER merges automatically to protected branches. The AI creates draft PRs labeled `ai-generated`.
- Admin review is required before any merge. Branch protections must be enabled on `main`/`staging`.
- Sandbox runs have network disabled by default. Only pre-approved endpoints may be whitelisted.
- Pre-execution checks: static analysis, SAST rules, dependency vulnerability scans, malware scan.
- Secrets are never stored in plaintext in the repo. Use ENCRYPTION_KEY + secret manager.
- Audit logs record all agent actions: who/what generated code, scan outputs, test results, sandbox logs.
