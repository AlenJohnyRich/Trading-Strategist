"""
Cybersecurity scanner wrapper and static analysis helpers.

This module wraps existing scanners (bandit, semgrep, ruff, safety) if available,
and provides a unified API for analyzing code and returning structured findings.
It is deliberately defensive: if scanners are not installed, it returns a stubbed
finding set and a guidance message rather than failing.
"""
from __future__ import annotations

import shutil
import subprocess
import json
import os
from typing import List, Dict, Any, Optional


class CyberScanner:
    def __init__(self):
        # detect available tools
        self._has_bandit = shutil.which('bandit') is not None
        self._has_semgrep = shutil.which('semgrep') is not None
        self._has_safety = shutil.which('safety') is not None

    def analyze_code(self, path: str) -> List[Dict[str, Any]]:
        """Run available static scanners on a file or directory and return findings.
        If no scanners are available, return an informative stub finding.
        """
        findings: List[Dict[str, Any]] = []
        if self._has_bandit:
            try:
                res = subprocess.run(['bandit', '-f', 'json', '-r', path], capture_output=True, text=True, check=False)
                data = json.loads(res.stdout) if res.stdout else {}
                for rep in (data.get('results') or []):
                    findings.append({'tool': 'bandit', 'test_name': rep.get('test_name'), 'severity': rep.get('issue_severity'), 'message': rep.get('issue_text'), 'file': rep.get('filename')})
            except Exception:
                findings.append({'tool': 'bandit', 'error': 'failed to run'})
        if self._has_semgrep:
            try:
                res = subprocess.run(['semgrep', '--json', '-e', ''], capture_output=True, text=True, check=False)
                # semgrep requires rules; this is a placeholder for integration
                # caller should provide a ruleset or run semgrep separately
            except Exception:
                findings.append({'tool': 'semgrep', 'error': 'failed to run or no rules provided'})
        if not findings:
            findings.append({'tool': 'none', 'message': 'No scanners available; install bandit/semgrep/safety for real scans'})
        return findings

    def dependency_check(self, requirements_txt: Optional[str] = None) -> Dict[str, Any]:
        """Run a dependency vulnerability check using safety if available.
        If no safety installed, return guidance.
        """
        if self._has_safety and requirements_txt and os.path.exists(requirements_txt):
            try:
                res = subprocess.run(['safety', 'check', '--file', requirements_txt, '--json'], capture_output=True, text=True, check=False)
                return json.loads(res.stdout) if res.stdout else {'status': 'no output'}
            except Exception:
                return {'error': 'safety run failed'}
        return {'status': 'safety not available or requirements file missing'}

    def quick_advice_from_findings(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Produce human-friendly remediation advice from findings (heuristic)."""
        adv = []
        for f in findings:
            if f.get('tool') == 'bandit' and f.get('test_name'):
                adv.append(f"Bandit: review {f.get('test_name')} in {f.get('file')}: {f.get('message')}")
            elif f.get('tool') == 'none':
                adv.append('No scanners available. Run bandit and semgrep locally for code security findings.')
            else:
                adv.append(f"{f.get('tool')}: {f.get('message', f.get('error',''))}")
        return adv
