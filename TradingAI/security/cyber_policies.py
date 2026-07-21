"""
Cybersecurity policy engine

Defines and evaluates simple security policies for code artifacts and
ingested intelligence. Policies are expressed as rules that inspect findings
from scanners and determine pass/fail and severity counts.
"""
from __future__ import annotations

from typing import List, Dict, Any


DEFAULT_POLICIES = [
    {'id': 'no-raw-sql', 'description': 'Flag raw SQL string concatenation usage', 'severity': 'high'},
    {'id': 'no-exec', 'description': 'Avoid use of exec/eval', 'severity': 'high'},
    {'id': 'dependency-vuln', 'description': 'Block known vulnerable dependency versions', 'severity': 'critical'},
]


class PolicyEngine:
    def __init__(self, policies: List[Dict[str, Any]] = None):
        self.policies = policies or DEFAULT_POLICIES

    def evaluate_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate scanner findings against policy rules and return report."""
        report = {'passed': True, 'violations': []}
        for f in findings:
            # simple heuristics
            msg = f.get('message','').lower()
            if 'exec(' in msg or 'eval(' in msg or f.get('test_name','').lower().startswith('exec'):
                report['violations'].append({'policy': 'no-exec', 'finding': f, 'severity': 'high'})
            if 'sql' in msg and 'execute' in msg:
                report['violations'].append({'policy': 'no-raw-sql', 'finding': f, 'severity': 'high'})
            if f.get('tool') == 'safety' and f.get('vulnerability'):
                report['violations'].append({'policy': 'dependency-vuln', 'finding': f, 'severity': 'critical'})
        if report['violations']:
            report['passed'] = False
        return report

    def summarize_report(self, report: Dict[str, Any]) -> str:
        if report['passed']:
            return 'All policies passed.'
        lines = [f"Policy violations: {len(report['violations'])}"]
        for v in report['violations']:
            lines.append(f" - {v['policy']} ({v.get('severity')}): {v['finding'].get('message', v['finding'])}")
        return '\n'.join(lines)
