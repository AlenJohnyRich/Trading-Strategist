"""
Unit tests for basic cybersecurity modules
"""
from TradingAI.security.cyber_kb import CyberKnowledgeBase
from TradingAI.security.cyber_scanner import CyberScanner
from TradingAI.security.cyber_policies import PolicyEngine


def test_kb_add_and_query():
    kb = CyberKnowledgeBase()
    kb.add_cve({'id':'CVE-2026-0001', 'summary':'Example RCE in package', 'product':['example/pkg']})
    res = kb.query_by_cve('CVE-2026-0001')
    assert res is not None
    hits = kb.search_keywords('RCE package')
    assert any('CVE-2026-0001' in (h.get('id') or h.get('cve') or '') for h in hits)


def test_scanner_stub():
    scanner = CyberScanner()
    findings = scanner.analyze_code('nonexistent_path_for_test')
    assert isinstance(findings, list)


def test_policy_engine():
    engine = PolicyEngine()
    # craft a fake bandit finding about exec usage
    fake = [{'tool':'bandit', 'test_name':'use_of_exec', 'message':'Possible use of exec() in file x.py', 'file':'x.py'}]
    report = engine.evaluate_findings(fake)
    assert not report['passed']
    assert any(v['policy']=='no-exec' for v in report['violations'])
