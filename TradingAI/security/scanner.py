"""
Security scanner wrappers. Each function returns a dict with scan results or raises
helpful errors if the tool is missing. Tools are executed via subprocess; quiet
failures return 'not available'.
"""
from __future__ import annotations

import shutil
import subprocess
import json
import logging
from typing import Dict, Any

logger = logging.getLogger('TradingAI.security')


def _run_cmd(cmd: list[str], timeout: int = 60) -> str:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return proc.stdout.decode('utf-8', errors='ignore')
    except Exception as exc:
        logger.exception('Scanner command failed: %s', exc)
        return ''


def run_bandit(path: str) -> Dict[str, Any]:
    if shutil.which('bandit') is None:
        return {'available': False}
    out = _run_cmd(['bandit', '-r', path, '-f', 'json'])
    try:
        return json.loads(out)
    except Exception:
        return {'output': out}


def run_semgrep(path: str, rules: str | None = None) -> Dict[str, Any]:
    if shutil.which('semgrep') is None:
        return {'available': False}
    cmd = ['semgrep', '--json', '--config', rules or 'p/ci']
    cmd.append(path)
    out = _run_cmd(cmd)
    try:
        return json.loads(out)
    except Exception:
        return {'output': out}


def run_safety(path: str) -> Dict[str, Any]:
    # safety typically analyzes requirements files
    if shutil.which('safety') is None:
        return {'available': False}
    out = _run_cmd(['safety', 'check', '--file', path, '--json'])
    try:
        return json.loads(out)
    except Exception:
        return {'output': out}


def run_trivy(image: str) -> Dict[str, Any]:
    if shutil.which('trivy') is None:
        return {'available': False}
    out = _run_cmd(['trivy', 'image', '--scanners', 'vuln,config', '--format', 'json', image], timeout=300)
    try:
        return json.loads(out)
    except Exception:
        return {'output': out}


def run_clamav(path: str) -> Dict[str, Any]:
    if shutil.which('clamscan') is None:
        return {'available': False}
    out = _run_cmd(['clamscan', '-r', '--no-summary', path])
    return {'output': out}
