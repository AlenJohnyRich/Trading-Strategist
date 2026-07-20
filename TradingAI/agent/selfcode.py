"""
Self-code agent orchestration: generate code via LLM, run security scans, run tests in sandbox,
commit to a new branch and open a PR (draft). The agent uses GitHub token from env.
"""
from __future__ import annotations

import os
import tempfile
import logging
import uuid
from typing import Optional

from TradingAI.ai.llm_clients import get_llm_client
from TradingAI.utils.github_ops import get_github_client, get_repo, create_branch, commit_file, create_pr
from TradingAI.security.scanner import run_bandit, run_semgrep, run_clamav
from TradingAI.sandbox.runner import run_code_in_sandbox

logger = logging.getLogger('TradingAI.selfcode')


class SelfCodeAgent:
    def __init__(self, github_repo: str, base_branch: Optional[str] = None, github_client_encrypted_token: Optional[bytes] = None):
        self.github_repo = github_repo
        self.base_branch = base_branch
        self.gh = get_github_client(github_client_encrypted_token)
        self.repo = get_repo(self.gh, github_repo)
        self.llm = get_llm_client()

    def generate_code(self, prompt: str) -> str:
        # Use underlying LLM to generate code. For OpenAI client the _call method exists.
        # Fall back to Mock client behavior if provider not configured.
        client = self.llm
        code = ''
        if hasattr(client, '_call'):
            # instruct model to provide only code block
            full_prompt = f"Implement the following Python function or module. Provide only code in your response.\n\n{prompt}"
            try:
                code = client._call(full_prompt)
            except Exception:
                code = '# failed to generate code'
        else:
            code = '# mock code generated\nprint("hello from self-code agent")\n'
        return code

    def run_security_scans(self, path: str) -> dict:
        results = {}
        results['bandit'] = run_bandit(path)
        results['semgrep'] = run_semgrep(path)
        results['clamav'] = run_clamav(path)
        return results

    def run_tests_in_sandbox(self, path: str) -> dict:
        # run pytest in sandbox
        rc, out = run_code_in_sandbox(path, ['pytest -q'], timeout=120)
        return {'returncode': rc, 'output': out}

    def submit_task(self, prompt: str, target_file: str = 'generated.py', pr_branch_prefix: str = 'ai-gen') -> dict:
        # generate code
        code = self.generate_code(prompt)
        # create temp workspace
        tid = uuid.uuid4().hex[:8]
        branch = f"{pr_branch_prefix}/{tid}"
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, target_file)
            with open(p, 'w', encoding='utf-8') as fh:
                fh.write(code)
            # run security scans
            scans = self.run_security_scans(td)
            # run tests
            tests = self.run_tests_in_sandbox(td)
            # commit to branch
            create_branch(self.repo, branch, base=self.base_branch)
            with open(p, 'r', encoding='utf-8') as fh:
                content = fh.read()
            commit_file(self.repo, branch, target_file, content, f"ai: add {target_file}")
            pr_url = create_pr(self.repo, f"AI-generated: {target_file}", head=branch, base=self.base_branch, body=f"Scans: {scans}\nTests: {tests}")
        return {'branch': branch, 'pr_url': pr_url, 'scans': scans, 'tests': tests}
