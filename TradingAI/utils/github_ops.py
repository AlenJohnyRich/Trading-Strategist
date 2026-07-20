"""
GitHub operations helper using PyGithub. Uses GITHUB_TOKEN env var or an encrypted token via EncryptionHelper.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from github import Github, Repository, InputGitAuthor

from TradingAI.utils.encryption import EncryptionHelper

logger = logging.getLogger('TradingAI.github')


def get_github_client(encrypted_token: bytes | None = None) -> Github:
    token = None
    if encrypted_token:
        enc = EncryptionHelper()
        token = enc.decrypt_text(encrypted_token)
    else:
        token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('GITHUB_TOKEN not configured')
    return Github(login_or_token=token, per_page=100)


def get_repo(client: Github, full_name: str) -> Repository.Repository:
    return client.get_repo(full_name)


def create_branch(repo: Repository.Repository, new_branch: str, base: str = None) -> str:
    base = base or repo.default_branch
    sb = repo.get_branch(base)
    ref = f"refs/heads/{new_branch}"
    try:
        repo.create_git_ref(ref=ref, sha=sb.commit.sha)
    except Exception:
        # branch may already exist
        logger.debug('Branch %s may already exist', new_branch)
    return new_branch


def commit_file(repo: Repository.Repository, branch: str, path: str, content: str, message: str, author_name: str = 'ai-bot') -> None:
    try:
        existing = repo.get_contents(path, ref=branch)
        repo.update_file(path, message, content, existing.sha, branch=branch)
    except Exception:
        repo.create_file(path, message, content, branch=branch)


def create_pr(repo: Repository.Repository, title: str, head: str, base: str = None, body: str = '') -> str:
    base = base or repo.default_branch
    pr = repo.create_pull(title=title, body=body, head=head, base=base)
    # label PR as ai-generated
    try:
        pr.add_to_labels('ai-generated')
    except Exception:
        pass
    return pr.html_url
