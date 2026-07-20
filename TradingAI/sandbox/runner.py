"""
Sandbox runner that executes commands in Docker with strict defaults. Falls back to
local subprocess with timeout when Docker isn't available.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger('TradingAI.sandbox')


def docker_available() -> bool:
    return shutil.which('docker') is not None


def run_in_docker(workdir: str, commands: List[str], image: str = 'python:3.10-slim', timeout: int = 300, network_whitelist: Optional[List[str]] = None) -> Tuple[int, str]:
    # create temporary tar or mount the workdir
    network_flag = '--network=none'
    if network_whitelist:
        # create a docker network or allowlist is complex; for now, we won't allow network
        network_flag = '--network=none'
    docker_cmd = [
        'docker', 'run', '--rm', network_flag, '--memory=1g', '--cpus=0.5', '-v', f"{os.path.abspath(workdir)}:/work:ro", '-w', '/work', image,
        '/bin/sh', '-c', ' && '.join(commands)
    ]
    try:
        proc = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.decode('utf-8', errors='ignore')
    except Exception as exc:
        logger.exception('Docker run failed: %s', exc)
        return 1, str(exc)


def run_locally(workdir: str, commands: List[str], timeout: int = 300) -> Tuple[int, str]:
    # execute commands in a shell in the given workdir with timeout
    joined = ' && '.join(commands)
    try:
        proc = subprocess.run(joined, cwd=workdir, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.decode('utf-8', errors='ignore')
    except Exception as exc:
        logger.exception('Local run failed: %s', exc)
        return 1, str(exc)


def run_code_in_sandbox(workdir: str, commands: List[str], image: str = 'python:3.10-slim', timeout: int = 300, network_whitelist: Optional[List[str]] = None) -> Tuple[int, str]:
    if docker_available():
        return run_in_docker(workdir, commands, image=image, timeout=timeout, network_whitelist=network_whitelist)
    return run_locally(workdir, commands, timeout=timeout)
