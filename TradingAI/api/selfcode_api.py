from fastapi import APIRouter, Body
from typing import Dict

from TradingAI.agent.selfcode import SelfCodeAgent

router = APIRouter(prefix='/selfcode', tags=['selfcode'])


@router.post('/submit')
def submit(payload: Dict = Body(...)):
    repo = payload.get('repo')
    prompt = payload.get('prompt')
    file = payload.get('file', 'generated.py')
    base = payload.get('base')
    if not repo or not prompt:
        return {'status': 'error', 'message': 'repo and prompt required'}
    agent = SelfCodeAgent(github_repo=repo, base_branch=base)
    res = agent.submit_task(prompt, target_file=file)
    return {'status': 'ok', 'result': res}


@router.get('/status/{branch}')
def status(branch: str):
    # minimal status: check PR existence
    from TradingAI.utils.github_ops import get_github_client, get_repo
    gh = get_github_client()
    repo = get_repo(gh, branch.split('/', 1)[0])
    prs = [p.html_url for p in repo.get_pulls(state='open') if p.head.ref == branch]
    return {'prs': prs}
