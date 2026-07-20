"""
Basic smoke tests for deep learning and RL modules. These tests run quickly and
are skipped if torch is not available to avoid CI failures on minimal runners.
"""
import pytest

try:
    import torch
except Exception:
    torch = None


def test_smoke_deep_learning():
    if torch is None:
        pytest.skip('torch not installed')
    from TradingAI.ai.deep_learning import LMWrapper
    lm = LMWrapper(model_name='distilgpt2')
    out = lm.generate('Hello world')
    assert isinstance(out, str)


def test_smoke_rl_agent():
    if torch is None:
        pytest.skip('torch not installed')
    from TradingAI.ai.rl_agent import PPOAgent
    import numpy as np
    agent = PPOAgent(obs_dim=4, act_dim=1)
    a, logp = agent.select_action(np.zeros(4))
    assert hasattr(a, 'shape') or hasattr(a, '__len__')
