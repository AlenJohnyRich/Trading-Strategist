"""
Tests for DeepSeek client wrapper (mocked / env-aware)
"""
import os
from TradingAI.ai.deepseek import DeepSeekClient


def test_deepseek_disabled_when_no_key():
    # ensure no key in env for test
    old = os.environ.pop('DEEPSEEK_API_KEY', None)
    client = DeepSeekClient()
    assert not client.enabled
    assert client.search('test') == []
    assert client.answer('Q?') == {}
    if old is not None:
        os.environ['DEEPSEEK_API_KEY'] = old
