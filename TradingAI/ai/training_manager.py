"""
Training manager orchestrates scheduled training jobs for LM fine-tuning and RL training.
This module provides simple APIs to kick off training runs using available compute and
to persist training metadata.
"""
from __future__ import annotations

import threading
import logging
from typing import Optional, List

from TradingAI.ai.deep_learning import LMWrapper
from TradingAI.ai.rl_agent import PPOAgent
from TradingAI.ai.experience_buffer import ExperienceBuffer

logger = logging.getLogger('TradingAI.training_manager')


class TrainingManager:
    def __init__(self):
        self._threads = []

    def train_lm_async(self, texts: List[str], model_name: str = 'distilgpt2', epochs: int = 1, batch_size: int = 4):
        def _task():
            logger.info('Starting LM fine-tune task')
            lm = LMWrapper(model_name=model_name)
            out = lm.fine_tune(texts, epochs=epochs, batch_size=batch_size)
            logger.info('LM fine-tune complete -> %s', out)

        t = threading.Thread(target=_task, daemon=True)
        t.start()
        self._threads.append(t)
        return True

    def train_ppo_async(self, obs_dim: int, act_dim: int, experience_buffer: ExperienceBuffer, epochs: int = 10):
        def _task():
            logger.info('Starting PPO training task')
            agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)
            # In a real loop, collect episodes and call agent.update() periodically.
            # Here we simulate consuming buffer items as trajectories for update.
            if experience_buffer.size() == 0:
                logger.warning('No experiences in buffer; aborting PPO task')
                return
            samples = experience_buffer.sample(min(10, experience_buffer.size()))
            # Map samples to expected format (very simplified)
            trajectories = []
            for s in samples:
                traj = {'obs': s.get('obs', []), 'acts': s.get('acts', []), 'logp': s.get('logp', []), 'adv': s.get('adv', []), 'ret': s.get('ret', [])}
                trajectories.append(traj)
            agent.update(trajectories)
            logger.info('PPO training task completed')

        t = threading.Thread(target=_task, daemon=True)
        t.start()
        self._threads.append(t)
        return True
