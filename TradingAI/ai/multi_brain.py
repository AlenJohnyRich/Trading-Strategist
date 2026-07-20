"""
Multi-brain orchestrator: route inputs to specialist 'brains' and aggregate outputs.
Brains are lightweight modules implementing a simple interface: process(input)->output.
A routing controller selects which brains to activate based on a small router network.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any


class BrainInterface:
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class Router(nn.Module):
    def __init__(self, input_dim: int, num_brains: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_brains))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class MultiBrainController:
    def __init__(self, brains: List[BrainInterface], router: Router):
        self.brains = brains
        self.router = router

    def decide_and_run(self, context_vector: torch.Tensor, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # context_vector: torch tensor (1, input_dim)
        weights = self.router(context_vector).squeeze(0).detach().cpu().numpy()
        # run all brains in parallel (or selectively) and weight their outputs
        outputs = []
        for b in self.brains:
            out = b.process(inputs)
            outputs.append(out)
        # naive aggregation: weighted sum for numeric fields, choose highest weight for text
        agg = {'text_candidates': [], 'numeric': {}}
        for w, o in zip(weights, outputs):
            if 'text' in o:
                agg['text_candidates'].append({'text': o['text'], 'weight': float(w)})
            if 'numeric' in o:
                for k, v in o['numeric'].items():
                    agg['numeric'].setdefault(k, 0.0)
                    agg['numeric'][k] += float(w) * float(v)
        # pick text from highest-weight brain
        if agg['text_candidates']:
            best = max(agg['text_candidates'], key=lambda x: x['weight'])
            agg['text'] = best['text']
        return agg
