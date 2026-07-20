"""
Extend transformer pipeline to accept memory context and save hidden activations.
Requires torch; if not available methods will raise informative errors.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass
class TransformerConfig:
    input_dim: int = 4
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    seq_len: int = 64
    device: str = 'cpu'


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len: int = 64):
        if torch is None:
            raise RuntimeError('torch not installed — install torch to use TransformerPipeline')
        import numpy as np
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.seq_len = seq_len
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('X and y must have the same length along time axis')

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        seq = self.X[start:end]
        target = self.y[end]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class SimpleTransformerModel(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        if torch is None:
            raise RuntimeError('torch not installed — install torch to use SimpleTransformerModel')
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead,
                                                   dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(cfg.d_model, 1)

    def forward(self, x, memory: Optional[torch.Tensor] = None, return_hidden: bool = False):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch, seq_len, d_model)
        x = x.permute(1, 0, 2)  # -> (seq_len, batch, d_model) for transformer
        out_enc = self.encoder(x)
        # incorporate memory by simple concatenation on the sequence axis if provided
        if memory is not None:
            # memory: (mem_seq_len, batch, d_model)
            try:
                out_enc = torch.cat([memory, out_enc], dim=0)
            except Exception:
                pass
        x2 = out_enc.permute(1, 2, 0)  # -> (batch, d_model, seq_len+mem)
        x2 = self.pool(x2).squeeze(-1)  # -> (batch, d_model)
        out = self.out(x2).squeeze(-1)
        if return_hidden:
            return out, x2
        return out


class TransformerPipeline:
    def __init__(self, cfg: TransformerConfig):
        if torch is None:
            raise RuntimeError('torch not installed — install torch to use TransformerPipeline')
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu')
        self.model = SimpleTransformerModel(cfg).to(self.device)

    def train(self, X, y, epochs: int = 10, batch_size: int = 64, lr: float = 1e-3, save_path: Optional[str] = None, memory: Optional[List[List[float]]] = None):
        dataset = TimeSeriesDataset(X, y, seq_len=self.cfg.seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(dataset) if len(dataset) > 0 else float('nan')
            print(f"Epoch {ep}/{epochs} avg_loss={avg_loss:.6f}")
        if save_path:
            self.save(save_path)

    def predict(self, X):
        self.model.eval()
        import numpy as np
        xs = []
        for i in range(0, max(0, len(X) - self.cfg.seq_len)):
            seq = X[i:i + self.cfg.seq_len]
            xs.append(seq)
        if not xs:
            return []
        xb = torch.tensor(xs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(xb)
        return out.cpu().numpy()

    def predict_with_memory(self, X, memory_embeddings: Optional[List[List[float]]] = None):
        """Run prediction using memory embeddings provided as list of vectors.
        Memory embeddings are projected into the model d_model space and concatenated.
        Returns predictions and optionally hidden activations.
        """
        self.model.eval()
        import numpy as np
        xs = []
        for i in range(0, max(0, len(X) - self.cfg.seq_len)):
            seq = X[i:i + self.cfg.seq_len]
            xs.append(seq)
        if not xs:
            return []
        xb = torch.tensor(xs, dtype=torch.float32).to(self.device)
        mem_tensor = None
        if memory_embeddings is not None:
            # convert memory embeddings to tensor shaped (mem_seq_len, batch, d_model)
            mem = torch.tensor(memory_embeddings, dtype=torch.float32).to(self.device)
            # expand for batch
            mem = mem.unsqueeze(1).repeat(1, xb.size(0), 1)
            mem_tensor = mem
        with torch.no_grad():
            out, hidden = self.model(xb, memory=mem_tensor, return_hidden=True)
        return out.cpu().numpy(), hidden.cpu().numpy()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'cfg': self.cfg}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
