"""
NVIDIA integration helpers.
Provide utilities to detect NVIDIA GPU availability and optional integration
points for NVIDIA NeMo or TensorRT acceleration. These are light wrappers that
attempt to import the necessary libraries and fall back gracefully if not
available.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger('TradingAI.nvidia')


def is_nvidia_available() -> bool:
    """Quick check for CUDA driver or nvidia-smi presence."""
    try:
        if shutil.which('nvidia-smi'):
            return True
    except Exception:
        pass
    # try torch.cuda
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def try_load_nemo():
    """Attempt to import NVIDIA NeMo (if installed) and return module or None."""
    try:
        import nemo
        import nemo.collections.nlp as nemo_nlp
        return nemo_nlp
    except Exception as exc:
        logger.debug('NeMo not available: %s', exc)
        return None


def try_tensorrt_compile(model, sample_input):
    """Attempt to compile a PyTorch model with torch-tensorrt if available.
    Returns compiled model or raises informative error if TensorRT unavailable.
    This is a best-effort helper — real deployments require careful tuning.
    """
    try:
        import torch_tensorrt
        compiled = torch_tensorrt.compile(model, inputs=[sample_input], enabled_precisions={torch_tensorrt.dtype.float16})
        return compiled
    except Exception as exc:
        logger.debug('TensorRT compile failed or not available: %s', exc)
        raise
