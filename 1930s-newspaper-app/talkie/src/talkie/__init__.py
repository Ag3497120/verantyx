"""Talkie — inference library for Talkie 13B language models."""

import warnings as _warnings

_warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")

__version__ = "0.1.0"

from talkie.chat import Message, format_chat, format_prompt
from talkie.config import MODELS, ModelSpec
from talkie.download import download_model, get_model_files
from talkie.generate import GenerationConfig, GenerationResult, Talkie

__all__ = [
    "__version__",
    "GenerationConfig",
    "GenerationResult",
    "Message",
    "MODELS",
    "ModelSpec",
    "Talkie",
    "download_model",
    "format_chat",
    "format_prompt",
    "get_model_files",
]
