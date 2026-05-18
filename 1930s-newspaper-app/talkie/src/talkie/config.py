"""Model registry — known Talkie model variants and their HuggingFace metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Style = Literal["base", "it"]


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a single model variant hosted on HuggingFace."""

    repo_id: str
    checkpoint_filename: str
    vocab_filename: str
    style: Style


MODELS: dict[str, ModelSpec] = {
    "talkie-1930-13b-base": ModelSpec(
        repo_id="talkie-lm/talkie-1930-13b-base",
        checkpoint_filename="final.ckpt",
        vocab_filename="vocab.txt",
        style="base",
    ),
    "talkie-1930-13b-it": ModelSpec(
        repo_id="talkie-lm/talkie-1930-13b-it",
        checkpoint_filename="rl-refined.pt",
        vocab_filename="vocab.txt",
        style="it",
    ),
    "talkie-web-13b-base": ModelSpec(
        repo_id="talkie-lm/talkie-web-13b-base",
        checkpoint_filename="base.ckpt",
        vocab_filename="vocab.txt",
        style="base",
    ),
}
