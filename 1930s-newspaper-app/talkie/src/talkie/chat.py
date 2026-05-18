"""Chat template formatting for the instruction-tuned (IT) model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    """A single chat message."""

    role: Literal["user", "assistant", "system"]
    content: str


def format_chat(messages: list[Message]) -> str:
    """Format a multi-turn conversation into the simple chat template.

    The template looks like::

        <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|><|user|>...<|end|><|assistant|>
    """
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>{msg.content}<|end|>")
        elif msg.role == "user":
            parts.append(f"<|user|>{msg.content}<|end|>")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>{msg.content}<|end|>")
    # End with the assistant turn marker so the model generates the reply.
    parts.append("<|assistant|>")
    return "".join(parts)


def format_prompt(prompt: str) -> str:
    """Convenience: wrap a single user message in the chat template."""
    return f"<|user|>{prompt}<|end|><|assistant|>"


# Strings that indicate the model is starting a new turn rather than
# continuing the assistant reply.
STOP_STRINGS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|end|>",
    "<|endoftext|>",
)
STOP_WINDOW = max(len(s) for s in STOP_STRINGS)


def truncate_at_stop(text: str) -> tuple[str, bool]:
    """Cut *text* at the first chat-template marker.

    Returns ``(truncated_text, was_truncated)``.
    """
    positions = [text.find(s) for s in STOP_STRINGS if s in text]
    if not positions:
        return text, False
    stop_at = min(positions)
    return text[:stop_at], True
