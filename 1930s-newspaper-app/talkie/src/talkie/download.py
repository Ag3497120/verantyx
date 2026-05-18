"""HuggingFace Hub download helpers."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

from talkie.config import MODELS, ModelSpec


def download_model(
    model_name: str, cache_dir: str | Path | None = None
) -> Path:
    """Download all files for *model_name* from HuggingFace Hub.

    Returns the local directory containing the downloaded files.  If the
    files are already cached, this is a fast no-op.
    """
    spec = _resolve_spec(model_name)
    kwargs: dict = {"repo_id": spec.repo_id}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    # Download both checkpoint and vocab — hf_hub_download returns the
    # local file path.  The parent directory is the repo snapshot dir.
    ckpt_path = hf_hub_download(filename=spec.checkpoint_filename, **kwargs)
    hf_hub_download(filename=spec.vocab_filename, **kwargs)
    return Path(ckpt_path).parent


def get_model_files(
    model_name: str, cache_dir: str | Path | None = None
) -> tuple[Path, Path]:
    """Return ``(checkpoint_path, vocab_path)`` for *model_name*.

    Downloads from HuggingFace if not already cached.
    """
    spec = _resolve_spec(model_name)
    kwargs: dict = {"repo_id": spec.repo_id}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    ckpt_path = Path(hf_hub_download(filename=spec.checkpoint_filename, **kwargs))
    vocab_path = Path(hf_hub_download(filename=spec.vocab_filename, **kwargs))
    return ckpt_path, vocab_path


def _resolve_spec(model_name: str) -> ModelSpec:
    if model_name not in MODELS:
        available = ", ".join(sorted(MODELS))
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: {available}"
        )
    return MODELS[model_name]
