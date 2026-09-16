from __future__ import annotations

from pathlib import Path


def resolve_local_model_path(model: str | Path) -> Path:
    path = Path(model)
    if (path / "config.json").exists():
        return path
    if "/" not in str(model):
        return path
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(model), local_files_only=True))
    except Exception as exc:
        raise FileNotFoundError(
            f"model {model!r} is not a local directory and is not available in the HuggingFace cache"
        ) from exc
