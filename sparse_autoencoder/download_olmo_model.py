"""Download OLMo-3-7B-Think directly onto the machine that will run training.

Download the Hugging Face model repository directly to the training machine.
The default destination is under the repository's `models/` folder, matching
the activation collector's default model path.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def env_path(name: str, default: Path) -> Path:
    """Read a path env var, falling back to a repo-root-relative default."""
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


# Anchor defaults to the repository root so this works from any shell directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO_ID = os.environ.get("OLMO_MODEL_REPO_ID", "allenai/Olmo-3-7B-Think")
MODELS_ROOT = env_path("SAE_MODELS_ROOT", REPO_ROOT / "models")
MODEL_DIR = env_path(
    "OLMO_MODEL_PATH",
    env_path("SAE_MODEL_DIR", MODELS_ROOT / "Olmo-3-7B-Think"),
)

# Required files for `AutoTokenizer.from_pretrained` and
# `AutoModelForCausalLM.from_pretrained`. Existing non-empty files allow the
# downloader to reuse the local model snapshot.
REQUIRED_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def is_complete_file(path: Path) -> bool:
    """A file counts as downloaded only if it exists and has real bytes."""
    return path.is_file() and path.stat().st_size > 0


def missing_required_files() -> list[str]:
    """Return the model files that still need to appear in MODEL_DIR."""
    return [filename for filename in REQUIRED_FILES if not is_complete_file(MODEL_DIR / filename)]


def main() -> None:
    """Download the model snapshot if the local model folder is incomplete."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    missing_before = missing_required_files()
    if not missing_before:
        print(f"using existing complete model at {MODEL_DIR}")
        return

    print(f"downloading {MODEL_REPO_ID} into {MODEL_DIR}")
    print(f"missing before download: {missing_before}")

    snapshot_download(
        repo_id=MODEL_REPO_ID,
        local_dir=MODEL_DIR,
    )

    missing_after = missing_required_files()
    if missing_after:
        raise RuntimeError(f"download finished, but these files are still missing: {missing_after}")

    print(f"model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
