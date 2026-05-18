"""Path resolution mirroring `probing/paths.py`. See that file for details."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PAIR_ROOT",
    str(Path(__file__).resolve().parents[1]),
))

DATA_ROOT = PROJECT_ROOT / "data"
PAIR_PROBE_DIR = DATA_ROOT / "models" / "methods" / "PAIR"

DATASET_DIRS = {
    "gta": DATA_ROOT / "gta",
    "toolbench": DATA_ROOT / "toolbench",
}

RUNS_DIR = PROJECT_ROOT / "runs"


def model_path(model_name: str) -> str:
    defaults = {
        "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
        "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    env_key = f"PAIR_MODEL_{model_name.upper()}"
    return os.environ.get(env_key, defaults.get(model_name, model_name))
