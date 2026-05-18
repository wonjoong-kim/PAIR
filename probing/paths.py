"""Path resolution for the PAIR codebase.

Paths default to the repo layout shipped with the official release:

    official_github/
    ├── data/{dataset}/<split>.json
    ├── data/features/{model}/{dataset}/{split}/features_*.npz
    ├── data/models/methods/PAIR/{model}/{dataset}/pair_{train_mode}.pkl
    └── data/models/methods/PAIR_NEW/{model}/{dataset}/pair_new_{train_mode}.pkl

Override the project root with the `PAIR_ROOT` environment variable when
data lives elsewhere (e.g. on a shared scratch volume).
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PAIR_ROOT",
    str(Path(__file__).resolve().parents[1]),
))

DATA_ROOT = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_ROOT / "features"
MODELS_DIR = DATA_ROOT / "models" / "methods"
PAIR_DIR = MODELS_DIR / "PAIR"
PAIR_NEW_DIR = MODELS_DIR / "PAIR_NEW"
RESULTS_DIR = DATA_ROOT / "results"

# Per-dataset raw dialog files (clean_train / contaminated_train / clean_test
# / contaminated_test, plus their `matched_*` balanced counterparts).
DATASET_DIRS = {
    "gta": DATA_ROOT / "gta",
    "toolbench": DATA_ROOT / "toolbench",
}


def model_path(model_name: str) -> str:
    """Resolve a short model alias (llama8b / qwen7b / mistral7b) to its
    HF identifier or local checkpoint path. Override with env vars
    `PAIR_MODEL_LLAMA8B`, `PAIR_MODEL_QWEN7B`, `PAIR_MODEL_MISTRAL7B`."""
    defaults = {
        "llama8b": "meta-llama/Meta-Llama-3-8B-Instruct",
        "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
        "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    }
    env_key = f"PAIR_MODEL_{model_name.upper()}"
    return os.environ.get(env_key, defaults.get(model_name, model_name))
