import json
import random
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_config(config: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    return float(accuracy_score(labels, preds))


def compute_f1(preds: Sequence[int], labels: Sequence[int]) -> float:
    return float(f1_score(labels, preds))


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def get_max_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


MODEL_README_TEXT = (
    "If automatic downloads fail, place distilbert-base-uncased under models/distilbert-base-uncased/ "
    "and set TRANSFORMERS_CACHE=models/ before rerunning `make` commands."
)


def ensure_models_readme() -> None:
    readme_path = Path("models") / "README.md"
    if readme_path.exists():
        return
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(MODEL_README_TEXT + "\n", encoding="utf-8")
