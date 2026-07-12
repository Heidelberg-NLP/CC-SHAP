"""Environment and determinism helpers shared by CC-SHAP.

``load_env`` reads ``.env`` (HF_TOKEN, HF_CACHE) and points the HuggingFace token and
cache-location env vars at it. It must run before ``transformers``/``huggingface_hub``
are imported, since those read ``HF_HOME`` and the token at import time.

``maybe_seed`` makes a run reproducible when ``CCSHAP_SEED`` is set: shap's permutation
explainer shuffles feature orderings with the global NumPy RNG, so the before/after
regression harness sets it to compare the two dependency stacks on equal footing. Unset
means the original stochastic behaviour.
"""
import os
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
ENV_FILE = REPO / ".env"


def _parse_env_file(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env(env_file: Path = ENV_FILE) -> None:
    for key, value in _parse_env_file(env_file).items():
        os.environ.setdefault(key, value)

    token = os.environ.get("HF_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    cache = os.environ.get("HF_CACHE")
    if cache:
        os.environ.setdefault("HF_HOME", cache)


def helper_model_key(default: str) -> str:
    """Which MODELS entry to use as the Lanham-test helper (paraphrase/insert mistakes).

    Defaults to the paper's ``llama2-13b-chat``; the regression harness overrides it via
    ``CCSHAP_HELPER_MODEL`` to keep the smoke test small and token-free.
    """
    return os.environ.get("CCSHAP_HELPER_MODEL", default)


def maybe_seed() -> Optional[int]:
    raw = os.environ.get("CCSHAP_SEED")
    if not raw:
        return None
    seed = int(raw)

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
