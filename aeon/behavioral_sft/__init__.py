"""Safety-balanced behavioral SFT data and validation helpers."""

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_EVAL_PATH",
    "DEFAULT_TRAIN_PATH",
    "DatasetValidationError",
    "ValidationReport",
    "validate_datasets",
]


def __getattr__(name: str) -> Any:
    """Load validator exports lazily so ``python -m ...validator`` stays clean."""

    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(".validator", __name__), name)
