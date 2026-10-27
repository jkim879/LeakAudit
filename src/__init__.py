from __future__ import annotations
import json
from pathlib import Path

from . import config, io_utils, features, evaluation, sfb, diagnostics, reporting

__all__ = ["config", "io_utils", "features", "evaluation", "sfb",
           "diagnostics", "reporting", "save_json", "load_json"]


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def save_json(obj, name) -> Path:
    """Write a result dictionary to ``results/<name>`` as JSON."""
    path = config.RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    return path


def load_json(name):
    """Read a JSON result written by :func:`save_json`."""
    with open(config.RESULTS / name, encoding="utf-8") as f:
        return json.load(f)
