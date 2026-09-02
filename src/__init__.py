"""
LeakAudit -- an empirical audit of evaluation validity in acoustic leak
detection for water distribution networks.

Public sub-modules
------------------
config      Paths, constants and plotting defaults.
io_utils    Dataset discovery and loaders for D1, D2, D3.
features    2023 acoustic pipeline (band-pass, window selection, mel features).
evaluation  Models, split protocols, metrics and conformal prediction.

The notebooks under ``notebooks/`` call into these modules so that every
experiment shares one tested implementation.
"""
from __future__ import annotations
import json
from pathlib import Path

from . import config, io_utils, features, evaluation, sfb

__all__ = ["config", "io_utils", "features", "evaluation", "sfb",
           "save_json", "load_json"]


def save_json(obj, name) -> Path:
    """Write a result dictionary to ``results/<name>`` as JSON."""
    path = config.RESULTS / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    return path


def load_json(name):
    """Read a JSON result written by :func:`save_json`."""
    with open(config.RESULTS / name, encoding="utf-8") as f:
        return json.load(f)
