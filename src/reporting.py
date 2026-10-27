"""
Number formatting and a local log of key results.

record(): appends a value to results/revision_ledger.jsonl (local, not tracked).
Formats: proportions 3 decimals, percentage points 1 decimal, counts as integers.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

from . import config as cfg

LEDGER = "revision_ledger.jsonl"


def fmt_p(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fmt_pp(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f} pp"


def fmt_n(n):
    return f"{int(n):,}"


def fmt_ci(lo, hi, nd=3):
    return f"[{fmt_p(lo, nd)}, {fmt_p(hi, nd)}]"


def record(key, value, source, comments="", note="", lo=None, hi=None):
    """Append one manuscript number to the ledger (overwritten by key later)."""
    rec = dict(key=key, value=None if value is None else float(value),
               lo=None if lo is None else float(lo),
               hi=None if hi is None else float(hi),
               source=source, comments=comments, note=note,
               quick_mode=cfg.QUICK)
    with open(cfg.RESULTS / LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_ledger():
    p = cfg.RESULTS / LEDGER
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    return df.drop_duplicates("key", keep="last").reset_index(drop=True)


def reset_ledger():
    p = cfg.RESULTS / LEDGER
    if p.exists():
        p.unlink()
