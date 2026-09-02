"""
The Signal-Free Baseline (SFB) and bootstrap confidence intervals.

Motivation
----------
A benchmark can be partly "solved" without ever opening the signal, purely from
metadata that travels with each record: which sensor produced it, when it was
recorded, or which experimental condition it belongs to. The accuracy reachable
from such metadata alone is the *signal-free baseline* (SFB).

Any acoustic accuracy that does not clearly exceed its SFB is not evidence of
acoustic discrimination. We therefore propose SFB as a mandatory reporting item
for the field, in the same spirit as the majority-class baseline.

The estimator is deliberately weak: a shallow decision tree (or a
majority-class lookup for categorical identifiers), scored under a
group-independent protocol so that the estimator itself cannot memorise the
grouping variable. This makes SFB a conservative lower bound on the confound.

British English spelling is used throughout.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (LeaveOneOut, GroupKFold, StratifiedKFold,
                                     cross_val_score)
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Reported alongside every small-sample result (notably the 140-file Hong Kong
    sample), where the seed-to-seed standard deviation alone understates the
    uncertainty.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(v, len(v), replace=True).mean()
                      for _ in range(n_boot)])
    return dict(mean=float(v.mean()),
                lo=float(np.percentile(means, 100 * alpha / 2)),
                hi=float(np.percentile(means, 100 * (1 - alpha / 2))),
                n=int(len(v)))


def bootstrap_paired_diff(a, b, n_boot=5000, alpha=0.05, seed=0):
    """Bootstrap CI for the paired difference ``a - b`` (same seeds/folds).

    Used to test whether the min-std window heuristic differs from a random
    window, and whether a foundation model collapses more than the baseline.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    rng = np.random.RandomState(seed)
    means = np.array([rng.choice(d, len(d), replace=True).mean()
                      for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return dict(diff=float(d.mean()), lo=lo, hi=hi,
                excludes_zero=bool(lo > 0 or hi < 0), n=int(len(d)))


# --------------------------------------------------------------------------- #
# SFB estimators
# --------------------------------------------------------------------------- #
def sfb_categorical(codes, y, groups=None, n_splits=5):
    """SFB for a categorical identifier (sensor id, site id, logger id, ...).

    The predictor is a majority-class lookup on the identifier. When ``groups``
    is supplied the lookup is fitted out-of-fold, so the score reflects only
    what the identifier carries beyond the held-out group.
    """
    codes = pd.Series(np.asarray(codes)).astype(str).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    global_major = y.value_counts().idxmax()

    if groups is None:
        # In-sample lookup: the ceiling that identity alone can reach.
        lookup = y.groupby(codes).agg(lambda s: s.value_counts().idxmax())
        pred = codes.map(lookup).fillna(global_major)
        return float(accuracy_score(y, pred))

    groups = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        return float("nan")
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(codes, y, groups=groups):
        lookup = y.iloc[tr].groupby(codes.iloc[tr]).agg(
            lambda s: s.value_counts().idxmax())
        pred = codes.iloc[te].map(lookup).fillna(global_major)
        accs.append(accuracy_score(y.iloc[te], pred))
    return float(np.mean(accs))


def sfb_numeric(x, y, groups=None, max_depth=1, n_splits=5):
    """SFB for a numeric metadata feature (recording hour, date ordinal, ...).

    A depth-limited decision tree is scored by leave-one-out when no grouping
    is given, or by group k-fold when it is, so the stump cannot memorise a
    logger or a site.
    """
    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    if groups is None:
        cv = LeaveOneOut() if len(y) <= 400 else StratifiedKFold(5, shuffle=True,
                                                                random_state=0)
        return float(cross_val_score(tree, X, y, cv=cv).mean())
    groups = np.asarray(groups).astype(str)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    return float(cross_val_score(tree, X, y,
                                 cv=GroupKFold(n_splits=n_splits),
                                 groups=groups).mean())


def sfb_multi(frame, y, numeric_cols=(), categorical_cols=(),
              groups=None, max_depth=3, n_splits=5):
    """SFB for several metadata columns used jointly (a small tree).

    This is the strongest signal-free predictor we report: it answers 'how far
    does the accompanying metadata get you, all of it together?'
    """
    parts = []
    for c in numeric_cols:
        parts.append(pd.to_numeric(frame[c], errors="coerce").fillna(-1).to_numpy())
    for c in categorical_cols:
        parts.append(pd.Categorical(frame[c].astype(str)).codes.astype(float))
    if not parts:
        return float("nan")
    X = np.column_stack(parts)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    if groups is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        return float(cross_val_score(tree, X, y, cv=cv).mean())
    groups = np.asarray(groups).astype(str)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return float("nan")
    return float(cross_val_score(tree, X, y,
                                 cv=GroupKFold(n_splits=n_splits),
                                 groups=groups).mean())


def majority_baseline(y):
    """Accuracy of always predicting the most frequent class."""
    return float(pd.Series(np.asarray(y)).value_counts(normalize=True).max())
