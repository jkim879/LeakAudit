from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (LeaveOneOut, GroupKFold, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.metrics import accuracy_score


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=0):

    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n=0)
    rng = np.random.RandomState(seed)
    means = rng.choice(v, (n_boot, len(v)), replace=True).mean(axis=1)
    return dict(mean=float(v.mean()),
                lo=float(np.percentile(means, 100 * alpha / 2)),
                hi=float(np.percentile(means, 100 * (1 - alpha / 2))),
                n=int(len(v)))


def bootstrap_paired_diff(a, b, n_boot=5000, alpha=0.05, seed=0):

    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    d = a[:n] - b[:n]
    rng = np.random.RandomState(seed)
    means = rng.choice(d, (n_boot, len(d)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return dict(diff=float(d.mean()), lo=lo, hi=hi,
                excludes_zero=bool(lo > 0 or hi < 0), n=int(len(d)))


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion k / n."""
    from scipy.stats import norm
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(c - h), float(c + h))


def cluster_bootstrap_acc(correct, clusters, n_boot=2000, alpha=0.05, seed=0,
                          base_correct=None):

    c = np.asarray(correct, float)
    g = pd.Series(np.asarray(clusters).astype(str))
    codes, uniq = pd.factorize(g)
    k = len(uniq)
    sums = np.bincount(codes, weights=c, minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(float)
    bsum = None
    if base_correct is not None:
        bsum = np.bincount(codes, weights=np.asarray(base_correct, float), minlength=k)
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, k, (n_boot, k))
    w = np.zeros((n_boot, k))
    for r in range(n_boot):
        w[r] = np.bincount(draws[r], minlength=k)
    acc_b = (w @ sums) / (w @ cnts)
    out = dict(acc=float(c.mean()), lo=float(np.percentile(acc_b, 100 * alpha / 2)),
               hi=float(np.percentile(acc_b, 100 * (1 - alpha / 2))),
               n_units=int(len(c)), n_clusters=int(k))
    if bsum is not None:
        ex_b = acc_b - (w @ bsum) / (w @ cnts)
        out.update(base=float(np.mean(base_correct)),
                   excess=float(c.mean() - np.mean(base_correct)),
                   excess_lo=float(np.percentile(ex_b, 100 * alpha / 2)),
                   excess_hi=float(np.percentile(ex_b, 100 * (1 - alpha / 2))))
    return out


def paired_cluster_bootstrap(correct_a, correct_b, clusters, n_boot=2000,
                             alpha=0.05, seed=0):

    a = np.asarray(correct_a, float)
    b = np.asarray(correct_b, float)
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters).astype(str)))
    k = len(uniq)
    sa = np.bincount(codes, weights=a, minlength=k)
    sb = np.bincount(codes, weights=b, minlength=k)
    n = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        w = np.bincount(rng.randint(0, k, k), minlength=k)
        diffs[r] = (w @ sa - w @ sb) / (w @ n)
    d = float(a.mean() - b.mean())
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # two-sided bootstrap p-value from the centred distribution
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(d)))
    return dict(diff=d, lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0),
                p_boot=p, n_units=int(len(a)), n_clusters=int(k))


def _cluster_stats(y, ok, codes, k, classes):
    """Per-cluster sums: total, correct, and per-class (n, correct)."""
    n = np.bincount(codes, minlength=k).astype(float)
    c = np.bincount(codes, weights=ok.astype(float), minlength=k)
    nc = np.stack([np.bincount(codes, weights=(y == cl).astype(float), minlength=k)
                   for cl in classes])
    cc = np.stack([np.bincount(codes, weights=((y == cl) & ok).astype(float), minlength=k)
                   for cl in classes])
    return n, c, nc, cc


def paired_cluster_bootstrap_metrics(y, pred_a, pred_b, clusters, n_boot=2000,
                                     alpha=0.05, seed=0, base_a=None, base_b=None):

    y = np.asarray(y).astype(str)
    oka = np.asarray(pred_a).astype(str) == y
    okb = np.asarray(pred_b).astype(str) == y
    classes = np.unique(y)
    codes, uniq = pd.factorize(pd.Series(np.asarray(clusters).astype(str)))
    k = len(uniq)
    na, ca, nca, cca = _cluster_stats(y, oka, codes, k, classes)
    _, cb, _, ccb = _cluster_stats(y, okb, codes, k, classes)
    ba_a = np.zeros(k) if base_a is None else np.bincount(codes, weights=np.asarray(base_a, float), minlength=k)
    ba_b = np.zeros(k) if base_b is None else np.bincount(codes, weights=np.asarray(base_b, float), minlength=k)

    def metrics(w):
        N = w @ na
        acc_a, acc_b = (w @ ca) / N, (w @ cb) / N
        rec_a = (cca @ w) / np.maximum(nca @ w, 1e-12)
        rec_b = (ccb @ w) / np.maximum(nca @ w, 1e-12)
        ex_a, ex_b = acc_a - (w @ ba_a) / N, acc_b - (w @ ba_b) / N
        return acc_a - acc_b, rec_a.mean() - rec_b.mean(), ex_a - ex_b

    obs = metrics(np.ones(k))
    rng = np.random.RandomState(seed)
    B = np.array([metrics(np.bincount(rng.randint(0, k, k), minlength=k).astype(float))
                  for _ in range(n_boot)])
    out = dict(n_units=int(len(y)), n_clusters=int(k))
    for j, name in enumerate(["acc", "bacc", "excess"]):
        if name == "excess" and base_a is None:
            continue
        lo, hi = np.percentile(B[:, j], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[f"d_{name}"] = float(obs[j])
        out[f"d_{name}_lo"], out[f"d_{name}_hi"] = float(lo), float(hi)
        out[f"d_{name}_excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


# --------------------------------------------------------------------------- #
# SFB estimators (v1 interface kept; out-of-fold variants added)
# --------------------------------------------------------------------------- #
def _lookup_predict(codes_tr, y_tr, codes_te, fallback):
    lookup = pd.Series(y_tr).groupby(pd.Series(codes_tr).values).agg(
        lambda s: s.value_counts().idxmax())
    return pd.Series(codes_te).map(lookup).fillna(fallback).to_numpy()


def sfb_categorical(codes, y, groups=None, n_splits=5, return_pred=False):

    codes = pd.Series(np.asarray(codes)).astype(str).reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)

    if groups is None:
        pred = _lookup_predict(codes.values, y.values, codes.values,
                               y.value_counts().idxmax())
        acc = float(accuracy_score(y, pred))
        return (acc, pred) if return_pred else acc

    groups = pd.Series(np.asarray(groups)).astype(str).reset_index(drop=True)
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        return (float("nan"), None) if return_pred else float("nan")
    pred = np.empty(len(y), dtype=object)
    accs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(codes, y, groups=groups):
        fb = y.iloc[tr].value_counts().idxmax()
        pred[te] = _lookup_predict(codes.iloc[tr].values, y.iloc[tr].values,
                                   codes.iloc[te].values, fb)
        accs.append(accuracy_score(y.iloc[te], pred[te]))
    acc = float(np.mean(accs))           # v1 convention: mean of fold accuracies
    return (acc, pred) if return_pred else acc


def sfb_numeric(x, y, groups=None, max_depth=1, n_splits=5, return_pred=False):

    X = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return (float("nan"), None) if return_pred else float("nan")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    if groups is None:
        cv = LeaveOneOut() if len(y) <= 400 else StratifiedKFold(5, shuffle=True,
                                                                random_state=0)
        acc = float(cross_val_score(tree, X, y, cv=cv).mean())
        if return_pred:
            return acc, cross_val_predict(tree, X, y, cv=cv)
        return acc
    groups = np.asarray(groups).astype(str)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return (float("nan"), None) if return_pred else float("nan")
    cv = GroupKFold(n_splits=n_splits)
    acc = float(cross_val_score(tree, X, y, cv=cv, groups=groups).mean())
    if return_pred:
        return acc, cross_val_predict(tree, X, y, cv=cv, groups=groups)
    return acc


def _design(frame, numeric_cols=(), categorical_cols=()):
    parts = []
    for c in numeric_cols:
        parts.append(pd.to_numeric(frame[c], errors="coerce").fillna(-1).to_numpy())
    for c in categorical_cols:
        parts.append(pd.Categorical(frame[c].astype(str)).codes.astype(float))
    return np.column_stack(parts) if parts else None


def sfb_multi(frame, y, numeric_cols=(), categorical_cols=(),
              groups=None, max_depth=3, n_splits=5, return_pred=False):
    """SFB for several metadata columns used jointly (a small tree)."""
    X = _design(frame, numeric_cols, categorical_cols)
    y = np.asarray(y)
    if X is None or len(np.unique(y)) < 2:
        return (float("nan"), None) if return_pred else float("nan")
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    if groups is None:
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        acc = float(cross_val_score(tree, X, y, cv=cv).mean())
        return (acc, cross_val_predict(tree, X, y, cv=cv)) if return_pred else acc
    groups = np.asarray(groups).astype(str)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        return (float("nan"), None) if return_pred else float("nan")
    cv = GroupKFold(n_splits=n_splits)
    acc = float(cross_val_score(tree, X, y, cv=cv, groups=groups).mean())
    if return_pred:
        return acc, cross_val_predict(tree, X, y, cv=cv, groups=groups)
    return acc


def sfb_official_split(frame, y, train_mask, numeric_cols=(), categorical_cols=(),
                       max_depth=3):

    y = np.asarray(y)
    tr, te = np.asarray(train_mask, bool), ~np.asarray(train_mask, bool)
    if len(categorical_cols) == 1 and not numeric_cols:
        c = categorical_cols[0]
        fb = pd.Series(y[tr]).value_counts().idxmax()
        pred = _lookup_predict(frame[c].astype(str).values[tr], y[tr],
                               frame[c].astype(str).values[te], fb)
    else:
        X = _design(frame, numeric_cols, categorical_cols)
        pred = DecisionTreeClassifier(max_depth=max_depth, random_state=0) \
            .fit(X[tr], y[tr]).predict(X[te])
    return float(accuracy_score(y[te], pred)), float(
        pd.Series(y[te]).value_counts(normalize=True).max()), int(te.sum())


def majority_baseline(y):
    """Accuracy of always predicting the most frequent class."""
    return float(pd.Series(np.asarray(y)).value_counts(normalize=True).max())


# --------------------------------------------------------------------------- #
# Dependence between metadata and label (beyond accuracy)
# --------------------------------------------------------------------------- #
def metadata_dependence(frame, col, y):

    from scipy.stats import chi2_contingency
    tab = pd.crosstab(frame[col].astype(str).values, np.asarray(y))
    obs = tab.to_numpy(float)
    n = obs.sum()
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        chi2, p, dof = 0.0, 1.0, 0
    else:
        chi2, p, dof, _ = chi2_contingency(obs, correction=False)
    k = min(obs.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0
    pj = obs / n
    pr, pc = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(pj * np.log2(pj / (pr @ pc)))
    cond = obs / obs.sum(1, keepdims=True)
    shift = float(np.abs(cond - pc).max())
    return tab, dict(field=col, n=int(n), levels=int(obs.shape[0]),
                     chi2=float(chi2), dof=int(dof), p=float(p),
                     cramers_v=v, mutual_info_bits=float(max(mi, 0.0)),
                     max_class_prob_shift=shift)


def permutation_null_margin(stat_fn, y, n_perm=1000, seed=0, groups=None):

    y = np.asarray(y)
    obs = float(stat_fn(y))
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    if groups is None:
        for i in range(n_perm):
            null[i] = stat_fn(rng.permutation(y))
    else:
        g = np.asarray(groups).astype(str)
        ug = pd.unique(g)
        first = pd.Series(y).groupby(g).first()
        for i in range(n_perm):
            mapping = dict(zip(ug, first.loc[rng.permutation(ug)].values))
            null[i] = stat_fn(np.array([mapping[x] for x in g]))
    return dict(observed=obs, null_mean=float(null.mean()),
                null_q95=float(np.quantile(null, 0.95)),
                null_q99=float(np.quantile(null, 0.99)),
                p_value=float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1)),
                n_perm=int(n_perm)), null
