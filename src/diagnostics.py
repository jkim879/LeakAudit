from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import evaluation as ev
from . import sfb


# --------------------------------------------------------------------------- #
# R2-1 (a) identity probe
# --------------------------------------------------------------------------- #
def identity_probe(X, ids, min_rows=20, max_ids=None, kind="rf", seed=0,
                   test_size=0.3, mask=None, max_rows=None):
    """Predict the identifier (site or sensor) from the spectrum alone.

    Only identifiers with at least ``min_rows`` rows (inside ``mask``) are used,
    rows are split at random stratified by identifier, and accuracy is compared
    with the majority-identifier rate ('chance'). A high probe accuracy shows
    that identity is decodable from the signal, i.e. that a label tied to
    identity can be learned without learning leak acoustics.
    """
    ids = np.asarray(ids).astype(str)
    sel = np.ones(len(ids), bool) if mask is None else np.asarray(mask, bool)
    vc = pd.Series(ids[sel]).value_counts()
    keep = vc[vc >= min_rows]
    if max_ids is not None and len(keep) > max_ids:
        keep = keep.sample(max_ids, random_state=seed)
    sel &= np.isin(ids, keep.index)
    idx = np.where(sel)[0]
    if max_rows is not None and len(idx) > max_rows:
        # equal cap per identifier (>= min(min_rows, 10) rows each), so every
        # identifier keeps enough rows for a stratified train/test split
        per = max(min(min_rows, 10), max_rows // len(keep))
        rs = np.random.RandomState(seed)
        idx = np.concatenate([rs.permutation(np.where(sel & (ids == k))[0])[:per]
                              for k in keep.index])
    a, b = train_test_split(idx, test_size=test_size, random_state=seed,
                            stratify=ids[idx])
    m = ev.make_model(kind, seed).fit(X[a], ids[a])
    acc = float(np.mean(m.predict(X[b]) == ids[b]))
    chance = float(pd.Series(ids[b]).value_counts(normalize=True).max())
    return dict(n_ids=int(len(keep)), n_rows=int(len(idx)), acc=acc,
                chance=chance, uniform_chance=1.0 / len(keep),
                lift_over_chance=acc / chance if chance else np.nan)


# --------------------------------------------------------------------------- #
# R2-1 (b) group-level label permutation
# --------------------------------------------------------------------------- #
def group_permuted_labels(y, groups, seed=0):
    """Give every group the (majority) label of another, randomly drawn group.

    Rows keep their own spectra; only the group -> label assignment is
    shuffled. Exact when groups are single-class (true of every D1 sensor).
    Returns the permuted label vector and the fraction of rows whose label
    changed.
    """
    y = np.asarray(y)
    g = np.asarray(groups).astype(str)
    lab = pd.Series(y).groupby(g).agg(lambda s: s.value_counts().idxmax())
    rng = np.random.RandomState(seed)
    new = pd.Series(rng.permutation(lab.values), index=lab.index)
    yp = new.loc[g].to_numpy()
    return yp, float(np.mean(yp != y))


def permutation_control(X, y, site, sid, dt, protocols, n_perm=5, kind="gbdt",
                        subsample=30000, seed0=0, true_reference=True):
    """Run :func:`group_permuted_labels` (sensor level) under several protocols.

    For each permutation replicate r and protocol p, the split is drawn with
    seed r, the training set is capped at ``subsample`` rows, and accuracy,
    balanced accuracy and the test-fold majority rate are recorded. With
    ``true_reference`` the same splits are also scored with the true labels.
    """
    rows = []
    for r in range(n_perm):
        yp, changed = group_permuted_labels(y, sid, seed=seed0 + 1000 + r)
        for label_kind, yy in ([("true", y)] if true_reference else []) + [("sensor_permuted", yp)]:
            for p in protocols:
                sp = ev.get_split(p, yy, site, sid, dt, seed0 + r)
                if sp is None:
                    continue
                a, b = sp
                if len(np.unique(yy[a])) < 2 or len(np.unique(yy[b])) < 2:
                    continue
                if len(a) > subsample:
                    a = np.random.RandomState(seed0 + r).choice(a, subsample, replace=False)
                res = ev.evaluate(X[a], yy[a], X[b], yy[b], kind, seed0 + r)
                rows.append(dict(rep=r, labels=label_kind, protocol=p,
                                 frac_rows_relabelled=changed if label_kind != "true" else 0.0,
                                 acc=res["acc"], bacc=res["bacc"], base=res["base"],
                                 excess=res["excess"], n_train=len(a), n_test=len(b)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# R2-3 metadata vs acoustic vs combined on identical folds
# --------------------------------------------------------------------------- #
def incremental_value(feature_sets, y, folds, kind="gbdt", seed=0,
                      clusters=None, subsample=None, n_boot=2000):
    """Out-of-fold comparison of several feature sets on identical folds.

    ``feature_sets`` maps a name to an (n, d) array; every set is scored on the
    same rows, the same target and the same folds. Returns (summary, correct)
    where ``correct`` holds per-row correctness for each set (for paired
    cluster bootstraps). With ``clusters`` an interval over independent groups
    is attached to every accuracy.
    """
    y = np.asarray(y)
    rule = ev.fold_majority_rule(y, folds)
    out, correct = [], {}
    for name, Xs in feature_sets.items():
        Xs = np.asarray(Xs, dtype=float)
        if Xs.ndim == 1:
            Xs = Xs.reshape(-1, 1)
        pred = np.empty(len(y), dtype=object)
        for k, (tr, te) in enumerate(folds):
            a = tr
            if subsample is not None and len(a) > subsample:
                a = np.random.RandomState(seed + k).choice(a, subsample, replace=False)
            m = ev.make_model(kind, seed).fit(Xs[a], y[a])
            pred[te] = m.predict(Xs[te])
        ok = pred.astype(str) == y.astype(str)
        correct[name] = ok
        s = ev.score(y.astype(str), pred.astype(str))
        rec = dict(feature_set=name, n=len(y), n_features=Xs.shape[1],
                   acc=s["acc"], bacc=s["bacc"], mf1=s["mf1"],
                   fold_majority_rule=float(rule.mean()),
                   excess_over_rule=float(s["acc"] - rule.mean()), cm=s["cm"])
        if clusters is not None:
            ci = sfb.cluster_bootstrap_acc(ok, clusters, n_boot=n_boot,
                                           base_correct=rule)
            rec.update(acc_lo=ci["lo"], acc_hi=ci["hi"],
                       excess_lo=ci["excess_lo"], excess_hi=ci["excess_hi"],
                       n_clusters=ci["n_clusters"])
        out.append(rec)
    return pd.DataFrame(out), correct


# --------------------------------------------------------------------------- #
# R2-11 uncertainty decomposition
# --------------------------------------------------------------------------- #
def uncertainty_decomposition(X, y, clusters, fold_fn, kind="gbdt", n_rep=10,
                              n_boot=2000, randomised_folds=True):
    """Three sources of uncertainty for one cross-validated accuracy.

    fold_fn(seed) -> list of (train, test) covering every row once.

    initialisation  folds fixed (seed 0); model seed varies 0..n_rep-1
    split           model seed fixed (0); fold seed varies (randomised
                    schemes only; leave-one-group-out has no split variability)
    group sampling  cluster bootstrap over ``clusters`` of the out-of-fold
                    correctness from fold seed 0 and model seed 0
    """
    y = np.asarray(y)
    res = {}
    f0 = fold_fn(0)
    init_acc = []
    for s in range(n_rep):
        pred, _ = ev.oof_predict(X, y, f0, kind, seed=s)
        init_acc.append(np.mean(pred == y))
        if s == 0:
            ok0 = pred == y
    res["acc_seed0"] = float(np.mean(ok0))
    res["init_sd"] = float(np.std(init_acc, ddof=1)) if n_rep > 1 else np.nan
    if randomised_folds:
        sp_acc = [init_acc[0]]
        for s in range(1, n_rep):
            pred, _ = ev.oof_predict(X, y, fold_fn(s), kind, seed=0)
            sp_acc.append(np.mean(pred == y))
        res["split_sd"] = float(np.std(sp_acc, ddof=1))
    else:
        res["split_sd"] = 0.0
    ci = sfb.cluster_bootstrap_acc(ok0, clusters, n_boot=n_boot,
                                   base_correct=ev.fold_majority_rule(y, f0))
    res.update(group_lo=ci["lo"], group_hi=ci["hi"],
               group_halfwidth=(ci["hi"] - ci["lo"]) / 2,
               n_clusters=ci["n_clusters"], excess=ci["excess"],
               excess_lo=ci["excess_lo"], excess_hi=ci["excess_hi"])
    return res, ok0
