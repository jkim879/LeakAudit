"""
Models, evaluation protocols, metrics and conformal prediction.

The central idea of the audit is that the *evaluation protocol* -- not the
model -- drives reported accuracy. :func:`get_split` therefore implements the
four protocols compared throughout the paper, and :func:`evaluate` holds the
model and metrics fixed so that only the protocol varies.

British English spelling ('normalise', 'behaviour') is used in comments.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_model(kind="gbdt", seed=0):
    """Return an unfitted classifier.

    'gbdt' (gradient-boosted trees) is the default because the inputs are
    numeric tabular spectra, where boosting is a strong and standard baseline.
    """
    if kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=150, random_state=seed, early_stopping=False)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed)
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                          random_state=seed))
    raise ValueError(f"unknown model kind: {kind}")


# --------------------------------------------------------------------------- #
# Split protocols
# --------------------------------------------------------------------------- #
def get_split(protocol, y, groups_site, groups_sid, dt, seed, test_size=0.3):
    """Return (train_idx, test_idx) for one of the four protocols.

    P1 naive_random    -- rows shuffled (what most papers report).
    P2 sensor_indep    -- no sensor shared between train and test.
    P3 site_indep      -- no site shared between train and test.
    P4 site_temporal   -- site-independent *and* trained on the past,
                          tested on the future (the strictest, most realistic).
    """
    idx = np.arange(len(y))
    if protocol == "P1_naive_random":
        return train_test_split(idx, test_size=test_size,
                                random_state=seed, stratify=y)
    if protocol == "P2_sensor_indep":
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                    .split(idx, y, groups=groups_sid))
    if protocol == "P3_site_indep":
        return next(GroupShuffleSplit(1, test_size=test_size, random_state=seed)
                    .split(idx, y, groups=groups_site))
    if protocol == "P4_site_temporal":
        sites = pd.unique(groups_site)
        rs = np.random.RandomState(seed)
        rs.shuffle(sites)
        test_sites = set(sites[:max(1, int(len(sites) * test_size))])
        test_mask = np.isin(groups_site, list(test_sites))
        cut = pd.Series(dt).quantile(0.7)
        tr = idx[(~test_mask) & (dt <= cut)]
        te = idx[test_mask & (dt > cut)]
        return (tr, te) if len(te) > 30 else None
    raise ValueError(f"unknown protocol: {protocol}")


PROTOCOLS = ["P1_naive_random", "P2_sensor_indep",
             "P3_site_indep", "P4_site_temporal"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate(X_tr, y_tr, X_te, y_te, kind="gbdt", seed=0):
    """Fit one model and return accuracy, balanced accuracy and macro-F1.

    'base' is the majority-class rate on the test fold, i.e. the accuracy of a
    trivial constant predictor -- the honest floor for every comparison.
    """
    model = make_model(kind, seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return dict(
        acc=accuracy_score(y_te, pred),
        bacc=balanced_accuracy_score(y_te, pred),
        mf1=f1_score(y_te, pred, average="macro"),
        base=pd.Series(y_te).value_counts(normalize=True).max(),
    )


# --------------------------------------------------------------------------- #
# Conformal prediction (split / inductive, with a class-conditional variant)
# --------------------------------------------------------------------------- #
def split_conformal(model, X_cal, y_cal, X_te, alpha=0.1,
                    classes=None, mondrian=False):
    """Split-conformal prediction sets with marginal coverage 1 - alpha.

    IMPORTANT -- what is and is not guaranteed
    ------------------------------------------
    The guarantee is *marginal coverage*:

        P(Y in C(X)) >= 1 - alpha

    i.e. the prediction SET contains the true label at least (1 - alpha) of the
    time. This is **not** a guarantee about the accuracy of any point decision,
    and in particular it says nothing about the error rate among the singleton
    (non-abstained) predictions. Use :func:`selective_risk_control` when an
    error-rate guarantee on accepted decisions is what is required.

    The finite-sample guarantee further relies on exchangeability between the
    calibration and test data. Our site-independent and site-plus-temporal
    protocols deliberately break exchangeability, so coverage under those
    protocols should be reported empirically rather than assumed.

    Returns a boolean matrix ``S`` of shape (n_test, n_class); ``S[i, k]`` is
    True when class ``k`` is retained in the prediction set for test point ``i``.
    With ``mondrian=True`` the quantile is computed per class
    (class-conditional / Mondrian conformal), which protects minority classes.
    """
    P_cal = model.predict_proba(X_cal)
    P_te = model.predict_proba(X_te)
    classes = model.classes_ if classes is None else classes
    col = {c: i for i, c in enumerate(classes)}
    # Non-conformity score: 1 - predicted probability of the true class.
    s_cal = 1.0 - P_cal[np.arange(len(y_cal)), [col[y] for y in y_cal]]

    if not mondrian:
        n = len(s_cal)
        q = np.quantile(s_cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
        return (1.0 - P_te) <= q, {"global": float(q)}

    quantiles = {}
    for c in classes:
        mask = (y_cal == c)
        n = int(mask.sum())
        quantiles[c] = (np.quantile(s_cal[mask],
                        min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
                        if n > 10 else 1.0)
    qv = np.array([quantiles[c] for c in classes])
    return (1.0 - P_te) <= qv[None, :], {str(k): float(v) for k, v in quantiles.items()}


def conformal_report(S, y_te, classes):
    """Summarise a set of conformal prediction sets.

    Returns
    -------
    coverage : empirical P(true label in prediction set). This is the quantity
        the conformal guarantee targets. It is NOT an accuracy.
    mean_set_size : average number of labels retained.
    abstain_rate : fraction of test points whose set is not a singleton, i.e.
        the fraction on which the system declines to commit.
    acc_on_decided : accuracy on the singleton subset. This is reported for
        information only and carries **no** distribution-free guarantee; it
        should always be compared against the majority-class baseline on the
        same subset.
    """
    col = {c: i for i, c in enumerate(classes)}
    covered = S[np.arange(len(y_te)), [col[y] for y in y_te]].mean()
    size = S.sum(axis=1)
    singleton = (size == 1)
    if singleton.sum():
        decided_pred = classes[S[singleton].argmax(axis=1)]
        acc_decided = (decided_pred == np.asarray(y_te)[singleton]).mean()
        base_decided = pd.Series(np.asarray(y_te)[singleton]).value_counts(
            normalize=True).max()
    else:
        acc_decided, base_decided = float("nan"), float("nan")
    return dict(coverage=float(covered),
                mean_set_size=float(size.mean()),
                abstain_rate=float(1 - singleton.mean()),
                acc_on_decided=float(acc_decided),
                baseline_on_decided=float(base_decided))


# --------------------------------------------------------------------------- #
# Selective risk control -- an error-rate guarantee on accepted decisions
# --------------------------------------------------------------------------- #
def selective_risk_control(model, X_cal, y_cal, X_te, y_te=None, alpha=0.1,
                           groups_cal=None):
    """Choose a confidence threshold that controls the *joint* error rate.

    Conformal prediction controls coverage, not the error rate of the decisions
    a utility acts on. Here we control instead

        P(prediction is wrong AND the system accepted it) <= alpha

    which is monotone in the confidence threshold and therefore admits the
    conformal risk control correction of Angelopoulos et al. (2024):

        lambda_hat = inf { lambda : (n * R_n(lambda) + B) / (n + 1) <= alpha }

    with loss bound B = 1. Read operationally: 'of every 100 alarms, at most
    alpha * 100 receive a confidently wrong automated decision'. Cases below
    the threshold are abstained and referred for manual verification.

    Exchangeability caveat
    ----------------------
    The bound assumes the calibration and test *units* are exchangeable. Field
    records are clustered by site and sensor, so treating individual rows as
    exchangeable is optimistic and the realised risk can exceed alpha by a few
    per cent. Passing ``groups_cal`` (e.g. the site label) calibrates on
    per-cluster mean losses instead of per-row losses, which respects the
    clustering and is the setting we report in the paper. We always report the
    realised risk so the reader can see the gap rather than take the bound on
    trust.

    Returns the chosen threshold, the abstention rate, the empirical joint
    risk, and (when ``y_te`` is given) the realised selective risk on the
    accepted subset together with its majority-class baseline.
    """
    P_cal = model.predict_proba(X_cal)
    classes = model.classes_
    conf_cal = P_cal.max(axis=1)
    pred_cal = classes[P_cal.argmax(axis=1)]
    wrong_cal = (pred_cal != np.asarray(y_cal)).astype(float)

    if groups_cal is None:
        units = None
        n = len(y_cal)
    else:
        units = pd.Series(np.asarray(groups_cal).astype(str))
        n = units.nunique()   # effective sample size = number of clusters

    def empirical_risk(threshold):
        accept = conf_cal >= threshold
        loss = wrong_cal * accept
        if units is None:
            return float(loss.mean())
        # Mean loss within each cluster, then averaged over clusters.
        return float(pd.Series(loss).groupby(units).mean().mean())

    # The joint risk is non-increasing in the threshold, so the first candidate
    # that satisfies the corrected bound is the infimum we want.
    grid = np.unique(np.concatenate([conf_cal, [0.0, 1.0]]))
    lam = 1.0
    for cand in np.sort(grid):
        if (n * empirical_risk(cand) + 1.0) / (n + 1.0) <= alpha:
            lam = float(cand)
            break

    P_te = model.predict_proba(X_te)
    conf_te = P_te.max(axis=1)
    pred_te = classes[P_te.argmax(axis=1)]
    accept_te = conf_te >= lam

    out = dict(alpha=float(alpha), threshold=lam, n_cal_units=int(n),
               clustered=bool(units is not None),
               accept_rate=float(accept_te.mean()),
               abstain_rate=float(1 - accept_te.mean()))
    if y_te is not None:
        y_te = np.asarray(y_te)
        wrong_te = (pred_te != y_te)
        out["joint_risk"] = float((wrong_te & accept_te).mean())
        out["risk_exceeds_alpha"] = bool(out["joint_risk"] > alpha)
        if accept_te.sum():
            out["selective_risk"] = float(wrong_te[accept_te].mean())
            out["acc_on_accepted"] = float(1 - wrong_te[accept_te].mean())
            out["baseline_on_accepted"] = float(
                pd.Series(y_te[accept_te]).value_counts(normalize=True).max())
        else:
            out["selective_risk"] = float("nan")
            out["acc_on_accepted"] = float("nan")
            out["baseline_on_accepted"] = float("nan")
    return out
