from __future__ import annotations
import json
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                     StratifiedKFold, GroupKFold,
                                     StratifiedGroupKFold, LeaveOneGroupOut)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, recall_score)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODELS = ["gbdt", "rf", "logreg", "mlp"]


def make_model(kind="gbdt", seed=0):
    """Return an unfitted classifier (no tuning; identical under every protocol).

    'gbdt' (gradient-boosted trees) is the default because the inputs are
    numeric tabular spectra, where boosting is a strong and standard baseline.
    """
    if kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=150, random_state=seed, early_stopping=False)
    if kind == "gbdt_small":           # cheaper learner for permutation refits
        return HistGradientBoostingClassifier(
            max_iter=60, random_state=seed, early_stopping=False)
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
# Protocols
# --------------------------------------------------------------------------- #
PROTOCOLS = ["P1_naive_random", "P2_sensor_indep",
             "P3_site_indep", "P4_site_temporal"]
"""The four D1 protocols of the factorial (4 models x 4 protocols x 10 seeds
= 160 fits per task; 480 over the three tasks)."""

PROTOCOLS_EXT = PROTOCOLS + ["P5_temporal_known"]

PROTOCOL_CATALOGUE = pd.DataFrame([
    # dataset, protocol, grouping held out, deployment setting estimated
    ("D1", "P1_naive_random", "nothing (rows shuffled)",
     "none: rows of the same sensor and date-series sit on both sides; "
     "an upper bound that includes identity memorisation", "record"),
    ("D1", "P2_sensor_indep", "sensor (sid)",
     "a new sensor at an already-instrumented site (site shared)", "record"),
    ("D1", "P3_site_indep", "site",
     "a new site, any period (sites unseen, time not ordered)", "record"),
    ("D1", "P4_site_temporal", "site + time (train past, test future)",
     "network expansion / new-site screening in a later period", "record"),
    ("D1", "P5_temporal_known", "time only (train past, test future)",
     "monitoring established sensors with their own history (R2-2); "
     "sensors and sites shared", "record"),
    ("D2", "P1_naive", "nothing (files shuffled)", "none (upper bound)", "file"),
    ("D2", "LOO", "one file", "none: files of the same logger/site on both "
     "sides; used only for the time stump", "file"),
    ("D2", "logger_indep", "logger / hydrophone site",
     "a new logger location", "file"),
    ("D3", "S1_segment_random", "nothing (1 s segments shuffled)",
     "none: segments of one recording on both sides", "1 s segment"),
    ("D3", "S2_recording_indep", "recording (one channel)",
     "a new recording; v1 definition: the other hydrophone of the same run may "
     "be in training", "1 s segment"),
    ("D3", "S2b_run_indep", "physical run (H1 + H2 together)",
     "a new run of a known operating condition", "1 s segment"),
    ("D3", "S3_condition_indep_v1", "topology x leak type (v1 'condition')",
     "documented for transparency only: holds out whole label levels, so most "
     "random draws leave one class absent from a fold", "1 s segment"),
    ("D3", "S3_opcond_LOCO", "operating condition = flow x noise "
     "(leave-one-condition-out, 6 folds)",
     "an unseen background flow / noise regime; prespecified, exhaustive, "
     "both classes in every fold", "1 s segment"),
    ("D3", "S3_topology_LOTO", "pipe topology (2 folds)",
     "an unseen network topology", "1 s segment"),
], columns=["dataset", "protocol", "held_out", "deployment_setting", "unit"])


def get_split(protocol, y, groups_site, groups_sid, dt, seed, test_size=0.3):
    """Return (train_idx, test_idx) for one D1 protocol, or None if unusable.

    P1 naive_random      rows shuffled (what most papers report).
    P2 sensor_indep      no sensor shared between train and test.
    P3 site_indep        no site shared between train and test.
    P4 site_temporal     site-independent *and* trained on the past, tested on
                         the future. Rows of training sites after the cut and of
                         test sites before the cut are excluded (reported by
                         :func:`split_inventory`).
    P5 temporal_known    trained on the past, tested on the future of sensors
                         that already appear in training (R2-2).

    The temporal cut is the 70th percentile of the acquisition date.
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
    if protocol == "P5_temporal_known":
        cut = pd.Series(dt).quantile(0.7)
        tr = idx[dt <= cut]
        known = np.isin(groups_sid, pd.unique(groups_sid[tr]))
        te = idx[(dt > cut) & known]
        return (tr, te) if len(te) > 30 else None
    raise ValueError(f"unknown protocol: {protocol}")


def split_inventory(tr, te, y, site=None, sid=None, dt=None, n_total=None):
    """Sizes, independent groups, class proportions and exclusions of a split."""
    y = np.asarray(y)
    n_total = len(y) if n_total is None else n_total
    out = dict(n_train=int(len(tr)), n_test=int(len(te)),
               n_excluded=int(n_total - len(np.union1d(tr, te))))
    for name, g in [("site", site), ("sensor", sid)]:
        if g is None:
            continue
        g = np.asarray(g)
        gtr, gte = set(g[tr]), set(g[te])
        out[f"n_{name}_train"] = len(gtr)
        out[f"n_{name}_test"] = len(gte)
        out[f"n_{name}_shared"] = len(gtr & gte)
    for part, ix in [("train", tr), ("test", te)]:
        vc = pd.Series(y[ix]).value_counts(normalize=True)
        for c, v in vc.items():
            out[f"p_{part}_{c}"] = float(v)
    if dt is not None:
        dt = pd.to_datetime(pd.Series(np.asarray(dt)))
        out["train_dates"] = f"{dt.iloc[tr].min().date()}..{dt.iloc[tr].max().date()}"
        out["test_dates"] = f"{dt.iloc[te].min().date()}..{dt.iloc[te].max().date()}"
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def score(y_te, pred, y_tr=None, labels=None):
    """All reported metrics for one set of test predictions.

    acc / bacc / mf1   accuracy, balanced accuracy, macro-F1
    base               majority-class rate *of the test fold* (the best any
                       constant predictor could do on this fold)
    base_trainrule     accuracy of predicting the training-fold majority class
                       (the trivial rule actually available at training time)
    excess             acc - base
    rec_<c>, n_<c>     per-class recall and support
    cm                 confusion matrix (rows = truth, columns = prediction),
                       JSON-encoded with its label order
    """
    y_te = np.asarray(y_te)
    pred = np.asarray(pred)
    labels = sorted(set(y_te) | set(pred) | (set(y_tr) if y_tr is not None else set())) \
        if labels is None else list(labels)
    base = float(pd.Series(y_te).value_counts(normalize=True).max())
    out = dict(
        acc=float(accuracy_score(y_te, pred)),
        bacc=float(balanced_accuracy_score(y_te, pred)),
        mf1=float(f1_score(y_te, pred, average="macro", labels=sorted(set(y_te)),
                           zero_division=0)),
        base=base,
    )
    if y_tr is not None:
        maj = pd.Series(np.asarray(y_tr)).value_counts().idxmax()
        out["base_trainrule"] = float(np.mean(y_te == maj))
    out["excess"] = out["acc"] - base
    rec = recall_score(y_te, pred, labels=labels, average=None, zero_division=0)
    for c, r in zip(labels, rec):
        n_c = int(np.sum(y_te == c))
        out[f"n_{c}"] = n_c
        out[f"rec_{c}"] = float(r) if n_c else float("nan")
    cm = confusion_matrix(y_te, pred, labels=labels)
    out["cm"] = json.dumps({"labels": [str(l) for l in labels], "m": cm.tolist()})
    return out


def evaluate(X_tr, y_tr, X_te, y_te, kind="gbdt", seed=0, return_pred=False,
             labels=None):
    """Fit one model and return every metric of :func:`score`."""
    model = make_model(kind, seed).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    out = score(y_te, pred, y_tr=y_tr, labels=labels)
    if return_pred:
        out["pred"] = pred
    return out


def operational_rates(cm_json, positive):

    d = json.loads(cm_json) if isinstance(cm_json, str) else cm_json
    labels, m = d["labels"], np.asarray(d["m"], float)
    if positive not in labels or len(labels) != 2:
        return {}
    p = labels.index(positive)
    n = 1 - p
    tp, fn, fp, tn = m[p, p], m[p, n], m[n, p], m[n, n]
    tot = m.sum()
    return dict(missed_leaks_per_1000=1000 * fn / tot,
                unnecessary_investigations_per_1000=1000 * fp / tot,
                leak_recall=tp / (tp + fn) if tp + fn else np.nan,
                leak_precision=tp / (tp + fp) if tp + fp else np.nan)


# --------------------------------------------------------------------------- #
# Cross-validated out-of-fold predictions (every unit predicted once)
# --------------------------------------------------------------------------- #
def make_folds(scheme, y, groups=None, n_splits=5, seed=0):

    idx = np.arange(len(y))
    if scheme == "stratified":
        cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(idx, y))
    if scheme == "group":
        return list(GroupKFold(n_splits).split(idx, y, groups))
    if scheme == "stratified_group":
        cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        return list(cv.split(idx, y, groups))
    if scheme == "logo":
        return list(LeaveOneGroupOut().split(idx, y, groups))
    raise ValueError(scheme)


def oof_predict(X, y, folds, kind="gbdt", seed=0, proba=False):

    y = np.asarray(y)
    classes = np.unique(y)
    pred = np.empty(len(y), dtype=object)
    fold_id = np.full(len(y), -1)
    P = np.full((len(y), len(classes)), np.nan) if proba else None
    for k, (tr, te) in enumerate(folds):
        m = make_model(kind, seed).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        fold_id[te] = k
        if proba:
            pm = m.predict_proba(X[te])
            for j, c in enumerate(m.classes_):
                P[te, np.searchsorted(classes, c)] = pm[:, j]
    pred = pred.astype(y.dtype) if y.dtype.kind in "US" else pred
    return (pred, fold_id, P) if proba else (pred, fold_id)


def fold_majority_rule(y, folds):
    """Correctness vector of the trivial rule 'predict the training majority'."""
    y = np.asarray(y)
    ok = np.zeros(len(y), bool)
    for tr, te in folds:
        maj = pd.Series(y[tr]).value_counts().idxmax()
        ok[te] = y[te] == maj
    return ok


# --------------------------------------------------------------------------- #
# Factorial variance attribution
# --------------------------------------------------------------------------- #
def factorial_ss(df, a="protocol", b="model", response="acc"):
    """Two-way sum-of-squares decomposition with replication (seeds).

    Model:   y_ijk = mu + alpha_i + beta_j + (alpha beta)_ij + e_ijk,
             i = protocol (a levels), j = model (b levels), k = seed (r reps).

        SS_A   = b r  sum_i (ybar_i.. - ybar...)^2
        SS_B   = a r  sum_j (ybar_.j. - ybar...)^2
        SS_AB  = r    sum_ij (ybar_ij. - ybar_i.. - ybar_.j. + ybar...)^2
        SS_E   =      sum_ijk (y_ijk - ybar_ij.)^2          (seed-to-seed)
        SS_T   =      sum_ijk (y_ijk - ybar...)^2

    """
    g = df[[a, b, response]].dropna()
    counts = g.groupby([a, b]).size()
    balanced = counts.nunique() == 1 and len(counts) == g[a].nunique() * g[b].nunique()
    y = g[response].to_numpy()
    gm = y.mean()
    ma = g.groupby(a)[response].transform("mean").to_numpy()
    mb = g.groupby(b)[response].transform("mean").to_numpy()
    mc = g.groupby([a, b])[response].transform("mean").to_numpy()
    ss = {
        a: float(((ma - gm) ** 2).sum()),
        b: float(((mb - gm) ** 2).sum()),
        f"{a} x {b}": float(((mc - ma - mb + gm) ** 2).sum()),
        "residual (seed)": float(((y - mc) ** 2).sum()),
    }
    ss_t = float(((y - gm) ** 2).sum())
    na, nb = g[a].nunique(), g[b].nunique()
    dfs = {a: na - 1, b: nb - 1, f"{a} x {b}": (na - 1) * (nb - 1),
           "residual (seed)": len(g) - na * nb}
    div = lambda num, den: num / den if den > 0 else float("nan")
    rows = []
    for term, s in ss.items():
        rows.append(dict(term=term, ss=s, df=dfs[term], ss_share=div(s, ss_t),
                         partial_eta2=(div(s, s + ss["residual (seed)"])
                                       if term != "residual (seed)" else np.nan)))
    tab = pd.DataFrame(rows)
    tab["ss_total"] = ss_t
    tab["identity_residual"] = ss_t - sum(ss.values())
    tab["balanced"] = bool(balanced)
    tab["n_fits"] = len(g)
    tab["reps_per_cell"] = int(counts.min())
    cells = (g.groupby([a, b])[response].agg(["mean", "std", "size"])
             .reset_index().rename(columns={"size": "n_seeds"}))
    return tab, cells


# --------------------------------------------------------------------------- #
# Conformal prediction (split / inductive, with a class-conditional variant)
# --------------------------------------------------------------------------- #
def conformal_quantile(scores, alpha):
    """Exact split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    Returns +inf when that rank exceeds n (the set must then contain every
    label). v1 used ``np.quantile`` with linear interpolation, which can fall
    *below* this order statistic and is not the finite-sample rule.
    """
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    return float("inf") if k > n or n == 0 else float(s[k - 1])


def split_conformal(model, X_cal, y_cal, X_te, alpha=0.1,
                    classes=None, mondrian=False):

    P_cal = model.predict_proba(X_cal)
    P_te = model.predict_proba(X_te)
    classes = model.classes_ if classes is None else classes
    col = {c: i for i, c in enumerate(classes)}
    y_cal = np.asarray(y_cal)
    s_cal = 1.0 - P_cal[np.arange(len(y_cal)), [col[y] for y in y_cal]]

    if not mondrian:
        q = conformal_quantile(s_cal, alpha)
        return (1.0 - P_te) <= q, {"q_global": q, "n_cal": int(len(s_cal))}

    info, qv = {}, []
    for c in classes:
        mask = (y_cal == c)
        q = conformal_quantile(s_cal[mask], alpha)
        info[f"q_{c}"], info[f"n_cal_{c}"] = q, int(mask.sum())
        qv.append(q)
    info["n_cal"] = int(len(s_cal))
    return (1.0 - P_te) <= np.asarray(qv)[None, :], info


def conformal_report(S, y_te, classes):

    y_te = np.asarray(y_te)
    col = {c: i for i, c in enumerate(classes)}
    covered = S[np.arange(len(y_te)), [col[y] for y in y_te]].mean()
    size = S.sum(axis=1)
    singleton = (size == 1)
    if singleton.sum():
        decided_pred = np.asarray(classes)[S[singleton].argmax(axis=1)]
        acc_decided = (decided_pred == y_te[singleton]).mean()
        base_decided = pd.Series(y_te[singleton]).value_counts(normalize=True).max()
    else:
        acc_decided, base_decided = float("nan"), float("nan")
    return dict(coverage=float(covered),
                mean_set_size=float(size.mean()),
                abstain_rate=float(1 - singleton.mean()),
                n_test=int(len(y_te)), n_decided=int(singleton.sum()),
                acc_on_decided=float(acc_decided),
                baseline_on_decided=float(base_decided),
                referrals_per_1000=float(1000 * (1 - singleton.mean())))


# --------------------------------------------------------------------------- #
# Selective risk control -- an error-rate guarantee on accepted decisions
# --------------------------------------------------------------------------- #
def _clopper_pearson_upper(k, n, delta):
    from scipy.stats import beta
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))


def selective_risk_control(model, X_cal, y_cal, X_te, y_te=None, alpha=0.1,
                           groups_cal=None, guarantee="expected", delta=0.1):

    P_cal = model.predict_proba(X_cal)
    classes = model.classes_
    conf_cal = P_cal.max(axis=1)
    pred_cal = classes[P_cal.argmax(axis=1)]
    wrong_cal = (pred_cal != np.asarray(y_cal)).astype(float)

    units = None if groups_cal is None else pd.Series(np.asarray(groups_cal).astype(str))
    n = len(y_cal) if units is None else units.nunique()

    def empirical_risk(threshold):
        loss = wrong_cal * (conf_cal >= threshold)
        if units is None:
            return float(loss.mean()), int(loss.sum())
        return float(pd.Series(loss).groupby(units.values).mean().mean()), None

    grid = np.unique(np.concatenate([conf_cal, [1.0 + 1e-9]]))
    if guarantee == "expected":
        lam = 1.0 + 1e-9
        for cand in np.sort(grid):                       # ascending: first passing
            r, _ = empirical_risk(cand)
            if (n * r + 1.0) / (n + 1.0) <= alpha:
                lam = float(cand)
                break
    elif guarantee == "high_prob":
        lam = 1.0 + 1e-9
        for cand in np.sort(grid)[::-1]:                 # descending: fixed sequence
            r, k = empirical_risk(cand)
            ucb = (_clopper_pearson_upper(k, n, delta) if units is None
                   else r + math.sqrt(math.log(1 / delta) / (2 * n)))
            if ucb <= alpha:
                lam = float(cand)
            else:
                break
    else:
        raise ValueError(guarantee)

    P_te = model.predict_proba(X_te)
    conf_te = P_te.max(axis=1)
    pred_te = classes[P_te.argmax(axis=1)]
    accept_te = conf_te >= lam

    out = dict(alpha=float(alpha), guarantee=guarantee,
               delta=float(delta) if guarantee == "high_prob" else np.nan,
               weighting="record" if units is None else "site",
               threshold=float(min(lam, 1.0)), n_cal=int(len(y_cal)),
               n_cal_units=int(n), clustered=bool(units is not None),
               n_test=int(len(conf_te)), n_accepted=int(accept_te.sum()),
               accept_rate=float(accept_te.mean()),
               abstain_rate=float(1 - accept_te.mean()),
               referrals_per_1000=float(1000 * (1 - accept_te.mean())))
    if y_te is not None:
        y_te = np.asarray(y_te)
        wrong_te = (pred_te != y_te)
        out["joint_risk"] = float((wrong_te & accept_te).mean())
        out["risk_exceeds_alpha"] = bool(out["joint_risk"] > alpha)
        if accept_te.sum():
            out["n_accepted_wrong"] = int((wrong_te & accept_te).sum())
            out["selective_risk"] = float(wrong_te[accept_te].mean())
            out["acc_on_accepted"] = float(1 - wrong_te[accept_te].mean())
            out["baseline_on_accepted"] = float(
                pd.Series(y_te[accept_te]).value_counts(normalize=True).max())
        else:
            out.update(n_accepted_wrong=0, selective_risk=float("nan"),
                       acc_on_accepted=float("nan"),
                       baseline_on_accepted=float("nan"))
    return out


def risk_acceptance_curve(model, X_te, y_te, n_points=101):

    P = model.predict_proba(X_te)
    conf = P.max(axis=1)
    pred = model.classes_[P.argmax(axis=1)]
    wrong = pred != np.asarray(y_te)
    qs = np.unique(np.quantile(conf, np.linspace(0, 1, n_points)))
    rows = []
    for t in qs:
        acc_ = conf >= t
        k = int(acc_.sum())
        rows.append(dict(threshold=float(t), n_accepted=k,
                         accept_rate=k / len(conf),
                         selective_risk=float(wrong[acc_].mean()) if k else np.nan,
                         joint_risk=float((wrong & acc_).mean()),
                         baseline_on_accepted=float(
                             pd.Series(np.asarray(y_te)[acc_]).value_counts(
                                 normalize=True).max()) if k else np.nan))
    return pd.DataFrame(rows)
