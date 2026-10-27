from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def _torch():
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


def make_groups(site, y, min_group=20):
    """(site, class) group labels; small groups merged into '<class>|rest'."""
    g = pd.Series(np.asarray(site).astype(str)) + "|" + pd.Series(np.asarray(y).astype(str))
    vc = g.value_counts()
    small = g.map(vc) < min_group
    g[small] = pd.Series(np.asarray(y).astype(str))[small] + "|rest"
    return g.to_numpy()


def worst_group_acc(y, pred, groups, min_group=20):
    df = pd.DataFrame(dict(ok=np.asarray(y) == np.asarray(pred), g=groups))
    vc = df.g.value_counts()
    df = df[df.g.map(vc) >= min_group]
    if not len(df):
        return float("nan")
    return float(df.groupby("g").ok.mean().min())


class _MLP:
    def __init__(self, d_in, n_cls, seed, hidden=128, dropout=0.1):
        torch = _torch()
        torch.manual_seed(seed)
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_cls))

    def logits(self, X):
        return self.net(X)


def _prep(Xtr, Xother):
    sc = StandardScaler().fit(np.log1p(np.clip(Xtr, 0, None)))
    f = lambda X: sc.transform(np.log1p(np.clip(X, 0, None))).astype(np.float32)
    return f(Xtr), [f(X) for X in Xother]


def _train(Xtr, ytr_i, n_cls, seed, epochs, wd, lr=1e-3, batch=512,
           method="erm", gtr=None, eta=0.01, weights=None, val=None,
           select="avg", min_group=20):
    """Train one model; returns (best_state_logits_fn, best_val, best_epoch)."""
    torch = _torch()
    F = torch.nn.functional
    rng = np.random.RandomState(seed)
    model = _MLP(Xtr.shape[1], n_cls, seed)
    opt = torch.optim.Adam(model.net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr_i.astype(np.int64))
    if method == "dro":
        gcodes, guniq = pd.factorize(pd.Series(gtr))
        gt = torch.from_numpy(gcodes.astype(np.int64))
        q = torch.ones(len(guniq)) / len(guniq)
    w_all = None if weights is None else torch.from_numpy(weights.astype(np.float32))
    best = (-np.inf, None, -1)
    for ep in range(epochs):
        model.net.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), batch):
            b = torch.from_numpy(order[i:i + batch])
            out = model.logits(Xt[b])
            loss_i = F.cross_entropy(out, yt[b], reduction="none")
            if method == "dro":
                gb = gt[b]
                gl = torch.zeros(len(guniq)).index_add(0, gb, loss_i)
                gn = torch.zeros(len(guniq)).index_add(0, gb, torch.ones_like(loss_i))
                present = gn > 0
                gmean = torch.where(present, gl / gn.clamp(min=1), torch.zeros_like(gl))
                with torch.no_grad():
                    q = q * torch.exp(eta * gmean.detach() * present)
                    q = q / q.sum()
                loss = (q * gmean).sum()
            elif w_all is not None:
                wb = w_all[b]
                loss = (wb * loss_i).sum() / wb.sum()
            else:
                loss = loss_i.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            Xv, yv_i, gv = val
            pv = predict_idx(model, Xv)
            sc = (accuracy_score(yv_i, pv) if select == "avg"
                  else worst_group_acc(yv_i, pv, gv, min_group))
            if sc > best[0]:
                best = (sc, {k: v.clone() for k, v in model.net.state_dict().items()}, ep + 1)
    if val is not None and best[1] is not None:
        model.net.load_state_dict(best[1])
    return model, best[0], best[2]


def predict_idx(model, X):
    torch = _torch()
    model.net.eval()
    with torch.no_grad():
        return model.logits(torch.from_numpy(X)).argmax(1).numpy()


def run_methods(X, y, site, train_idx, test_idx, seed=0, max_epochs=20,
                min_group=20, quick=False):

    y = np.asarray(y).astype(str)
    site = np.asarray(site).astype(str)
    classes = np.unique(y)
    to_i = {c: i for i, c in enumerate(classes)}
    yi = np.array([to_i[c] for c in y])

    # validation sites carved out of the training sites
    gss = GroupShuffleSplit(1, test_size=0.2, random_state=seed)
    a_rel, v_rel = next(gss.split(train_idx, y[train_idx], site[train_idx]))
    a, v = train_idx[a_rel], train_idx[v_rel]
    Xa, (Xv, Xte) = _prep(X[a], [X[v], X[test_idx]])
    ga, gv = make_groups(site[a], y[a], min_group), make_groups(site[v], y[v], min_group)
    val = (Xv, yi[v], gv)
    epochs = 3 if quick else max_epochs
    wds = [1e-4] if quick else [1e-4, 1e-2]

    def test_row(method, model, params, val_score, ep):
        p = predict_idx(model, Xte)
        yt = yi[test_idx]
        rec = [np.mean(p[yt == k] == k) for k in range(len(classes)) if np.any(yt == k)]
        return dict(method=method, seed=seed, acc=accuracy_score(yt, p),
                    bacc=balanced_accuracy_score(yt, p),
                    mf1=f1_score(yt, p, average="macro"),
                    worst_class_recall=float(np.min(rec)),
                    base=float(pd.Series(yt).value_counts(normalize=True).max()),
                    val_score=val_score, best_epoch=ep, params=str(params),
                    n_train=len(a), n_val=len(v), n_test=len(test_idx),
                    n_groups_train=int(pd.Series(ga).nunique()))

    rows = []
    # ERM
    best = None
    for wd in wds:
        m, s, ep = _train(Xa, yi[a], len(classes), seed, epochs, wd, val=val,
                          select="avg", min_group=min_group)
        if best is None or s > best[1]:
            best = (m, s, ep, dict(wd=wd))
    rows.append(test_row("ERM", best[0], best[3], best[1], best[2]))

    # Group-DRO
    best = None
    for eta, wd in itertools.product([0.01] if quick else [0.01, 0.1], wds):
        m, s, ep = _train(Xa, yi[a], len(classes), seed, epochs, wd, method="dro",
                          gtr=ga, eta=eta, val=val, select="worst", min_group=min_group)
        if best is None or s > best[1]:
            best = (m, s, ep, dict(eta=eta, wd=wd))
    rows.append(test_row("GroupDRO", best[0], best[3], best[1], best[2]))

    # JTT
    best = None
    for T, lam in itertools.product([1] if quick else [1, 2], [5] if quick else [5, 20]):
        m1, _, _ = _train(Xa, yi[a], len(classes), seed, T, wds[0])
        err = predict_idx(m1, Xa) != yi[a]
        w = np.where(err, float(lam), 1.0)
        m, s, ep = _train(Xa, yi[a], len(classes), seed, epochs, wds[0], weights=w,
                          val=val, select="worst", min_group=min_group)
        if best is None or s > best[1]:
            best = (m, s, ep, dict(T=T, lam=lam, error_set=int(err.sum())))
    rows.append(test_row("JTT", best[0], best[3], best[1], best[2]))
    return rows
