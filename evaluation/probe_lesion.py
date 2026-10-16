"""Does the lesion token encode nodule size and TW_Lung_RADS?

Reads one or more .npz files from extract_lesion_features.py (e.g. the B2
checkpoint and the --random_init control) and fits cross-validated linear probes,
GroupKFold on patient, with patient-bootstrap CIs -- same conventions as
probe_local_info.py.

Targets
    size  eq_diam_mm, ridge on log(diameter). Spearman rho, MAE, MAE on >= 6 mm.
    RADS  TW_Lung_RADS 1-4. Multinomial logistic: accuracy, quadratic-weighted
          kappa; plus binary AUC for RADS >= 3.

Reference rows, so every number has something to be read against:
    intensity     model-free HU statistics of the same crop (no learning)
    size oracle   the mask-derived size features themselves (eq_diam, bbox long
                  axis, log voxel count) -- for RADS only; the ceiling a pure
                  "measure size, apply thresholds" rule reaches. On the full
                  5392-nodule table a depth-4 tree on these gets weighted kappa
                  ~0.68.
    prior         predict the training-fold median / majority class

Reading it:
    post_fc rho high, but generated sizes uncorrelated  -> decoder ignores the token
    post_fc rho ~ intensity or ~ random_init            -> training put no size into it
    post_fc RADS kappa > size oracle                    -> token carries more than size

If train patients are included the encoder has seen them; report val alone as
the clean number and train+val only for its tighter CIs (as HANDOFF.md §9 did).
"""
import argparse
import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings('ignore')   # ill-conditioned ridge solves and lbfgs chatter drown the tables


def inner_cv(X, y, g, stratify=False):
    # a list, not a generator: newer joblib cannot pickle generators
    if stratify:
        return list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(X, y, g))
    return list(GroupKFold(n_splits=3).split(X, y, g))


def pipe(model):
    # median imputation: a few nodules (doctor-bbox only, no ME mask) have no
    # bbox_long_mm / size_vox, which made the size-oracle row crash
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('m', model)])


def oof_regress(X, y, g, folds):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
        gs = GridSearchCV(pipe(Ridge()),
                          {'m__alpha': [1e-1, 1, 1e1, 1e2, 1e3, 1e4]},
                          scoring='neg_mean_absolute_error', cv=inner_cv(X[tr], y[tr], g[tr]), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        oof[te] = gs.best_estimator_.predict(X[te])
    return oof


def oof_classify(X, y, g, folds, binary):
    classes = np.unique(y)
    oof = np.zeros((len(y), 1 if binary else len(classes)))
    # Stratified by class within patient groups: RADS 4 is ~3% of nodules, and a
    # plain GroupKFold left whole folds without it (log-loss then refused to score).
    outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    for tr, te in outer.split(X, y, g):
        gs = GridSearchCV(
            pipe(LogisticRegression(max_iter=20000, class_weight='balanced')),
            {'m__C': [1e-3, 1e-2, 1e-1, 1.0]},
            scoring='roc_auc' if binary else 'balanced_accuracy',
            cv=inner_cv(X[tr], y[tr], g[tr], stratify=True), n_jobs=-1)
        gs.fit(X[tr], y[tr])
        p = gs.best_estimator_.predict_proba(X[te])
        if binary:
            oof[te, 0] = p[:, 1]
        else:
            full = np.zeros((len(te), len(classes)))
            full[:, np.searchsorted(classes, gs.best_estimator_.classes_)] = p
            oof[te] = full
    return oof[:, 0] if binary else classes[oof.argmax(1)]


def boot(fn, g, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    by = {u: np.flatnonzero(g == u) for u in uniq}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    return np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)


def size_metrics(y, pred, g):
    rho = spearmanr(y, pred).statistic
    lo, hi = boot(lambda i: spearmanr(y[i], pred[i]).statistic, g)
    big = y >= 6
    return dict(rho=rho, rho_ci=f'[{lo:.2f},{hi:.2f}]', MAE=np.abs(pred - y).mean(),
                MAE_ge6=np.abs(pred - y)[big].mean())


def rads_metrics(y, pred_cls, p_ge3, g):
    k = cohen_kappa_score(y, pred_cls, weights='quadratic')
    lo, hi = boot(lambda i: cohen_kappa_score(y[i], pred_cls[i], weights='quadratic'), g)
    auc = roc_auc_score(y >= 3, p_ge3) if p_ge3 is not None else np.nan
    return dict(kappa=k, kappa_ci=f'[{lo:.2f},{hi:.2f}]', acc=(pred_cls == y).mean(), auc_ge3=auc)


def load(path, splits):
    z = np.load(path, allow_pickle=False)
    keep = np.isin(z['split'].astype(str), splits)
    return {k: z[k][keep] if z[k].ndim and len(z[k]) == len(keep) else z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', nargs='+', required=True,
                    help='label=path.npz pairs, e.g. B2=results/lesion_B2.npz rand=results/lesion_rand.npz')
    ap.add_argument('--splits', default='val')
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()
    splits = [s.strip() for s in a.splits.split(',')]

    sets, base = {}, None
    for spec in a.features:
        label, path = spec.split('=', 1)
        d = load(path, splits)
        if base is None:
            base = d
        elif not np.array_equal(d['nodule_id'], base['nodule_id']) or not np.array_equal(d['pid'], base['pid']):
            raise SystemExit(f'{path}: nodule rows do not line up with the first file')
        sets[f'{label}:post_fc'] = d['post_fc'].astype(np.float32)
        sets[f'{label}:pre_fc'] = d['pre_fc'].astype(np.float32)
    sets['intensity'] = base['intensity']

    g = base['pid'].astype(int)
    diam = base['eq_diam_mm'].astype(float)
    rads = base['tw_lung_rads'].astype(int)
    ylog = np.log(diam)
    print(f'splits={splits}  nodules={len(g)}  patients={len(np.unique(g))}  '
          f'RADS counts={dict(zip(*np.unique(rads, return_counts=True)))}')

    print('\n== size (eq_diam_mm) ==')
    rows = []
    prior = np.zeros(len(diam))
    for tr, te in GroupKFold(n_splits=a.folds).split(diam, diam, g):
        prior[te] = np.median(diam[tr])
    rows.append(('prior (median)', dict(rho=np.nan, rho_ci='', MAE=np.abs(prior - diam).mean(),
                                        MAE_ge6=np.abs(prior - diam)[diam >= 6].mean())))
    for name, X in sets.items():
        rows.append((name, size_metrics(diam, np.exp(oof_regress(X, ylog, g, a.folds)), g)))
    for name, m in rows:
        print(f'  {name:22s} rho={m["rho"]:.3f} {m["rho_ci"]:13s} MAE={m["MAE"]:.2f}  MAE(>=6mm)={m["MAE_ge6"]:.2f}')

    print('\n== TW_Lung_RADS ==')
    oracle = np.c_[diam, base['bbox_long_mm'].astype(float), np.log1p(base['size_vox'].astype(float))]
    rsets = dict(sets, **{'size oracle': oracle})
    maj = np.full(len(rads), np.bincount(rads).argmax())
    m = rads_metrics(rads, maj, None, g)
    print(f'  {"prior (majority)":22s} kappa={m["kappa"]:.3f} {m["kappa_ci"]:13s} acc={m["acc"]:.3f}')
    for name, X in rsets.items():
        cls = oof_classify(X, rads, g, a.folds, binary=False)
        p3 = oof_classify(X, (rads >= 3).astype(int), g, a.folds, binary=True)
        m = rads_metrics(rads, cls, p3, g)
        print(f'  {name:22s} kappa={m["kappa"]:.3f} {m["kappa_ci"]:13s} acc={m["acc"]:.3f}  AUC(RADS>=3)={m["auc_ge3"]:.3f}')


if __name__ == '__main__':
    main()
