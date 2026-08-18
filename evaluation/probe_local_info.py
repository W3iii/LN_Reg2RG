"""Probe cached region features for lesion-scale information.

The question this answers, without involving the LLM at all: given the visual
representation of one lobe, can anything recover how many nodules are in it and
how big the largest one is? If a well-regularised probe cannot beat the trivial
controls, the information is not in the representation -- which is a direct,
low-variance measurement of the claim that HANDOFF.md §6 currently infers
indirectly from generated text.

Feature sets are ordered along the encoder pathway so the comparison localises
*where* the information dies:

    lobe_voxels     raw lobe size            <- trivial control, must be beaten
    mask_token      the shape/mask embedding <- second trivial control
    pre_perceiver   ViT3D patch tokens
    post_perceiver  after the 32-latent resampler
    post_fc         what the LLM actually sees

`prior` is the no-feature floor. A feature set that ties `prior` carries nothing;
one that ties `lobe_voxels` carries only "how big is this lobe", which the report
text correlates with for reasons that have nothing to do with seeing a nodule.

Everything is cross-validated with GroupKFold on patient id, so the five lobes of
one patient never straddle the train/test boundary -- without that, a probe can
score well by memorising per-patient quirks shared across lobes.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from regions import REGIONS  # noqa: E402

SHORT2REGION = {
    'RUL': 'right upper lobe',
    'RML': 'right middle lobe',
    'RLL': 'right lower lobe',
    'LUL': 'left upper lobe',
    'LLL': 'left lower lobe',
}

FEATURE_SETS = ['lobe_voxels', 'mask_token', 'pre_perceiver', 'post_perceiver', 'post_fc']


def pick_column(df, candidates, what):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f'could not find the {what} column in nodule_metadata.csv.\n'
        f'  tried: {candidates}\n'
        f'  available: {list(df.columns)}\n'
        'Pass the right name with the matching --*_col flag.'
    )


def load_labels(csv_path, acc_col=None, lobe_col=None, diam_col=None):
    """(acc_num, region) -> nodule count and max equivalent diameter."""
    df = pd.read_csv(csv_path)

    acc_col = acc_col or pick_column(
        df, ['Volumename', 'volumename', 'folder', 'AccNum', 'acc_num', 'accnum'], 'volume/accession')
    lobe_col = lobe_col or pick_column(df, ['lobe', 'Lobe', 'region', 'Anatomy'], 'lobe')
    diam_col = diam_col or pick_column(
        df, ['eq_diam_mm', 'eq_diameter_mm', 'diameter_mm', 'bbox_long_mm'], 'diameter')

    acc = df[acc_col].astype(str)
    if not acc.str.endswith('.nii.gz').all():
        acc = acc.str.replace(r'\.(nii\.gz|npz|npy)$', '', regex=True) + '.nii.gz'

    lobe = df[lobe_col].astype(str).str.strip()
    lobe = lobe.map(lambda v: SHORT2REGION.get(v.upper(), v))
    unknown = set(lobe.unique()) - set(REGIONS)
    if unknown:
        print(f'  warning: dropping {len(unknown)} unrecognised lobe value(s): {sorted(unknown)[:5]}')

    tidy = pd.DataFrame({'acc_num': acc, 'region': lobe, 'diam': df[diam_col].astype(float)})
    tidy = tidy[tidy.region.isin(REGIONS)]

    grouped = tidy.groupby(['acc_num', 'region'])
    return grouped.size().rename('n_nodules'), grouped.diam.max().rename('max_diam')


def assemble(npz_path, counts, diams):
    d = np.load(npz_path, allow_pickle=True)
    acc, region = d['acc_num'].astype(str), d['region'].astype(str)

    idx = pd.MultiIndex.from_arrays([acc, region])
    n = counts.reindex(idx).fillna(0).to_numpy()
    dm = diams.reindex(idx).fillna(0.0).to_numpy()

    feats = {
        'lobe_voxels': d['lobe_voxels'].reshape(-1, 1).astype(np.float64),
        'mask_token': d['mask_token'].astype(np.float32),
        'pre_perceiver': d['pre_perceiver'].astype(np.float32),
        'post_perceiver': d['post_perceiver'].astype(np.float32),
        'post_fc': d['post_fc'].astype(np.float32),
    }
    ok = np.isfinite(feats['lobe_voxels']).ravel()
    if not ok.all():
        print(f'  dropping {(~ok).sum()} rows with missing lobe volume')
        feats = {k: v[ok] for k, v in feats.items()}
        acc, n, dm = acc[ok], n[ok], dm[ok]

    return feats, acc, n, dm


def cv_binary(X, y, groups, n_splits=5):
    """Out-of-fold AUC for 'this lobe contains >=1 nodule'."""
    if len(np.unique(y)) < 2:
        return float('nan')
    oof = np.zeros(len(y))
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        gs = GridSearchCV(
            Pipeline([('sc', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))]),
            {'clf__C': [1e-3, 1e-2, 1e-1, 1.0]},
            scoring='roc_auc',
            cv=GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]),
            n_jobs=-1,
        )
        gs.fit(X[tr], y[tr])
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def cv_regression(X, y, groups, n_splits=5):
    """Out-of-fold Spearman rho, computed over lobes that actually have a nodule."""
    oof = np.zeros(len(y))
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(X, y, groups):
        gs = GridSearchCV(
            Pipeline([('sc', StandardScaler()), ('reg', Ridge())]),
            {'reg__alpha': [1.0, 1e1, 1e2, 1e3, 1e4]},
            scoring='r2',
            cv=GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]),
            n_jobs=-1,
        )
        gs.fit(X[tr], y[tr])
        oof[te] = gs.best_estimator_.predict(X[te])
    mask = y > 0
    if mask.sum() < 10:
        return float('nan')
    return spearmanr(y[mask], oof[mask]).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='npz from extract_region_features.py')
    ap.add_argument('--nodule_metadata', required=True)
    ap.add_argument('--acc_col', default=None)
    ap.add_argument('--lobe_col', default=None)
    ap.add_argument('--diam_col', default=None)
    ap.add_argument('--folds', type=int, default=5)
    a = ap.parse_args()

    counts, diams = load_labels(a.nodule_metadata, a.acc_col, a.lobe_col, a.diam_col)
    feats, acc, n_nod, max_diam = assemble(a.features, counts, diams)

    has = (n_nod > 0).astype(int)
    print(f'\nregion rows: {len(has)}   with >=1 nodule: {has.sum()} ({has.mean():.1%})')
    print(f'patients: {len(np.unique(acc))}   folds: {a.folds} (grouped by patient)')

    print('\n' + '=' * 62)
    print('does this lobe contain a nodule?   (out-of-fold AUC, 0.5 = chance)')
    print('=' * 62)
    print(f'{"prior":<18}{0.5:>10.3f}')
    auc = {}
    for name in FEATURE_SETS:
        auc[name] = cv_binary(feats[name], has, acc, a.folds)
        print(f'{name:<18}{auc[name]:>10.3f}   dim={feats[name].shape[1]}')

    print('\n' + '=' * 62)
    print('how many nodules / how big?   (out-of-fold Spearman rho)')
    print('=' * 62)
    print(f'{"":<18}{"count":>10}{"max_diam":>12}')
    for name in FEATURE_SETS:
        rc = cv_regression(feats[name], n_nod.astype(float), acc, a.folds)
        rd = cv_regression(feats[name], max_diam.astype(float), acc, a.folds)
        print(f'{name:<18}{rc:>10.3f}{rd:>12.3f}')

    print('\n' + '=' * 62)
    print('reading this')
    print('=' * 62)
    print("""\
A feature set at ~0.5 AUC carries nothing about lesions. One that matches
lobe_voxels carries only lobe size. The decisive comparison is pre_perceiver vs
post_perceiver: if pre is clearly higher, the 32-latent resampler is where the
information dies, and changing the resize schedule (docs/LESION_TOKENS.md B1)
cannot recover it -- 32 latents in, 32 latents out. If both are at the floor, the
loss happens earlier, in the resize or the frozen ViT.""")


if __name__ == '__main__':
    main()
