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


def assemble(npz_path, counts, diams, sets=None):
    d = np.load(npz_path, allow_pickle=True)
    acc, region = d['acc_num'].astype(str), d['region'].astype(str)

    idx = pd.MultiIndex.from_arrays([acc, region])
    n = counts.reindex(idx).fillna(0).to_numpy()
    dm = diams.reindex(idx).fillna(0.0).to_numpy()

    names = sets or FEATURE_SETS
    feats = {}
    for name in names:
        if name not in d:
            raise SystemExit(
                f'"{name}" is not in {npz_path}.\n  available: {sorted(d.files)}')
        arr = d[name]
        feats[name] = (arr.reshape(len(acc), -1).astype(np.float64) if arr.ndim == 1
                       else arr.astype(np.float32))
    ok = np.ones(len(acc), dtype=bool)
    if 'lobe_voxels' in feats:
        ok = np.isfinite(feats['lobe_voxels']).ravel()
    if not ok.all():
        print(f'  dropping {(~ok).sum()} rows with missing lobe volume')
        feats = {k: v[ok] for k, v in feats.items()}
        acc, n, dm = acc[ok], n[ok], dm[ok]

    return feats, acc, n, dm


def bootstrap_auc_ci(y, scores, groups, n_boot=2000, seed=0):
    """Percentile CI for AUC, resampling *patients* rather than rows.

    Resampling rows would treat the five lobes of one patient as independent and
    give a CI roughly sqrt(5) too narrow -- which is how a 0.02 AUC gap between
    two tap points gets mistaken for a finding.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], scores[idx]))
    if not out:
        return float('nan'), float('nan')
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def cv_binary(X, y, groups, n_splits=5):
    """Out-of-fold AUC for 'this lobe contains >=1 nodule', with a bootstrap CI."""
    if len(np.unique(y)) < 2:
        return float('nan'), (float('nan'), float('nan')), np.zeros(len(y))
    oof = np.zeros(len(y))
    cv = GroupKFold(n_splits=n_splits)
    for tr, te in cv.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = y[tr].mean()
            continue
        gs = GridSearchCV(
            # max_iter is generous because lbfgs on the 8192-dim post_fc features
            # hit the 2000 cap in a handful of folds on the 5580-row train split.
            # A non-converged fit still returns scores, so this fails silently
            # into slightly-wrong AUCs rather than erroring.
            Pipeline([('sc', StandardScaler()),
                      ('clf', LogisticRegression(max_iter=20000, class_weight='balanced'))]),
            {'clf__C': [1e-3, 1e-2, 1e-1, 1.0]},
            scoring='roc_auc',
            cv=GroupKFold(n_splits=3).split(X[tr], y[tr], groups[tr]),
            n_jobs=-1,
        )
        gs.fit(X[tr], y[tr])
        oof[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), bootstrap_auc_ci(y, oof, groups), oof


def paired_delta_ci(y, s_a, s_b, groups, n_boot=2000, seed=0):
    """CI on AUC(a) - AUC(b), bootstrapping the same patients for both.

    Pairing matters: two feature sets scored on identical folds share most of
    their error, so independent CIs on each overlap heavily even when the
    difference is consistent -- and vice versa.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    if not out:
        return float('nan'), float('nan')
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


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
    ap.add_argument('--sets', default=None,
                    help='comma-separated feature keys to probe; defaults to the '
                         'encoder tap points. Use e.g. lobe_voxels,native,resized '
                         'for the model-free voxel features.')
    ap.add_argument('--compare', default=None,
                    help='comma-separated a-b pairs for paired CIs, e.g. '
                         '"native-resized,native-lobe_voxels"')
    a = ap.parse_args()

    sets = [s.strip() for s in a.sets.split(',')] if a.sets else None
    counts, diams = load_labels(a.nodule_metadata, a.acc_col, a.lobe_col, a.diam_col)
    feats, acc, n_nod, max_diam = assemble(a.features, counts, diams, sets)
    names = sets or FEATURE_SETS

    has = (n_nod > 0).astype(int)
    print(f'\nregion rows: {len(has)}   with >=1 nodule: {has.sum()} ({has.mean():.1%})')
    print(f'patients: {len(np.unique(acc))}   folds: {a.folds} (grouped by patient)')

    print('\n' + '=' * 70)
    print('does this lobe contain a nodule?   (out-of-fold AUC, 0.5 = chance)')
    print('95% CI bootstraps patients, not rows -- lobes within a patient are not')
    print('independent, and ignoring that makes every CI ~sqrt(5) too narrow.')
    print('=' * 70)
    print(f'{"prior":<18}{0.5:>8.3f}')
    auc, oof = {}, {}
    for name in names:
        auc[name], ci, oof[name] = cv_binary(feats[name], has, acc, a.folds)
        print(f'{name:<18}{auc[name]:>8.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]   dim={feats[name].shape[1]}')

    if a.compare:
        pairs = [tuple(p.split('-', 1)) for p in a.compare.split(',')]
    else:
        pairs = [('pre_perceiver', 'post_perceiver'),
                 ('pre_perceiver', 'lobe_voxels'),
                 ('post_fc', 'lobe_voxels')]
    pairs = [(x, y) for x, y in pairs if x in oof and y in oof]

    if pairs:
        print('\n' + '-' * 70)
        print('paired differences  (95% CI, same patients resampled for both arms)')
        print('-' * 70)
        for a_name, b_name in pairs:
            lo, hi = paired_delta_ci(has, oof[a_name], oof[b_name], acc)
            d = auc[a_name] - auc[b_name]
            verdict = 'CI excludes 0' if (lo > 0 or hi < 0) else 'CI includes 0 -- not resolved'
            print(f'{a_name} - {b_name}:  delta {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]   {verdict}')

    print('\n' + '=' * 62)
    print('how many nodules / how big?   (out-of-fold Spearman rho)')
    print('=' * 62)
    print(f'{"":<18}{"count":>10}{"max_diam":>12}')
    for name in names:
        rc = cv_regression(feats[name], n_nod.astype(float), acc, a.folds)
        rd = cv_regression(feats[name], max_diam.astype(float), acc, a.folds)
        print(f'{name:<18}{rc:>10.3f}{rd:>12.3f}')

    print('\n' + '=' * 62)
    print('reading this')
    print('=' * 62)
    print("""\
A feature set at ~0.5 AUC carries nothing about lesions; one that matches
lobe_voxels carries only lobe size. The comparison that pays is
post_fc - lobe_voxels: whether what the LLM receives beats a single scalar.

Reference values already measured on checkpoint-1390 (docs/LESION_TOKENS.md §9):
post_fc 0.605, lobe_voxels 0.562, delta +0.044 with the CI excluding zero, and
an oracle crop at the lesion reaching 0.837 on mean HU alone. So the lesion
signal is present in the region representation but roughly an order of magnitude
weaker than it is at lesion scale.

Note pre_perceiver < post_perceiver there (-0.030, CI excludes 0): the learned
resampler helps rather than bottlenecks, so a low pre_perceiver is not evidence
against it.""")


if __name__ == '__main__':
    main()
