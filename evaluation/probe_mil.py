"""Multiple-instance probe over unpooled ViT patch tokens.

probe_local_info.py summarises a lobe by mean- and max-pooling its 1024 patch
tokens, which is the wrong question for a lesion. A nodule under one patch is
diluted ~1000x by the mean, and survives the max only if it happens to dominate
some dimension. So a low score there cannot distinguish "the encoder does not
represent nodules" from "the pooling threw the representation away" -- and that
ambiguity is what this script exists to remove.

The honest framing is multiple-instance learning: a lobe is a *bag* of 1024 patch
instances, and the label "this lobe contains a nodule" applies to the bag without
saying which instance is responsible. So score each patch and pool the *scores*,
not the features:

    bag_score = max_i (w . x_i + b)        # or a smooth top-k variant

If the MIL probe scores clearly above the pooled probe, the pooling was hiding
information the ViT does encode, and the earlier conclusion has to be withdrawn.
If it lands in the same place, the information really is not in the patch tokens,
and it has to be injected from outside the current pathway.

Reports both max pooling (a single decisive patch) and top-k mean (a small
cluster of patches), because a nodule spanning a patch boundary shows up in the
second and not necessarily the first.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_local_info import load_labels, bootstrap_auc_ci  # noqa: E402


class MILScorer(nn.Module):
    """Linear score per patch, pooled over patches into one bag score.

    Deliberately linear at the instance level: the point is to measure what is
    *linearly decodable* from a patch token, matching the pooled probe's capacity
    so the comparison isolates the pooling and not the probe's expressiveness.
    """

    def __init__(self, dim, pool='max', topk=8):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.pool, self.topk = pool, topk

    def forward(self, x):                      # x: (B, n_patches, dim)
        s = self.score(x).squeeze(-1)          # (B, n_patches)
        if self.pool == 'max':
            return s.amax(dim=1)
        k = min(self.topk, s.shape[1])
        return s.topk(k, dim=1).values.mean(dim=1)


def fit_fold(Xtr, ytr, Xte, pool, dim, epochs=60, lr=1e-3, wd=1e-2, device='cuda', seed=0):
    torch.manual_seed(seed)
    model = MILScorer(dim, pool=pool).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # class_weight equivalent: positives are the minority in most splits
    pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    yt = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    n, bs = len(ytr), 32
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs].numpy()
            xb = torch.as_tensor(np.asarray(Xtr[np.sort(idx)], dtype=np.float32), device=device)
            yb = yt[torch.as_tensor(np.sort(idx), device=device)]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xte), 64):
            xb = torch.as_tensor(np.asarray(Xte[i:i + 64], dtype=np.float32), device=device)
            outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True, help='npz from extract_region_features.py')
    ap.add_argument('--patch_tokens', required=True, help='.npy written by --patch_tokens_out')
    ap.add_argument('--nodule_metadata', required=True)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=60)
    a = ap.parse_args()

    d = np.load(a.features, allow_pickle=True)
    acc, region = d['acc_num'].astype(str), d['region'].astype(str)
    counts, _ = load_labels(a.nodule_metadata)
    idx = pd.MultiIndex.from_arrays([acc, region])
    has = (counts.reindex(idx).fillna(0).to_numpy() > 0).astype(int)

    X = np.load(a.patch_tokens, mmap_mode='r')
    if len(X) != len(has):
        raise SystemExit(f'patch tokens ({len(X)}) and feature rows ({len(has)}) disagree')
    dim = X.shape[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'{len(has)} lobes, {has.sum()} with a nodule ({has.mean():.1%})')
    print(f'patch tokens: {X.shape}   device={device}')

    print('\n' + '=' * 66)
    print('MIL over unpooled patch tokens   (out-of-fold AUC)')
    print('=' * 66)
    for pool in ('max', 'topk'):
        oof = np.zeros(len(has), dtype=np.float64)
        for tr, te in GroupKFold(n_splits=a.folds).split(X, has, acc):
            oof[te] = fit_fold(X[tr], has[tr], X[te], pool, dim,
                               epochs=a.epochs, device=device)
        auc = roc_auc_score(has, oof)
        lo, hi = bootstrap_auc_ci(has, oof, acc)
        print(f'  patch_tokens ({pool:<4}) {auc:.3f}  [{lo:.3f}, {hi:.3f}]')

    print("""
Compare against pre_perceiver from probe_local_info.py, which is the same tokens
mean/max-pooled. A clear gain here means the pooling was hiding information and
the pooled result understates what the ViT encodes. No gain means the patch
tokens genuinely do not carry it.""")


if __name__ == '__main__':
    main()
