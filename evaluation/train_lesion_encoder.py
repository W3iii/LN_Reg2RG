"""Train the lesion encoder alone (no LLM) on size + TW_Lung_RADS, compare poolings.

Motivation (scripts/probe_lesion.sh, B2 seed 1, val):

    representation                 size rho   RADS kappa
    B2 lesion token                  0.175      0.112
    random-init LesionEncoder        0.182      0.052
    12 centre-window HU statistics   0.719      0.613
    mask-derived size (ceiling)        --       0.683

The trained token is no better than a random one, while a dozen numbers read
off the *centre* of the same crop get most of the way to the ceiling. The
information is in the crop; LesionEncoder's global average pool over a
64x64x32 window (median nodule ~6x7x4 voxels, <0.2% of it) dilutes it -- the
lobe-level scale mismatch of HANDOFF.md §6, one level down. And the only
training signal B2 gave the encoder was the LM loss on digit tokens.

This script tests both fixes before paying for another 4.5 h LLM run:

    --arch gap      the current LesionEncoder, unchanged     (does supervision alone fix it?)
    --arch center   same convs, keep 16x16x8 resolution, Gaussian-weighted pool
                    centred on the nodule, concatenated with the global pool
    --arch attn     same trunk, learned spatial-attention pool + centre-distance bias

All three emit the same 4096-d token, and the auxiliary heads sit on that token,
so whatever is learned is in the vector the LLM would receive.

Losses: smooth-L1 on log(eq_diam_mm) + class-weighted CE on TW_Lung_RADS.
Train on the train split, select on val, report val. Test is never touched.

Crops are cached once (outputs/, git-ignored): reading 1258 NIfTI took ~11 min.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, HERE)

from Model.my_embedding_layer import LesionEncoder, PooledLesionEncoder  # noqa: E402  (single source: MyEmbedding loads these state_dicts)
from lesion_utils import crop_lesion, normalize_lesion, BBOX_COLS  # noqa: E402
from extract_lesion_features import find_image                    # noqa: E402


# ------------------------------------------------------------------ data
def build_cache(data_root, meta_csv, cache, splits=('train', 'val')):
    import nibabel as nib
    nm = pd.read_csv(meta_csv)
    nm = nm[nm['split'].isin(list(splits))].dropna(subset=BBOX_COLS + ['eq_diam_mm', 'tw_lung_rads'])
    nm = nm.reset_index(drop=True)
    X = np.zeros((len(nm), 64, 64, 32), dtype=np.float16)
    t0 = time.time()
    vols = list(dict.fromkeys(nm['Volumename']))
    for vi, vol in enumerate(vols):
        img = np.asarray(nib.load(find_image(os.path.join(data_root, 'images'), vol), mmap=True).dataobj).astype(np.float32)
        for r in nm.index[nm['Volumename'] == vol]:
            b = tuple(int(nm.at[r, c]) for c in ('y0_crop', 'y1_crop', 'x0_crop', 'x1_crop', 'z0_crop', 'z1_crop'))
            X[r] = normalize_lesion(crop_lesion(img, b))
        if (vi + 1) % 100 == 0:
            print(f'  cache {vi+1}/{len(vols)} volumes {time.time()-t0:.0f}s', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    np.save(cache, X)
    keep = [c for c in ('pid', 'Volumename', 'split', 'nodule_id', 'region', 'lobe', 'eq_diam_mm', 'tw_lung_rads') if c in nm.columns]
    nm[keep].to_csv(cache + '.csv', index=False)
    print(f'wrote {cache} {X.shape} ({X.nbytes/1e9:.2f} GB)')


class Crops(torch.utils.data.Dataset):
    def __init__(self, X, logd, rads, augment):
        self.X, self.logd, self.rads, self.augment = X, logd, rads, augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32)
        if self.augment:
            for ax in (0, 1):                       # in-plane flips; z kept (cranio-caudal)
                if np.random.rand() < 0.5:
                    x = np.flip(x, ax)
            sy, sx, sz = np.random.randint(-2, 3, 3)  # small jitter; the nodule stays central
            x = np.roll(x, (sy, sx, int(sz) // 2), axis=(0, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(x))[None], self.logd[i], self.rads[i]


# ------------------------------------------------------------------ models
class Net(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.enc = LesionEncoder(dim=4096) if arch == 'gap' else PooledLesionEncoder(arch)
        self.size_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1))
        self.rads_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 4))

    def forward(self, x):
        t = self.enc(x)
        return self.size_head(t)[:, 0], self.rads_head(t), t


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, L, Y, R = [], [], [], []
    for x, logd, rads in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            s, r, _ = model(x.to(device))
        P.append(s.float().cpu()); L.append(F.softmax(r.float(), 1).cpu()); Y.append(logd); R.append(rads)
    P, L, Y, R = (torch.cat(v).numpy() for v in (P, L, Y, R))
    d, dhat = np.exp(Y), np.exp(P)
    big = d >= 6
    rads1 = R + 1
    pred = L.argmax(1) + 1
    return dict(rho=spearmanr(d, dhat).statistic, MAE=float(np.abs(dhat - d).mean()),
                MAE_ge6=float(np.abs(dhat - d)[big].mean()),
                kappa=cohen_kappa_score(rads1, pred, weights='quadratic'),
                auc_ge3=roc_auc_score(rads1 >= 3, L[:, 2:].sum(1)), acc=float((pred == rads1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['gap', 'center', 'attn'], required=True)
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--cache', default=os.path.join(HERE, '..', 'outputs', 'lesion_crops_trainval.npy'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w_rads', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='dir for best.pt + metrics.json')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    if not os.path.exists(a.cache):
        build_cache(a.data_root, os.path.join(a.data_root, 'nodule_metadata.csv'), a.cache)
    X = np.load(a.cache, mmap_mode='r')
    meta = pd.read_csv(a.cache + '.csv')
    logd = np.log(meta['eq_diam_mm'].to_numpy(np.float32))
    rads = (meta['tw_lung_rads'].to_numpy(np.int64) - 1)
    tr = np.flatnonzero(meta['split'].to_numpy() == 'train')
    va = np.flatnonzero(meta['split'].to_numpy() == 'val')
    print(f'arch={a.arch}  train={len(tr)}  val={len(va)}  val patients={meta.pid.iloc[va].nunique()}')

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else '  <-- no GPU visible, this will be slow'))
    mk = lambda idx, aug, sh: torch.utils.data.DataLoader(
        Crops(np.asarray(X[idx]), logd[idx], rads[idx], aug),
        batch_size=a.bs, shuffle=sh, num_workers=a.workers, drop_last=sh)
    dl_tr, dl_va = mk(tr, True, True), mk(va, False, False)

    model = Net(a.arch).to(device)
    cw = torch.tensor(len(tr) / (4 * np.bincount(rads[tr], minlength=4).clip(1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl_tr))
    out = a.out or os.path.join(HERE, '..', 'outputs', f'lesion_enc_{a.arch}_seed{a.seed}')
    os.makedirs(out, exist_ok=True)

    best, best_score, hist = None, -1e9, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, ld, r in dl_tr:
            x, ld, r = x.to(device), ld.to(device), r.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                s, logits, _ = model(x)
            loss = F.smooth_l1_loss(s.float(), ld, beta=0.1) + a.w_rads * F.cross_entropy(logits.float(), r, weight=cw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        m = evaluate(model, dl_va, device)
        m.update(epoch=ep + 1, train_loss=tot / len(dl_tr))
        hist.append(m)
        score = m['rho'] + m['kappa']           # model selection on val, both targets equally
        flag = ''
        if score > best_score:
            best_score, best = score, m
            torch.save({'arch': a.arch, 'encoder': model.enc.state_dict(),
                        'size_head': model.size_head.state_dict(), 'rads_head': model.rads_head.state_dict(),
                        'metrics': {k: float(v) for k, v in m.items()}}, os.path.join(out, 'best.pt'))
            flag = ' *'
        print(f'ep {ep+1:3d} loss {m["train_loss"]:.3f} | val rho {m["rho"]:.3f} MAE(>=6) {m["MAE_ge6"]:.2f} '
              f'kappa {m["kappa"]:.3f} AUC>=3 {m["auc_ge3"]:.3f} | {time.time()-t0:.0f}s{flag}', flush=True)

    json.dump({'best': best, 'history': hist, 'args': vars(a)}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    print(f'\nBEST arch={a.arch}: rho {best["rho"]:.3f}  MAE {best["MAE"]:.2f}  MAE(>=6) {best["MAE_ge6"]:.2f}  '
          f'kappa {best["kappa"]:.3f}  AUC>=3 {best["auc_ge3"]:.3f}  (epoch {best["epoch"]})')
    print('reference (probe, val): token B2 rho 0.175 / kappa 0.112; HU stats 0.719 / 0.613; mask-size kappa 0.683')


if __name__ == '__main__':
    main()
