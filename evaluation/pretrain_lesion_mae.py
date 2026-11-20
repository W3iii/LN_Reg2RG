"""Masked-image pretraining (SimMIM-style MAE) of the vitq lesion encoder on unlabelled crops.

  python evaluation/pretrain_lesion_mae.py --data outputs/lidc_mae_crops.npy --out outputs/lesion_mae_vitq_lidc
  python evaluation/train_lesion_encoder.py --arch vitq --meta $DATA_ROOT/nodule_metadata_v2.csv \
      --init_encoder outputs/lesion_mae_vitq_lidc/mae.pt ...          # then size / type / RADS on ME

What is pretrained: the part of QueryLesionEncoder that sees the image --
    conv trunk (net) -> proj -> + pos -> transformer encoder -> mem_norm   (the 16x16x8 grid memory)
The 3 attribute queries, their decoder blocks and fc have nothing to reconstruct and are left to
the supervised stage (train_lesion_encoder.py loads only the keys in PRETRAINED_PREFIXES).

Masking (SimMIM, Xie et al. 2022, adapted to the conv trunk): the 16x16x8 grid of 4x4x4-voxel
cells is masked in blocks of --block cells (default 2x2x2 = 8x8x8 voxels = 6.4x6.4x8 mm) at
--mask_ratio (default 0.6). Masked voxels are set to 0 (= -400 HU after normalize_lesion) before
the conv trunk, so no masked intensity reaches the network, and the masked cells' tokens are
replaced by a learned mask token before the transformer. A linear head predicts each masked
cell's 64 voxels from the grid memory; loss = L1 on masked cells only.

Data: one or more crop arrays (N, 64, 64, 32) float16 in normalize_lesion units:
  data_prep/lidc/build_lidc_mae.py   LIDC nodules (any reader count, < 3 mm too) + random lung crops
  data_prep/lidc/build_lidc_lesions.py  the 1,854 labelled LIDC nodules (subset of the above)
ME crops are not used by default (MAE = LIDC only, ME is for the supervised stage); --me_cache
adds the ME *train* rows of a train_lesion_encoder crop cache if wanted. 2% of the crops
(whole patients, by pid in <npy>.csv) are held out to track the reconstruction loss.
"""
import os
import sys
import json
import time
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from Model.my_embedding_layer import QueryLesionEncoder  # noqa: E402

PRETRAINED_PREFIXES = ('net.', 'proj.', 'pos', 'encoder.', 'mem_norm.')
CELL = (4, 4, 4)                    # input voxels per grid cell: (64, 64, 32) / (16, 16, 8)


def patchify(x):
    """(N, 1, 64, 64, 32) -> (N, 2048, 64) in the grid's flatten order (y, x, z)."""
    n = x.shape[0]
    gy, gx, gz = (s // c for s, c in zip(x.shape[2:], CELL))
    x = x.reshape(n, gy, CELL[0], gx, CELL[1], gz, CELL[2])
    return x.permute(0, 1, 3, 5, 2, 4, 6).reshape(n, gy * gx * gz, -1)


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = QueryLesionEncoder(dim=4096)
        d = self.enc.proj.out_features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.head = nn.Linear(d, CELL[0] * CELL[1] * CELL[2])

    def forward(self, x, m):
        """x (N, 1, 64, 64, 32); m bool (N, 16, 16, 8), True = masked. -> (loss, prediction)."""
        e = self.enc
        mv = m.repeat_interleave(CELL[0], 1).repeat_interleave(CELL[1], 2).repeat_interleave(CELL[2], 3)
        f = e.net(x.masked_fill(mv[:, None], 0.0))
        t = e.proj(f.flatten(2).transpose(1, 2))
        mf = m.flatten(1)
        t = torch.where(mf[..., None], self.mask_token.to(t.dtype).expand_as(t), t) + e.pos.to(t.dtype)
        pred = self.head(e.mem_norm(e.encoder(t)))
        err = (pred.float() - patchify(x).float()).abs().mean(-1)
        loss = (err * mf).sum() / mf.sum().clamp(min=1)
        return loss, pred


def random_mask(n, grid, block, ratio, gen=None, device='cpu'):
    """Block mask over the cell grid: (n, *grid) bool, `ratio` of the blocks masked."""
    coarse = tuple(g // b for g, b in zip(grid, block))
    nb = int(np.prod(coarse))
    k = int(round(ratio * nb))
    order = torch.rand(n, nb, generator=gen, device=device).argsort(1)
    m = torch.zeros(n, nb, dtype=torch.bool, device=device)
    m.scatter_(1, order[:, :k], True)
    m = m.view(n, *coarse)
    for ax, b in enumerate(block, 1):
        m = m.repeat_interleave(b, ax)
    return m


class Crops(torch.utils.data.Dataset):
    def __init__(self, X, idx, augment):
        self.X, self.idx, self.augment = X, np.asarray(idx), augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = np.asarray(self.X[self.idx[i]], dtype=np.float32)
        if self.augment:
            for ax in (0, 1):                               # in-plane flips; z kept (cranio-caudal)
                if np.random.rand() < 0.5:
                    x = np.flip(x, ax)
            if np.random.rand() < 0.5:                      # in-plane 90 deg (square 0.8 x 0.8 mm pixels)
                x = np.swapaxes(x, 0, 1)
            sy, sx, sz = np.random.randint(-2, 3, 3)
            x = np.roll(x, (sy, sx, int(sz) // 2), axis=(0, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(x))[None]


def load_data(paths, me_cache):
    arrays, groups = [], []
    for p in paths:
        X = np.load(p, mmap_mode='r')
        meta = pd.read_csv(p + '.csv') if os.path.exists(p + '.csv') else None
        g = meta['pid'].astype(str).to_numpy() if meta is not None and 'pid' in meta else np.arange(len(X)).astype(str)
        if len(g) != len(X):
            raise SystemExit(f'{p}: {len(X)} crops but {len(g)} rows in {p}.csv')
        arrays.append(X); groups.append(np.char.add(os.path.basename(p) + ':', g.astype(str)))
        print(f'[data] {p}: {len(X)} crops' + (f'; kinds {meta.kind.value_counts().to_dict()}' if meta is not None and 'kind' in meta else ''))
    if me_cache:
        X = np.load(me_cache, mmap_mode='r')
        meta = pd.read_csv(me_cache + '.csv')
        tr = np.flatnonzero(meta['split'].to_numpy() == 'train')
        arrays.append(X[tr]); groups.append(np.char.add('me:', meta['pid'].astype(str).to_numpy()[tr]))
        print(f'[data] {me_cache}: {len(tr)} ME train crops (val / test rows are never read)')
    X = np.concatenate([np.asarray(a, dtype=np.float16) for a in arrays]) if len(arrays) > 1 else arrays[0]
    return X, np.concatenate(groups)


def save_qc(model, X, idx, device, path, args):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('qc png skipped:', e); return
    model.eval()
    x = torch.from_numpy(np.asarray(X[idx[:6]], dtype=np.float32))[:, None].to(device)
    g = torch.Generator(device=device).manual_seed(0)
    m = random_mask(len(x), model.enc.grid, args.block, args.mask_ratio, g, device)
    with torch.no_grad():
        _, pred = model(x, m)
    n, (gy, gx, gz) = len(x), model.enc.grid
    rec = pred.float().view(n, gy, gx, gz, *CELL).permute(0, 1, 4, 2, 5, 3, 6).reshape(n, 64, 64, 32)
    mv = m.repeat_interleave(4, 1).repeat_interleave(4, 2).repeat_interleave(4, 3)
    shown = torch.where(mv, rec, x[:, 0])
    fig, ax = plt.subplots(3, n, figsize=(2 * n, 6))
    for j in range(n):
        for r, (img, t) in enumerate(((x[j, 0], 'input'), (x[j, 0].masked_fill(mv[j], -1.0), 'masked'), (shown[j], 'recon'))):
            ax[r, j].imshow(img[:, :, 16].cpu().numpy(), cmap='gray', vmin=-1, vmax=1); ax[r, j].axis('off')
            ax[r, j].set_title(t, fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True, help='crop .npy files (+ optional <npy>.csv with pid)')
    ap.add_argument('--me_cache', default=None, help='also use the ME train rows of this crop cache')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--warmup', type=int, default=5, help='epochs of linear warm-up')
    ap.add_argument('--mask_ratio', type=float, default=0.6)
    ap.add_argument('--block', type=int, nargs=3, default=[2, 2, 2], help='mask block in grid cells (y x z)')
    ap.add_argument('--val_frac', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    X, groups = load_data(a.data, a.me_cache)
    ug = np.unique(groups)
    rng = np.random.default_rng(a.seed)
    held = set(rng.choice(ug, size=max(1, int(round(a.val_frac * len(ug)))), replace=False))
    va = np.flatnonzero(np.isin(groups, list(held)))
    tr = np.flatnonzero(~np.isin(groups, list(held)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'train {len(tr)} crops, held-out {len(va)} crops ({len(held)} patients); device={device}'
          + ('' if device.type == 'cuda' else '  <-- no GPU visible, this will be slow'))

    dl = torch.utils.data.DataLoader(Crops(X, tr, True), batch_size=a.bs, shuffle=True, drop_last=True,
                                     num_workers=a.workers, persistent_workers=a.workers > 0)
    dv = torch.utils.data.DataLoader(Crops(X, va, False), batch_size=a.bs, shuffle=False, num_workers=0)
    model = MAE().to(device)
    # only what the MAE trains: the image path of the encoder + mask token + head (queries / fc idle)
    used = [p for n, p in model.named_parameters()
            if n.startswith(tuple('enc.' + q for q in PRETRAINED_PREFIXES)) or not n.startswith('enc.')]
    decay = [p for p in used if p.dim() > 1]
    no_decay = [p for p in used if p.dim() <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': no_decay, 'weight_decay': 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    total, warm = a.epochs * len(dl), a.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    print(f'MAE: {sum(p.numel() for p in used) / 1e6:.2f}M trained params, mask {a.mask_ratio} in {tuple(a.block)}-cell '
          f'blocks, {len(dl)} steps/epoch x {a.epochs}')

    os.makedirs(a.out, exist_ok=True)
    gv = torch.Generator(device=device)
    hist, best = [], float('inf')
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x in dl:
            x = x.to(device, non_blocking=True)
            m = random_mask(len(x), model.enc.grid, a.block, a.mask_ratio, device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss, _ = model(x, m)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(used, 3.0)
            opt.step(); sched.step(); tot += loss.item()
        model.eval(); gv.manual_seed(1234); vl, vn = 0.0, 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for x in dv:
                x = x.to(device)
                m = random_mask(len(x), model.enc.grid, a.block, a.mask_ratio, gv, device)
                vl += model(x, m)[0].item() * len(x); vn += len(x)
        vl /= max(vn, 1)
        hist.append(dict(epoch=ep + 1, train_l1=tot / len(dl), val_l1=vl))
        blob = {'arch': 'vitq', 'encoder': model.enc.state_dict(), 'pretrained_prefixes': list(PRETRAINED_PREFIXES),
                'mae': dict(vars(a), epoch=ep + 1, val_l1=vl)}
        torch.save(blob, os.path.join(a.out, 'mae.pt'))      # the last epoch is the one used
        flag = ''
        if vl < best:
            best, flag = vl, ' *'
        print(f'ep {ep + 1:3d} train L1 {tot / len(dl):.4f} | held-out L1 {vl:.4f}{flag} | lr {sched.get_last_lr()[0]:.2e} '
              f'| {time.time() - t0:.0f}s', flush=True)
    json.dump({'history': hist, 'args': vars(a)}, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1)
    save_qc(model, X, va, device, os.path.join(a.out, 'recon_qc.png'), a)
    print(f'\nwrote {a.out}/mae.pt (last epoch; held-out L1 {hist[-1]["val_l1"]:.4f}, best {best:.4f}) + recon_qc.png')


if __name__ == '__main__':
    main()
