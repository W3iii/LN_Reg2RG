"""Train the lesion encoder alone (no LLM) on size + TW_Lung_RADS (+ nodule type), compare poolings.

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
    --arch vitq     small 3D ViT on the 16x16x8 grid + 3 learnable attribute queries
                    (QueryLesionEncoder): token 0 -> size head, 1 -> type head, 2 -> RADS head

gap/center/attn emit one 4096-d token and all heads sit on it; vitq emits three and each
head sits on its own token. Either way the heads read exactly what the LLM would receive.

Extra training crops (--extra_train, e.g. LIDC-IDRI from data_prep/lidc/build_lidc_lesions.py):
appended to the training set only, with size and type labels and no RADS label (its loss is
masked for those rows). Validation and model selection stay on ME val.

MAE initialisation (--init_encoder, vitq only): start the image path of the encoder (conv trunk,
proj, pos, grid transformer, mem_norm) from evaluation/pretrain_lesion_mae.py's mae.pt, masked-image
pretraining on unlabelled LIDC crops (data_prep/lidc/build_lidc_mae.py). The queries, their decoder
blocks, fc and the heads start random and everything is trained on ME as usual.

Data (split C, 2026-11): --meta $DATA_ROOT/nodule_metadata_v2.csv. Only train / val rows with
a TW_Lung_RADS grade are cached, i.e. ME only; LDCT is test_ldct and never read here. The
crop cache and the type labels are named after the metadata file, so the old-split cache
(lesion_crops_trainval.npy, outputs/nodule_type.csv) is never reused by a v2 run.

Losses
    L = SmoothL1(log size)                       all nodules
      + w_rads * CE_w(TW_Lung_RADS)              all nodules
      + w_type * L_type                          nodules whose own report text names a type
      + w_mil  * L_mil                           lobes whose text names a type no labelled
                                                 nodule explains (after --mil_warmup epochs)

Type labels come from data_prep/build_nodule_type.py and are *sets*: {GGO} is a hard
label, {GGO, part-solid} ('subsolid') a partial one. "The report does not say" is
never a negative.

    L_type = weighted mean over labelled nodules of  -log sum_{t in S_i} p_i(t)
             (= ordinary CE when S_i has one type; unlabelled nodules contribute 0,
              and the mean is over labelled nodules only, so sparse labels are not diluted)
    L_mil  = weighted mean over (lobe bag B, mentioned set M) of
             -log P(at least one nodule of B is in M),  P = 1 - prod_{i in B} (1 - sum_{t in M} p_i(t))
             (noisy-OR; only what the text says is used, never what it leaves out)

Class weights 1/sqrt(freq) on the hard labels, normalised to mean 1 over them.
Batches are built from whole lobe bags so every MIL bag is complete in one batch.

--type_loss none | ce | ce_mil  selects the ablation arm (none = the B3e encoder).

--concept adds a concept bottleneck for the grade: g(size, type) -> RADS, a small MLP
that sees only the predicted size and type probabilities (see Net), trained with
    + w_cbm * CE_w(g -> RADS) + w_consist * symmetric KL(direct grade, concept grade)
The direct rads_head stays. Reported per epoch: the concept path's kappa / AUC, its
agreement with the direct head, and how much higher g grades a solid than a GGO at the
same predicted size (Lung-RADS says it should).

--concept frozen (arm C) keeps g fixed instead: before the encoder trains, g is fitted on
the training nodules that have a hard type label -- TRUE size + one-hot type -> RADS --
and then frozen. While the encoder trains, the grade loss still flows through the frozen
g back into the type head, but only for nodules whose own text gives no type, and only
after --mil_warmup. A fixed g is a fixed rule: the type head can only move toward a type
that explains the grade under that rule, not redefine "solid" as "graded high", which is
what the jointly trained g did (OOF type bacc -0.046).

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
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
sys.path.insert(0, os.path.join(HERE, '..', 'data_prep'))
sys.path.insert(0, HERE)

from Model.my_embedding_layer import LesionEncoder, PooledLesionEncoder, QueryLesionEncoder, vitq_options  # noqa: E402  (single source: MyEmbedding loads these state_dicts)
from lesion_utils import crop_lesion, normalize_lesion, lesion_views, BBOX_COLS  # noqa: E402
from extract_lesion_features import find_image                    # noqa: E402
from build_nodule_type import TYPES, build as build_type_labels   # noqa: E402

N_TYPES = len(TYPES)
DEFAULT_TYPE_CSV = os.path.join(HERE, '..', 'outputs', 'nodule_type.csv')


# ------------------------------------------------------------------ data
def cache_matches(cache, meta_csv, splits=('train', 'val')):
    """The cache's rows must be exactly the current metadata's train/val nodules, same split."""
    c = pd.read_csv(cache + '.csv')
    nm = pd.read_csv(meta_csv)
    nm = nm[nm['split'].isin(list(splits))].dropna(subset=BBOX_COLS + ['eq_diam_mm', 'tw_lung_rads'])
    a = c[['Volumename', 'nodule_id', 'split']].astype(str).reset_index(drop=True)
    b = nm[['Volumename', 'nodule_id', 'split']].astype(str).reset_index(drop=True)
    return a.equals(b)


def default_paths(meta_csv):
    """(crop cache, type csv) for a metadata file; the original file keeps its original names."""
    tag = os.path.splitext(os.path.basename(meta_csv))[0]
    out = os.path.join(HERE, '..', 'outputs')
    if tag == 'nodule_metadata':
        return os.path.join(out, 'lesion_crops_trainval.npy'), DEFAULT_TYPE_CSV
    return os.path.join(out, f'lesion_crops_trainval_{tag}.npy'), os.path.join(out, f'nodule_type_{tag}.csv')


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


VIEW_FILES = {'m': '_mask.npy', 'z': '_zoom.npy', 'zm': '_zoommask.npy'}


def view_paths(cache):
    base = cache[:-4] if cache.endswith('.npy') else cache
    return {k: base + suf for k, suf in VIEW_FILES.items()}


def build_view_cache(data_root, meta_csv, cache):
    """Mask channel + zoom view (lesion_utils.lesion_views) for the rows of an existing crop cache."""
    import nibabel as nib
    rows = pd.read_csv(cache + '.csv')
    nm = pd.read_csv(meta_csv)[['Volumename', 'nodule_id'] + BBOX_COLS]
    rows = rows[['Volumename', 'nodule_id', 'eq_diam_mm']].merge(nm, on=['Volumename', 'nodule_id'], how='left',
                                                                validate='one_to_one')
    n = len(rows)
    out = {'m': np.zeros((n, 64, 64, 32), np.uint8), 'z': np.zeros((n, 32, 32, 16), np.float16),
           'zm': np.zeros((n, 32, 32, 16), np.uint8)}
    t0 = time.time()
    vols = list(dict.fromkeys(rows['Volumename']))
    for vi, vol in enumerate(vols):
        im = nib.load(find_image(os.path.join(data_root, 'images'), vol), mmap=True)
        img = np.asarray(im.dataobj).astype(np.float32)
        folder = vol[:-7] if vol.endswith('.nii.gz') else vol
        lab = np.asarray(nib.load(os.path.join(data_root, 'masks', f'seg_{folder}', 'nodules.nii.gz')).dataobj)
        sp = tuple(float(s) for s in im.header.get_zooms()[:3])
        for r in rows.index[rows['Volumename'] == vol]:
            b = tuple(int(rows.at[r, c]) for c in BBOX_COLS)
            _, m, z, zm = lesion_views(img, lab, int(rows.at[r, 'nodule_id']), b, rows.at[r, 'eq_diam_mm'], sp)
            out['m'][r], out['z'][r], out['zm'][r] = m, z, zm
        if (vi + 1) % 100 == 0:
            print(f'  views {vi + 1}/{len(vols)} volumes {time.time() - t0:.0f}s', flush=True)
    for k, p in view_paths(cache).items():
        np.save(p, out[k])
    print(f'wrote {sorted(view_paths(cache).values())}; empty native masks: {int((out["m"].reshape(n, -1).max(1) == 0).sum())}')


def load_views(X, cache, arch, data_root, meta_csv):
    """X (the crop array) alone, or a dict of the views the arch reads: x [+ m] [+ z (+ zm)]."""
    o = vitq_options(arch) if arch.startswith('vitq') else dict(in_ch=1, zoom=False)
    if o['in_ch'] == 1 and not o['zoom']:
        return X
    paths = view_paths(cache)
    if not all(os.path.exists(p) for p in paths.values()):
        print(f'[views] building {sorted(paths.values())} (mask channel + zoom view, one pass over the volumes)')
        build_view_cache(data_root, meta_csv, cache)
    V = {k: np.load(p, mmap_mode='r') for k, p in paths.items()}
    if any(len(v) != len(X) for v in V.values()):
        raise SystemExit(f'view caches {sorted(paths.values())} do not match {cache}; delete them to rebuild')
    out = {'x': X}
    if o['in_ch'] == 2:
        out['m'] = V['m']
    if o['zoom']:
        out['z'] = V['z']
        if o['in_ch'] == 2:
            out['zm'] = V['zm']
    print(f'[views] {arch}: ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in out.items()))
    return out


def subset(X, idx):
    """Rows idx of the crop array, or of every view in a dict of views (loaded into memory)."""
    if isinstance(X, dict):
        return {k: np.asarray(v[idx]) for k, v in X.items()}
    return np.asarray(X[idx])


def to_device(x, device):
    return {k: v.to(device) for k, v in x.items()} if isinstance(x, dict) else x.to(device)


class TypeTargets:
    """Type supervision aligned to the rows of the crop cache.

    tset   [N, 4] float, multi-hot allowed types (all zero = no label)
    hard   [N] int, the type when tset has exactly one, else -1
    bag    [N] int, MIL bag id (-1 = none); bags never cross lobes, so never cross patients
    ments  {bag_id: [N_TYPES]-float masks}, the type sets the lobe text names
    """

    def __init__(self, meta, type_csv, data_root=None, meta_csv=None):
        if not os.path.exists(type_csv):
            if data_root is None:
                raise FileNotFoundError(type_csv)
            print(f'[type] {type_csv} missing, building it from {data_root} ({meta_csv or "nodule_metadata.csv"})')
            build_type_labels(data_root, type_csv, meta_csv=meta_csv)
        t = pd.read_csv(type_csv, keep_default_na=False)
        if t.duplicated(['Volumename', 'nodule_id']).any():
            raise ValueError(f'{type_csv}: duplicate (Volumename, nodule_id)')
        t = t.drop(columns=[c for c in ('split', 'lobe') if c in t.columns])
        m = meta[['Volumename', 'nodule_id']].merge(t, on=['Volumename', 'nodule_id'], how='left', validate='one_to_one')
        if m['type_set'].isna().any():
            raise ValueError(f'{type_csv} has no row for {int(m.type_set.isna().sum())} cached nodules; rebuild it')
        idx = {n: i for i, n in enumerate(TYPES)}
        self.tset = np.zeros((len(m), N_TYPES), dtype=np.float32)
        for r, s in enumerate(m['type_set']):
            for n in filter(None, str(s).split('|')):
                self.tset[r, idx[n]] = 1
        self.hard = np.where(self.tset.sum(1) == 1, self.tset.argmax(1), -1).astype(np.int64)
        self.bag = m['bag_id'].astype(int).to_numpy()
        self.ments = {}
        for b, ms in m.loc[self.bag >= 0, ['bag_id', 'bag_mentions']].drop_duplicates('bag_id').itertuples(index=False):
            masks = []
            for s in ms.split(';'):
                v = np.zeros(N_TYPES, dtype=np.float32)
                for n in s.split('|'):
                    v[idx[n]] = 1
                masks.append(v)
            self.ments[int(b)] = masks

    def append_sets(self, sets):
        """Extra rows (e.g. LIDC) with their own type sets ('GGO|part-solid', ...), no MIL bag."""
        idx = {n: i for i, n in enumerate(TYPES)}
        ts = np.zeros((len(sets), N_TYPES), dtype=np.float32)
        for r, s in enumerate(sets):
            for n in filter(None, str(s).split('|')):
                ts[r, idx[n]] = 1
        self.tset = np.concatenate([self.tset, ts])
        self.hard = np.where(self.tset.sum(1) == 1, self.tset.argmax(1), -1).astype(np.int64)
        self.bag = np.concatenate([self.bag, np.full(len(sets), -1, dtype=self.bag.dtype)])

    def class_weights(self, idx):
        """1/sqrt(freq) over the hard labels in idx, normalised to mean 1 over them."""
        h = self.hard[idx]
        n = np.bincount(h[h >= 0], minlength=N_TYPES).astype(np.float64)
        w = 1.0 / np.sqrt(n.clip(1))
        w /= (n * w).sum() / max(n.sum(), 1)
        return w.astype(np.float32)


class Crops(torch.utils.data.Dataset):
    """Returns (crop, log diam, rads, row) -- row indexes the cache, for type targets.

    With views (a dict from load_views) the crop is a dict too: 'x' (C, 64, 64, 32) with the mask
    as channel 1 when present, and 'z' (C, 32, 32, 16) the zoom view (+ its mask). Flips are shared
    by all views; the jitter is per view (+-2 voxels native, +-1 zoom)."""

    def __init__(self, X, logd, rads, augment, rows=None):
        self.X, self.logd, self.rads, self.augment = X, logd, rads, augment
        n = len(X['x']) if isinstance(X, dict) else len(X)
        self.rows = np.arange(n) if rows is None else np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def _aug(self, arrs, flips, jit):
        out = []
        for a in arrs:
            for ax, f in zip((0, 1), flips):         # in-plane flips; z kept (cranio-caudal)
                if f:
                    a = np.flip(a, ax)
            if jit is not None:
                a = np.roll(a, jit, axis=(0, 1, 2))
            out.append(np.ascontiguousarray(a))
        return out

    def __getitem__(self, i):
        if not isinstance(self.X, dict):            # unchanged single-crop path (same random stream)
            x = self.X[i].astype(np.float32)
            if self.augment:
                for ax in (0, 1):                   # in-plane flips; z kept (cranio-caudal)
                    if np.random.rand() < 0.5:
                        x = np.flip(x, ax)
                sy, sx, sz = np.random.randint(-2, 3, 3)  # small jitter; the nodule stays central
                x = np.roll(x, (sy, sx, int(sz) // 2), axis=(0, 1, 2))
            return torch.from_numpy(np.ascontiguousarray(x))[None], self.logd[i], self.rads[i], int(self.rows[i])
        flips = np.random.rand(2) < 0.5 if self.augment else (False, False)
        def jitter(r):                              # small jitter; the nodule stays central
            if not self.augment:
                return None
            sy, sx, sz = np.random.randint(-r, r + 1, 3)
            return (int(sy), int(sx), int(sz) // 2 if r > 1 else int(sz))
        nat = [self.X['x'][i].astype(np.float32)] + ([self.X['m'][i].astype(np.float32)] if 'm' in self.X else [])
        out = {'x': torch.from_numpy(np.stack(self._aug(nat, flips, jitter(2))))}
        if 'z' in self.X:
            zz = [self.X['z'][i].astype(np.float32)] + ([self.X['zm'][i].astype(np.float32)] if 'zm' in self.X else [])
            out['z'] = torch.from_numpy(np.stack(self._aug(zz, flips, jitter(1))))
        return out, self.logd[i], self.rads[i], int(self.rows[i])


class BagBatchSampler(torch.utils.data.Sampler):
    """Shuffled batches of ~bs crops that never split a MIL bag.

    Units are whole bags or single nodules; units are shuffled and packed until a
    batch reaches bs. The last batch is dropped when it is under half full.
    """

    def __init__(self, bag, bs, seed=0):
        self.bs = bs
        self.rng = np.random.default_rng(seed)
        units = {}
        for i, b in enumerate(bag):
            units.setdefault(('b', int(b)) if b >= 0 else ('n', i), []).append(i)
        self.units = list(units.values())

    def _batches(self, order):
        batch = []
        for u in order:
            batch.extend(self.units[u])
            if len(batch) >= self.bs:
                yield batch
                batch = []
        if len(batch) >= self.bs // 2:
            yield batch

    def __iter__(self):
        return self._batches(self.rng.permutation(len(self.units)))

    def __len__(self):
        # exact for the identity order; a shuffled epoch differs by at most a batch or so
        return sum(1 for _ in self._batches(range(len(self.units))))


def train_loader(X, logd, rads, idx, bag, bs, workers, seed):
    ds = Crops(subset(X, idx), logd[idx], rads[idx], True, rows=idx)
    return torch.utils.data.DataLoader(ds, batch_sampler=BagBatchSampler(bag[idx], bs, seed), num_workers=workers)


G_LR_MULT = 10.0   # g is a 5 -> 32 -> 4 MLP on top of the heads: at the encoder's 3e-4 it is still
                   # near chance after a few epochs, so it learns 10x faster


def make_optim(model, lr):
    g = list(model.g.parameters())
    gid = {id(q) for q in g}
    rest = [q for q in model.parameters() if id(q) not in gid]
    return torch.optim.AdamW([{'params': rest, 'lr': lr}, {'params': g, 'lr': lr * G_LR_MULT}], weight_decay=1e-2)


def one_cycle(opt, lr, epochs, dl):
    # +2 per epoch: a shuffled epoch can pack a batch more than len() counts, and
    # OneCycleLR raises if stepped past total_steps; the unused tail is ~0.1% of steps
    return torch.optim.lr_scheduler.OneCycleLR(opt, [grp['lr'] for grp in opt.param_groups],
                                               total_steps=epochs * (len(dl) + 2))


def eval_loader(X, logd, rads, idx, bs, workers=0):
    return torch.utils.data.DataLoader(Crops(subset(X, idx), logd[idx], rads[idx], False, rows=idx),
                                       batch_size=bs, shuffle=False, num_workers=workers)


# ------------------------------------------------------------------ models
LOG_MM_CENTRE, LOG_MM_SCALE = 1.7, 0.5     # log(eq_diam_mm): median ~5.5 mm, most nodules 3-15 mm


class Net(nn.Module):
    """Encoder + three parallel heads on the 4096-d token, plus the concept path g.

    g (--concept) grades a nodule from its *concepts* only -- the predicted size and
    the predicted type probabilities -- the way TW Lung-RADS itself does (a 7 mm solid
    nodule is category 3, a 7 mm GGO category 2). It never sees the token, so it is a
    real bottleneck: change the size or the type and the grade follows. The direct
    rads_head stays as the unconstrained reference.
    """

    def __init__(self, arch):
        super().__init__()
        if arch == 'gap':
            self.enc = LesionEncoder(dim=4096)
        elif arch.startswith('vitq'):
            self.enc = QueryLesionEncoder(dim=4096, **vitq_options(arch))
        else:
            self.enc = PooledLesionEncoder(arch)
        self.size_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1))
        self.rads_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 4))
        self.type_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, N_TYPES))
        self.g = nn.Sequential(nn.Linear(1 + N_TYPES, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, x, size_mm=None):
        # vitq_s* gets the mask size as an input (its size head is then a sanity check, not a result);
        # x is a dict of views ('x', 'z') for vitq_*m / vitq_*z
        if isinstance(x, dict):
            t = self.enc(x['x'], size_mm, zoom=x.get('z'))
        else:
            t = self.enc(x, size_mm) if getattr(self.enc, 'size_token', False) else self.enc(x)
        if t.dim() == 3:          # vitq: one token per attribute query (QueryLesionEncoder.QUERY_NAMES)
            return self.size_head(t[:, 0])[:, 0], self.rads_head(t[:, 2]), self.type_head(t[:, 1]), t
        return self.size_head(t)[:, 0], self.rads_head(t), self.type_head(t), t

    def concept(self, log_mm, type_probs):
        """RADS logits from (log size, type probabilities) alone."""
        z = ((log_mm.float() - LOG_MM_CENTRE) / LOG_MM_SCALE)[:, None]
        return self.g(torch.cat([z, type_probs.float()], 1))


def load_mae_init(enc, path, arch):
    """Image path of the encoder from pretrain_lesion_mae.py; queries / decoder / fc stay random."""
    blob = torch.load(path, map_location='cpu', weights_only=False)
    if blob.get('arch') != ('vitq' if arch.startswith('vitq') else arch):    # vitq_* = vitq + options
        raise SystemExit(f'{path} is an arch={blob.get("arch")!r} encoder, not {arch!r}')
    pre = tuple(blob.get('pretrained_prefixes', ()))
    sd = {k: v for k, v in blob['encoder'].items() if pre and k.startswith(pre)}
    own = enc.state_dict()
    for k, v in list(sd.items()):
        # mask channel (vitq_*m): the first conv gets a 2nd input channel; MAE weights go to the
        # image channel and the mask channel starts at zero, so the initial output is the MAE one
        if k in own and v.dim() == 5 and own[k].shape[1] != v.shape[1] and own[k].shape[0] == v.shape[0]:
            w = torch.zeros_like(own[k]); w[:, :v.shape[1]] = v; sd[k] = w
            print(f'[init] {k}: MAE {tuple(v.shape)} -> {tuple(w.shape)} (extra input channels start at 0)')
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    if unexpected or not sd:
        raise SystemExit(f'{path}: {len(sd)} tensors to load, unexpected {unexpected[:5]}')
    mae = blob.get('mae', {})
    print(f'[init] {path}: {len(sd)} tensors ({sum(v.numel() for v in sd.values()) / 1e6:.2f}M) from MAE epoch '
          f'{mae.get("epoch")} (held-out L1 {mae.get("val_l1", float("nan")):.4f}); start random: '
          + ', '.join(sorted({k.split('.')[0] for k in missing})))


def checkpoint(model, arch, metrics, **extra):
    return {'arch': arch, 'encoder': model.enc.state_dict(),
            'size_head': model.size_head.state_dict(), 'rads_head': model.rads_head.state_dict(),
            'type_head': model.type_head.state_dict(), 'g': model.g.state_dict(), 'type_names': list(TYPES),
            'metrics': {k: float(v) for k, v in metrics.items() if np.isscalar(v)}, **extra}


# ------------------------------------------------------------------ losses
def type_loss(type_logits, tset, cw):
    """Partial-label CE: -log sum_{t in S_i} p_i(t), class-weighted mean over labelled rows only."""
    lab = tset.sum(1) > 0
    if not lab.any():
        return type_logits.sum() * 0.0
    logp = F.log_softmax(type_logits[lab].float(), 1)
    s = tset[lab]
    lp = torch.logsumexp(logp.masked_fill(s == 0, float('-inf')), 1)
    w = (s * cw).sum(1) / s.sum(1)
    return -(w * lp).sum() / w.sum()


def mil_loss(type_logits, bag, ments, cw):
    """Noisy-OR over each lobe bag: -log(1 - prod_i (1 - P(nodule i is in M))), per mentioned set M."""
    p = F.softmax(type_logits.float(), 1)
    losses, weights = [], []
    for b in torch.unique(bag[bag >= 0]).tolist():
        pb = p[bag == b]
        for M in ments[b]:
            M = torch.as_tensor(M, device=p.device)
            q = (pb * M).sum(1).clamp(0.0, 1.0 - 1e-6)           # P(nodule i is one of M)
            log_none = torch.log1p(-q).sum().clamp(max=-1e-6)     # log P(no nodule of the bag is in M)
            losses.append(-torch.log(-torch.expm1(log_none)))      # -log P(at least one is)
            weights.append((M * cw).sum() / M.sum())
    if not losses:
        return type_logits.sum() * 0.0
    w = torch.stack(weights)
    return (w * torch.stack(losses)).sum() / w.sum()


class Objective:
    """The full Stage-1 loss; shared by this script and predict_lesion_attributes.py."""

    def __init__(self, a, rads_train, tt, idx, device, model=None):
        self.a, self.tt, self.device, self.model = a, tt, device, model
        if a.concept != 'none' and model is None:
            raise ValueError('--concept needs the model (for its g)')
        rt = np.asarray(rads_train)
        rt = rt[rt >= 0]                         # extra rows (--extra_train) carry no RADS label
        self.cw_rads = torch.tensor(len(rt) / (4 * np.bincount(rt, minlength=4).clip(1)), dtype=torch.float32, device=device)
        self.cw_type = torch.tensor(tt.class_weights(idx), device=device)
        self.tset = torch.tensor(tt.tset, device=device)
        self.bag = torch.tensor(tt.bag, device=device)

    def __call__(self, out, ld, r, rows, epoch):
        s, rl, tl, _ = out
        a = self.a
        has_r = r >= 0
        parts = {'size': F.smooth_l1_loss(s.float(), ld, beta=0.1),
                 'rads': (F.cross_entropy(rl.float()[has_r], r[has_r], weight=self.cw_rads) if has_r.any()
                          else rl.float().sum() * 0.0)}
        loss = parts['size'] + a.w_rads * parts['rads']
        if a.type_loss in ('ce', 'ce_mil'):
            parts['type'] = type_loss(tl, self.tset[rows], self.cw_type)
            loss = loss + a.w_type * parts['type']
        if a.type_loss == 'ce_mil' and epoch >= a.mil_warmup:
            parts['mil'] = mil_loss(tl, self.bag[rows], self.tt.ments, self.cw_type)
            loss = loss + a.w_mil * parts['mil']
        if a.concept == 'frozen' and epoch >= a.mil_warmup:
            # arm C: the grade loss reaches the type head through the FROZEN g, only for
            # nodules whose own text names no type (labelled ones keep their own CE)
            free = self.tset[rows].sum(1) == 0
            if free.any():
                rc = self.model.concept(s.detach()[free], F.softmax(tl.float()[free], 1))
                w = self.cw_rads[r[free]]
                parts['rads_cbm'] = (w * F.cross_entropy(rc, r[free], reduction='none')).sum() / w.sum()
                loss = loss + a.w_cbm * parts['rads_cbm']
        if a.concept == 'joint':
            # size is detached: it has its own full supervision, and the grade must not
            # bend it. The type probabilities are not: RADS labels exist for every nodule,
            # so this path also teaches the type head on the ones the reports never type.
            rc = self.model.concept(s.detach(), F.softmax(tl.float(), 1))
            parts['rads_cbm'] = F.cross_entropy(rc, r, weight=self.cw_rads)
            lp, lq = F.log_softmax(rl.float(), 1), F.log_softmax(rc, 1)
            parts['consist'] = 0.5 * (F.kl_div(lp, lq, log_target=True, reduction='batchmean')
                                      + F.kl_div(lq, lp, log_target=True, reduction='batchmean'))
            loss = loss + a.w_cbm * parts['rads_cbm'] + a.w_consist * parts['consist']
        return loss, {k: v.item() for k, v in parts.items()}


def concept_suffix(a):
    return {'none': '', 'joint': '_cbm', 'frozen': '_cbmfrozen'}[a.concept]


@torch.enable_grad()
def pretrain_g(model, logd, rads, tt, idx, device, steps=4000, seed=0):
    """Arm C: fit g on (true log size, one-hot hard type) -> RADS over the hard-labelled
    nodules in idx (training nodules only), then freeze it."""
    lab = idx[tt.hard[idx] >= 0]
    x = torch.as_tensor(logd[lab], device=device)
    oh = F.one_hot(torch.as_tensor(tt.hard[lab], device=device), N_TYPES).float()
    y = torch.as_tensor(rads[lab], device=device)
    cw = torch.tensor(len(lab) / (4 * np.bincount(rads[lab], minlength=4).clip(1)), dtype=torch.float32, device=device)
    for q in model.g.parameters():
        q.requires_grad_(True)
    opt = torch.optim.AdamW(model.g.parameters(), lr=3e-3, weight_decay=1e-2)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(lab), (64,), generator=gen).to(device)
        loss = F.cross_entropy(model.concept(x[i], oh[i]), y[i], weight=cw)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for q in model.g.parameters():
        q.requires_grad_(False)
    with torch.no_grad():
        pred = model.concept(x, oh).argmax(1).cpu().numpy() + 1
        e = {}
        for name in ('solid', 'GGO'):
            o = torch.zeros(1, N_TYPES, device=device); o[0, TYPES.index(name)] = 1
            pr = F.softmax(model.concept(torch.tensor([float(np.log(7.0))], device=device), o), 1)
            e[name] = float((pr * torch.arange(1, 5, device=device)).sum())
    print(f'[g] pretrained on {len(lab)} hard-labelled nodules, frozen: train kappa '
          f'{cohen_kappa_score(rads[lab] + 1, pred, weights="quadratic"):.3f}; at 7 mm solid {e["solid"]:.2f} vs GGO {e["GGO"]:.2f}', flush=True)


def add_loss_args(ap):
    ap.add_argument('--type_loss', choices=['none', 'ce', 'ce_mil'], default='ce_mil',
                    help='none = size + RADS only (the B3e encoder); ce = + nodule-level type; ce_mil = + lobe-level MIL')
    ap.add_argument('--type_csv', default=None, help='data_prep/build_nodule_type.py output (built if missing); '
                    'default outputs/nodule_type.csv for nodule_metadata.csv, outputs/nodule_type_<meta>.csv otherwise')
    ap.add_argument('--w_rads', type=float, default=1.0)
    ap.add_argument('--w_type', type=float, default=1.0)
    ap.add_argument('--w_mil', type=float, default=0.5)
    ap.add_argument('--mil_warmup', type=int, default=10, help='epochs before L_mil is switched on')
    ap.add_argument('--concept', nargs='?', const='joint', default='none', choices=['none', 'joint', 'frozen'],
                    help='concept bottleneck g(size, type) -> RADS, see Net. joint = g trained with the encoder '
                         '(bare --concept); frozen = arm C, g pretrained on labelled nodules then fixed')
    ap.add_argument('--w_cbm', type=float, default=1.0, help='weight of CE on the concept-path grade')
    ap.add_argument('--w_consist', type=float, default=0.1, help='symmetric KL between direct and concept grades')


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, device, tt=None, concept=False):
    model.eval()
    P, L, T, C, Y, R, rows = [], [], [], [], [], [], []
    for x, logd, rads, row in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            s, r, t, _ = model(to_device(x, device), logd.to(device).exp())
        tp = F.softmax(t.float(), 1)
        P.append(s.float().cpu()); L.append(F.softmax(r.float(), 1).cpu()); T.append(tp.cpu())
        C.append(F.softmax(model.concept(s, tp), 1).cpu())
        Y.append(logd); R.append(rads); rows.append(row)
    P, L, T, C, Y, R, rows = (torch.cat(v).numpy() for v in (P, L, T, C, Y, R, rows))
    d, dhat = np.exp(Y), np.exp(P)
    big = d >= 6
    rads1 = R + 1
    pred = L.argmax(1) + 1
    m = dict(rho=spearmanr(d, dhat).statistic, MAE=float(np.abs(dhat - d).mean()),
             MAE_ge6=float(np.abs(dhat - d)[big].mean()),
             kappa=cohen_kappa_score(rads1, pred, weights='quadratic'),
             auc_ge3=roc_auc_score(rads1 >= 3, L[:, 2:].sum(1)), acc=float((pred == rads1).mean()))
    if tt is not None:
        m.update(type_metrics(T, rows, tt))
    if concept and concept != 'none':
        pc = C.argmax(1) + 1
        m.update(kappa_cbm=cohen_kappa_score(rads1, pc, weights='quadratic'),
                 auc_ge3_cbm=roc_auc_score(rads1 >= 3, C[:, 2:].sum(1)), acc_cbm=float((pc == rads1).mean()),
                 agree_cbm=float((pc == pred).mean()))
        m.update(concept_type_effect(model, P, device))
    return m


@torch.no_grad()
def concept_type_effect(model, log_mm, device):
    """Does g use the type? Grade every nodule at its own predicted size twice, as a pure
    solid and as a pure GGO, and compare the expected category (1-4). Lung-RADS says solid
    should grade higher at the same size; ~0 means g is reading the size alone."""
    s = torch.as_tensor(log_mm, device=device)
    out = {}
    exp = {}
    for name in ('solid', 'GGO'):
        oh = torch.zeros(len(s), N_TYPES, device=device)
        oh[:, TYPES.index(name)] = 1
        p = F.softmax(model.concept(s, oh), 1)
        exp[name] = (p * torch.arange(1, 5, device=device)).sum(1).cpu().numpy()
    d = exp['solid'] - exp['GGO']
    mm = np.exp(log_mm)
    out['cbm_solid_minus_ggo'] = float(d.mean())
    band = (mm >= 6) & (mm < 8)
    if band.any():
        out['cbm_solid_minus_ggo_6to8mm'] = float(d[band].mean())
    return out


def type_metrics(T, rows, tt):
    """Hard-labelled nodules: acc, macro-F1 / balanced acc over the classes present, per-class recall.
    MIL bags: share of (bag, mention) pairs where some member's argmax is in the mentioned set."""
    m = {}
    h = tt.hard[rows]
    lab = h >= 0
    tp = T.argmax(1)
    if lab.any():
        present = sorted(set(h[lab].tolist()))
        m['type_n'] = int(lab.sum())
        m['type_acc'] = float((tp[lab] == h[lab]).mean())
        m['type_f1'] = float(f1_score(h[lab], tp[lab], labels=present, average='macro', zero_division=0))
        rec = [float((tp[lab & (h == c)] == c).mean()) for c in present]
        m['type_bacc'] = float(np.mean(rec))
        for c, rc in zip(present, rec):
            m[f'recall_{TYPES[c]}'] = rc
            m[f'n_{TYPES[c]}'] = int((h[lab] == c).sum())
    bag = tt.bag[rows]
    hits = []
    for b in np.unique(bag[bag >= 0]):
        mem = tp[bag == b]
        for M in tt.ments[int(b)]:
            hits.append(bool(M[mem].any()))
    if hits:
        m['bag_hit'] = float(np.mean(hits)); m['bag_n'] = len(hits)
    return m


def fmt_cbm(m):
    if 'kappa_cbm' not in m:
        return ''
    return (f' | concept kappa {m["kappa_cbm"]:.3f} AUC>=3 {m["auc_ge3_cbm"]:.3f} agree {m["agree_cbm"]:.2f} '
            f'solid-GGO {m["cbm_solid_minus_ggo"]:+.2f} (6-8mm {m.get("cbm_solid_minus_ggo_6to8mm", float("nan")):+.2f})')


def fmt_type(m):
    if 'type_f1' not in m:
        return ''
    rec = ' '.join(f'{t}:{m[f"recall_{t}"]:.2f}' for t in TYPES if f'recall_{t}' in m)
    s = f' | type F1 {m["type_f1"]:.3f} bacc {m["type_bacc"]:.3f} (n={m["type_n"]}; {rec})'
    if 'bag_hit' in m:
        s += f' bag-hit {m["bag_hit"]:.2f}'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', choices=['gap', 'center', 'attn', 'vitq', 'vitq_s', 'vitq_sm', 'vitq_sz', 'vitq_szm'],
                    required=True, help='vitq_<flags>: s = mask size as input, m = nodule mask channel, z = zoom view')
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--meta', default=None, help='nodule metadata csv (default <data_root>/nodule_metadata.csv; '
                    'split C: <data_root>/nodule_metadata_v2.csv)')
    ap.add_argument('--cache', default=None, help='crop cache .npy (default named after --meta)')
    ap.add_argument('--extra_train', default=None, help='extra training crops .npy with labels in <npy>.csv '
                    '(Volumename, nodule_id, eq_diam_mm, type_set), e.g. LIDC; no RADS, train only')
    ap.add_argument('--init_encoder', default=None, help='mae.pt of evaluation/pretrain_lesion_mae.py: start the '
                    "encoder's image path from MAE pretraining (vitq)")
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    add_loss_args(ap)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='dir for best.pt + metrics.json')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()

    meta_csv = a.meta or os.path.join(a.data_root, 'nodule_metadata.csv')
    d_cache, d_type = default_paths(meta_csv)
    a.cache = a.cache or d_cache
    a.type_csv = a.type_csv or d_type
    if not os.path.exists(a.cache):
        build_cache(a.data_root, meta_csv, a.cache)
    elif not cache_matches(a.cache, meta_csv):
        raise SystemExit(f'{a.cache} was built from a different nodule list / split than {meta_csv}; '
                         f'delete it (and {a.cache}.csv) or pass another --cache')
    print(f'[data] meta={meta_csv} cache={a.cache} type_csv={a.type_csv}')
    X = np.load(a.cache, mmap_mode='r')
    meta = pd.read_csv(a.cache + '.csv')
    logd = np.log(meta['eq_diam_mm'].to_numpy(np.float32))
    rads = (meta['tw_lung_rads'].to_numpy(np.int64) - 1)
    tr = np.flatnonzero(meta['split'].to_numpy() == 'train')
    va = np.flatnonzero(meta['split'].to_numpy() == 'val')
    tt = TypeTargets(meta, a.type_csv, a.data_root, meta_csv)
    if a.extra_train and a.arch.startswith('vitq') and (vitq_options(a.arch)['in_ch'] > 1 or vitq_options(a.arch)['zoom']):
        raise SystemExit('--extra_train crops have no mask / zoom view; use vitq or vitq_s with it')
    if a.extra_train:
        if a.concept != 'none':
            raise SystemExit('--extra_train rows have no RADS label; use --concept none')
        ex = pd.read_csv(a.extra_train + '.csv', keep_default_na=False, na_values={'eq_diam_mm': ['']})
        Xe = np.load(a.extra_train, mmap_mode='r')
        if len(ex) != len(Xe):
            raise SystemExit(f'{a.extra_train}: {len(Xe)} crops but {len(ex)} label rows')
        n0 = len(X)
        X = np.concatenate([np.asarray(X), np.asarray(Xe).astype(np.float16)])
        logd = np.concatenate([logd, np.log(ex['eq_diam_mm'].to_numpy(np.float32))])
        rads = np.concatenate([rads, np.full(len(ex), -100, dtype=np.int64)])
        tr = np.concatenate([tr, n0 + np.arange(len(ex))])
        tt.append_sets(ex['type_set'].tolist())
        print(f'[extra] {a.extra_train}: +{len(ex)} training crops; type sets {ex.type_set.value_counts().to_dict()}')
    X = load_views(X, a.cache, a.arch, a.data_root, meta_csv)
    print(f'arch={a.arch}  type_loss={a.type_loss}  train={len(tr)}  val={len(va)}  val patients={meta.pid.iloc[va].nunique()}')
    print(f'[type] train: {int((tt.hard[tr] >= 0).sum())} hard, {int(((tt.tset[tr].sum(1) > 1)).sum())} partial, '
          f'{int((tt.bag[tr] >= 0).sum())} in {len(np.unique(tt.bag[tr][tt.bag[tr] >= 0]))} MIL bags; '
          f'class weights {dict(zip(TYPES, tt.class_weights(tr).round(2).tolist()))}')

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else '  <-- no GPU visible, this will be slow'))
    dl_tr = train_loader(X, logd, rads, tr, tt.bag, a.bs, a.workers, a.seed)
    dl_va = eval_loader(X, logd, rads, va, a.bs, a.workers)

    model = Net(a.arch)
    if a.init_encoder:
        load_mae_init(model.enc, a.init_encoder, a.arch)
    model = model.to(device)
    if a.concept == 'frozen':
        pretrain_g(model, logd, rads, tt, tr, device, seed=a.seed)
    obj = Objective(a, rads[tr], tt, tr, device, model)
    opt = make_optim(model, a.lr)
    sched = one_cycle(opt, a.lr, a.epochs, dl_tr)
    # type_loss is always in the name, so the 'none' arm never overwrites the B3a encoder
    # (outputs/lesion_enc_<arch>_seed<seed>/best.pt, trained before the type head existed)
    out = a.out or os.path.join(HERE, '..', 'outputs', f'lesion_enc_{a.arch}_{a.type_loss}{concept_suffix(a)}_seed{a.seed}')
    os.makedirs(out, exist_ok=True)

    best, best_score, hist = None, -1e9, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0; parts = {}; nb = 0
        for x, ld, r, rows in dl_tr:
            x, ld, r, rows = to_device(x, device), ld.to(device), r.to(device), rows.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                o = model(x, ld.exp())
            loss, p = obj(o, ld, r, rows, ep)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); nb += 1
            for k, v in p.items():
                parts[k] = parts.get(k, 0.0) + v
        m = evaluate(model, dl_va, device, tt, a.concept)
        m.update(epoch=ep + 1, train_loss=tot / nb, **{f'loss_{k}': v / nb for k, v in parts.items()})
        hist.append(m)
        # model selection on val; the type term only once type is being trained
        score = m['rho'] + m['kappa'] + (m.get('type_f1', 0.0) if a.type_loss != 'none' else 0.0)
        flag = ''
        if score > best_score:
            best_score, best = score, m
            torch.save(checkpoint(model, a.arch, m, type_loss=a.type_loss, concept=a.concept,
                                  init_encoder=a.init_encoder), os.path.join(out, 'best.pt'))
            flag = ' *'
        lp = ' '.join(f'{k} {v / nb:.3f}' for k, v in parts.items())
        print(f'ep {ep+1:3d} loss {m["train_loss"]:.3f} ({lp}) | val rho {m["rho"]:.3f} MAE(>=6) {m["MAE_ge6"]:.2f} '
              f'kappa {m["kappa"]:.3f} AUC>=3 {m["auc_ge3"]:.3f}{fmt_type(m)}{fmt_cbm(m)} | {time.time()-t0:.0f}s{flag}', flush=True)

    json.dump({'best': best, 'history': hist, 'args': vars(a)}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    print(f'\nBEST arch={a.arch} type_loss={a.type_loss}{"" if a.concept == "none" else " +concept=" + a.concept}: rho {best["rho"]:.3f}  MAE {best["MAE"]:.2f}  MAE(>=6) {best["MAE_ge6"]:.2f}  '
          f'kappa {best["kappa"]:.3f}  AUC>=3 {best["auc_ge3"]:.3f}{fmt_type(best)}{fmt_cbm(best)}  (epoch {best["epoch"]})')
    last = hist[-1]
    print(f'LAST  epoch {last["epoch"]} (no val selection): rho {last["rho"]:.3f}  kappa {last["kappa"]:.3f}  '
          f'AUC>=3 {last["auc_ge3"]:.3f}{fmt_type(last)}{fmt_cbm(last)}')
    print('reference (val): attn size+RADS only rho 0.805 / kappa 0.694; report-text GGO recall of B3e_clean 0.18')


if __name__ == '__main__':
    main()
