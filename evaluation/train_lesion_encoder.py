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

All three emit the same 4096-d token, and the auxiliary heads sit on that token,
so whatever is learned is in the vector the LLM would receive.

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

from Model.my_embedding_layer import LesionEncoder, PooledLesionEncoder  # noqa: E402  (single source: MyEmbedding loads these state_dicts)
from lesion_utils import crop_lesion, normalize_lesion, BBOX_COLS  # noqa: E402
from extract_lesion_features import find_image                    # noqa: E402
from build_nodule_type import TYPES, build as build_type_labels   # noqa: E402

N_TYPES = len(TYPES)
DEFAULT_TYPE_CSV = os.path.join(HERE, '..', 'outputs', 'nodule_type.csv')


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


class TypeTargets:
    """Type supervision aligned to the rows of the crop cache.

    tset   [N, 4] float, multi-hot allowed types (all zero = no label)
    hard   [N] int, the type when tset has exactly one, else -1
    bag    [N] int, MIL bag id (-1 = none); bags never cross lobes, so never cross patients
    ments  {bag_id: [N_TYPES]-float masks}, the type sets the lobe text names
    """

    def __init__(self, meta, type_csv, data_root=None):
        if not os.path.exists(type_csv):
            if data_root is None:
                raise FileNotFoundError(type_csv)
            print(f'[type] {type_csv} missing, building it from {data_root}')
            build_type_labels(data_root, type_csv)
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

    def class_weights(self, idx):
        """1/sqrt(freq) over the hard labels in idx, normalised to mean 1 over them."""
        h = self.hard[idx]
        n = np.bincount(h[h >= 0], minlength=N_TYPES).astype(np.float64)
        w = 1.0 / np.sqrt(n.clip(1))
        w /= (n * w).sum() / max(n.sum(), 1)
        return w.astype(np.float32)


class Crops(torch.utils.data.Dataset):
    """Returns (crop, log diam, rads, row) -- row indexes the cache, for type targets."""

    def __init__(self, X, logd, rads, augment, rows=None):
        self.X, self.logd, self.rads, self.augment = X, logd, rads, augment
        self.rows = np.arange(len(X)) if rows is None else np.asarray(rows)

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
        return torch.from_numpy(np.ascontiguousarray(x))[None], self.logd[i], self.rads[i], int(self.rows[i])


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
    ds = Crops(np.asarray(X[idx]), logd[idx], rads[idx], True, rows=idx)
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
    return torch.utils.data.DataLoader(Crops(np.asarray(X[idx]), logd[idx], rads[idx], False, rows=idx),
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
        self.enc = LesionEncoder(dim=4096) if arch == 'gap' else PooledLesionEncoder(arch)
        self.size_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 1))
        self.rads_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 4))
        self.type_head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, N_TYPES))
        self.g = nn.Sequential(nn.Linear(1 + N_TYPES, 32), nn.GELU(), nn.Linear(32, 4))

    def forward(self, x):
        t = self.enc(x)
        return self.size_head(t)[:, 0], self.rads_head(t), self.type_head(t), t

    def concept(self, log_mm, type_probs):
        """RADS logits from (log size, type probabilities) alone."""
        z = ((log_mm.float() - LOG_MM_CENTRE) / LOG_MM_SCALE)[:, None]
        return self.g(torch.cat([z, type_probs.float()], 1))


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
        if a.concept and model is None:
            raise ValueError('--concept needs the model (for its g)')
        self.cw_rads = torch.tensor(len(idx) / (4 * np.bincount(rads_train, minlength=4).clip(1)), dtype=torch.float32, device=device)
        self.cw_type = torch.tensor(tt.class_weights(idx), device=device)
        self.tset = torch.tensor(tt.tset, device=device)
        self.bag = torch.tensor(tt.bag, device=device)

    def __call__(self, out, ld, r, rows, epoch):
        s, rl, tl, _ = out
        a = self.a
        parts = {'size': F.smooth_l1_loss(s.float(), ld, beta=0.1),
                 'rads': F.cross_entropy(rl.float(), r, weight=self.cw_rads)}
        loss = parts['size'] + a.w_rads * parts['rads']
        if a.type_loss in ('ce', 'ce_mil'):
            parts['type'] = type_loss(tl, self.tset[rows], self.cw_type)
            loss = loss + a.w_type * parts['type']
        if a.type_loss == 'ce_mil' and epoch >= a.mil_warmup:
            parts['mil'] = mil_loss(tl, self.bag[rows], self.tt.ments, self.cw_type)
            loss = loss + a.w_mil * parts['mil']
        if a.concept:
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


def add_loss_args(ap):
    ap.add_argument('--type_loss', choices=['none', 'ce', 'ce_mil'], default='ce_mil',
                    help='none = size + RADS only (the B3e encoder); ce = + nodule-level type; ce_mil = + lobe-level MIL')
    ap.add_argument('--type_csv', default=DEFAULT_TYPE_CSV, help='data_prep/build_nodule_type.py output (built if missing)')
    ap.add_argument('--w_rads', type=float, default=1.0)
    ap.add_argument('--w_type', type=float, default=1.0)
    ap.add_argument('--w_mil', type=float, default=0.5)
    ap.add_argument('--mil_warmup', type=int, default=10, help='epochs before L_mil is switched on')
    ap.add_argument('--concept', action='store_true',
                    help='concept bottleneck: also grade RADS through g(size, type), see Net')
    ap.add_argument('--w_cbm', type=float, default=1.0, help='weight of CE on the concept-path grade')
    ap.add_argument('--w_consist', type=float, default=0.1, help='symmetric KL between direct and concept grades')


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, loader, device, tt=None, concept=False):
    model.eval()
    P, L, T, C, Y, R, rows = [], [], [], [], [], [], []
    for x, logd, rads, row in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            s, r, t, _ = model(x.to(device))
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
    if concept:
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
    ap.add_argument('--arch', choices=['gap', 'center', 'attn'], required=True)
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--cache', default=os.path.join(HERE, '..', 'outputs', 'lesion_crops_trainval.npy'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    add_loss_args(ap)
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
    tt = TypeTargets(meta, a.type_csv, a.data_root)
    print(f'arch={a.arch}  type_loss={a.type_loss}  train={len(tr)}  val={len(va)}  val patients={meta.pid.iloc[va].nunique()}')
    print(f'[type] train: {int((tt.hard[tr] >= 0).sum())} hard, {int(((tt.tset[tr].sum(1) > 1)).sum())} partial, '
          f'{int((tt.bag[tr] >= 0).sum())} in {len(np.unique(tt.bag[tr][tt.bag[tr] >= 0]))} MIL bags; '
          f'class weights {dict(zip(TYPES, tt.class_weights(tr).round(2).tolist()))}')

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else '  <-- no GPU visible, this will be slow'))
    dl_tr = train_loader(X, logd, rads, tr, tt.bag, a.bs, a.workers, a.seed)
    dl_va = eval_loader(X, logd, rads, va, a.bs, a.workers)

    model = Net(a.arch).to(device)
    obj = Objective(a, rads[tr], tt, tr, device, model)
    opt = make_optim(model, a.lr)
    sched = one_cycle(opt, a.lr, a.epochs, dl_tr)
    # type_loss is always in the name, so the 'none' arm never overwrites the B3a encoder
    # (outputs/lesion_enc_<arch>_seed<seed>/best.pt, trained before the type head existed)
    out = a.out or os.path.join(HERE, '..', 'outputs', f'lesion_enc_{a.arch}_{a.type_loss}{"_cbm" if a.concept else ""}_seed{a.seed}')
    os.makedirs(out, exist_ok=True)

    best, best_score, hist = None, -1e9, []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0; parts = {}; nb = 0
        for x, ld, r, rows in dl_tr:
            x, ld, r, rows = x.to(device), ld.to(device), r.to(device), rows.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                o = model(x)
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
            torch.save(checkpoint(model, a.arch, m, type_loss=a.type_loss, concept=a.concept), os.path.join(out, 'best.pt'))
            flag = ' *'
        lp = ' '.join(f'{k} {v / nb:.3f}' for k, v in parts.items())
        print(f'ep {ep+1:3d} loss {m["train_loss"]:.3f} ({lp}) | val rho {m["rho"]:.3f} MAE(>=6) {m["MAE_ge6"]:.2f} '
              f'kappa {m["kappa"]:.3f} AUC>=3 {m["auc_ge3"]:.3f}{fmt_type(m)}{fmt_cbm(m)} | {time.time()-t0:.0f}s{flag}', flush=True)

    json.dump({'best': best, 'history': hist, 'args': vars(a)}, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, default=float)
    print(f'\nBEST arch={a.arch} type_loss={a.type_loss}{" +concept" if a.concept else ""}: rho {best["rho"]:.3f}  MAE {best["MAE"]:.2f}  MAE(>=6) {best["MAE_ge6"]:.2f}  '
          f'kappa {best["kappa"]:.3f}  AUC>=3 {best["auc_ge3"]:.3f}{fmt_type(best)}{fmt_cbm(best)}  (epoch {best["epoch"]})')
    last = hist[-1]
    print(f'LAST  epoch {last["epoch"]} (no val selection): rho {last["rho"]:.3f}  kappa {last["kappa"]:.3f}  '
          f'AUC>=3 {last["auc_ge3"]:.3f}{fmt_type(last)}{fmt_cbm(last)}')
    print('reference (val): attn size+RADS only rho 0.805 / kappa 0.694; report-text GGO recall of B3e_clean 0.18')


if __name__ == '__main__':
    main()
