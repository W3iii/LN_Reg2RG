"""Cache per-nodule lesion-token features from a lesion-arm (B2) checkpoint.

Companion to extract_region_features.py, one level down. That script asks whether
lesion-scale information survives *lobe* pooling; this one asks whether the
lesion token itself -- the single 4096-d vector the LLM receives per nodule --
encodes the nodule's size and TW_Lung_RADS.

Why it is needed: B2 seed 1 reports nodules far more often than B0, but its stated
sizes are uncorrelated with the reference (Spearman ~0.01 on val, worse than
predicting 5 mm for everything). Text cannot tell the two possible causes apart:

    the token carries size, the decoder ignores it   -> fix the decoding side
    the token never encoded size                     -> fix the encoder / its loss

A linear probe on the token separates them, with no LLM, no generation, and no
regex parsing involved -- the same reasoning as the lobe-level probe.

Only the LesionEncoder weights are loaded (`embedding_layer.lesion_encoder.*`),
so this runs in a few GB of memory and does not need LLaMA. Crops go through the
exact functions the Dataset uses (lesion_utils.crop_lesion / normalize_lesion) on
the NIfTI array as nibabel returns it, which is what the Dataset does too.

Taps per nodule:
    pre_fc     pooled conv features before the projection   (128 gap, 256 center/attn)
    post_fc    the lesion token the LLM sees                (4096,)
    intensity  model-free crop statistics (HU means / dense-voxel fractions in
               nested windows around the box centre) -- a reference for how much
               size is recoverable from the crop without any learning

`--random_init` builds the encoder with fixed random weights instead of loading
a checkpoint. That is the control for "did training put anything into the token
beyond what random conv features + average pooling give for free".

Output: an .npz consumed by probe_lesion.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from Model.my_embedding_layer import build_lesion_encoder as make_encoder, LESION_ARCHS  # noqa: E402
from lesion_utils import (crop_lesion, normalize_lesion,       # noqa: E402
                          BBOX_COLS, LESION_CROP_SIZE)

PREFIX = 'embedding_layer.lesion_encoder.'


def build_lesion_encoder(ckpt_path, random_init, device, dtype, arch='gap', seed=0):
    torch.manual_seed(seed)
    enc = make_encoder(arch, 4096)
    if not random_init:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = {k[len(PREFIX):]: v for k, v in ckpt.items() if k.startswith(PREFIX)}
        if not state:
            raise RuntimeError(
                f'{ckpt_path} has no {PREFIX}* keys -- this is a B0 (no lesion token) '
                f'checkpoint, or the key prefix changed. First keys: {list(ckpt)[:5]}')
        enc.load_state_dict(state, strict=True)
        print(f'loaded {len(state)} lesion-encoder tensors')
    return enc.to(device=device, dtype=dtype).eval()


def intensity_features(raw_crop):
    """Model-free statistics of the raw HU crop, centred like crop_lesion centres it.

    Nested cubes of half-size 2/4/8/16 voxels (z halved, matching the 64x64x32
    crop's anisotropy). For each: mean HU, and fraction of voxels above -500 HU
    and above -200 HU. Dense-voxel fraction in a fixed window is a crude size
    estimate, which is exactly the point: it tells the probe how much size signal
    a trivial reader gets from the same pixels.
    """
    cy, cx, cz = (s // 2 for s in raw_crop.shape)
    feats = []
    for r in (2, 4, 8, 16):
        rz = max(r // 2, 1)
        w = raw_crop[cy - r:cy + r, cx - r:cx + r, cz - rz:cz + rz]
        feats += [float(w.mean()), float((w > -500).mean()), float((w > -200).mean())]
    return np.array(feats, dtype=np.float32)


def find_image(data_folder, volumename):
    folder = volumename.replace('.nii.gz', '')
    p = os.path.join(data_folder, folder, folder, volumename)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(p)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='B2 checkpoint pytorch_model.bin')
    ap.add_argument('--lesion_arch', default='gap', choices=LESION_ARCHS,
                    help='must match the checkpoint (B2 = gap, B3a = attn/center)')
    ap.add_argument('--random_init', action='store_true',
                    help='control arm: fixed random LesionEncoder weights, no checkpoint')
    ap.add_argument('--data_root', required=True,
                    help='reg2rg_nifti root (images/ and nodule_metadata.csv)')
    ap.add_argument('--nodule_metadata', default=None,
                    help='defaults to <data_root>/nodule_metadata.csv')
    ap.add_argument('--splits', default='train,val',
                    help='comma-separated values of the split column; test stays untouched by default')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='max volumes (smoke test)')
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    if not a.random_init and not a.ckpt:
        ap.error('--ckpt is required unless --random_init')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float32
    print(f'device={device} dtype={dtype}')

    enc = build_lesion_encoder(a.ckpt, a.random_init, device, dtype, arch=a.lesion_arch)
    tap = {}
    # input of the final projection = pooled conv features, for every arch
    enc.fc.register_forward_pre_hook(lambda m, inp: tap.__setitem__('pre_fc', inp[0]))

    nm = pd.read_csv(a.nodule_metadata or os.path.join(a.data_root, 'nodule_metadata.csv'))
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    nm = nm[nm['split'].isin(splits)].dropna(subset=BBOX_COLS).reset_index(drop=True)
    vols = list(dict.fromkeys(nm['Volumename']))
    if a.limit:
        vols = vols[:a.limit]
        nm = nm[nm['Volumename'].isin(vols)].reset_index(drop=True)
    print(f'{len(nm)} nodules in {len(vols)} volumes, splits={splits}')

    data_folder = os.path.join(a.data_root, 'images')
    post_fc = np.zeros((len(nm), 4096), dtype=np.float16)
    pre_fc = None
    inten = np.zeros((len(nm), 12), dtype=np.float32)
    t0 = time.time()
    for vi, vol in enumerate(vols):
        img = np.asarray(nib.load(find_image(data_folder, vol), mmap=True).dataobj).astype(np.float32)
        rows = nm.index[nm['Volumename'] == vol].tolist()
        crops, raws = [], []
        for r in rows:
            b = tuple(int(nm.at[r, c]) for c in ('y0_crop', 'y1_crop', 'x0_crop',
                                                  'x1_crop', 'z0_crop', 'z1_crop'))
            raw = crop_lesion(img, b)                 # same call the Dataset makes
            raws.append(raw)
            crops.append(normalize_lesion(raw))
        for j, raw in zip(rows, raws):
            inten[j] = intensity_features(raw)
        x = torch.from_numpy(np.stack(crops)[:, None]).to(device=device, dtype=dtype)
        for s in range(0, len(rows), a.batch):
            out = enc(x[s:s + a.batch]).float().cpu().numpy()
            pf = tap['pre_fc'].float().cpu().numpy()
            if pre_fc is None:
                pre_fc = np.zeros((len(nm), pf.shape[1]), dtype=np.float32)
            idx = rows[s:s + a.batch]
            post_fc[idx] = out.astype(np.float16)
            pre_fc[idx] = pf
        if (vi + 1) % 50 == 0:
            print(f'  {vi+1}/{len(vols)} volumes  {time.time()-t0:.0f}s', flush=True)

    keep = ['pid', 'folder', 'Volumename', 'split', 'nodule_id', 'region', 'lobe',
            'eq_diam_mm', 'bbox_long_mm', 'size_vox', 'tw_lung_rads', 'matched_sentence']
    # str arrays, not object arrays, so np.load works without allow_pickle
    out = {c: (nm[c].to_numpy() if pd.api.types.is_numeric_dtype(nm[c])
              else np.asarray(nm[c].astype(str), dtype=str))
           for c in keep if c in nm.columns}
    out.update(post_fc=post_fc, pre_fc=pre_fc, intensity=inten,
               source=np.array('random_init' if a.random_init else os.path.abspath(a.ckpt)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out}: {len(nm)} nodules  post_fc {post_fc.shape}  pre_fc {pre_fc.shape}  '
          f'({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
