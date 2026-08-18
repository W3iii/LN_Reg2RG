"""Model-free voxel statistics per lobe, at native resolution and after the resize.

The encoder probes put every tap point in the same narrow band just above "how
big is this lobe", and MIL over unpooled patch tokens did not beat pooling. So
the information is already gone at the ViT's input, and two suspects remain:

  (a) the resize to (256, 256, 64), which scales z by ~0.38x
  (b) the frozen RadFM ViT3D, never trained at nodule scale

This separates them without involving any model at all. It computes the same
statistics twice -- once on the native-resolution masked lobe, once on exactly
the volume the dataset hands the ViT -- and the probe then asks which of them
still knows whether the lobe contains a nodule:

  native predictive, resized not  -> the resize destroys it. No encoder change
                                     can help, and lesion crops at native
                                     resolution (docs/LESION_TOKENS.md) are the
                                     right fix.
  both predictive                 -> the signal survives into the ViT's input and
                                     the frozen encoder is discarding it. Unfreeze
                                     or replace it; lesion tokens are a detour.
  neither predictive              -> nothing here separates the classes at lobe
                                     level. Check the masks and the labels before
                                     touching the model at all.

Statistics are chosen for what a nodule actually is -- a small dense blob inside
dark lung -- rather than anything learned: an HU histogram plus a crude
connected-component count in the nodule density and size range.
"""
import argparse
import os
import sys

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from regions import REGIONS  # noqa: E402

HU_MIN, HU_MAX = -1000.0, 200.0
N_BINS = 32
# Nodules sit well above aerated lung (about -800 HU) and around soft tissue.
DENSE_HU = -400.0
# 10..5000 voxels at (0.8, 0.8, 1.0) mm is roughly 2..20 mm equivalent diameter,
# i.e. the range the reports actually describe (HANDOFF.md §4: median 5 mm).
MIN_BLOB, MAX_BLOB = 10, 5000


def lobe_stats(vol, inside):
    """HU histogram + crude blob counts for the voxels inside one lobe mask."""
    v = vol[inside]
    if v.size == 0:
        return np.zeros(N_BINS + 4, dtype=np.float32)

    hist, _ = np.histogram(np.clip(v, HU_MIN, HU_MAX), bins=N_BINS, range=(HU_MIN, HU_MAX))
    hist = hist.astype(np.float32) / max(v.size, 1)

    dense = (vol > DENSE_HU) & inside
    frac_dense = float(dense.sum()) / max(v.size, 1)

    lab, n = ndimage.label(dense)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        in_range = sizes[(sizes >= MIN_BLOB) & (sizes <= MAX_BLOB)]
        n_blob = float(len(in_range))
        max_blob = float(in_range.max()) if in_range.size else 0.0
        biggest = float(sizes.max())
    else:
        n_blob = max_blob = biggest = 0.0

    return np.concatenate([
        hist,
        np.array([frac_dense, n_blob, max_blob, biggest], dtype=np.float32),
    ]).astype(np.float32)


def resize_to(vol, shape):
    """Trilinear resize, matching what monai's Resize does in the dataset."""
    import torch
    t = torch.as_tensor(vol, dtype=torch.float32)[None, None]
    out = torch.nn.functional.interpolate(t, size=shape, mode='trilinear', align_corners=False)
    return out[0, 0].numpy()


def crop_foreground(vol, keep):
    idx = np.argwhere(keep)
    if idx.size == 0:
        return vol
    lo, hi = idx.min(0), idx.max(0) + 1
    return vol[tuple(slice(a, b) for a, b in zip(lo, hi))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--target', default='256,256,64')
    a = ap.parse_args()

    target = tuple(int(x) for x in a.target.split(','))
    import pandas as pd
    df = pd.read_csv(os.path.join(a.data_root, f'region_report_{a.split}.csv'))
    volumes = sorted(df['Volumename'].astype(str).unique())
    if a.limit:
        volumes = volumes[:a.limit]
    print(f'{len(volumes)} volumes, target resize {target}')

    rows = []
    for i, vol_name in enumerate(volumes):
        stem = vol_name.replace('.nii.gz', '')
        img_p = os.path.join(a.data_root, 'images', stem, stem, vol_name)
        seg_d = os.path.join(a.data_root, 'masks', f'seg_{stem}')
        if not os.path.exists(img_p):
            continue
        img = np.asarray(nib.load(img_p, mmap=True).dataobj).astype(np.float32)

        for region in REGIONS:
            mp = os.path.join(seg_d, f'{region}.nii.gz')
            if not os.path.exists(mp):
                continue
            mask = np.asarray(nib.load(mp, mmap=True).dataobj) > 0
            if mask.sum() == 0:
                continue

            # Same construction the dataset uses: keep the lobe, blank the rest.
            masked = np.where(mask, img, -1024.0).astype(np.float32)
            keep = masked > -1000.0

            native = crop_foreground(masked, keep)
            resized = resize_to(native, target)

            rows.append({
                'acc_num': vol_name,
                'region': region,
                'lobe_voxels': float(mask.sum()),
                'native': lobe_stats(native, native > -1000.0),
                'resized': lobe_stats(resized, resized > -1000.0),
            })

        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(volumes)}', flush=True)

    out = {
        'acc_num': np.array([r['acc_num'] for r in rows]),
        'region': np.array([r['region'] for r in rows]),
        'lobe_voxels': np.array([r['lobe_voxels'] for r in rows], dtype=np.float64),
        'native': np.stack([r['native'] for r in rows]),
        'resized': np.stack([r['resized'] for r in rows]),
    }
    # Probe-compatible aliases so probe_local_info.py can read this file directly.
    out['pre_perceiver'] = out['native']
    out['post_perceiver'] = out['resized']
    out['post_fc'] = out['resized']
    out['mask_token'] = out['native']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f'wrote {a.out}: {len(rows)} region rows')
    print(f"  native  {out['native'].shape}")
    print(f"  resized {out['resized'].shape}")


if __name__ == '__main__':
    main()
