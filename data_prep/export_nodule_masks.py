"""
Export per-patient nodule instance masks + the crop origin.

Satisfies the data contract in Reg2RG/docs/LESION_TOKENS.md §5.

  masks/seg_<folder>/nodules.nii.gz    uint8, 0 = background, 1..N = nodule_id
  crop_offsets.csv                     folder, crop_y0/x0/z0, crop shape

Why an instance mask instead of coordinates: export_reg2rg.py crops each volume to the
lung bounding box and does not record the origin, while nodules.csv holds coordinates in
the *uncropped* npy frame. Shipping the mask in the same cropped frame as images/ and
masks/ means the loader can take lesion k as `mask == k` with no bookkeeping, and it stays
correct if the crop is ever changed.

The crop is recomputed here exactly as export_reg2rg.py computes it (lobe-union bbox +
MARGIN), so the frames line up. MARGIN must stay in sync with that script.

Only writes nodules.nii.gz — the images and lobe masks are untouched, so this does not
re-export the 47 GB.
"""
import os, io, json, argparse
import numpy as np
import nibabel as nib
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
DS = os.path.join(ROOT, 'reg2rg_dataset')
OUT = os.path.join(ROOT, 'reg2rg_nifti')
TGT = (0.8, 0.8, 1.0)
MARGIN = 8                      # must match export_reg2rg.py


def read_meta(folder):
    p = os.path.join(ME, folder, 'npy', 'series_metadata.txt')
    lines = [l.strip() for l in open(p).read().strip().split('\n') if l.strip()]
    return [float(v) for v in lines[-2].split(',')], [float(v) for v in lines[-1].split(',')]


def make_affine(folder):
    """Identical to export_reg2rg.py / build_lobe_masks.py. z is negative: slice index
    increases toward the feet, and a positive z gives a mirrored volume."""
    sp, org = read_meta(folder)
    oy = org[0] + 256 * (sp[0] - TGT[0])
    ox = org[1] + 256 * (sp[1] - TGT[1])
    oz = org[2]
    sy, sx, sz = TGT
    return np.array([[0.0, -sx, 0.0, -ox],
                     [-sy, 0.0, 0.0, -oy],
                     [0.0, 0.0, -sz, oz],
                     [0.0, 0.0, 0.0, 1.0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--patients', default='')      # comma-separated folder names
    a = ap.parse_args()

    N = pd.read_csv(os.path.join(DS, 'nodules.csv'))
    man = pd.read_csv(os.path.join(DS, 'manifest.csv'))
    folders = man.folder.tolist()
    if a.patients:
        folders = a.patients.split(',')
    if a.limit:
        folders = folders[:a.limit]

    rows, qc = [], []
    n_ok = n_skip = n_err = 0
    for i, folder in enumerate(folders, 1):
        msk_dir = os.path.join(OUT, 'masks', f'seg_{folder}')
        out_p = os.path.join(msk_dir, 'nodules.nii.gz')
        try:
            lobe = np.load(os.path.join(DS, 'lobe', f'{folder}_lobe5.npz'))['image']
            idx = np.argwhere(lobe > 0)
            lo = np.maximum(idx.min(0) - MARGIN, 0)
            hi = np.minimum(idx.max(0) + MARGIN + 1, lobe.shape)
            shape = tuple(int(b - c) for c, b in zip(lo, hi))
            rows.append(dict(folder=folder, crop_y0=int(lo[0]), crop_x0=int(lo[1]), crop_z0=int(lo[2]),
                             crop_h=shape[0], crop_w=shape[1], crop_d=shape[2]))
            if os.path.exists(out_p) and not a.force:
                n_skip += 1
                continue

            binary = np.load(os.path.join(ME, folder, 'mask', f'{folder}.npz'))['image'] > 0
            inst = np.zeros(lobe.shape, dtype=np.uint8)
            g = N[N.folder == folder]
            n_overlap = n_empty = 0

            def box(r):
                y0, x0, z0 = int(r.y0), int(r.x0), int(r.z0)
                y1 = max(int(np.ceil(r.y1)), y0 + 1)
                x1 = max(int(np.ceil(r.x1)), x0 + 1)
                z1 = max(int(np.ceil(r.z1)), z0 + 1)
                return y0, y1, x0, x1, z0, z1

            # me_mask first: those have voxel evidence. Where two bboxes overlap, the
            # shared binary mask cannot say which instance a voxel belongs to, so assign
            # it to the nearer bbox centre rather than to whichever was written last.
            me = [r for r in g.itertuples() if r.region_source == 'me_mask']
            for r in me:
                y0, y1, x0, x1, z0, z1 = box(r)
                sub = inst[y0:y1, x0:x1, z0:z1]
                sel = binary[y0:y1, x0:x1, z0:z1]
                if not sel.any():
                    n_empty += 1
                    continue
                clash = (sub > 0) & sel
                if clash.any():
                    n_overlap += int(clash.sum())
                    prev = {int(v): next(q for q in me if int(q.nodule_id) == int(v))
                            for v in np.unique(sub[clash])}
                    idx3 = np.argwhere(clash)
                    here = np.array([r.cy - y0, r.cx - x0, r.cz - z0])
                    d_here = ((idx3 - here) ** 2).sum(1)
                    keep = np.ones(len(idx3), dtype=bool)
                    for v, q in prev.items():
                        m = sub[clash] == v
                        there = np.array([q.cy - y0, q.cx - x0, q.cz - z0])
                        keep[m] = d_here[m] < ((idx3[m] - there) ** 2).sum(1)
                    sel = sel.copy()
                    sel[tuple(idx3[~keep].T)] = False
                sub[sel] = int(r.nodule_id)

            # doctor-bbox nodules have no voxel mask, so the whole box would be filled.
            # Only claim voxels nobody else owns — otherwise a box that encloses a real
            # segmented nodule erases it (this was 296 voxels on CHESTCT1225 alone).
            for r in (q for q in g.itertuples() if q.region_source != 'me_mask'):
                y0, y1, x0, x1, z0, z1 = box(r)
                sub = inst[y0:y1, x0:x1, z0:z1]
                sel = sub == 0
                if not sel.any():
                    n_empty += 1
                    continue
                sub[sel] = int(r.nodule_id)

            aff = make_affine(folder)
            aff[:3, 3] += aff[:3, :3] @ lo
            crop = inst[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            os.makedirs(msk_dir, exist_ok=True)
            n = nib.Nifti1Image(np.ascontiguousarray(crop), aff)
            n.header.set_xyzt_units('mm')
            nib.save(n, out_p)

            present = set(np.unique(crop)) - {0}
            qc.append(dict(folder=folder, n_nodule=len(g), n_in_mask=len(present),
                           n_lost_by_crop=len(g) - len(present),
                           n_overlap_vox=n_overlap, n_empty_bbox=n_empty,
                           vox=int((crop > 0).sum())))
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f'[{i}] {folder} ERROR {repr(e)[:120]}', flush=True)
        if i % 200 == 0:
            print(f'  {i}/{len(folders)} ok={n_ok} skip={n_skip} err={n_err}', flush=True)

    # Merge rather than replace: a --patients run must not truncate the full-cohort files.
    def upsert(new, path):
        new = pd.DataFrame(new)
        if os.path.exists(path) and len(new):
            old = pd.read_csv(path)
            new = pd.concat([old[~old.folder.isin(new.folder)], new], ignore_index=True)
        new = new.sort_values('folder')
        new.to_csv(path, index=False, encoding='utf-8-sig')
        return new

    upsert(rows, os.path.join(OUT, 'crop_offsets.csv'))
    Q = upsert(qc, os.path.join(DS, 'nodule_mask_qc.csv'))

    print(f'\nok={n_ok} skip={n_skip} err={n_err}')
    if len(Q):
        print('=' * 60)
        print('nodules written        : %d / %d expected' % (Q.n_in_mask.sum(), Q.n_nodule.sum()))
        print('lost to the lung crop  : %d  (patients affected: %d)'
              % (Q.n_lost_by_crop.sum(), (Q.n_lost_by_crop > 0).sum()))
        print('empty bbox (no voxels) : %d' % Q.n_empty_bbox.sum())
        print('overlapping voxels     : %d  (patients affected: %d)'
              % (Q.n_overlap_vox.sum(), (Q.n_overlap_vox > 0).sum()))
        print('  overlaps mean two bboxes claim the same voxel; the later nodule_id wins')
        print('mask voxels per patient: median %d' % Q.vox.median())


if __name__ == '__main__':
    main()
