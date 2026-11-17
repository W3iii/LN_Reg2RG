"""
LDCT_test -> reg2rg_nifti, with exactly the ME crop logic (export_reg2rg.py + export_nodule_masks.py).

Per patient (test_ldct in reg2rg_nifti/splits_merged.csv, 566 patients):
  images/<f>/<f>/<f>.nii.gz           int16 CT, cropped
  masks/seg_<f>/<region>.nii.gz       uint8, 5 lobes (TotalSegmentator, lobe_ldct/<f>_lobe5.npz)
  masks/seg_<f>/nodules.nii.gz        uint8 instance mask, value = nodule_id (json order, same ids as
                                      ldct_nodules.csv / fill_table_v2 / region_report_v2)
  crop_offsets_ldct.csv               folder, crop_y0/x0/z0, crop_h/w/d
  nodule_mask_qc_ldct.csv

Identical to ME:
  * source npy is already (0.8, 0.8, 1.0) mm, (y, x, z), k toward the feet
  * crop = lobe-union bbox +- MARGIN 8, clipped to the volume
  * affine = make_affine() of export_reg2rg.py, shifted by the crop origin
  * nodule voxels = binary mask inside each bbox; overlaps go to the nearer bbox centre
Only LDCT-specific: prone scans (origin x, y > 0; only Test0374 among the 566) store y and x
reversed (axcodes A,R,I instead of ME's P,L,I). The Reg2RG loader reads raw voxel arrays without
reorienting, so after cropping the prone arrays (CT, lobes, nodules) are flipped on y and x and the
affine is adjusted: every exported volume is P,L,I in voxel order. crop_offsets_ldct.csv marks this
with flip_yx = 1; crop-frame coordinates are then  y' = crop_h - y  (and x likewise) for bbox edges /
centres in the convention cy = (y0 + y1) / 2 with exclusive ends.

  python export_reg2rg_ldct.py --limit 3
  python export_reg2rg_ldct.py               # all 566, resumable (skips finished patients)
  python export_reg2rg_ldct.py --budget 160  # stop after ~160 s (for short sessions)
"""
import os, io, json, time, argparse
import numpy as np
import nibabel as nib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LD = os.path.join(ROOT, 'LDCT_test_dataset')
LOBE_DIR = os.path.join(HERE, 'lobe_ldct')
OUT = os.path.join(ROOT, 'reg2rg_nifti')
TGT = (0.8, 0.8, 1.0)
MARGIN = 8                      # must match export_reg2rg.py / export_nodule_masks.py
LAB2REGION = {1: 'right upper lobe', 2: 'right middle lobe', 3: 'right lower lobe',
              4: 'left upper lobe', 5: 'left lower lobe'}


def read_meta(folder):
    p = os.path.join(LD, folder, 'npy', 'series_metadata.txt')
    lines = [l.strip() for l in open(p).read().strip().split('\n') if l.strip()]
    return [float(v) for v in lines[-2].split(',')], [float(v) for v in lines[-1].split(',')]


def is_prone(folder):
    org = read_meta(folder)[1]
    return org[0] > 0 and org[1] > 0


def make_affine(folder):
    """export_reg2rg.make_affine with the prone sign of build_lobe_masks_ldct.make_affine (d = +1 supine)."""
    sp, org = read_meta(folder)
    d = -1.0 if is_prone(folder) else 1.0
    oy = org[0] + d * 256 * (sp[0] - TGT[0])
    ox = org[1] + d * 256 * (sp[1] - TGT[1])
    oz = org[2]
    sy, sx, sz = TGT
    return np.array([[0.0, -d * sx, 0.0, -ox],
                     [-d * sy, 0.0, 0.0, -oy],
                     [0.0, 0.0, -sz, oz],
                     [0.0, 0.0, 0.0, 1.0]])


def save_nii(arr, aff, path, dtype):
    n = nib.Nifti1Image(np.ascontiguousarray(arr.astype(dtype)), aff)
    n.header.set_xyzt_units('mm')
    nib.save(n, path)


def instance_mask(folder, shape):
    """Same rule as the me_mask branch of export_nodule_masks.py."""
    binary = np.load(os.path.join(LD, folder, 'mask', f'{folder}.npz'))['image'] > 0
    if binary.shape != tuple(shape):
        raise ValueError(f'mask {binary.shape} vs ct {shape}')
    j = json.load(open(os.path.join(LD, folder, 'mask', f'{folder}_nodule_count.json')))
    boxes = []
    for k, bb in enumerate(j['bboxes'], 1):
        (y0, x0, z0), (y1, x1, z1) = bb                      # start inclusive, end exclusive
        boxes.append((k, int(y0), max(int(y1), int(y0) + 1), int(x0), max(int(x1), int(x0) + 1),
                      int(z0), max(int(z1), int(z0) + 1)))
    ctr = {k: np.array([(y0 + y1) / 2, (x0 + x1) / 2, (z0 + z1) / 2]) for k, y0, y1, x0, x1, z0, z1 in boxes}
    inst = np.zeros(shape, dtype=np.uint8)
    n_overlap = n_empty = 0
    for k, y0, y1, x0, x1, z0, z1 in boxes:
        sub = inst[y0:y1, x0:x1, z0:z1]
        sel = binary[y0:y1, x0:x1, z0:z1].copy()
        if not sel.any():
            n_empty += 1
            continue
        clash = (sub > 0) & sel
        if clash.any():
            n_overlap += int(clash.sum())
            idx3 = np.argwhere(clash)
            off = np.array([y0, x0, z0])
            d_here = ((idx3 + off - ctr[k]) ** 2).sum(1)
            owner = sub[clash]
            keep = np.ones(len(idx3), dtype=bool)
            for v in np.unique(owner):
                m = owner == v
                keep[m] = d_here[m] < ((idx3[m] + off - ctr[int(v)]) ** 2).sum(1)
            sel[tuple(idx3[~keep].T)] = False
        sub[sel] = k
    return inst, len(boxes), n_overlap, n_empty


KNOWN = {}


def export_one(folder, force=False):
    img_dir = os.path.join(OUT, 'images', folder, folder)
    msk_dir = os.path.join(OUT, 'masks', f'seg_{folder}')
    img_p = os.path.join(img_dir, f'{folder}.nii.gz')
    nod_p = os.path.join(msk_dir, 'nodules.nii.gz')

    if not force and folder in KNOWN and os.path.exists(img_p) and os.path.exists(nod_p):
        return 'skip', KNOWN[folder], None
    lobe = np.load(os.path.join(LOBE_DIR, f'{folder}_lobe5.npz'))['image']
    idx = np.argwhere(lobe > 0)
    lo = np.maximum(idx.min(0) - MARGIN, 0)
    hi = np.minimum(idx.max(0) + MARGIN + 1, lobe.shape)
    shape = tuple(int(b - c) for c, b in zip(lo, hi))
    flip = is_prone(folder)
    off = dict(folder=folder, crop_y0=int(lo[0]), crop_x0=int(lo[1]), crop_z0=int(lo[2]),
               crop_h=shape[0], crop_w=shape[1], crop_d=shape[2], flip_yx=int(flip))
    if os.path.exists(img_p) and os.path.exists(nod_p) and not force:
        return 'skip', off, None

    ct = np.load(os.path.join(LD, folder, 'npy', f'{folder}.npy'))
    if ct.shape != lobe.shape:
        raise ValueError(f'ct {ct.shape} vs lobe {lobe.shape}')
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    aff = make_affine(folder)
    aff[:3, 3] += aff[:3, :3] @ lo
    fl = (lambda v: v[::-1, ::-1, :]) if flip else (lambda v: v)
    if flip:
        for ax in (0, 1):
            aff[:3, 3] += aff[:3, ax] * (shape[ax] - 1)
            aff[:3, ax] *= -1

    inst, n_nod, n_overlap, n_empty = instance_mask(folder, ct.shape)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    lobe_c = fl(lobe[sl])
    for lab, region in LAB2REGION.items():
        save_nii(lobe_c == lab, aff, os.path.join(msk_dir, f'{region}.nii.gz'), np.uint8)
    crop = fl(inst[sl])
    save_nii(crop, aff, nod_p + '.part.nii.gz', np.uint8)
    save_nii(fl(ct[sl]), aff, img_p + '.part.nii.gz', np.int16)
    os.replace(nod_p + '.part.nii.gz', nod_p)      # images last: its presence marks a finished patient
    os.replace(img_p + '.part.nii.gz', img_p)
    present = set(np.unique(crop).tolist()) - {0}
    qc = dict(folder=folder, prone=is_prone(folder), n_nodule=n_nod, n_in_mask=len(present),
              n_lost_by_crop=n_nod - n_empty - len(present), n_overlap_vox=n_overlap,
              n_empty_bbox=n_empty, vox=int((crop > 0).sum()))
    return 'ok', off, qc


def upsert(new, path):
    new = pd.DataFrame(new)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old.folder.isin(new.folder)], new], ignore_index=True) if len(new) else old
    if len(new):
        new.sort_values('folder').to_csv(path, index=False, encoding='utf-8-sig')
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--patients', default='')
    ap.add_argument('--shard', default='', help='i/n: process every n-th patient starting at i')
    ap.add_argument('--budget', type=float, default=0, help='stop after this many seconds (0 = no limit)')
    a = ap.parse_args()

    S = pd.read_csv(os.path.join(OUT, 'splits_merged.csv'))
    folders = S[S.split == 'test_ldct'].folder.tolist()
    if a.patients:
        folders = a.patients.split(',')
    if a.limit:
        folders = folders[:a.limit]
    sh = ''
    if a.shard:
        i_, n_ = map(int, a.shard.split('/'))
        folders = folders[i_::n_]
        sh = f'_{i_}of{n_}'

    import glob
    for q in glob.glob(os.path.join(OUT, 'crop_offsets_ldct*.csv')):
        for r in pd.read_csv(q).to_dict('records'):
            if 'flip_yx' in r and r['flip_yx'] == r['flip_yx']:
                KNOWN[r['folder']] = {k: (int(v) if k != 'folder' else v) for k, v in r.items()}
    t0 = time.time()
    offs, qcs = [], []
    n = dict(ok=0, skip=0, err=0)
    log = io.open(os.path.join(OUT, 'export_log_ldct.txt'), 'a', encoding='utf-8')
    for i, f in enumerate(folders, 1):
        if a.budget and time.time() - t0 > a.budget:
            print(f'budget reached at {i - 1}/{len(folders)}', flush=True)
            break
        try:
            st, off, qc = export_one(f, a.force)
            offs.append(off)
            if qc:
                qcs.append(qc)
            n[st] += 1
        except Exception as e:
            n['err'] += 1
            msg = f'[{i}] {f} ERROR {repr(e)[:160]}'
            print(msg, flush=True); log.write(msg + '\n')
        if i % 25 == 0:
            print(f'  {i}/{len(folders)} {n}  {time.time() - t0:.0f}s', flush=True)
    upsert(offs, os.path.join(OUT, f'crop_offsets_ldct{sh}.csv'))
    Q = upsert(qcs, os.path.join(HERE, f'nodule_mask_qc_ldct{sh}.csv'))
    log.write(f'{time.strftime("%Y-%m-%d %H:%M")} {n}\n'); log.close()
    print(f'\n{n}  {time.time() - t0:.0f}s')
    done = sum(os.path.exists(os.path.join(OUT, 'images', f, f, f'{f}.nii.gz'))
               for f in S[S.split == 'test_ldct'].folder)
    print(f'test_ldct exported: {done} / {(S.split == "test_ldct").sum()}')
    if len(Q):
        print(f'nodules {Q.n_in_mask.sum()} / {Q.n_nodule.sum()}  lost_by_crop {Q.n_lost_by_crop.sum()}  '
              f'empty_bbox {Q.n_empty_bbox.sum()}  overlap_vox {Q.n_overlap_vox.sum()}')


if __name__ == '__main__':
    main()
