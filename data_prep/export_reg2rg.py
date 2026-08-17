"""
方案 A: 匯出成 Reg2RG 吃的磁碟格式 (5 肺葉 region)

輸出結構 (完全遷就 Reg2RG 既有慣例, 讓程式碼改動最小):
  reg2rg_nifti/
    images/<folder>/<folder>/<folder>.nii.gz        <- 它 glob 兩層
    masks/seg_<folder>/<region>.nii.gz              <- 它組 'seg_'+volume_name
    region_report_{train,val,test}.csv              <- Volumename / Anatomy / Sentence
    nodule_metadata.csv                             <- 保留給之後的 lesion token
    splits.csv

Volumename 必須含 .nii.gz (它拿 nii 檔的 basename 當 key)
region 檔名必須與 REGIONS 常數逐字相符
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
MARGIN = 8

LAB2REGION = {1: 'right upper lobe', 2: 'right middle lobe', 3: 'right lower lobe',
              4: 'left upper lobe', 5: 'left lower lobe'}
SHORT2LAB = {'RUL': 1, 'RML': 2, 'RLL': 3, 'LUL': 4, 'LLL': 5}
NO_FINDING = 'No significant finding.'


def read_meta(folder):
    p = os.path.join(ME, folder, 'npy', 'series_metadata.txt')
    lines = [l.strip() for l in open(p).read().strip().split('\n') if l.strip()]
    return [float(v) for v in lines[-2].split(',')], [float(v) for v in lines[-1].split(',')]


def make_affine(folder):
    """與 build_lobe_masks.py 同一套 (已用肺面積剖面 + 報告肺葉驗證過方位)"""
    sp, org = read_meta(folder)
    oy = org[0] + 256 * (sp[0] - TGT[0])
    ox = org[1] + 256 * (sp[1] - TGT[1])
    oz = org[2]
    sy, sx, sz = TGT
    return np.array([[0.0, -sx, 0.0, -ox],
                     [-sy, 0.0, 0.0, -oy],
                     [0.0, 0.0, -sz, oz],
                     [0.0, 0.0, 0.0, 1.0]])


def save_nii(arr, aff, path, dtype):
    n = nib.Nifti1Image(np.ascontiguousarray(arr.astype(dtype)), aff)
    n.header.set_xyzt_units('mm')
    nib.save(n, path)


def export_one(folder, force=False):
    img_dir = os.path.join(OUT, 'images', folder, folder)
    msk_dir = os.path.join(OUT, 'masks', f'seg_{folder}')
    img_p = os.path.join(img_dir, f'{folder}.nii.gz')
    if os.path.exists(img_p) and not force:
        return 'skip', None
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    ct = np.load(os.path.join(ME, folder, 'npy', f'{folder}.npy'))
    lobe = np.load(os.path.join(DS, 'lobe', f'{folder}_lobe5.npz'))['image']

    # 用 TotalSegmentator 的肺葉聯集裁切 (比 lobe_info 的粗 bbox 準)
    idx = np.argwhere(lobe > 0)
    lo = np.maximum(idx.min(0) - MARGIN, 0)
    hi = np.minimum(idx.max(0) + MARGIN + 1, ct.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    ct_c, lobe_c = ct[sl], lobe[sl]

    aff = make_affine(folder)
    aff[:3, 3] += aff[:3, :3] @ lo

    save_nii(ct_c, aff, img_p, np.int16)
    for lab, region in LAB2REGION.items():
        save_nii((lobe_c == lab), aff, os.path.join(msk_dir, f'{region}.nii.gz'), np.uint8)
    return 'ok', ct_c.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    man = pd.read_csv(os.path.join(DS, 'manifest.csv'))
    folders = man.folder.tolist()
    if a.limit:
        folders = folders[:a.limit]

    os.makedirs(OUT, exist_ok=True)
    log = io.open(os.path.join(OUT, 'export_log.txt'), 'a', encoding='utf-8')
    n_ok = n_skip = n_err = 0
    for i, f in enumerate(folders, 1):
        try:
            st, shp = export_one(f, a.force)
        except Exception as e:
            st, shp = 'error', repr(e)[:160]
        if st == 'ok':
            n_ok += 1
        elif st == 'skip':
            n_skip += 1
        else:
            n_err += 1
            print(f'[{i}] {f} {st} {shp}', flush=True)
        if i % 50 == 0:
            print(f'  {i}/{len(folders)} ok={n_ok} skip={n_skip} err={n_err}', flush=True)
            log.write(f'{i}/{len(folders)} ok={n_ok} skip={n_skip} err={n_err}\n'); log.flush()
    log.write(f'== done ok={n_ok} skip={n_skip} err={n_err}\n'); log.close()
    print(f'影像/遮罩匯出: ok={n_ok} skip={n_skip} err={n_err}')


if __name__ == '__main__':
    main()
