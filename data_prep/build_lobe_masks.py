"""
Step 3: 用 TotalSegmentator 產生五肺葉 mask

ME_dataset npy 是 (y, x, z) int16, 已 resample 到 (0.8, 0.8, 1.0) mm。
轉成 NIfTI (RAS+) -> TotalSegmentator -> 合併成單一 label volume 存回 npz。

label: 1=RUL 2=RML 3=RLL 4=LUL 5=LLL   (0 = 非肺葉)

用法:
    python build_lobe_masks.py --limit 5          # 冒煙測試
    python build_lobe_masks.py                    # 全跑
    python build_lobe_masks.py --fast             # 3mm 模型, 快很多
"""
import os, sys, json, argparse, time, shutil, tempfile
import numpy as np
import nibabel as nib

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
OUT_DIR = os.path.join(ROOT, 'reg2rg_dataset', 'lobe')
TGT = (0.8, 0.8, 1.0)

LOBE_FILES = {                       # TotalSegmentator 輸出檔名 -> 我們的 label
    'lung_upper_lobe_right': 1,
    'lung_middle_lobe_right': 2,
    'lung_lower_lobe_right': 3,
    'lung_upper_lobe_left': 4,
    'lung_lower_lobe_left': 5,
}
LOBE_NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}


def read_meta(folder):
    """回傳原始 spacing / world origin (y, x, z)"""
    p = os.path.join(ME, folder, 'npy', 'series_metadata.txt')
    lines = [l.strip() for l in open(p).read().strip().split('\n') if l.strip()]
    sp = [float(v) for v in lines[-2].split(',')]
    org = [float(v) for v in lines[-1].split(',')]
    return sp, org


def make_affine(folder):
    """
    npy index (i=row=y_LPS 後側+, j=col=x_LPS 病人左+, k=slice 往下 inferior) -> RAS+ mm

    in-plane 是「以影像中心等比縮放」到 0.8mm, 所以新 origin 要補償:
        origin_new = origin_orig + 256 * (spacing_orig - 0.8)

    z 方向: 實測肺面積剖面 (k=20 肺尖 3k voxel -> k=298 肺底 15k) 確認
    k 增加是往 inferior, 所以 dS/dk = -sz。寫成 +sz 會讓 affine 行列式為負
    (鏡像), 模型看到的是頭腳顛倒的胸腔, RML 會幾乎消失。
    """
    sp, org = read_meta(folder)
    oy = org[0] + 256 * (sp[0] - TGT[0])
    ox = org[1] + 256 * (sp[1] - TGT[1])
    oz = org[2]
    sy, sx, sz = TGT
    # LPS -> RAS: R=-X, A=-Y, S=Z
    return np.array([
        [0.0, -sx, 0.0, -ox],
        [-sy, 0.0, 0.0, -oy],
        [0.0, 0.0, -sz, oz],
        [0.0, 0.0, 0.0, 1.0],
    ])


def run_one(folder, fast, tmproot):
    from totalsegmentator.python_api import totalsegmentator

    out_npz = os.path.join(OUT_DIR, f'{folder}_lobe5.npz')
    if os.path.exists(out_npz):
        return 'skip', None

    full = np.load(os.path.join(ME, folder, 'npy', f'{folder}.npy'))
    # 裁到肺部 bbox (含 margin) 再送模型: 體積少 ~4x, 對肺葉分割無損
    lob = [l.strip() for l in open(os.path.join(ME, folder, 'npy', 'lobe_info.txt')).read().strip().split('\n') if l.strip()]
    lz0, lz1 = [int(v) for v in lob[-2].split(',')]
    ty0, ty1, tx0, tx1 = [int(v) for v in lob[-1].split(',')]
    M = 24
    y0, y1 = max(ty0 - M, 0), min(ty1 + M, full.shape[0])
    x0, x1 = max(tx0 - M, 0), min(tx1 + M, full.shape[1])
    z0, z1 = max(lz0 - M, 0), min(lz1 + M, full.shape[2])
    vol = full[y0:y1, x0:x1, z0:z1]

    aff = make_affine(folder)
    aff[:3, 3] += aff[:3, :3] @ np.array([y0, x0, z0])      # 平移原點到裁切起點
    nii = nib.Nifti1Image(np.ascontiguousarray(vol.astype(np.int16)), aff)
    nii.header.set_xyzt_units('mm')

    work = tempfile.mkdtemp(dir=tmproot)
    try:
        in_p = os.path.join(work, 'ct.nii.gz')
        nib.save(nii, in_p)
        seg_d = os.path.join(work, 'seg')
        totalsegmentator(in_p, seg_d, task='total', fast=fast, quiet=True,
                         roi_subset=list(LOBE_FILES.keys()))
        sub = np.zeros(vol.shape, dtype=np.uint8)
        found = 0
        for fn, lab in LOBE_FILES.items():
            p = os.path.join(seg_d, fn + '.nii.gz')
            if not os.path.exists(p):
                continue
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.shape != vol.shape:
                return 'shape_mismatch', f'{m.shape} vs {vol.shape}'
            sub[m] = lab
            found += 1
        if found == 0:
            return 'empty', None
        lobe = np.zeros(full.shape, dtype=np.uint8)         # 貼回原尺寸
        lobe[y0:y1, x0:x1, z0:z1] = sub
        np.savez_compressed(out_npz, image=lobe)
        vox = {LOBE_NAME[l]: int((lobe == l).sum()) for l in LOBE_NAME}
        return 'ok', vox
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--patients', default='')     # 逗號分隔 folder 名, 供冒煙測試
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    import pandas as pd
    man = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'manifest.csv'))
    todo = man.folder.tolist()
    if a.patients:
        todo = a.patients.split(',')
    if a.limit:
        todo = todo[:a.limit]

    tmproot = os.path.join(os.environ.get('TEMP', '.'), 'ts_work')
    os.makedirs(tmproot, exist_ok=True)
    log = open(os.path.join(ROOT, 'reg2rg_dataset', 'lobe_seg_log.txt'), 'a', encoding='utf-8')
    t0 = time.time()
    n_ok = n_skip = n_err = 0
    consec = 0
    MAX_CONSEC = 5          # GPU 掛掉時 CUDA context 不會自己復原, 直接中止讓外層重啟
    for i, f in enumerate(todo, 1):
        t = time.time()
        try:
            st, info = run_one(f, a.fast, tmproot)
        except Exception as e:
            st, info = 'error', repr(e)[:200]
        if st == 'ok':
            n_ok += 1; consec = 0
        elif st == 'skip':
            n_skip += 1; consec = 0
        else:
            n_err += 1; consec += 1
        line = f'[{i}/{len(todo)}] {f} {st} {time.time()-t:.1f}s {info if st != "ok" else info}'
        print(line, flush=True)
        log.write(line + '\n'); log.flush()
        if consec >= MAX_CONSEC:
            msg = f'== ABORT: 連續 {consec} 次失敗 (多半是 GPU/CUDA context 掛了), 中止讓外層以新 process 重啟'
            print(msg, flush=True); log.write(msg + '\n')
            log.write(f'== partial ok={n_ok} skip={n_skip} err={n_err} total={time.time()-t0:.0f}s\n')
            log.close()
            sys.exit(2)
    log.write(f'== done ok={n_ok} skip={n_skip} err={n_err} total={time.time()-t0:.0f}s\n')
    log.close()
    print(f'ok={n_ok} skip={n_skip} err={n_err} elapsed={time.time()-t0:.0f}s')
    sys.exit(0)


if __name__ == '__main__':
    main()
