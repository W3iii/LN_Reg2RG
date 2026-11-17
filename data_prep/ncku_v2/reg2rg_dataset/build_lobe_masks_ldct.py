"""
LDCT_test 五肺葉 mask (TotalSegmentator) — build_lobe_masks.py 的 LDCT 版

與 ME 版完全相同的前處理 (已確認 LDCT npy 一樣是 (0.8,0.8,1.0) mm、k 往 inferior、metadata 格式相同):
  npy (y,x,z) -> 裁肺部 bbox + 24 voxel -> RAS+ NIfTI -> TotalSegmentator task=total, 只取 5 個肺葉
輸出: reg2rg_dataset/lobe_ldct/<folder>_lobe5.npz   label 1=RUL 2=RML 3=RLL 4=LUL 5=LLL

用法 (在 lobeseg 環境):
    python build_lobe_masks_ldct.py --limit 3        # 冒煙測試
    python build_lobe_masks_ldct.py                  # 有 nodule 的 615 位 (配對需要的)
    python build_lobe_masks_ldct.py --all            # 全部 1000 位
    python build_lobe_masks_ldct.py --dry --limit 1  # 不跑模型, 只輸出 NIfTI 檢查方向
"""
import os, sys, json, argparse, time, shutil, tempfile
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DS = os.path.join(ROOT, 'LDCT_test_dataset')
OUT_DIR = os.path.join(HERE, 'lobe_ldct')
LOG = os.path.join(HERE, 'lobe_seg_log_ldct.txt')
TGT = (0.8, 0.8, 1.0)
LOBE_FILES = {'lung_upper_lobe_right': 1, 'lung_middle_lobe_right': 2, 'lung_lower_lobe_right': 3,
              'lung_upper_lobe_left': 4, 'lung_lower_lobe_left': 5}
LOBE_NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}


def read_lines(p):
    return [l.strip() for l in open(p).read().strip().split('\n') if l.strip()]


def make_affine(folder):
    """同 ME 版: in-plane 以影像中心等比縮放到 0.8 mm; k 增加 = inferior -> dS/dk = -sz"""
    ln = read_lines(os.path.join(DS, folder, 'npy', 'series_metadata.txt'))
    sp = [float(v) for v in ln[-2].split(',')]; org = [float(v) for v in ln[-1].split(',')]
    # 俯臥 (HFP, 例: CHESTCT_Test0374): IOP = [-1,0,0, 0,-1,0], origin 在病人左後方 -> origin x,y 為正
    # 欄往病人右、列往病人前 -> in-plane 兩軸方向相反。仰臥時 d = +1, 與 ME 版完全相同。
    d = -1.0 if is_prone(folder) else 1.0
    oy = org[0] + d * 256 * (sp[0] - TGT[0]); ox = org[1] + d * 256 * (sp[1] - TGT[1]); oz = org[2]
    sy, sx, sz = TGT
    return np.array([[0.0, -d * sx, 0.0, -ox], [-d * sy, 0.0, 0.0, -oy], [0.0, 0.0, -sz, oz], [0.0, 0.0, 0.0, 1.0]])


def is_prone(folder):
    org = [float(v) for v in read_lines(os.path.join(DS, folder, 'npy', 'series_metadata.txt'))[-1].split(',')]
    return org[0] > 0 and org[1] > 0


def prepare(folder):
    full = np.load(os.path.join(DS, folder, 'npy', f'{folder}.npy'))
    lob = read_lines(os.path.join(DS, folder, 'npy', 'lobe_info.txt'))
    lz0, lz1 = [int(v) for v in lob[-2].split(',')]
    ty0, ty1, tx0, tx1 = [int(v) for v in lob[-1].split(',')]
    M = 24
    y0, y1 = max(ty0 - M, 0), min(ty1 + M, full.shape[0])
    x0, x1 = max(tx0 - M, 0), min(tx1 + M, full.shape[1])
    z0, z1 = max(lz0 - M, 0), min(lz1 + M, full.shape[2])
    vol = full[y0:y1, x0:x1, z0:z1]
    aff = make_affine(folder)
    aff[:3, 3] += aff[:3, :3] @ np.array([y0, x0, z0])
    nii = nib.Nifti1Image(np.ascontiguousarray(vol.astype(np.int16)), aff)
    nii.header.set_xyzt_units('mm')
    return full.shape, (y0, y1, x0, x1, z0, z1), vol.shape, nii


def run_one(folder, fast, tmproot, dry=False, force=False):
    out_npz = os.path.join(OUT_DIR, f'{folder}_lobe5.npz')
    if os.path.exists(out_npz) and not dry and not force:
        return 'skip', None
    shape, (y0, y1, x0, x1, z0, z1), vshape, nii = prepare(folder)
    if dry:
        p = os.path.join(OUT_DIR, f'{folder}_dry_ct.nii.gz'); nib.save(nii, p)
        return 'dry', f'{p} orient={nib.aff2axcodes(nii.affine)} det={np.linalg.det(nii.affine[:3, :3]):.3f}'
    from totalsegmentator.python_api import totalsegmentator
    work = tempfile.mkdtemp(dir=tmproot)
    try:
        in_p = os.path.join(work, 'ct.nii.gz'); nib.save(nii, in_p)
        seg_d = os.path.join(work, 'seg')
        totalsegmentator(in_p, seg_d, task='total', fast=fast, quiet=True, roi_subset=list(LOBE_FILES.keys()))
        sub = np.zeros(vshape, dtype=np.uint8); found = 0
        for fn, lab in LOBE_FILES.items():
            p = os.path.join(seg_d, fn + '.nii.gz')
            if not os.path.exists(p): continue
            m = np.asanyarray(nib.load(p).dataobj) > 0
            if m.shape != vshape: return 'shape_mismatch', f'{m.shape} vs {vshape}'
            sub[m] = lab; found += 1
        if found == 0: return 'empty', None
        lobe = np.zeros(shape, dtype=np.uint8); lobe[y0:y1, x0:x1, z0:z1] = sub
        np.savez_compressed(out_npz, image=lobe)
        return 'ok', {LOBE_NAME[l]: int((lobe == l).sum()) for l in LOBE_NAME}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def todo_list(all_):
    out = []
    for f in sorted(os.listdir(DS)):
        j = os.path.join(DS, f, 'mask', f'{f}_nodule_count.json')
        if not os.path.exists(j): continue
        if all_ or len(json.load(open(j))['nodule_size']) > 0: out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--fast', action='store_true')
    ap.add_argument('--all', action='store_true'); ap.add_argument('--dry', action='store_true')
    ap.add_argument('--patients', default=''); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = a.patients.split(',') if a.patients else todo_list(a.all)
    if a.limit: todo = todo[:a.limit]
    tmproot = os.path.join(os.environ.get('TEMP', tempfile.gettempdir()), 'ts_work_ldct'); os.makedirs(tmproot, exist_ok=True)
    log = open(LOG, 'a', encoding='utf-8'); t0 = time.time(); n_ok = n_skip = n_err = consec = 0
    for i, f in enumerate(todo, 1):
        t = time.time()
        try: st, info = run_one(f, a.fast, tmproot, a.dry, a.force)
        except Exception as e: st, info = 'error', repr(e)[:200]
        if st in ('ok', 'dry'): n_ok += 1; consec = 0
        elif st == 'skip': n_skip += 1; consec = 0
        else: n_err += 1; consec += 1
        line = f'[{i}/{len(todo)}] {f} {st} {time.time()-t:.1f}s {info}'
        print(line, flush=True); log.write(line + '\n'); log.flush()
        if consec >= 5:
            msg = f'== ABORT: 連續 {consec} 次失敗 (多半是 GPU/CUDA context 掛了), 中止讓外層以新 process 重啟'
            print(msg, flush=True); log.write(msg + '\n'); log.close(); sys.exit(2)
    log.write(f'== done ok={n_ok} skip={n_skip} err={n_err} total={time.time()-t0:.0f}s\n'); log.close()
    print(f'ok={n_ok} skip={n_skip} err={n_err} elapsed={time.time()-t0:.0f}s'); sys.exit(0)


if __name__ == '__main__':
    main()
