"""方位與覆蓋率健檢: 確認 affine 沒轉錯、裁切沒切掉肺。"""
import os, glob
import numpy as np

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
LOBE = os.path.join(ROOT, 'reg2rg_dataset', 'lobe')
NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}

for p in sorted(glob.glob(os.path.join(LOBE, '*_lobe5.npz'))):
    f = os.path.basename(p).replace('_lobe5.npz', '')
    lobe = np.load(p)['image']
    ct = np.load(os.path.join(ME, f, 'npy', f'{f}.npy'), mmap_mode='r')
    lung = np.load(os.path.join(ME, f, 'npy', f'{f}_lobe.npz'))['image'] > 0

    # 1) 方位: 脊椎(高 HU 骨)在 LPS 應位於 row 大的一側(後側)
    z = ct.shape[2] // 2
    sl = np.asarray(ct[:, :, z])
    bone = np.argwhere(sl > 400)
    spine_row = np.median(bone[:, 0]) if len(bone) else np.nan
    # 2) 心臟/前縱膈在 row 小的一側 -> 用肺的 row 重心對比
    lung_row = np.argwhere(lung[:, :, z])[:, 0].mean() if lung[:, :, z].any() else np.nan

    # 3) 覆蓋率: TS 五葉聯集 vs 既有二值肺 mask
    ts = lobe > 0
    inter = (ts & lung).sum()
    cov = inter / max(lung.sum(), 1)          # 既有肺 mask 有多少被 TS 蓋到
    frac = {NAME[k]: (lobe == k).sum() / max(ts.sum(), 1) for k in NAME}
    print('%-12s spine_row=%5.0f lung_row=%5.0f | TS/lung覆蓋=%.3f  TS體積/lung體積=%.2f' %
          (f, spine_row, lung_row, cov, ts.sum() / max(lung.sum(), 1)))
    print('              比例 ' + '  '.join('%s=%.1f%%' % (k, 100 * v) for k, v in frac.items()))
print('\n參考: 正常肺葉體積佔比 RUL~21%  RML~9%  RLL~24%  LUL~24%  LLL~22%')
print('spine_row 應明顯大於 lung_row (脊椎在後 = row 大) 才代表 affine 方位正確')
