"""
Step 3b: 把肺葉指派給每顆 nodule, 並做全量品管

輸出:
  nodules.csv       (就地更新 lobe 欄)
  lobe_qc.csv       每位病患的肺葉體積佔比 (偏離常態代表分割可疑)
  lobe_validation.csv  分割肺葉 vs 報告寫的肺葉 (方位/品質總驗證)
"""
import os, json, glob
import numpy as np
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
OUT = os.path.join(ROOT, 'reg2rg_dataset')
LOBE = os.path.join(OUT, 'lobe')
NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}
REF = {'RUL': .21, 'RML': .09, 'RLL': .24, 'LUL': .24, 'LLL': .22}

N = pd.read_csv(os.path.join(OUT, 'nodules.csv'))
NS = pd.read_csv(os.path.join(OUT, 'nodule_sentence.csv')).fillna({'match_lobe': '', 'matched_sentence': ''})

assigned, qc, bad = {}, [], []
folders = sorted(N.folder.unique())
for fi, folder in enumerate(folders, 1):
    p = os.path.join(LOBE, f'{folder}_lobe5.npz')
    if not os.path.exists(p):
        bad.append((folder, 'missing')); continue
    try:
        lobe = np.load(p)['image']
    except Exception as e:
        bad.append((folder, repr(e)[:60])); continue
    if lobe.ndim != 3 or lobe.max() == 0 or lobe.max() > 5:
        bad.append((folder, f'bad content max={lobe.max()}')); continue

    cnt = {NAME[k]: int((lobe == k).sum()) for k in NAME}
    tot = sum(cnt.values())
    frac = {k: v / max(tot, 1) for k, v in cnt.items()}
    dev = max(abs(frac[k] - REF[k]) for k in REF)
    qc.append(dict(folder=folder, total_vox=tot, lung_L=round(tot * 0.64 / 1e6, 2),
                   **{f'f_{k}': round(v, 4) for k, v in frac.items()}, max_dev=round(dev, 3)))

    g = N[N.folder == folder]
    for r in g.itertuples():
        cy, cx, cz = int(r.cy), int(r.cx), int(r.cz)
        y0, y1 = max(cy - 5, 0), min(cy + 6, lobe.shape[0])
        x0, x1 = max(cx - 5, 0), min(cx + 6, lobe.shape[1])
        z0, z1 = max(cz - 3, 0), min(cz + 4, lobe.shape[2])
        patch = lobe[y0:y1, x0:x1, z0:z1]
        nz = patch[patch > 0]
        if nz.size:
            lab = int(np.bincount(nz).argmax())
        else:                                   # 落在肺葉外(貼胸膜/裂隙) -> 找最近的肺葉體素
            y0, y1 = max(cy - 20, 0), min(cy + 21, lobe.shape[0])
            x0, x1 = max(cx - 20, 0), min(cx + 21, lobe.shape[1])
            z0, z1 = max(cz - 12, 0), min(cz + 13, lobe.shape[2])
            wide = lobe[y0:y1, x0:x1, z0:z1]
            nz2 = wide[wide > 0]
            lab = int(np.bincount(nz2).argmax()) if nz2.size else 0
        assigned[r.Index] = NAME.get(lab, '')
    if fi % 200 == 0:
        print(f'  {fi}/{len(folders)}', flush=True)

N['lobe'] = pd.Series(assigned)
N.to_csv(os.path.join(OUT, 'nodules.csv'), index=False, encoding='utf-8-sig')
Q = pd.DataFrame(qc)
Q.to_csv(os.path.join(OUT, 'lobe_qc.csv'), index=False, encoding='utf-8-sig')

# ---- 驗證: 分割肺葉 vs 報告句子講的肺葉 ----
M = N.merge(NS[['pid', 'nodule_id', 'match_lobe', 'matched_sentence']], on=['pid', 'nodule_id'], how='left')
V = M[(M.match_lobe.fillna('') != '') & (~M.match_lobe.fillna('').str.contains(',')) & (M.lobe != '')].copy()
V['side_ok'] = V.lobe.str[0] == V.match_lobe.str[0]
V['exact'] = V.lobe == V.match_lobe
V.to_csv(os.path.join(OUT, 'lobe_validation.csv'), index=False, encoding='utf-8-sig')

print('\n' + '=' * 62)
print('肺葉分割總檢  (%d 位病患)' % len(Q))
print('=' * 62)
print('損壞/缺檔 : %d %s' % (len(bad), bad[:5]))
print('全肺體積 L: median %.2f  p5 %.2f  p95 %.2f' % (Q.lung_L.median(), Q.lung_L.quantile(.05), Q.lung_L.quantile(.95)))
print('肺葉佔比 (參考 RUL.21 RML.09 RLL.24 LUL.24 LLL.22):')
for k in REF:
    print('  %-4s median %.3f   p5 %.3f  p95 %.3f' % (k, Q[f'f_{k}'].median(), Q[f'f_{k}'].quantile(.05), Q[f'f_{k}'].quantile(.95)))
susp = Q[Q.max_dev > 0.12]
print('佔比明顯偏離常態 (max_dev>0.12) : %d 位 (%.1f%%)' % (len(susp), 100 * len(susp) / len(Q)))

print('\n' + '=' * 62)
print('nodule 肺葉指派  (%d 顆)' % len(N))
print('=' * 62)
print(N.lobe.value_counts(dropna=False).to_string())
print('指派不到肺葉 : %d (%.1f%%)' % ((N.lobe == '').sum(), 100 * (N.lobe == '').mean()))

print('\n' + '=' * 62)
print('方位/品質驗證: 分割肺葉 vs 報告肺葉  (%d 顆可比對)' % len(V))
print('=' * 62)
print('左右一致 : %d / %d (%.1f%%)   <- 低於 90%% 代表方位有問題' % (V.side_ok.sum(), len(V), 100 * V.side_ok.mean()))
print('肺葉全對 : %d / %d (%.1f%%)' % (V.exact.sum(), len(V), 100 * V.exact.mean()))
print('\n混淆 (列=分割, 欄=報告):')
print(pd.crosstab(V.lobe, V.match_lobe).to_string())
