"""驗證 lobe 分割的方位是否正確 (左右翻了是最危險的靜默錯誤)。

作法: 取「單顆 nodule + 報告明確指名肺葉」的病患, 比對 TotalSegmentator
在該 nodule 位置給的 lobe 是否等於報告寫的 lobe。
"""
import os, re, json, glob
import numpy as np
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
LOBE = os.path.join(ROOT, 'reg2rg_dataset', 'lobe')
L = {0: '-', 1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}
NAME2LAB = {v: k for k, v in L.items()}

reps = {r['pid']: r for r in (json.loads(l) for l in
        open(os.path.join(ROOT, 'reg2rg_dataset', 'reports.jsonl'), encoding='utf-8'))}
man = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'manifest.csv'))
f2p = dict(zip(man.folder, man.pid))
NS = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'nodule_sentence.csv')).fillna({'match_lobe': ''})

rows = []
for p in sorted(glob.glob(os.path.join(LOBE, '*_lobe5.npz'))):
    folder = os.path.basename(p).replace('_lobe5.npz', '')
    pid = f2p.get(folder)
    if pid is None:
        continue
    lobe = np.load(p)['image']
    j = json.load(open(os.path.join(ME, folder, 'mask', f'{folder}_nodule_count.json')))
    if len(j['bboxes']) != 1:
        continue
    # 用「配到這顆 nodule 的句子」所指的肺葉當 ground truth
    g = NS[(NS.pid == pid) & (NS.match_lobe != '')]
    if len(g) != 1:
        continue
    said = g.match_lobe.iloc[0]
    if said not in NAME2LAB:
        continue                       # 跳過跨葉句 (e.g. "RUL,RML")

    b = j['bboxes'][0]
    cy, cx, cz = [int((b[0][i] + b[1][i]) / 2) for i in range(3)]
    y0, y1 = max(cy - 5, 0), cy + 6
    x0, x1 = max(cx - 5, 0), cx + 6
    z0, z1 = max(cz - 3, 0), cz + 4
    patch = lobe[y0:y1, x0:x1, z0:z1]
    nz = patch[patch > 0]
    got = L[int(np.bincount(nz).argmax())] if nz.size else '-'
    rows.append(dict(folder=folder, pid=pid, report_lobe=said, seg_lobe=got,
                     side_ok=(said[0] == got[0]) if got != '-' else None,
                     exact=(said == got), cy=cy, cx=cx, cz=cz))

R = pd.DataFrame(rows)
if len(R) == 0:
    print('尚無可驗證案例 (需要單顆 nodule + 報告只提一個肺葉)')
else:
    print('可驗證案例 %d' % len(R))
    ok = R.side_ok.dropna()
    print('左右一致  : %d / %d (%.1f%%)   <- 低於 90%% 代表方位翻了' % (ok.sum(), len(ok), 100 * ok.mean()))
    print('肺葉全對  : %d / %d (%.1f%%)' % (R.exact.sum(), len(R), 100 * R.exact.mean()))
    print('nodule 落在肺葉外(-): %d' % (R.seg_lobe == '-').sum())
    print()
    print(R[['folder', 'report_lobe', 'seg_lobe', 'side_ok', 'exact']].to_string(index=False))
    R.to_csv(os.path.join(ROOT, 'reg2rg_dataset', 'lobe_validation.csv'), index=False, encoding='utf-8-sig')
