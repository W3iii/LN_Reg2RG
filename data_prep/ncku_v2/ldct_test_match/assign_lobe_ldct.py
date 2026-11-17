"""LDCT nodule -> 五肺葉指派 (同 assign_lobe.py), 結果快取在 ldct_nodule_lobe.csv, 可續跑 / 指定重算
排除: 影像不完整 (只掃到肺底, 全肺 < 1.5 L) -> lobe 留空並標記
"""
import os, sys, csv, time, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
T0 = time.time(); BUDGET = float(os.environ.get('QC_BUDGET', '1e9'))
REDO = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else set()
NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}
OUT = os.path.join(HERE, 'ldct_nodule_lobe.csv')
Nd = pd.read_csv(os.path.join(HERE, 'ldct_nodules.csv'))
Q = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'lobe_qc_ldct.csv')).set_index('folder')
old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=['folder'])
old = old[~old.folder.isin(REDO)]
done = set(old.folder); rows = old.to_dict('records')


def lobe_at(lobe, cy, cx, cz):
    cy, cx, cz = int(cy), int(cx), int(cz)
    for ry, rz in ((5, 3), (20, 12)):
        p = lobe[max(cy - ry, 0):cy + ry + 1, max(cx - ry, 0):cx + ry + 1, max(cz - rz, 0):cz + rz + 1]
        nz = p[p > 0]
        if nz.size: return NAME.get(int(np.bincount(nz).argmax()), '')
    return ''


for f, g in Nd.groupby('folder'):
    if f in done: continue
    if time.time() - T0 > BUDGET: print('budget reached'); break
    flag = ''
    if f in Q.index and Q.loc[f, 'lung_L'] < 1.5: flag = 'incomplete_scan'
    org = [float(v) for v in open(os.path.join(ROOT, 'LDCT_test_dataset', f, 'npy', 'series_metadata.txt')).read().strip().split('\n')[-1].split(',')]
    prone = org[0] > 0 and org[1] > 0
    p = os.path.join(ROOT, 'reg2rg_dataset', 'lobe_ldct', f'{f}_lobe5.npz')
    lobe = np.load(p)['image'] if (os.path.exists(p) and not flag) else None
    for r in g.itertuples():
        rows.append(dict(folder=f, nodule_id=r.nodule_id, lobe=lobe_at(lobe, r.cy, r.cx, r.cz) if lobe is not None else '',
                         prone=prone, flag=flag))
pd.DataFrame(rows).to_csv(OUT, index=False, encoding='utf-8-sig')
print('folders done', pd.DataFrame(rows).folder.nunique(), '/', Nd.folder.nunique())
