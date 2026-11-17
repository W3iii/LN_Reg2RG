"""LDCT 五肺葉 mask 全量品管 (可續跑): 體積佔比 / 方向 / 是否在全肺 mask 內 -> lobe_qc_ldct.csv"""
import os, csv, time, numpy as np
T0 = time.time(); BUDGET = float(os.environ.get('QC_BUDGET', '1e9'))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}
REF = {'RUL': .21, 'RML': .09, 'RLL': .24, 'LUL': .24, 'LLL': .22}
OUT = os.path.join(HERE, 'lobe_qc_ldct.csv')
done = set()
if os.path.exists(OUT):
    done = {r['folder'] for r in csv.DictReader(open(OUT, encoding='utf-8'))}
files = sorted(f[:-10] for f in os.listdir(os.path.join(HERE, 'lobe_ldct')) if f.endswith('_lobe5.npz'))
cols = ['folder', 'total_vox', 'lung_L'] + [f'f_{k}' for k in REF] + ['max_dev', 'missing_lobe', 'lr_ok', 'z_ok', 'inside_lung']
new = not os.path.exists(OUT)
fh = open(OUT, 'a', newline='', encoding='utf-8'); w = csv.DictWriter(fh, cols)
if new: w.writeheader()
for i, f in enumerate(files, 1):
    if f in done: continue
    if time.time() - T0 > BUDGET: print('budget reached'); break
    L = np.load(os.path.join(HERE, 'lobe_ldct', f'{f}_lobe5.npz'))['image']
    lung = np.load(os.path.join(ROOT, 'LDCT_test_dataset', f, 'npy', f'{f}_lobe.npz'))['image'] > 0
    tot = int((L > 0).sum())
    cnt = {NAME[k]: int((L == k).sum()) for k in NAME}
    fr = {k: v / max(tot, 1) for k, v in cnt.items()}
    s = L[::2, ::2, ::2]
    cx, cz = {}, {}
    for k, n in NAME.items():
        idx = np.nonzero(s == k)
        cx[n] = idx[1].mean() if idx[1].size else np.nan; cz[n] = idx[2].mean() if idx[2].size else np.nan
    lr = bool(np.nanmax([cx['RUL'], cx['RML'], cx['RLL']]) < np.nanmin([cx['LUL'], cx['LLL']]))
    zz = bool(cz['RUL'] < cz['RLL'] and cz['LUL'] < cz['LLL'])
    w.writerow(dict(folder=f, total_vox=tot, lung_L=round(tot * 0.64 / 1e6, 2),
                    **{f'f_{k}': round(fr[k], 4) for k in REF}, max_dev=round(max(abs(fr[k] - REF[k]) for k in REF), 3),
                    missing_lobe=','.join(k for k in REF if cnt[k] == 0), lr_ok=lr, z_ok=zz,
                    inside_lung=round(((L > 0) & lung).sum() / max(tot, 1), 4)))
    fh.flush()
fh.close()
print('QC done', len(files))
