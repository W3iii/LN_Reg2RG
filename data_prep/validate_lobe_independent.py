"""
獨立驗證肺葉分割 (不依賴 nodule<->句子配對器, 避免循環論證)

只取「該病患剛好 1 顆 nodule」且「報告的肺部病灶句剛好只提到 1 個肺葉」的案例。
這種情況下那顆 nodule 必然就是那句話講的病灶, 完全不需要配對演算法介入。
"""
import os, json, io
import numpy as np
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
DS = os.path.join(ROOT, 'reg2rg_dataset')
LOBE = os.path.join(DS, 'lobe')
LOBES5 = {'RUL', 'RML', 'RLL', 'LUL', 'LLL'}

N = pd.read_csv(os.path.join(DS, 'nodules.csv'))
reps = {r['pid']: r for r in (json.loads(l) for l in io.open(os.path.join(DS, 'reports.jsonl'), encoding='utf-8'))}

single = N.groupby('pid').filter(lambda g: len(g) == 1)
rows = []
for r in single.itertuples():
    rep = reps.get(r.pid)
    if not rep:
        continue
    named = set()
    for s in rep['sentences']:
        if s['is_lung_lesion']:
            named.update(L for L in s['lobes'] if L in LOBES5)
    if len(named) != 1:
        continue
    said = named.pop()
    rows.append(dict(pid=r.pid, folder=r.folder, seg_lobe=r.lobe, report_lobe=said,
                     side_ok=r.lobe[0] == said[0], exact=r.lobe == said,
                     dia_mm=r.eq_diam_mm, rads=r.tw_lung_rads))

V = pd.DataFrame(rows)
V.to_csv(os.path.join(DS, 'lobe_validation_independent.csv'), index=False, encoding='utf-8-sig')
print('獨立可驗證案例: %d (單顆 nodule + 報告只提一個肺葉)' % len(V))
print('左右一致 : %d / %d (%.1f%%)' % (V.side_ok.sum(), len(V), 100 * V.side_ok.mean()))
print('肺葉全對 : %d / %d (%.1f%%)' % (V.exact.sum(), len(V), 100 * V.exact.mean()))
print('\n混淆 (列=TotalSegmentator, 欄=報告):')
print(pd.crosstab(V.seg_lobe, V.report_lobe).to_string())
print('\n不一致案例的 nodule 直徑: median %.1f mm (一致者 %.1f mm)'
      % (V[~V.exact].dia_mm.median() if (~V.exact).any() else float('nan'), V[V.exact].dia_mm.median()))
