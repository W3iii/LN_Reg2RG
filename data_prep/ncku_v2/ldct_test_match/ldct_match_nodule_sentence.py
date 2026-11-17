"""
LDCT_test_dataset <-> LDCT_test1000_report: nodule mask <-> 報告句子配對
(ME_dataset 版 match_nodule_sentence.py 的 LDCT 移植版)

跟 ME 版的差異:
  * 沒有醫師逐顆 Excel / bbox_ME / TW_Lung_RADS -> nodule 直接取自 mask json (bbox + size)
  * LDCT 的 *_lobe.npz 只是 0/1 全肺 mask, 沒有五肺葉 -> 若 reg2rg_dataset/lobe_ldct/<folder>_lobe5.npz
    存在 (TotalSegmentator) 就用真實肺葉, 否則退回「x 對肺中線定側別 + z 相對高度」弱訊號
  * 報告清理規則 (BOILER / DROP_SENT / SLICE_REF / LOBE_RULES ...) 直接從 build_dataset.py 載入, 兩邊一致

座標: LDCT npy 與 ME 相同, 已 resample 到 (0.8, 0.8, 1.0) mm; json bbox = [[y0,x0,z0],[y1,x1,z1]] npy 座標
"""
import os, re, json, io, sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DS = os.path.join(ROOT, 'LDCT_test_dataset')
OUT = HERE
LOBE_DIR = os.path.join(ROOT, 'reg2rg_dataset', 'lobe_ldct')
XY, Z = 0.8, 1.0

# ---- 從 build_dataset.py 只取「報告正規化」那段 (常數 + 函式), 不跑它的主流程 ----
src = io.open(os.path.join(ROOT, 'reg2rg_dataset', 'build_dataset.py'), encoding='utf-8').read()
a = src.index('# ============================================================ 報告正規化')
b = src.index('# ============================================================ 主流程')
exec(src[a:b])

# ---- match_nodule_sentence.py 的規則 (照抄) ----
LOBE_SIDE = {'RUL': 'R', 'RML': 'R', 'RLL': 'R', 'LUL': 'L', 'LLL': 'L'}
LOBE_ZONE = {'RUL': (0.00, 0.55), 'RML': (0.40, 0.85), 'RLL': (0.35, 1.00),
             'LUL': (0.00, 0.65), 'LLL': (0.35, 1.00)}
NAME = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}


def stated_sizes_mm(t):
    out = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(mm|cm)\b', t, re.I):
        v = float(m.group(1)); out.append(v * 10 if m.group(2).lower() == 'cm' else v)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(mm|cm)\b', t, re.I):
        v = max(float(m.group(1)), float(m.group(2))); out.append(v * 10 if m.group(3).lower() == 'cm' else v)
    return [v for v in out if 0.5 <= v <= 120]


def sent_side(s):
    sides = {LOBE_SIDE[L] for L in s['lobes'] if L in LOBE_SIDE}
    if not sides:
        if re.search(r'\bright\b|\bRt\b|\bR\'t\b', s['text'], re.I): sides.add('R')
        if re.search(r'\bleft\b|\bLt\b|\bL\'t\b', s['text'], re.I): sides.add('L')
    return sides


def lobe_of(lobe, cy, cx, cz):
    cy, cx, cz = int(cy), int(cx), int(cz)
    for ry, rz in ((5, 3), (20, 12)):
        p = lobe[max(cy-ry, 0):cy+ry+1, max(cx-ry, 0):cx+ry+1, max(cz-rz, 0):cz+rz+1]
        nz = p[p > 0]
        if nz.size:
            return NAME.get(int(np.bincount(nz).argmax()), '')
    return ''


xl = pd.read_excel(os.path.join(ROOT, 'LDCT_test1000_report.xlsx'))
xl.columns = ['folder', 'report']
xl['report'] = xl.report.fillna('').astype(str)

nod_rows, rows, per_patient, skipped = [], [], [], []
for r0 in xl.itertuples():
    folder, t = r0.folder, r0.report
    j = json.load(open(os.path.join(DS, folder, 'mask', f'{folder}_nodule_count.json')))
    lob = [l.strip() for l in open(os.path.join(DS, folder, 'npy', 'lobe_info.txt')).read().strip().split('\n') if l.strip()]
    z0, z1 = [int(v) for v in lob[-2].split(',')]
    bbv = [int(v) for v in lob[-1].split(',')]
    mid_x, zspan = (bbv[2] + bbv[3]) / 2, max(z1 - z0, 1)
    lp = os.path.join(LOBE_DIR, f'{folder}_lobe5.npz')
    lobe5 = np.load(lp)['image'] if os.path.exists(lp) else None

    nods = []
    for k, (bb, sz) in enumerate(zip(j['bboxes'], j['nodule_size']), 1):
        (y0, x0, zz0), (y1, x1, zz1) = bb
        cy, cx, cz = (y0+y1)/2, (x0+x1)/2, (zz0+zz1)/2
        dia = 2 * (3 * sz * XY * XY * Z / (4 * np.pi)) ** (1/3)
        lobe = lobe_of(lobe5, cy, cx, cz) if lobe5 is not None else ''
        side = lobe[0] if lobe else ('L' if cx > mid_x else 'R')
        zrel = (cz - z0) / zspan
        nods.append(dict(nodule_id=k, side=side, lobe=lobe, zrel=zrel, dia=dia))
        nod_rows.append(dict(folder=folder, nodule_id=k, y0=y0, x0=x0, z0=zz0, y1=y1, x1=x1, z1=zz1,
                             cy=cy, cx=cx, cz=cz, size_vox=sz, eq_diam_mm=round(dia, 2),
                             bbox_long_mm=round(max(y1-y0, x1-x0) * XY, 2),
                             side=side, lobe=lobe, zrel=round(zrel, 3)))

    def emit_unmatched(reason):
        for n_ in nods:
            rows.append(dict(folder=folder, nodule_id=n_['nodule_id'], side=n_['side'], lobe=n_['lobe'],
                             zrel=round(n_['zrel'], 3), eq_diam_mm=round(n_['dia'], 2), matched_sentence='',
                             match_score=0.0, match_lobe='', lobe_agree=None, match_section='', note=reason))

    # ---- 報告篩選 (同 ME 版) ----
    nzh = len(re.findall(r'[一-鿿]', t))
    is_zh_cad = bool(re.search(r'Syngo\.Lung CAD', t)) or (nzh > 20 and not re.search(r'lung apex|Low dose chest CT|Chest CT', t, re.I))
    if not t.strip():
        skipped.append((folder, 'no_report', len(nods))); emit_unmatched('no_report'); continue
    if is_zh_cad:
        skipped.append((folder, 'zh_cad_template', len(nods))); emit_unmatched('zh_cad_template'); continue

    body = strip_boiler(t)
    find_b, imp_b = split_sections(body)
    sents = []
    for sec, blk in (('findings', find_b), ('impression', imp_b)):
        for s0 in to_sentences(blk):
            s = clean_sentence(s0)
            if s is None: continue
            sents.append(dict(section=sec, text=s, lobes=sentence_lobes(s),
                              is_lung_lesion=bool(re.search(LESION, s, re.I)) and not re.search(EXTRA_PULM, s, re.I)))

    cands = []
    for si, s in enumerate(sents):
        if not s['is_lung_lesion']: continue
        if re.match(r'^\s*no\b', s['text'], re.I): continue        # "No lesion over bilateral lungs." 之類否定句
        sd = sent_side(s)
        if not sd: continue
        cands.append(dict(si=si, text=s['text'], section=s['section'], lobes=s['lobes'],
                          sides=sd, sizes=stated_sizes_mm(s['text'])))

    # findings + impression 同一顆 -> 合併 (同 ME 版)
    merged, bylobe = [], defaultdict(list)
    for c in cands:
        bylobe[(tuple(sorted(c['lobes'])), tuple(sorted(c['sides'])))].append(c)
    for _, grp in bylobe.items():
        fs = [c for c in grp if c['section'] == 'findings']
        im = [c for c in grp if c['section'] == 'impression']
        if fs and im:
            cost = np.zeros((len(fs), len(im)))
            for a_, f in enumerate(fs):
                for b_, i_ in enumerate(im):
                    cost[a_, b_] = (min(abs(x-y) for x in f['sizes'] for y in i_['sizes'])
                                    if f['sizes'] and i_['sizes'] else 2.0)
            ra, rb = linear_sum_assignment(cost)
            paired = set()
            for a_, b_ in zip(ra, rb):
                if cost[a_, b_] <= 3.0:
                    f, i_ = fs[a_], im[b_]
                    merged.append(dict(si=f['si'], text=f['text'] + ' ' + i_['text'], section='both',
                                       lobes=f['lobes'], sides=f['sides'], sizes=f['sizes'] + i_['sizes']))
                    paired.add(b_); fs[a_] = None
            merged += [c for c in fs if c is not None] + [c for b_, c in enumerate(im) if b_ not in paired]
        else:
            merged += grp
    cands = merged

    if not nods or not cands:
        per_patient.append(dict(folder=folder, n_nodule=len(nods), n_cand_sent=len(cands), n_matched=0))
        emit_unmatched('no_candidate_sentence' if nods else '')
        continue

    S = np.full((len(nods), len(cands)), -1e3)
    for i, n_ in enumerate(nods):
        for jj, c in enumerate(cands):
            if n_['side'] not in c['sides']: continue
            sc = 3.0 + (0.5 if len(c['sides']) == 1 else 0)
            lobes5 = [L for L in c['lobes'] if L in LOBE_SIDE]
            if n_['lobe'] and lobes5:
                sc += 3.0 if n_['lobe'] in lobes5 else -1.0
            else:
                zs = [LOBE_ZONE[L] for L in c['lobes'] if L in LOBE_ZONE]
                if zs:
                    sc += 1.0 if any(lo <= n_['zrel'] <= hi for lo, hi in zs) else -0.5
            if c['sizes']:
                d = min(abs(n_['dia'] - v) for v in c['sizes'])
                sc += 2.0 if d <= 2 else (1.0 if d <= 4 else (-0.5 if d > 8 else 0.0))
            S[i, jj] = sc
    ri, ci = linear_sum_assignment(-S)
    assign = {i: jj for i, jj in zip(ri, ci) if S[i, jj] >= 3.0}
    for i, n_ in enumerate(nods):
        jj = assign.get(i); c = cands[jj] if jj is not None else None
        rows.append(dict(folder=folder, nodule_id=n_['nodule_id'], side=n_['side'], lobe=n_['lobe'],
                         zrel=round(n_['zrel'], 3), eq_diam_mm=round(n_['dia'], 2),
                         matched_sentence=c['text'] if c else '', match_score=round(S[i, jj], 2) if c else 0.0,
                         match_lobe=','.join(c['lobes']) if c else '',
                         lobe_agree=(n_['lobe'] in c['lobes']) if (c and n_['lobe']) else None,
                         match_section=c['section'] if c else '', note=''))
    per_patient.append(dict(folder=folder, n_nodule=len(nods), n_cand_sent=len(cands), n_matched=len(assign)))

# ============================================================ 輸出
Nd = pd.DataFrame(nod_rows); R = pd.DataFrame(rows); P = pd.DataFrame(per_patient)
Sk = pd.DataFrame(skipped, columns=['folder', 'reason', 'n_nodule'])
Nd.to_csv(os.path.join(OUT, 'ldct_nodules.csv'), index=False, encoding='utf-8-sig')
R.to_csv(os.path.join(OUT, 'ldct_nodule_sentence.csv'), index=False, encoding='utf-8-sig')
P.to_csv(os.path.join(OUT, 'ldct_per_patient.csv'), index=False, encoding='utf-8-sig')
Sk.to_csv(os.path.join(OUT, 'ldct_skipped.csv'), index=False, encoding='utf-8-sig')

# Excel: 跟 lung_M_..._with_perNodule_text.xlsx 同樣「一顆 nodule 一列」, Report 只放在每人第一列
X = Nd.merge(R[['folder', 'nodule_id', 'match_lobe', 'matched_sentence', 'match_score', 'note']],
             on=['folder', 'nodule_id'], how='left')
X['size_mm_text'] = X.matched_sentence.fillna('').apply(lambda s: max(stated_sizes_mm(s)) if stated_sizes_mm(s) else None)
X = X.rename(columns={'folder': 'Rename', 'nodule_id': 'nodule', 'match_lobe': 'lobe_text',
                      'matched_sentence': 'per_nodule_text', 'lobe': 'lobe_seg'})
X = X.merge(xl.rename(columns={'folder': 'Rename', 'report': 'Report'}), on='Rename', how='left')
X.loc[X.duplicated('Rename'), 'Report'] = None
cols = ['Rename', 'nodule', 'cy', 'cx', 'cz', 'size_vox', 'eq_diam_mm', 'side', 'lobe_seg', 'zrel',
        'Report', 'lobe_text', 'size_mm_text', 'per_nodule_text', 'match_score', 'note']
X[cols].to_excel(os.path.join(ROOT, 'LDCT_test1000_report_with_perNodule_text.xlsx'), index=False)

m = R.matched_sentence != ''
L = [f'病患 1000 位, 有 mask 的 {(Nd.groupby("folder").size() > 0).sum()} 位, nodule 共 {len(Nd)} 顆',
     f'五肺葉 mask: {"有" if (Nd.lobe != "").any() else "沒有 -> 側別用 x 中線, 肺葉用 z 高度弱訊號"}',
     f'排除報告: {Sk.reason.value_counts().to_dict()}  (其中有 nodule 的病患 {(Sk.n_nodule > 0).sum()} 位 / {Sk.n_nodule.sum()} 顆)',
     f'配到句子的 nodule: {m.sum()} / {len(R)} ({100*m.mean():.1f}%)',
     f'分數 (已配對): median {R[m].match_score.median():.1f}  p10 {R[m].match_score.quantile(.1):.1f}  p90 {R[m].match_score.quantile(.9):.1f}',
     f'  側別+尺寸都對 (>=5.5): {(R[m].match_score >= 5.5).sum()}   只有側別 (<4): {(R[m].match_score < 4).sum()}',
     f'完全沒配到的病患 (有 nodule 且英文報告): {((P.n_nodule > 0) & (P.n_matched == 0)).sum()}',
     f'nodule 數 > 候選句數的病患: {(P.n_nodule > P.n_cand_sent).sum()}']
io.open(os.path.join(OUT, 'ldct_match_log.txt'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
