"""
Step 4b: nodule instance <-> 報告句子 配對 (位置 + 大小規則匹配)

訊號:
  1. 側別  — nodule cx 對肺中線 (已驗證可靠) vs 句子提到的 lobe/左右
  2. 大小  — mask 等效球徑 vs 句子敘述的尺寸 (單顆病患驗證 r=0.932)
  3. 縱向  — nodule cz 在肺範圍內的相對位置 vs lobe 的 upper/middle/lower (弱訊號)

側別為硬條件, 其餘為評分。用 Hungarian 做最佳指派, 低於門檻不配。
lobe mask 到位後把 `lobe` 欄填上, ZONE 弱訊號可換成硬條件。
"""
import os, re, json, io
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
OUT = os.path.join(ROOT, 'reg2rg_dataset')

LOBE_SIDE = {'RUL': 'R', 'RML': 'R', 'RLL': 'R', 'LUL': 'L', 'LLL': 'L'}
# lobe 在肺 z 範圍內的大致相對高度 (0=肺尖, 1=肺底)
LOBE_ZONE = {'RUL': (0.00, 0.55), 'RML': (0.40, 0.85), 'RLL': (0.35, 1.00),
             'LUL': (0.00, 0.65), 'LLL': (0.35, 1.00)}

N = pd.read_csv(os.path.join(OUT, 'nodules.csv'))
reps = {r['pid']: r for r in (json.loads(l) for l in io.open(os.path.join(OUT, 'reports.jsonl'), encoding='utf-8'))}

# 每位病患的肺中線 / 肺 z 範圍
geom = {}
for folder in N.folder.unique():
    lob = [l.strip() for l in open(os.path.join(ME, folder, 'npy', 'lobe_info.txt')).read().strip().split('\n') if l.strip()]
    z0, z1 = [int(v) for v in lob[-2].split(',')]
    bbv = [int(v) for v in lob[-1].split(',')]
    geom[folder] = dict(mid_x=(bbv[2] + bbv[3]) / 2, z0=z0, z1=z1)


def stated_sizes_mm(t):
    out = []
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*[-\s]?\s*(mm|cm)\b', t, re.I):
        v = float(m.group(1))
        out.append(v * 10 if m.group(2).lower() == 'cm' else v)
    for m in re.finditer(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(mm|cm)\b', t, re.I):
        v = max(float(m.group(1)), float(m.group(2)))
        out.append(v * 10 if m.group(3).lower() == 'cm' else v)
    return [v for v in out if 0.5 <= v <= 120]


def sent_side(s):
    sides = {LOBE_SIDE[L] for L in s['lobes'] if L in LOBE_SIDE}
    if not sides:
        if re.search(r'\bright\b|\bRt\b', s['text'], re.I):
            sides.add('R')
        if re.search(r'\bleft\b|\bLt\b', s['text'], re.I):
            sides.add('L')
    return sides


rows, per_patient = [], []
for pid, g in N.groupby('pid'):
    rep = reps.get(pid)
    if rep is None:
        continue
    folder = g.folder.iloc[0]
    gm = geom[folder]
    zspan = max(gm['z1'] - gm['z0'], 1)

    nods = []
    for r in g.itertuples():
        lobe = getattr(r, 'lobe', '') or ''
        # 有 TotalSegmentator 肺葉就用它定側別 (99.8% 與報告一致), 沒有才退回 x 座標
        side = lobe[0] if lobe else ('L' if r.cx > gm['mid_x'] else 'R')
        zrel = (r.cz - gm['z0']) / zspan
        nods.append(dict(idx=r.Index, nodule_id=r.nodule_id, side=side, zrel=zrel,
                         lobe=lobe, dia=r.eq_diam_mm, rads=r.tw_lung_rads))

    # 候選句: 肺部病灶句 (impression 與 findings 都收, 之後取分數高者)
    cands = []
    for si, s in enumerate(rep['sentences']):
        if not s['is_lung_lesion']:
            continue
        sizes = stated_sizes_mm(s['text'])
        sd = sent_side(s)
        if not sd:
            continue
        cands.append(dict(si=si, text=s['text'], section=s['section'],
                          lobes=s['lobes'], sides=sd, sizes=sizes))

    # findings 與 impression 常在講同一顆病灶 -> 合併, 否則同一顆會被拆給兩個 nodule
    merged = []
    from collections import defaultdict
    bylobe = defaultdict(list)
    for c in cands:
        bylobe[(tuple(sorted(c['lobes'])), tuple(sorted(c['sides'])))].append(c)
    for _, grp in bylobe.items():
        fs = [c for c in grp if c['section'] == 'findings']
        im = [c for c in grp if c['section'] == 'impression']
        if fs and im:
            cost = np.zeros((len(fs), len(im)))
            for a, f in enumerate(fs):
                for b, i_ in enumerate(im):
                    if f['sizes'] and i_['sizes']:
                        cost[a, b] = min(abs(x - y) for x in f['sizes'] for y in i_['sizes'])
                    else:
                        cost[a, b] = 2.0                 # 缺尺寸 -> 中性
            ra, rb = linear_sum_assignment(cost)
            paired_i = set()
            for a, b in zip(ra, rb):
                if cost[a, b] <= 3.0:                    # 尺寸差 3mm 內視為同一顆
                    f, i_ = fs[a], im[b]
                    merged.append(dict(si=f['si'], text=f['text'] + ' ' + i_['text'],
                                       section='both', lobes=f['lobes'], sides=f['sides'],
                                       sizes=f['sizes'] + i_['sizes']))
                    paired_i.add(b); fs[a] = None
            merged += [c for c in fs if c is not None]
            merged += [c for b, c in enumerate(im) if b not in paired_i]
        else:
            merged += grp
    cands = merged

    if not nods or not cands:
        per_patient.append(dict(pid=pid, folder=folder, n_nodule=len(nods),
                                n_cand_sent=len(cands), n_matched=0))
        for n_ in nods:
            rows.append(dict(pid=pid, folder=folder, nodule_id=n_['nodule_id'],
                             side=n_['side'], lobe=n_['lobe'], eq_diam_mm=n_['dia'],
                             tw_lung_rads=n_['rads'], matched_sentence='', match_score=0.0,
                             match_lobe='', lobe_agree=None, match_section=''))
        continue

    # 評分矩陣
    S = np.full((len(nods), len(cands)), -1e3)
    for i, n_ in enumerate(nods):
        for j, c in enumerate(cands):
            if n_['side'] not in c['sides']:
                continue                                    # 側別硬條件
            sc = 3.0
            if len(c['sides']) == 1:
                sc += 0.5                                   # 句子只指一側 -> 更明確
            lobes5 = [L for L in c['lobes'] if L in LOBE_SIDE]
            if n_['lobe'] and lobes5:
                # 真實肺葉 (TotalSegmentator) vs 句子肺葉: 主要訊號
                # 同側不同葉不直接排除 — 裂隙旁 nodule 與分割誤差都會落在這裡
                sc += 3.0 if n_['lobe'] in lobes5 else -1.0
            else:
                zs = [LOBE_ZONE[L] for L in c['lobes'] if L in LOBE_ZONE]
                if zs:                                      # 無肺葉時退回縱向弱訊號
                    sc += 1.0 if any(lo <= n_['zrel'] <= hi for lo, hi in zs) else -0.5
            if c['sizes']:
                d = min(abs(n_['dia'] - v) for v in c['sizes'])
                sc += 2.0 if d <= 2 else (1.0 if d <= 4 else (-0.5 if d > 8 else 0.0))
            S[i, j] = sc

    ri, ci = linear_sum_assignment(-S)
    assign = {i: j for i, j in zip(ri, ci) if S[i, j] >= 3.0}   # 門檻: 至少側別對上
    for i, n_ in enumerate(nods):
        j = assign.get(i)
        c = cands[j] if j is not None else None
        rows.append(dict(pid=pid, folder=folder, nodule_id=n_['nodule_id'],
                         side=n_['side'], lobe=n_['lobe'],
                         eq_diam_mm=n_['dia'], tw_lung_rads=n_['rads'],
                         matched_sentence=c['text'] if c else '',
                         match_score=round(S[i, j], 2) if c else 0.0,
                         match_lobe=','.join(c['lobes']) if c else '',
                         lobe_agree=(n_['lobe'] in c['lobes']) if (c and n_['lobe']) else None,
                         match_section=c['section'] if c else ''))
    per_patient.append(dict(pid=pid, folder=folder, n_nodule=len(nods),
                            n_cand_sent=len(cands), n_matched=len(assign)))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, 'nodule_sentence.csv'), index=False, encoding='utf-8-sig')
P = pd.DataFrame(per_patient)

log = io.open(os.path.join(OUT, 'match_log.txt'), 'w', encoding='utf-8')
log.write('nodule 總數 %d / 病患 %d\n' % (len(R), len(P)))
m = R.matched_sentence != ''
log.write('配到句子的 nodule : %d (%.1f%%)\n' % (m.sum(), 100 * m.mean()))
log.write('沒配到的 nodule   : %d (%.1f%%)  -> region 文字給 "No specific description."\n'
          % ((~m).sum(), 100 * (~m).mean()))
log.write('\n分數分布 (已配對): median %.1f  p10 %.1f  p90 %.1f\n'
          % (R[m].match_score.median(), R[m].match_score.quantile(.1), R[m].match_score.quantile(.9)))
log.write('  高信心 (>=5.5, 側別+尺寸都對) : %d (%.1f%% of 已配對)\n'
          % ((R[m].match_score >= 5.5).sum(), 100 * (R[m].match_score >= 5.5).mean()))
log.write('  只有側別對上 (<4)             : %d\n' % (R[m].match_score < 4).sum())
log.write('\n每位病患: nodule median %d, 候選句 median %d, 配對 median %d\n'
          % (P.n_nodule.median(), P.n_cand_sent.median(), P.n_matched.median()))
log.write('nodule 數 > 候選句數的病患 : %d (%.1f%%)  <- 報告只寫有意義的病灶, 屬正常\n'
          % ((P.n_nodule > P.n_cand_sent).sum(), 100 * (P.n_nodule > P.n_cand_sent).mean()))
log.write('完全沒配到任何 nodule 的病患: %d\n' % (P.n_matched == 0).sum())
log.write('\n配對率 by TW_Lung_RADS (惡性度越高, 報告越該提到):\n')
for k, gg in R.groupby('tw_lung_rads'):
    log.write('  class %d : %4d 顆, 配到 %.1f%%\n' % (k, len(gg), 100 * (gg.matched_sentence != '').mean()))
la = R.lobe_agree.dropna()
if len(la):
    log.write('\n已配對且雙方都有肺葉的 %d 顆中, 肺葉一致 %d (%.1f%%)\n'
              % (len(la), la.sum(), 100 * la.mean()))
    log.write('  (不一致多為裂隙旁 nodule 或分割誤差, 已保留但配對分數較低)\n')
log.close()
print(open(os.path.join(OUT, 'match_log.txt'), encoding='utf-8').read())
