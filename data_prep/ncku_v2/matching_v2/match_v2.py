"""
Nodule <-> 報告句子配對 v2 (ME_dataset + LDCT_test 共用)

v1 的問題 (CHEST1011):
  "There are small GGOs in the LLL, RML, and RLL (0.4cm, 0.4cm, 0.6cm)" 一句描述 3 顆,
  v1 整句只能給 1 顆 (Hungarian 一對一) -> 該顆的 per_nodule_text 混入別顆的描述, 其他肺葉的 nodule 配不到.

v2 改動 (評分規則與 v1 相同):
  1. lobe_split : 句子明確列出 >=2 個肺葉 -> 拆成每肺葉一個候選; 尺寸數 = 肺葉數時照順序配對 (respectively)
  2. size_split : 單一肺葉但列出多個尺寸的複數句 ("LLL nodules, 0.5cm and 0.5cm") -> 依尺寸拆
  3. shared     : 複數且沒寫數量/尺寸列表 ("RUL small nodules") -> 可同時給多顆 nodule (同側)
  4. "No ..." 開頭的否定句不當候選
  5. "up to X" 視為上限: nodule <= X 不扣分
  拆開的片段會改寫成單數、單一肺葉、單一尺寸 -> per_nodule_text; 原句保留在 matched_sentence
"""
import os, re, io, json, sys
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
src = io.open(os.path.join(ROOT, 'reg2rg_dataset', 'build_dataset.py'), encoding='utf-8').read()
exec(src[src.index('# ============================================================ 報告正規化'):
         src.index('# ============================================================ 主流程')])

L5 = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
LOBE_SIDE = {'RUL': 'R', 'RML': 'R', 'RLL': 'R', 'LUL': 'L', 'LLL': 'L'}
LOBE_ZONE = {'RUL': (0.00, 0.55), 'RML': (0.40, 0.85), 'RLL': (0.35, 1.00), 'LUL': (0.00, 0.65), 'LLL': (0.35, 1.00)}
NODULE_LIKE = r'nodul|GGN|GGO|ground[\s-]?glass|opacit|mass\b|tumou?r|lesion|densit|calcif'
PLURAL = r'\b(?:nodules|opacities|GGNs|GGOs|densities|lesions|calcifications|calcified nodules|masses)\b'
COUNT = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'both': 2}

SIZE_RE = re.compile(
    r'(?P<list>(?:\d+(?:\.\d+)?\s*(?:,\s*|\s+and\s+|\s*&\s*))+\d+(?:\.\d+)?)\s*(?P<lu>mm|cm)\b'
    r'|(?P<a>\d+(?:\.\d+)?)\s*x\s*(?P<b>\d+(?:\.\d+)?)(?:\s*x\s*(?P<c>\d+(?:\.\d+)?))?\s*(?P<xu>mm|cm)\b'
    r'|(?P<s>\d+(?:\.\d+)?)\s*-?\s*(?P<su>mm|cm)\b', re.I)


def sizes_ordered(t):
    """[(start, end, [(value_mm, unit, raw_number)])] in text order"""
    out = []
    for m in SIZE_RE.finditer(t):
        if m.group('list'):
            u = m.group('lu').lower(); nums = re.findall(r'\d+(?:\.\d+)?', m.group('list'))
            vals = [(float(x) * (10 if u == 'cm' else 1), u, x) for x in nums]
        elif m.group('a'):
            u = m.group('xu').lower(); x = max(float(m.group('a')), float(m.group('b')))
            vals = [(x * (10 if u == 'cm' else 1), u, f'{x:g}')]
        else:
            u = m.group('su').lower(); vals = [(float(m.group('s')) * (10 if u == 'cm' else 1), u, m.group('s'))]
        vals = [v for v in vals if 0.5 <= v[0] <= 120]
        if vals: out.append((m.start(), m.end(), vals))
    return out


def lobe_mentions(t):
    ms = []
    for name, pat in LOBE_RULES:
        for m in re.finditer(pat, t, re.I):
            e = m.end()
            while e < len(t) and t[e].isalpha(): e += 1          # "lingul" -> "lingula"
            ms.append((m.start(), e, name))
    ms.sort()
    return ms


def singular(t):
    t = re.sub(r'^\s*(?:There\s+are|There\s+were)\s+', 'There is a ', t, flags=re.I)
    t = re.sub(r'^\s*(?:Two|Three|Four|Five|Several|Multiple|Some|A\s+few|Few)\s+', 'A ', t, flags=re.I)
    for a, b in (('opacities', 'opacity'), ('densities', 'density'), ('nodules', 'nodule'), ('GGOs', 'GGO'),
                 ('GGNs', 'GGN'), ('lesions', 'lesion'), ('calcifications', 'calcification'),
                 ('infiltrations', 'infiltration'), ('masses', 'mass')):
        t = re.sub(r'\b' + a + r'\b', b, t)
    t = re.sub(r'\bare\s+(noted|seen|found|present)\b', r'is \1', t)
    t = re.sub(r'\bA\s+(?=[aeiouAEIOU])', 'An ', t)
    t = re.sub(r'\bThere is a\s+(?=[aeiou])', 'There is an ', t)
    return t[:1].upper() + t[1:]


SEP_OK = r'^[\s,/&;:\-]*(?:(?:and|or|the|both|lobes?|lungs?|segments?|L\'t|R\'t|Lt\.?|Rt\.?|left|right)\b[\s,/&;:\-]*)*$'


def rewrite(t, lobe_span, lobe_txt, size_span, size_txt):
    """replace the lobe enumeration / size enumeration; edits applied right-to-left"""
    edits = []
    if lobe_span: edits.append((lobe_span[0], lobe_span[1], lobe_txt))
    if size_span: edits.append((size_span[0], size_span[1], size_txt))
    edits.sort(reverse=True)
    for s, e, r in edits:
        t = t[:s] + r + t[e:]
    t = re.sub(r',?\s*respectively', '', t, flags=re.I)
    t = re.sub(r'\(\s*\)', '', t); t = re.sub(r'\s{2,}', ' ', t).strip()
    return singular(t)


def explode(c, n_nod):
    """one candidate sentence -> list of pieces (each a candidate)"""
    t = c['text']
    base = dict(section=c['section'], orig=t, upto=bool(re.search(r'up to|largest', t, re.I)))
    ms = lobe_mentions(t)
    order, first = [], {}
    for s, e, L in ms:
        if L not in first: first[L] = (s, e); order.append(L)
    szs = sizes_ordered(t)
    flat = [v for _, _, vals in szs for v in vals]
    plural = bool(re.search(PLURAL, t, re.I))

    # ---- 1. lobe_split
    if len(order) >= 2:
        s0, e1 = ms[0][0], ms[-1][1]
        left = t[s0:e1]
        for s, e, _ in sorted(ms, reverse=True):
            left = left[:s - s0] + left[e - s0:]
        lobe_ok = bool(re.match(SEP_OK, left, re.I))
        size_span = None
        if len(flat) == len(order) and szs:
            ss, se = szs[0][0], szs[-1][1]
            rest = t[ss:se]
            for s, e, _ in sorted(szs, reverse=True): rest = rest[:s - ss] + rest[e - ss:]
            if re.match(r'^[\s,;&]*(?:(?:and|respectively)\b[\s,;&]*)*$', rest, re.I): size_span = (ss, se)
        # 肺葉與尺寸交錯: "GGN at RUL(0.3cm), and RLL(0.3cm)" -> 依肺葉切段, 每段一個尺寸
        if not lobe_ok and len(ms) == len(order):
            prefix = t[:ms[0][0]]
            segs = [t[ms[k][0]:(ms[k + 1][0] if k + 1 < len(ms) else len(t))] for k in range(len(ms))]
            seg_sz = [[v for _, _, vals in sizes_ordered(sg) for v in vals] for sg in segs]
            if all(len(x) == 1 for x in seg_sz):
                out = []
                for (s_, e_, L), sg, sz in zip(ms, segs, seg_sz):
                    sg = re.sub(r'[\s,;]*(?:and|&)?[\s,;]*$', '', sg)
                    txt = singular((prefix + sg).strip().rstrip('.,;') + '.')
                    out.append(dict(base, text=txt, lobes=[L], sides={LOBE_SIDE[L]}, sizes=[sz[0][0]],
                                    kind='lobe_split', rewritten=True))
                return out
        # 不同肺葉、不同病灶用 and / ; 連接: "RUL small nodule and LLL GGN" -> 依子句拆, 子句內再遞迴拆
        if not lobe_ok and not c.get('_clause'):
            parts = re.split(r";\s*|,?\s+and\s+(?=(?:a\s+|an\s+)?(?:RUL|RML|RLL|LUL|LLL|L't|R't|Lt\b|Rt\b|left|right|lingul))",
                             t.rstrip('.'), flags=re.I)
            parts = [x.strip(' ,') for x in parts if x and x.strip(' ,')]
            if len(parts) >= 2 and all(lobe_mentions(x) for x in parts):
                out = []
                for x in parts:
                    x = x[0].upper() + x[1:] + '.'
                    lb = sentence_lobes(x)
                    sub = explode(dict(text=x, section=c['section'], lobes=lb, sides=sent_side(lb, x), _clause=True), n_nod)
                    for q in sub:
                        q['orig'] = t; q['kind'] = 'lobe_split'; q['rewritten'] = True
                    out += sub
                return out
        out = []
        for i, L in enumerate(order):
            if len(flat) == len(order):
                sz = [flat[i][0]]
                stxt = f'{flat[i][2]}{flat[i][1]}'
            else:
                sz = [v[0] for v in flat]; stxt = None
            if lobe_ok:
                ltxt = t[first[L][0]:first[L][1]]
                txt = rewrite(t, (s0, e1), ltxt, size_span if stxt else None, stxt or '')
            else:
                txt = t                                    # 結構太複雜, 不改寫, 只把候選限定在該肺葉
            out.append(dict(base, text=txt, lobes=[L], sides={LOBE_SIDE[L]}, sizes=sz, kind='lobe_split',
                            rewritten=lobe_ok))
        return out

    lobes = [L for L in c['lobes'] if L in L5]
    # ---- 2. size_split
    if plural and 2 <= len(flat) <= 6:
        ss, se = szs[0][0], szs[-1][1]
        rest = t[ss:se]
        for s, e, _ in sorted(szs, reverse=True): rest = rest[:s - ss] + rest[e - ss:]
        ok = bool(re.match(r'^[\s,;&]*(?:(?:and|respectively)\b[\s,;&]*)*$', rest, re.I))
        return [dict(base, text=rewrite(t, None, '', (ss, se), f'{v[2]}{v[1]}') if ok else t,
                     lobes=lobes, sides=c['sides'], sizes=[v[0]], kind='size_split', rewritten=ok) for v in flat]
    # ---- 3. count word / shared plural
    m = re.match(r'^\s*(two|three|four|five|both)\b', t, re.I)
    if plural and m:
        k = COUNT[m.group(1).lower()]
        return [dict(base, text=t, lobes=lobes, sides=c['sides'], sizes=[v[0] for v in flat], kind='count_copy',
                     rewritten=False) for _ in range(k)]
    if plural:
        return [dict(base, text=t, lobes=lobes, sides=c['sides'], sizes=[v[0] for v in flat], kind='shared',
                     rewritten=False) for _ in range(max(n_nod, 1))]
    return [dict(base, text=t, lobes=lobes, sides=c['sides'], sizes=[v[0] for v in flat], kind='single',
                 rewritten=False)]


def sent_side(lobes, text):
    sides = {LOBE_SIDE[L] for L in lobes if L in LOBE_SIDE}
    if not sides:
        if re.search(r"\bright\b|\bRt\b|\bR't\b", text, re.I): sides.add('R')
        if re.search(r"\bleft\b|\bLt\b|\bL't\b", text, re.I): sides.add('L')
    return sides


def match_patient(nods, sents):
    cands = []
    for s in sents:
        if not s['is_lung_lesion'] or re.match(r'^\s*no\b', s['text'], re.I): continue
        sd = sent_side(s['lobes'], s['text'])
        if not sd: continue
        for q in explode(dict(text=s['text'], section=s['section'], lobes=s['lobes'], sides=sd), len(nods)):
            q.setdefault('order', len(cands)); q['nodule_like'] = bool(re.search(NODULE_LIKE, q['text'], re.I)); cands.append(q)
    # findings + impression 同一病灶合併 (同 v1, 但現在多為單一肺葉片段)
    merged, groups = [], defaultdict(list)
    for c in cands:
        groups[(tuple(sorted(c['lobes'])), tuple(sorted(c['sides'])), c['kind'] == 'shared')].append(c)
    for _, grp in groups.items():
        fs = [c for c in grp if c['section'] == 'findings']; im = [c for c in grp if c['section'] == 'impression']
        if fs and im:
            cost = np.array([[min(abs(x - y) for x in f['sizes'] for y in i['sizes']) if f['sizes'] and i['sizes'] else 2.0
                              for i in im] for f in fs])
            ra, rb = linear_sum_assignment(cost); used_i = set()
            for a, b in zip(ra, rb):
                if cost[a, b] <= 3.0:
                    f, i = fs[a], im[b]
                    same = f['text'].strip().lower() == i['text'].strip().lower()   # Findings 與 Impression 一字不差 -> 只留一次
                    merged.append(dict(f, text=f['text'] if same else f['text'] + ' ' + i['text'], orig=f['orig'] + ' ' + i['orig'],
                                       section='both', sizes=f['sizes'] + i['sizes'],
                                       upto=f['upto'] or i['upto'], rewritten=f['rewritten'] or i['rewritten']))
                    used_i.add(b); fs[a] = None
            merged += [c for c in fs if c] + [c for b, c in enumerate(im) if b not in used_i]
        else:
            merged += grp
    cands = merged
    if not nods or not cands:
        return {}, cands
    S = np.full((len(nods), len(cands)), -1e3)
    for i, n in enumerate(nods):
        for j, c in enumerate(cands):
            if n['side'] not in c['sides']: continue
            sc = 3.0 + (0.5 if len(c['sides']) == 1 else 0) - (0.25 if c['kind'] == 'shared' else 0)   # 具體描述優先
            if c.get('nodule_like'): sc += 0.2                     # 平手時優先結節類句子, 而非肺塌陷/支氣管擴張
            lb = [L for L in c['lobes'] if L in L5]
            if n['lobe'] and lb:
                sc += 3.0 if n['lobe'] in lb else -1.0
            else:
                zs = [LOBE_ZONE[L] for L in lb]
                if zs: sc += 1.0 if any(lo <= n['zrel'] <= hi for lo, hi in zs) else -0.5
            if c['sizes']:
                d = min((max(0.0, n['dia'] - v) if c['upto'] else abs(n['dia'] - v)) for v in c['sizes'])
                sc += 2.0 if d <= 2 else (1.0 if d <= 4 else (-0.5 if d > 8 else 0.0))
            S[i, j] = sc
    ri, ci = linear_sum_assignment(-S)
    return {i: (j, S[i, j]) for i, j in zip(ri, ci) if S[i, j] >= 3.0}, cands


def run(dataset, patients):
    """patients: list of (key, nods, sents)"""
    rows, rep_lesion, pieces = [], [], []
    for key, nods, sents in patients:
        asg, cands = match_patient(nods, sents)
        for i, n in enumerate(nods):
            a = asg.get(i); c = cands[a[0]] if a else None
            rows.append(dict(key=key, nodule_id=n['nodule_id'], side=n['side'], lobe=n['lobe'], zrel=round(n['zrel'], 3),
                             eq_diam_mm=round(n['dia'], 2), per_nodule_text=c['text'] if c else '',
                             matched_sentence=c['orig'] if c else '', piece_lobe=','.join(c['lobes']) if c else '',
                             piece_size_mm=(c['sizes'][0] if c and len(c['sizes']) == 1 else None),
                             kind=c['kind'] if c else '', rewritten=c['rewritten'] if c else None,
                             match_score=round(a[1], 2) if a else 0.0,
                             lobe_agree=(n['lobe'] in c['lobes']) if (c and n['lobe'] and c['lobes']) else None))
        by_cand = {}
        for i, (j, sc) in asg.items(): by_cand.setdefault(j, []).append(nods[i]['nodule_id'])
        for j, c in enumerate(cands):
            pieces.append(dict(key=key, piece=j, order=c.get('order', j), section=c['section'], kind=c['kind'],
                               lobes=','.join(c['lobes']), sides=','.join(sorted(c['sides'])),
                               sizes=';'.join(f'{v:g}' for v in c['sizes']), upto=c['upto'],
                               nodule_like=bool(re.search(NODULE_LIKE, c['text'], re.I)),
                               assigned=','.join(str(n) for n in sorted(by_cand.get(j, []))), text=c['text'], orig=c['orig']))
        used = {a[0] for a in asg.values()}
        for j, c in enumerate(cands):
            if j in used or c['kind'] == 'shared': continue
            if not re.search(NODULE_LIKE, c['text'], re.I): continue
            rep_lesion.append(dict(key=key, lobes=','.join(c['lobes']), size=c['sizes'][0] if len(c['sizes']) == 1 else None,
                                   kind=c['kind'], text=c['text']))
    R = pd.DataFrame(rows); U = pd.DataFrame(rep_lesion)
    R.to_csv(os.path.join(HERE, f'{dataset}_nodule_sentence_v2.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(pieces).to_csv(os.path.join(HERE, f'{dataset}_pieces_v2.csv'), index=False, encoding='utf-8-sig')
    U.to_csv(os.path.join(HERE, f'{dataset}_report_lesion_without_mask.csv'), index=False, encoding='utf-8-sig')
    return R, U


# ============================================================ ME_dataset
N = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'nodules.csv'))
reps = {r['pid']: r for r in (json.loads(l) for l in io.open(os.path.join(ROOT, 'reg2rg_dataset', 'reports.jsonl'), encoding='utf-8'))}
pts = []
for pid, g in N.groupby('pid'):
    if pid not in reps: continue
    folder = g.folder.iloc[0]
    lob = [l.strip() for l in open(os.path.join(ROOT, 'ME_dataset', folder, 'npy', 'lobe_info.txt')).read().strip().split('\n') if l.strip()]
    z0, z1 = [int(v) for v in lob[-2].split(',')]; bbv = [int(v) for v in lob[-1].split(',')]
    mid = (bbv[2] + bbv[3]) / 2
    nods = []
    for r in g.itertuples():
        lobe = r.lobe if isinstance(r.lobe, str) else ''
        nods.append(dict(nodule_id=r.nodule_id, lobe=lobe, side=lobe[0] if lobe else ('L' if r.cx > mid else 'R'),
                         zrel=(r.cz - z0) / max(z1 - z0, 1), dia=r.eq_diam_mm))
    pts.append((folder, nods, reps[pid]['sentences']))
ME_R, ME_U = run('me', pts)

# ============================================================ LDCT_test
NAME5 = {1: 'RUL', 2: 'RML', 3: 'RLL', 4: 'LUL', 5: 'LLL'}


def lobe_at(lobe, cy, cx, cz):
    """同 assign_lobe.py: nodule 中心 11x11x7 鄰域多數決; 落在肺葉外 (貼胸膜/裂隙) 擴大到 41x41x25"""
    cy, cx, cz = int(cy), int(cx), int(cz)
    for ry, rz in ((5, 3), (20, 12)):
        p = lobe[max(cy - ry, 0):cy + ry + 1, max(cx - ry, 0):cx + ry + 1, max(cz - rz, 0):cz + rz + 1]
        nz = p[p > 0]
        if nz.size: return NAME5.get(int(np.bincount(nz).argmax()), '')
    return ''


n_lobe_pts = 0; EXCLUDED = []
Nd = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodules.csv'))
_lc = os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_lobe.csv')
LOBE_CACHE = {(r.folder, r.nodule_id): dict(lobe=r.lobe, prone=bool(r.prone), flag=r.flag if isinstance(r.flag, str) else '')
              for r in pd.read_csv(_lc).itertuples()} if os.path.exists(_lc) else {}
xl = pd.read_excel(os.path.join(ROOT, 'LDCT_test1000_report.xlsx')); xl.columns = ['folder', 'report']
xl['report'] = xl.report.fillna('')
pts = []
for r0 in xl.itertuples():
    t = r0.report; nzh = len(re.findall(r'[一-鿿]', t))
    if re.search(r'Syngo\.Lung CAD', t) or (nzh > 20 and not re.search(r'lung apex|Low dose chest CT|Chest CT', t, re.I)):
        continue
    body = strip_boiler(t); fb, ib = split_sections(body); sents = []
    for sec, blk in (('findings', fb), ('impression', ib)):
        for s0 in to_sentences(blk):
            s = clean_sentence(s0)
            if s is None: continue
            sents.append(dict(section=sec, text=s, lobes=sentence_lobes(s),
                              is_lung_lesion=bool(re.search(LESION, s, re.I)) and not re.search(EXTRA_PULM, s, re.I)))
    g = Nd[Nd.folder == r0.folder]
    nods = []
    for r in g.itertuples():
        a = LOBE_CACHE.get((r0.folder, r.nodule_id), {})
        lobe = a.get('lobe', '') if isinstance(a.get('lobe', ''), str) else ''
        side = lobe[0] if lobe else ({'L': 'R', 'R': 'L'}[r.side] if a.get('prone') else r.side)   # 俯臥: x 方向相反
        nods.append(dict(nodule_id=r.nodule_id, lobe=lobe, side=side, zrel=r.zrel, dia=r.eq_diam_mm))
    if any(LOBE_CACHE.get((r0.folder, r.nodule_id), {}).get('flag') == 'incomplete_scan' for r in g.itertuples()):
        EXCLUDED.append(r0.folder); continue                       # 影像只掃到肺底, 不配對
    n_lobe_pts += any(n['lobe'] for n in nods)
    pts.append((r0.folder, nods, sents))
LD_R, LD_U = run('ldct', pts)
print('done', len(ME_R), len(LD_R), '| LDCT 有五肺葉的病患:', n_lobe_pts, '| 排除 (影像不完整):', EXCLUDED)
