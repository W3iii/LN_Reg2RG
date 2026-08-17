"""
Reg2RG 資料集建置 (Step 1-2, 4-5)

輸出:
  manifest.csv        英文報告病患清單
  nodules.csv         nodule instance region (ME mask ∩ 醫師 bbox)
  reports.jsonl       正規化報告 + 句子層級 lobe 標記
  region_report.jsonl 五分區 lobe matching report

座標: ME npy = (0.8, 0.8, 1.0) mm; 醫師 bbox 為原始 DICOM 座標, 需轉換
"""
import os, re, json, io
import numpy as np
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
ME = os.path.join(ROOT, 'ME_dataset')
DB = os.path.join(ROOT, 'ME_db1_20241210')
OUT = os.path.join(ROOT, 'reg2rg_dataset')
TGT_XY, TGT_Z = 0.8, 1.0
os.makedirs(OUT, exist_ok=True)


def fid(n):
    m = re.match(r'^(CHESTCT|CHEST)(\d+)$', n)
    return int(m.group(2)) if m.group(1) == 'CHESTCT' else int(m.group(2)) - 1000


def id2f(p):
    return f'CHEST{p+1000}' if p <= 908 else f'CHESTCT{p:04d}'


# ============================================================ 載入
me = {}
for d in sorted(os.listdir(ME)):
    j = json.load(open(os.path.join(ME, d, 'mask', f'{d}_nodule_count.json')))
    lines = [l.strip() for l in open(os.path.join(ME, d, 'npy', 'series_metadata.txt')).read().strip().split('\n') if l.strip()]
    lob = [l.strip() for l in open(os.path.join(ME, d, 'npy', 'lobe_info.txt')).read().strip().split('\n') if l.strip()]
    me[fid(d)] = dict(folder=d, bboxes=j['bboxes'], sizes=j['nodule_size'],
                      orig_spacing=[float(v) for v in lines[-2].split(',')],
                      orig_shape=[int(v) for v in lines[-3].split(',')],
                      lung_z=[int(v) for v in lob[-2].split(',')],
                      lung_bb=[int(v) for v in lob[-1].split(',')])

bb = {int(f.split('.')[0]): json.load(open(os.path.join(DB, 'bbox_ME', f)))
      for f in sorted(os.listdir(os.path.join(DB, 'bbox_ME')))}

cls = pd.read_csv(os.path.join(DB, 'check_nodule_cls_20241207_v1b4.csv'))
cls['pid'] = cls['patient_id'].astype(int)
# bbox_index -> (nodule_index, category)
bidx2nod = {}
for p, g in cls.groupby('pid'):
    bidx2nod[p] = {int(r.bbox_index): (int(r.nodule_index), int(r.category)) for r in g.itertuples()}

xl = pd.read_excel(os.path.join(ROOT, 'lung_M_class_0001-1800_report.xlsx'), sheet_name=0)
xl.columns = ['pid', 'nodule', 'img', 'nlabel', 'rads', 'report']
rep_raw = xl.dropna(subset=['report']).drop_duplicates('pid').set_index('pid')['report'].to_dict()
rads_map = {(int(r.pid), int(r.nodule)): int(r.rads) for r in xl.itertuples()}

pass_ids = set()
sect = ''
for line in open(os.path.join(DB, 'patient_ids_pass.txt'), encoding='utf-8').read().replace('\r', '').split('\n'):
    line = line.strip()
    if not line:
        continue
    if re.match(r'^(CHESTCT|CHEST)\d+$', line):
        pass_ids.add(fid(line))
    elif re.match(r'^\d+$', line):
        pass_ids.add(int(line))

BAD_REPORT = {113, 622, 662}          # 報告錯置: 骨密度 / CXR / 左膝X光

# ============================================================ 報告正規化
BOILER = [
    r'Appendix\s*:\s*Low dose chest CT is only limited.*',          # LDCT 免責樣板
    r'此報告藉由自動化電腦輔助偵測軟體.*',                            # Syngo CAD 樣板
    r'開單目的\s*:[^\n]*\n?',
    r'低劑量電腦斷層肺癌篩檢\s*\n?',
    r'Low dose chest CT \(120KVP[^\n]*\n?',                          # 掃描參數
    r'Chest CT (?:with(?:out)?|without) IV contrast[^\n]*\n?',
]
# 整句刪除: 純病史 / 純前次比較
DROP_SENT = [
    r'^\s*Clinical (?:Information|history)\s*:',
    r'compared\s+(?:with|to)\s+(?:the\s+)?(?:previous|prior|earlier|\d{4})',
    r'^\s*Th(?:is|e)\s+(?:CT|study|examination)\s+is compared',
    r'^\s*\[S\]|^\s*\[A\]',
    r'status post|\bs/p\b',
    r'health examination showed',
    r'^\s*Findings\s*:\s*$',
    r'^\s*IMP(?:RESSION)?\s*:\s*$',
    r'^\s*no interval change\b',                  # 純比較句, 無 finding
    r'previous[^.;]{0,45}(?:available|comparison)',   # "No previous CT for comparison"
    r'^\s*(?:low dose\s+)?(?:no\s+)?(?:available\s+)?previous\b',
    r'^\s*(?:Without\s+)?(?:As\s+)?[Cc]ompar(?:ing|ed) to previous',
    r'^\s*(?:stationary|persistent|unchanged|no change)\s*[.;]?\s*$',
]
# 行內時序修飾 (名詞前的形容詞 / 附加子句)
TEMPORAL_INLINE = [
    (r'^\s*(?:interval\s+)?(?:regression|progression|resolution|increase|decrease)\s+of\s+(?:the\s+)?', ''),
    (r',?\s*some (?:are\s+)?(?:sa?tationary|stationary)[^,.;]*', ''),
    (r',?\s*(?:and\s+)?as compared\s+(?:with|to)[^.;]*', ''),
    (r'^\s*The previously noted\s+', ''),
    (r',?\s*(?:mild(?:ly)?|slight(?:ly)?)?\s*interval\s+(?:enlargement|change|increase|decrease)\b', ''),
    (r'\s*with\s+(?:stationary|unchanged|increased|decreased|no)\s+(?:size|change)\b', ''),
    (r'\b(?:stationary|persistent|persisted|unchanged|newly developed)\s+(?=[a-z])', ''),
    (r'\s+(?:stationary|persistent|unchanged)\b(?=\s*[,.;]|\s*$)', ''),
]
# 行內刪除: 時序形容詞 (保留 finding 本體)
TEMPORAL_LEAD = (r'\b(?:stationary|persistent|persisted|unchanged|newly developed|new|again seen|'
                 r'previously noted|residual|slowly enlarged|slightly enlarged|enlarged|'
                 r'progressive|regressed|improved|increased|decreased)\b')
# 逗號夾住的時序子句 (句中或句尾皆可)
TEMPORAL_CLAUSE = (r',\s*(?:stationary(?: after prior regression| in size)?|persistent(?: in size)?|'
                   r'unchanged|no interval change|without interval change|not changed|'
                   r'slowly enlarged|slightly enlarged|newly developed|progressive|regressed|'
                   r'improved|as before|as previous)\s*(?=[,.;]|$)')
# 影像索引 -> 對不到 npy slice, 移除避免模型亂編號碼
# 形式: (Srs/Img:4/20,602/27) / (Srs/Img:) / (S/I: 6/74) / 6/74
SLICE_REF = [
    r'\s*\(\s*S(?:rs?)?\s*/\s*I(?:mg)?\s*:[^)]*\)',       # (Srs/Img:4/20) 整個括號
    r'[,;]?\s*S(?:rs?)?\s*/\s*I(?:mg)?\s*:\s*[\d/,\s]*',  # (0.5cm, Srs/Img:6/74) 只拔 ref 留大小
    r'\s*\(\s*Srs?\s*:[^)]*\)',                           # (Srs:3,arrows)
    r'\s*\(?\s*\d{1,4}\s*/\s*\d{1,4}\s*\)?(?=[\s.,;]|$)',  # 裸的 6/74
    r'\(\s*[,;\s]*\)',                                    # 清掉拔完剩下的空括號
    r'[,;]\s*\)',                                         # (0.5cm, ) -> (0.5cm)
]

LOBE_RULES = [
    ('RUL', r'\bRUL\b|right upper lobe|Rt\.?\s*upper lobe'),
    ('RML', r'\bRML\b|right middle lobe|Rt\.?\s*middle lobe'),
    ('RLL', r'\bRLL\b|right lower lobe|Rt\.?\s*lower lobe'),
    ('LUL', r'\bLUL\b|left upper lobe|Lt\.?\s*upper lobe|lingul'),   # lingula ⊂ LUL
    ('LLL', r'\bLLL\b|left lower lobe|Lt\.?\s*lower lobe'),
]
WHOLE_SIDE = [
    (('RUL', 'RML', 'RLL'), r'right lung|Rt\.?\s*lung'),
    (('LUL', 'LLL'), r'left lung|Lt\.?\s*lung'),
    (('RUL', 'RML', 'RLL', 'LUL', 'LLL'), r'bilateral lungs|both lungs|bilateral lung fields'),
    (('RLL', 'LLL'), r'bilateral lower lung|both lower lung'),
    (('RUL', 'LUL'), r'bilateral upper lung|both upper lung'),
]
LOBES5 = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
# 沒有敘述的 lobe 要給明確文字, 不能留空白, 否則模型會學成「看到 region 就一定要生一句」
NO_FINDING = 'No significant finding.'
LESION = (r'nodul|GGN|GGO|ground[\s-]?glass|mass\b|tumou?r|densit|opacit|lesion|'
          r'infiltrat|calcif|bronchiect|consolidat|atelectas|fibro|scar|cyst|emphysem')
# 肺外器官 -> 不算肺部病灶句 (No adrenal lesion / No hepatic tumor 這類樣板負句)
EXTRA_PULM = (r'adrenal|hepatic|liver|gallbladder|\bGB\b|gallstone|renal|kidney|spleen|'
              r'pancrea|breast|thyroid|spine|spondylo|aorta|coronary|myocard|lymph node|'
              r'lymphadenopathy|mediastin|pleural effusion|bone|rib\b|vertebra')


def strip_boiler(t):
    for p in BOILER:
        t = re.sub(p, '', t, flags=re.S | re.I)
    return t


def split_sections(t):
    m = re.search(r'\n\s*(?:IMPRESSION|IMP)\b\s*:?', t, re.I)   # IMPRESSION 要排前面
    if m:
        return t[:m.start()].strip(), t[m.end():].strip()
    return t.strip(), ''


def to_sentences(block):
    out = []
    for line in block.split('\n'):
        line = line.strip()
        line = re.sub(r'^[\-\>\*\u2022]+\s*', '', line)         # bullet
        line = re.sub(r'^\d+[\.\)]\s*', '', line)               # 1. 2)
        if not line:
            continue
        for s in re.split(r'(?<=[.;])\s*(?=[A-Z\u4e00-\u9fff])', line):   # \u5141\u8a31 "...region.Stationary."
            s = s.strip()
            if len(s) >= 3:
                out.append(s)
    return out


def clean_sentence(s):
    """刪時序詞、保留 finding。回傳 None 代表整句應刪除。"""
    for p in DROP_SENT:
        if re.search(p, s, re.I):
            return None
    for p in SLICE_REF:
        s = re.sub(p, '', s, flags=re.I)
    for _ in range(3):                                   # 一句可能有多個時序子句
        s2 = re.sub(TEMPORAL_CLAUSE, '', s, flags=re.I)
        if s2 == s:
            break
        s = s2
    s = re.sub(r'^\s*' + TEMPORAL_LEAD + r'\s+', '', s, flags=re.I)
    s = re.sub(r'\b(?:is|are|was|were)\s+' + TEMPORAL_LEAD + r'\b', '', s, flags=re.I)
    for pat, rep in TEMPORAL_INLINE:
        s = re.sub(pat, rep, s, flags=re.I)
    s = re.sub(r'\s{2,}', ' ', s).strip(' ,;')
    if re.fullmatch(r'(?:noted|seen|found|is noted|are noted|also noted)\s*[.;]?', s, re.I):
        return None                                   # 剝完只剩動詞
    if not s or len(s) < 3:
        return None
    if not re.search(r'[.;]$', s):
        s += '.'
    return s[0].upper() + s[1:]


def sentence_lobes(s):
    hit = set()
    for name, pat in LOBE_RULES:
        if re.search(pat, s, re.I):
            hit.add(name)
    if not hit:
        for names, pat in WHOLE_SIDE:
            if re.search(pat, s, re.I):
                hit.update(names)
    return sorted(hit)


# ============================================================ 主流程
manifest, nod_rows, rep_rows, rr_rows = [], [], [], []
skipped = []

for pid in sorted(me):
    folder = me[pid]['folder']
    t = rep_raw.get(pid)
    # ---- Step 1: 只留英文報告病患 ----
    if pid in pass_ids:
        skipped.append((pid, folder, 'pass_list')); continue
    if pid in BAD_REPORT:
        skipped.append((pid, folder, 'report_mismatched_study')); continue
    if not isinstance(t, str) or not t.strip():
        skipped.append((pid, folder, 'no_report')); continue
    nz = len(re.findall(r'[\u4e00-\u9fff]', t))
    is_zh_cad = bool(re.search(r'Syngo\.Lung CAD', t)) or (nz > 20 and not re.search(r'lung apex|Low dose chest CT|Chest CT', t, re.I))
    if is_zh_cad:
        skipped.append((pid, folder, 'zh_cad_template')); continue

    # ---- Step 5: 報告正規化 ----
    body = strip_boiler(t)
    find_b, imp_b = split_sections(body)
    sents, dropped = [], 0
    for src, blk in (('findings', find_b), ('impression', imp_b)):
        for s0 in to_sentences(blk):
            s = clean_sentence(s0)
            if s is None:
                dropped += 1; continue
            sents.append(dict(section=src, text=s, raw=s0,
                              lobes=sentence_lobes(s),
                              is_lesion=bool(re.search(LESION, s, re.I)),
                              is_lung_lesion=bool(re.search(LESION, s, re.I)) and
                                             not re.search(EXTRA_PULM, s, re.I)))
    clean_txt = ' '.join(x['text'] for x in sents)
    if len(clean_txt) < 60:
        skipped.append((pid, folder, 'low_info_after_clean')); continue

    # ---- Step 2: nodule instance region (ME mask ∩ 醫師 bbox) ----
    sy, sx, sz = me[pid]['orig_spacing']

    def tf(pt):     # 原始 DICOM 座標 -> ME npy 座標
        return (256 + (pt[0] - 256) * sy / TGT_XY,
                256 + (pt[1] - 256) * sx / TGT_XY,
                pt[2] * sz / TGT_Z)

    # box_ids 才是真正的 bbox_index (會跳號, 例如 pid232 是 [0,2,3,4]);
    # 用 enumerate 的位置去查 check_nodule_cls 會撞號, 造成兩顆 nodule 共用同一個 nodule_id
    box_ids = bb[pid].get('box_ids') or list(range(len(bb[pid]['bboxes'])))
    doc = []
    for k, b in enumerate(bb[pid]['bboxes']):
        c0, c1 = tf(b[0]), tf(b[1])
        bid = box_ids[k] if k < len(box_ids) else k
        nod_i, cat = bidx2nod.get(pid, {}).get(bid, (None, -1))
        doc.append(dict(k=k, bbox_index=bid, nodule_id=nod_i, cat=cat,
                        lo=c0, hi=c1, cen=tuple((a + c) / 2 for a, c in zip(c0, c1))))
    # 查不到對照的補流水號, 避開已用掉的 nodule_id
    taken = {d['nodule_id'] for d in doc if d['nodule_id'] is not None}
    nxt = 1
    for d in doc:
        if d['nodule_id'] is None:
            while nxt in taken:
                nxt += 1
            d['nodule_id'] = nxt; taken.add(nxt)
            if d['cat'] == -1:
                d['cat'] = rads_map.get((pid, nxt), -1)
    mes = [dict(i=i, lo=b[0], hi=b[1], cen=tuple((b[0][j] + b[1][j]) / 2 for j in range(3)),
                vox=me[pid]['sizes'][i] if i < len(me[pid]['sizes']) else 0)
           for i, b in enumerate(me[pid]['bboxes'])]

    used, pairs = set(), []
    for m_ in mes:
        best, bj = 1e9, -1
        for d_ in doc:
            if d_['k'] in used:
                continue
            dd = np.sqrt(((m_['cen'][0] - d_['cen'][0]) * TGT_XY) ** 2 +
                         ((m_['cen'][1] - d_['cen'][1]) * TGT_XY) ** 2 +
                         (m_['cen'][2] - d_['cen'][2]) ** 2)
            if dd < best:
                best, bj = dd, d_['k']
        if bj >= 0 and best <= 10.0:            # 10mm 內才算配對成功
            used.add(bj); pairs.append((m_, doc[bj], best))

    paired_doc = {d['k'] for _, d, _ in pairs}
    n_me_extra = len(mes) - len(pairs)
    for m_, d_, dist in pairs:                  # voxel-level mask region
        vol = m_['vox'] * TGT_XY * TGT_XY * TGT_Z
        nod_rows.append(dict(
            pid=pid, folder=folder, nodule_id=d_['nodule_id'], region_source='me_mask',
            y0=m_['lo'][0], x0=m_['lo'][1], z0=m_['lo'][2],
            y1=m_['hi'][0], x1=m_['hi'][1], z1=m_['hi'][2],
            cy=round(m_['cen'][0], 1), cx=round(m_['cen'][1], 1), cz=round(m_['cen'][2], 1),
            size_vox=m_['vox'], eq_diam_mm=round(2 * (3 * vol / (4 * np.pi)) ** (1 / 3), 2),
            bbox_long_mm=round(max(m_['hi'][0] - m_['lo'][0], m_['hi'][1] - m_['lo'][1]) * TGT_XY, 2),
            tw_lung_rads=d_['cat'], match_dist_mm=round(dist, 2), lobe=''))
    for d_ in doc:                              # 醫師有標但 ME 缺 mask -> 用 bbox
        if d_['k'] in paired_doc:
            continue
        dy = d_['hi'][0] - d_['lo'][0]; dx = d_['hi'][1] - d_['lo'][1]; dz = d_['hi'][2] - d_['lo'][2]
        vol = dy * dx * dz * TGT_XY * TGT_XY * TGT_Z * 0.52     # 橢球體積近似
        nod_rows.append(dict(
            pid=pid, folder=folder, nodule_id=d_['nodule_id'], region_source='doc_bbox',
            y0=round(d_['lo'][0], 1), x0=round(d_['lo'][1], 1), z0=round(d_['lo'][2], 1),
            y1=round(d_['hi'][0], 1), x1=round(d_['hi'][1], 1), z1=round(d_['hi'][2], 1),
            cy=round(d_['cen'][0], 1), cx=round(d_['cen'][1], 1), cz=round(d_['cen'][2], 1),
            size_vox=None, eq_diam_mm=round(2 * (3 * vol / (4 * np.pi)) ** (1 / 3), 2),
            bbox_long_mm=round(max(dy, dx) * TGT_XY, 2),
            tw_lung_rads=d_['cat'], match_dist_mm=None, lobe=''))

    # ---- Step 4: 五分區 lobe matching report ----
    per_lobe = {L: [] for L in LOBES5}
    globals_ = []

    def push(lst, txt):                     # findings/impression 完全重複的句子只留一次
        if txt.lower() not in {x.lower() for x in lst}:
            lst.append(txt)

    for s in sents:
        if s['lobes']:
            for L in s['lobes']:
                push(per_lobe[L], s['text'])
        else:
            push(globals_, s['text'])
    rr_rows.append(dict(pid=pid, folder=folder,
                        regions={L: (' '.join(v) if v else NO_FINDING) for L, v in per_lobe.items()},
                        regions_empty=[L for L, v in per_lobe.items() if not v],
                        global_text=' '.join(globals_),
                        n_lobe_sent=sum(len(v) for v in per_lobe.values()),
                        n_global_sent=len(globals_)))

    rep_rows.append(dict(pid=pid, folder=folder, clean_report=clean_txt,
                         n_sent=len(sents), n_dropped_sent=dropped,
                         sentences=sents))
    manifest.append(dict(pid=pid, folder=folder, n_nodule_excel=len(doc),
                         n_region=len(pairs) + (len(doc) - len(paired_doc)),
                         n_me_extra_dropped=n_me_extra,
                         n_sent=len(sents), report_chars=len(clean_txt)))

# ============================================================ 輸出
M = pd.DataFrame(manifest)
M.to_csv(os.path.join(OUT, 'manifest.csv'), index=False, encoding='utf-8-sig')
N = pd.DataFrame(nod_rows)
N.to_csv(os.path.join(OUT, 'nodules.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(skipped, columns=['pid', 'folder', 'reason']).to_csv(
    os.path.join(OUT, 'skipped.csv'), index=False, encoding='utf-8-sig')
with io.open(os.path.join(OUT, 'reports.jsonl'), 'w', encoding='utf-8') as f:
    for r in rep_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
with io.open(os.path.join(OUT, 'region_report.jsonl'), 'w', encoding='utf-8') as f:
    for r in rr_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ============================================================ 統計
S = pd.DataFrame(skipped, columns=['pid', 'folder', 'reason'])
log = io.open(os.path.join(OUT, 'build_log.txt'), 'w', encoding='utf-8')
log.write('病患: 收 %d / 排除 %d\n' % (len(M), len(S)))
log.write(S.reason.value_counts().to_string() + '\n\n')
log.write('nodule region: %d 顆 (me_mask %d, doc_bbox %d)\n'
          % (len(N), (N.region_source == 'me_mask').sum(), (N.region_source == 'doc_bbox').sum()))
log.write('  丟棄的 ME 多標 mask: %d 顆\n' % M.n_me_extra_dropped.sum())
log.write('  配對距離 median %.2f mm / p99 %.2f mm\n'
          % (N.match_dist_mm.median(), N.match_dist_mm.quantile(.99)))
log.write('  等效球徑 median %.1f mm (p10 %.1f, p90 %.1f)\n'
          % (N.eq_diam_mm.median(), N.eq_diam_mm.quantile(.1), N.eq_diam_mm.quantile(.9)))
log.write('  TW_Lung_RADS: %s\n\n' % N.tw_lung_rads.value_counts().sort_index().to_dict())
log.write('報告: 句子 median %d / 病患, 刪除句 median %d\n'
          % (M.n_sent.median(), pd.Series([r['n_dropped_sent'] for r in rep_rows]).median()))
log.write('  正規化後字元 median %d (p10 %d, p90 %d)\n'
          % (M.report_chars.median(), M.report_chars.quantile(.1), M.report_chars.quantile(.9)))
RR = pd.DataFrame([{'pid': r['pid'], **{L: (L not in r['regions_empty']) for L in LOBES5},
                    'glob': r['n_global_sent'], 'lobe_sent': r['n_lobe_sent']} for r in rr_rows])
log.write('\n五分區覆蓋 (有實際敘述的病患數 / %d):\n' % len(RR))
for L in LOBES5:
    log.write('  %-4s %5d (%.1f%%)  其餘標為 "%s"\n'
              % (L, RR[L].sum(), 100 * RR[L].mean(), NO_FINDING))
log.write('  至少一個 lobe 有敘述 : %d (%.1f%%)\n'
          % ((RR.lobe_sent > 0).sum(), 100 * (RR.lobe_sent > 0).mean()))
log.write('  全部落在 global(無 lobe): %d\n' % (RR.lobe_sent == 0).sum())
log.write('  global 句數 median %d\n' % RR.glob.median())
ll = pd.Series([s['is_lung_lesion'] for r in rep_rows for s in r['sentences']])
lo = pd.Series([bool(s['lobes']) for r in rep_rows for s in r['sentences'] if s['is_lung_lesion']])
log.write('\n句子: 總 %d, 肺部病灶句 %d (%.1f%%), 其中有 lobe 標記 %d (%.1f%%)\n'
          % (len(ll), ll.sum(), 100 * ll.mean(), lo.sum(), 100 * lo.mean()))
log.close()
print(open(os.path.join(OUT, 'build_log.txt'), encoding='utf-8').read())
