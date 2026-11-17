"""
Region-report GT v2 for Reg2RG (ME + LDCT, split C)

Output (reg2rg_nifti/):
  region_report_v2_{train,val,test_me,test_ldct}.csv
      Volumename, Anatomy, Sentence, nodule_ids, loss_mask, source, split
      - 5 lobe rows per patient + 1 global row (Anatomy blank, ignored by Reg2RG as before)
      - Sentence: v2 split pieces of that lobe (Findings + Impression merged, as in the old GT)
          size of a piece given to exactly one nodule -> [SIZE#<nodule_id>]
          any other size (group description, unmatched)  -> [SIZE]   (existing fill_sizes)
      - nodule_ids: nodules in that lobe (TotalSegmentator lobe), mask size descending, ';'-joined
      - loss_mask: 0 if the report describes a lesion in this lobe that has no mask
  fill_table_v2.csv
      Volumename, nodule_id, lobe, eq_diam_mm, bbox_long_mm, size_cm_eq, size_cm_cal
"""
import os, re, io, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, 'reg2rg_nifti')
src = io.open(os.path.join(ROOT, 'reg2rg_dataset', 'build_dataset.py'), encoding='utf-8').read()
exec(src[src.index('# ============================================================ 報告正規化'):
         src.index('# ============================================================ 主流程')])

# ---- copied from Reg2RG/src/size_placeholder.py (kept identical so GT and model code agree) ----
PLACEHOLDER = '[SIZE]'
_NUM = r'\d+(?:\.\d+)?'; _UNIT = r'(?:cm|mm)\b'; _X = r'\s*(?:x|\*|×)\s*'
_NOT_DISTANCE = r'(?!\s*(?:from|away|beneath|below|above|distal|proximal)\b)'
_DIMS = re.compile(rf'{_NUM}{_X}{_NUM}(?:{_X}{_NUM})?\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_PAIR = re.compile(rf'{_NUM}\s*(?:{_UNIT}\s*)?(and|,)\s*{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_RANGE = re.compile(rf'{_NUM}\s*(?:{_UNIT}\s*)?(?:-|~|to)\s*{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_SINGLE = re.compile(rf'{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)
def mask_sizes(text):
    t = _DIMS.sub(PLACEHOLDER, text)
    t = _PAIR.sub(lambda m: f'{PLACEHOLDER} {m.group(1)} {PLACEHOLDER}'.replace(' ,', ','), t)
    t = _RANGE.sub(PLACEHOLDER, t)
    return _SINGLE.sub(PLACEHOLDER, t)
def format_size(mm):
    return f'{max(0.1, round(mm / 10.0, 1)):.1f}cm'

# ---- residual text artifacts (Reg2RG HANDOFF_TO_LOCAL.md §1-4) ----
ARTIFACTS = [
    (r'\s*(?:,\s*)?(?:without|with)?\s*as\s+compar(?:ed|ing)\b[^.;]*', ''),     # "... without as comparing (to)."
    (r'\s+(?:without|with)\s+interval\b(?=\s*[.;,]|$)', ''),                    # "... without interval."
    (r'\.?,\s*\d{1,4}\s*,\s*\d{1,4}\s*\)', ''),                                 # "likely benign.,104,123)."
    (r'\s+(?:without|with)\s*(?=[.;]|$)', ''),                                  # dangling "without."
    (r'[一-鿿]+', ''),                                                  # stray CJK characters
]
def tidy(t):
    for p, r in ARTIFACTS: t = re.sub(p, r, t, flags=re.I)
    t = re.sub(r'\s+([.,;])', r'\1', t); t = re.sub(r'([.;])\s*\.', r'\1', t)
    t = re.sub(r'\(\s*\)', '', t); t = re.sub(r'\s{2,}', ' ', t).strip()
    return t

def dedupe(text):
    """drop a sentence that already appeared in the same lobe text (case-insensitive)"""
    out, seen = [], set()
    for sent in re.split(r'(?<=[.;])\s+', text.strip()):
        k = sent.strip().lower()
        if k and k not in seen: seen.add(k); out.append(sent.strip())
    return ' '.join(out)
HEALTH = re.compile(r'(?:^|(?<=[.;]\s))(?:Clinical\s*(?:Information|history)\s*:?\s*)?Health examination\.?\s*', re.I)

L5 = ['RUL', 'RML', 'RLL', 'LUL', 'LLL']
NAME = {'RUL': 'right upper lobe', 'RML': 'right middle lobe', 'RLL': 'right lower lobe',
        'LUL': 'left upper lobe', 'LLL': 'left lower lobe'}
NO_FINDING = 'No significant finding.'

S = pd.read_csv(os.path.join(OUT, 'splits_merged.csv'))
# nodules: lobe + size
me = pd.read_csv(os.path.join(ROOT, 'reg2rg_dataset', 'nodules.csv'))[['folder', 'nodule_id', 'lobe', 'eq_diam_mm', 'bbox_long_mm']]
ld = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodules.csv'))[['folder', 'nodule_id', 'eq_diam_mm', 'bbox_long_mm']]
ld = ld.merge(pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_lobe.csv'))[['folder', 'nodule_id', 'lobe']], on=['folder', 'nodule_id'])
NOD = pd.concat([me, ld], ignore_index=True)
NOD = NOD[NOD.folder.isin(S.folder)]
NOD['lobe'] = NOD.lobe.fillna('')

# tiers: same rules as export_excel_v2.py, computed from the matching output (no dependency on the Excel files)
NODW = r'nodul|GGN|GGO|ground[\s-]?glass|opacit|mass\b|lesion|densit|calcif|tumou?r|granuloma|LN\b|lymph'
def _tier(r):
    t = r.per_nodule_text if isinstance(r.per_nodule_text, str) else ''
    if not t: return 'none'
    if not re.search(NODW, t, re.I): return 'X'
    if r.kind != 'shared': return 'A'
    return 'B' if re.fullmatch(r'RUL|RML|RLL|LUL|LLL', str(r.piece_lobe) if isinstance(r.piece_lobe, str) else '') else 'C'
_pn = pd.concat([pd.read_csv(os.path.join(HERE, 'me_nodule_sentence_v2.csv')), pd.read_csv(os.path.join(HERE, 'ldct_nodule_sentence_v2.csv'))])
TIER = {(r.key, int(r.nodule_id)): _tier(r) for r in _pn.itertuples()}

P = pd.concat([pd.read_csv(os.path.join(HERE, 'me_pieces_v2.csv')), pd.read_csv(os.path.join(HERE, 'ldct_pieces_v2.csv'))], ignore_index=True)
P = P.rename(columns={'key': 'folder'}).fillna({'lobes': '', 'assigned': ''})
P['assigned'] = P.assigned.astype(str).str.replace(r'\.0\b', '', regex=True).replace('nan', '')

# all cleaned sentences per patient (for the global row)
def ldct_sents():
    x = pd.read_excel(os.path.join(ROOT, 'LDCT_test1000_report.xlsx')); x.columns = ['folder', 'report']
    out = {}
    for r in x.itertuples():
        if r.folder not in set(S.folder): continue
        b = strip_boiler(str(r.report)); f, i = split_sections(b); ss = []
        for blk in (f, i):
            for s0 in to_sentences(blk):
                s = clean_sentence(s0)
                if s: ss.append(s)
        out[r.folder] = ss
    return out
SENT = {json.loads(l)['folder']: [s['text'] for s in json.loads(l)['sentences']]
        for l in io.open(os.path.join(ROOT, 'reg2rg_dataset', 'reports.jsonl'), encoding='utf-8')}
SENT.update(ldct_sents())

rows, n_idx, n_gen = [], 0, 0
for r in S.itertuples():
    f = r.folder
    nods = NOD[NOD.folder == f]
    lobe_of = dict(zip(nods.nodule_id.astype(int), nods.lobe))
    per_lobe = {L: [] for L in L5}; masked = set(); used_orig = set(); glob = []
    g = P[P.folder == f].sort_values(['order', 'piece'])
    shared_hit = g[g.kind == 'shared'].groupby('orig').assigned.apply(lambda s: (s != '').any()).to_dict()
    seen = set()
    for p in g.itertuples():
        used_orig.add(p.orig)
        ids = [int(x) for x in p.assigned.split(',') if x]
        lobes = [x for x in p.lobes.split(',') if x in L5]
        if p.kind == 'shared' and not ids:
            if shared_hit.get(p.orig): continue                     # an unused copy of a group sentence
        targets = sorted({lobe_of[i] for i in ids if lobe_of.get(i)} ) if ids else []
        if not targets: targets = [lobes[0]] if len(lobes) == 1 else lobes
        text = mask_sizes(tidy(p.text))
        if len(ids) == 1 and p.kind != 'shared':
            text = text.replace(PLACEHOLDER, f'[SIZE#{ids[0]}]'); n_idx += text.count('[SIZE#')
        n_gen += text.count(PLACEHOLDER)
        if not ids and p.nodule_like: masked.update(targets)      # report lesion without a mask
        if not targets: glob.append(text); continue
        for L in targets:
            if (L, text) in seen: continue
            seen.add((L, text)); per_lobe[L].append(text)
    # global row: every remaining sentence
    for s in SENT.get(f, []):
        if re.fullmatch(r'\s*(?:Clinical\s*(?:Information|history)\s*:?\s*)?Health examination\.?\s*', s, re.I): continue   # 病史
        if not any(s in o for o in used_orig): glob.append(tidy(s))
    vol = f + '.nii.gz'
    for L in L5:
        ids = nods[nods.lobe == L].sort_values('eq_diam_mm', ascending=False).nodule_id.astype(int).tolist()
        rows.append(dict(Volumename=vol, Anatomy=NAME[L], Sentence=dedupe(' '.join(per_lobe[L])) or NO_FINDING,
                         nodule_ids=';'.join(map(str, ids)), loss_mask=0 if L in masked else 1, source=r.source, split=r.split))
    rows.append(dict(Volumename=vol, Anatomy=np.nan, Sentence=dedupe(HEALTH.sub('', ' '.join(x for x in glob if x))), nodule_ids='', loss_mask=1,
                     source=r.source, split=r.split))
G = pd.DataFrame(rows)
for sp, d in G.groupby('split'):
    d.to_csv(os.path.join(OUT, f'region_report_v2_{sp}.csv'), index=False, encoding='utf-8')

# ---- fill table: mask size -> report-style cm; linear calibration fitted on train A-tier pairs ----
pn = pd.concat([pd.read_csv(os.path.join(HERE, 'me_nodule_sentence_v2.csv')), pd.read_csv(os.path.join(HERE, 'ldct_nodule_sentence_v2.csv'))])
pn = pn.rename(columns={'key': 'folder'}).merge(NOD[['folder', 'nodule_id', 'bbox_long_mm']], on=['folder', 'nodule_id']).merge(S[['folder', 'split']], on='folder')
pn['tier'] = [TIER.get((a, int(b))) for a, b in zip(pn.folder, pn.nodule_id)]
cal = pn[(pn.tier == 'A') & pn.piece_size_mm.notna() & ~pn.per_nodule_text.fillna('').str.contains('up to|largest', case=False)]
tr = cal[cal.split == 'train']
w = np.linalg.lstsq(np.c_[tr.eq_diam_mm, tr.bbox_long_mm, np.ones(len(tr))], tr.piece_size_mm, rcond=None)[0]
F = NOD.copy(); F['Volumename'] = F.folder + '.nii.gz'
F['size_cm_eq'] = F.eq_diam_mm.apply(format_size)
F['cal_mm'] = np.c_[F.eq_diam_mm, F.bbox_long_mm, np.ones(len(F))] @ w
F['size_cm_cal'] = F.cal_mm.apply(format_size)
F[['Volumename', 'nodule_id', 'lobe', 'eq_diam_mm', 'bbox_long_mm', 'size_cm_eq', 'size_cm_cal']].to_csv(
    os.path.join(OUT, 'fill_table_v2.csv'), index=False, encoding='utf-8')
va = cal[cal.split == 'val'].merge(F[['folder', 'nodule_id', 'cal_mm']], on=['folder', 'nodule_id'])
print('calibration w =', np.round(w, 3), '| val MAE eq %.2f mm, cal %.2f mm (n=%d)' % (
    (va.eq_diam_mm - va.piece_size_mm).abs().mean(), (va.cal_mm - va.piece_size_mm).abs().mean(), len(va)))
print('indexed [SIZE#k]:', n_idx, '| generic [SIZE]:', n_gen)
lobe = G[G.Anatomy.notna()]
print(lobe.groupby('split').agg(patients=('Volumename', 'nunique'), lobes=('Anatomy', 'size'),
      loss_masked=('loss_mask', lambda s: int((s == 0).sum())),
      no_finding=('Sentence', lambda s: int((s == NO_FINDING).sum()))).reindex(['train', 'val', 'test_me', 'test_ldct']).to_string())
