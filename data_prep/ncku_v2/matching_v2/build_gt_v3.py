"""
Region-report GT v3 for Reg2RG (ME + LDCT, split C): Findings per lobe + one Impression,
size AND type placeholders.

Same inputs as build_gt_v2.py (its definitions are reused by exec), plus the v3 columns of the
matching output: <ds>_pieces_v2.csv text_f / text_i (Findings and Impression wording of a merged
piece, kept apart) and <ds>_sentences_v2.csv (every cleaned sentence with its section).

Output (reg2rg_nifti_ME_LDCT/):
  region_report_v3_{train,val,test_me,test_ldct}.csv
      Volumename, Anatomy, Sentence, nodule_ids, loss_mask, source, split
      - 5 lobe rows: FINDINGS wording only of the lobe's lesion pieces (v2 used Findings +
        Impression, which repeated each nodule twice)
      - 1 row Anatomy = 'impression': the report's Impression, in order -- lung-lesion pieces in
        their Impression wording, plus recommendation / judgement sentences ("Suggest follow up.",
        "Probably benign."); extrapulmonary sentences (fatty liver, renal stone, ...) and
        comparison-only sentences are left out. Empty -> 'No significant finding.'
        loss_mask 0 when it describes a lung lesion that has no mask.
      - 1 global row (Anatomy blank, ignored by Reg2RG): everything else
      - placeholders, only in pieces given to exactly one (non-shared) nodule:
          [SIZE#<nodule_id>]  as in v2
          [TYPE#<nodule_id>=<word>]  the piece's density wording (GGN / GGO / ground-glass / part-solid /
                              subsolid / non-solid / solid / calcified); filled at inference from
                              the lesion encoder's type head, or removed when it is not confident
        other sizes stay [SIZE]; other type words stay as written
"""
import os, re, io, json
src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'build_gt_v2.py'), encoding='utf-8').read()
# the data folder was renamed reg2rg_nifti -> reg2rg_nifti_ME_LDCT (2026-11-17)
src = src.replace("OUT = os.path.join(ROOT, 'reg2rg_nifti')", "OUT = os.path.join(ROOT, 'reg2rg_nifti_ME_LDCT')")
exec(src[:src.index('rows, n_idx, n_gen = [], 0, 0')])          # helpers, NOD, P (pieces), SENT, S, TIER ...

SECT = pd.concat([pd.read_csv(os.path.join(HERE, f'{d}_sentences_v2.csv')) for d in ('me', 'ldct')], ignore_index=True)
SECT = SECT.rename(columns={'key': 'folder'})
P = P.fillna({'text_f': '', 'text_i': ''})

# ---- type placeholders -------------------------------------------------------------------
_TYPE_RULES = [            # (pattern, what follows the tag, the GT word kept in the tag)
    (r'\bGGNs?\b', ' nodule', 'ground-glass'),
    (r'\bGGOs?\b', ' opacity', 'ground-glass'),
    (r'\bnon[\s-]?solid\b', '', 'ground-glass'),
    (r'\b(?:part[\s-]?solid|sub[\s-]?solid|semi[\s-]?solid)\b', '', None),        # None: the matched words
    (r'\b(?:pure\s+)?ground[\s-]?glass(?:\s+(?:density|attenuation))?\b', '', 'ground-glass'),
    (r'\bsolid\b', '', 'solid'),
    (r'\bcalcified\b', '', 'calcified'),
]
_TYPE_RE = re.compile('|'.join(f'(?P<r{i}>{p})' for i, (p, _, _) in enumerate(_TYPE_RULES)), re.I)
def mask_types(text, nid):
    """Density words of a one-nodule piece -> [TYPE#<nid>=<GT word>]. The dataset turns the tag into
    [TYPE#k] for the target (the word is not shown to the model) and back into the word for the GT
    side of evaluation. One alternation, one pass: a tag is never re-matched by a later rule."""
    def sub(m):
        i = int(m.lastgroup[1:])
        _, tail, word = _TYPE_RULES[i]
        return f'[TYPE#{nid}={word or m.group(0).lower()}]{tail}'
    text = _TYPE_RE.sub(sub, text)
    tag = rf'\[TYPE#{nid}=[^\]]*\]'
    text = re.sub(rf'({tag})(?:[\s,-]*(?:and\s+|or\s+|/)?{tag})+', r'\1', text)        # "part-solid GGO" -> one tag
    return re.sub(r'\s{2,}', ' ', text).strip()

# ---- which Impression sentences to keep ----------------------------------------------------
KEEP_IMP = re.compile(r'suggest|recommend|follow|correlat|benign|malignan|neoplas|cancer|infect|inflamm|'
                      r'non-?specific|nature|ddx|consult|survey|biops|r/o|probabl|likely|suspici|lung-?rads|ldct', re.I)
COMPARE = re.compile(r'\b(?:no\s+(?:obvious\s+)?(?:interval\s+)?change|stable|unchanged|interval|previous|prior|'
                     r'recurrence|progress|regress|decreas|increas)\w*', re.I)
NON_LUNG = re.compile(EXTRA_PULM + r'|cyst|stone|goiter|athero|hernia|fatty|spondyl|osteo|fracture|'
                      r'scapula|clavic|stern|humer|lytic|sclerotic', re.I)

def placeholders(text, ids, kind):
    t = mask_sizes(tidy(text))
    if len(ids) == 1 and kind != 'shared':
        t = mask_types(t.replace(PLACEHOLDER, f'[SIZE#{ids[0]}]'), ids[0])
    return t

rows, stat = [], dict(size_idx=0, size_gen=0, type_idx=0, imp_masked=0, imp_empty=0, imp_kept=0, imp_dropped=0)
for r in S.itertuples():
    f = r.folder
    nods = NOD[NOD.folder == f]
    lobe_of = dict(zip(nods.nodule_id.astype(int), nods.lobe))
    per_lobe = {L: [] for L in L5}; masked = set(); used = set(); glob = []
    g = P[P.folder == f].sort_values(['order', 'piece'])
    shared_hit = g[g.kind == 'shared'].groupby('orig').assigned.apply(lambda s: (s != '').any()).to_dict()
    seen = set(); imp_piece = {}; imp_unmatched = False
    for p in g.itertuples():
        ids = [int(x) for x in p.assigned.split(',') if x]
        lobes = [x for x in p.lobes.split(',') if x in L5]
        if p.kind == 'shared' and not ids and shared_hit.get(p.orig):
            continue
        targets = sorted({lobe_of[i] for i in ids if lobe_of.get(i)}) if ids else []
        if not targets: targets = [lobes[0]] if len(lobes) == 1 else lobes
        if not ids and p.nodule_like:
            if p.text_f: masked.update(targets)
            if p.text_i and not NON_LUNG.search(p.text_i): imp_unmatched = True
        if p.text_f:
            t = placeholders(p.text_f, ids, p.kind)
            if not targets: glob.append(t)
            for L in targets:
                if (L, t) not in seen:
                    seen.add((L, t)); per_lobe[L].append(t)
        if p.text_i and not (NON_LUNG.search(p.text_i) and not ids):   # e.g. a scapula lesion the matcher kept
            imp_piece.setdefault(p.orig, []).append(placeholders(p.text_i, ids, p.kind))
        used.add(p.orig)
    # Impression in report order
    imp, done = [], set()
    sec = SECT[SECT.folder == f].sort_values('idx')
    for s in sec.itertuples():
        if s.section == 'findings':
            if not any(s.text in o for o in used): glob.append(tidy(s.text))
            continue
        hit = [o for o in imp_piece if s.text in o and o not in done]
        if hit:
            for o in hit:
                done.add(o)
                for t in imp_piece[o]:
                    if t not in imp: imp.append(t)
            continue
        t = tidy(s.text)
        if not any(s.text in o for o in used) and KEEP_IMP.search(t) and not NON_LUNG.search(t) and not COMPARE.search(t):
            imp.append(mask_sizes(t)); stat['imp_kept'] += 1
        else:
            glob.append(t); stat['imp_dropped'] += 1
    vol = f + '.nii.gz'
    for L in L5:
        ids = nods[nods.lobe == L].sort_values('eq_diam_mm', ascending=False).nodule_id.astype(int).tolist()
        txt = dedupe(' '.join(per_lobe[L])) or NO_FINDING
        rows.append(dict(Volumename=vol, Anatomy=NAME[L], Sentence=txt, nodule_ids=';'.join(map(str, ids)),
                         loss_mask=0 if L in masked else 1, source=r.source, split=r.split))
    itxt = dedupe(' '.join(imp)) or NO_FINDING
    stat['imp_masked'] += imp_unmatched; stat['imp_empty'] += itxt == NO_FINDING
    allids = nods.sort_values('eq_diam_mm', ascending=False).nodule_id.astype(int).tolist()
    rows.append(dict(Volumename=vol, Anatomy='impression', Sentence=itxt, nodule_ids=';'.join(map(str, allids)),
                     loss_mask=0 if imp_unmatched else 1, source=r.source, split=r.split))
    rows.append(dict(Volumename=vol, Anatomy=np.nan, Sentence=dedupe(HEALTH.sub('', ' '.join(x for x in glob if x))),
                     nodule_ids='', loss_mask=1, source=r.source, split=r.split))
G = pd.DataFrame(rows)
for c in ('size_idx', 'size_gen', 'type_idx'):
    pass
txt = G.Sentence.fillna('')
stat['size_idx'] = int(txt.str.count(r'\[SIZE#').sum()); stat['size_gen'] = int(txt.str.count(r'\[SIZE\]').sum())
stat['type_idx'] = int(txt.str.count(r'\[TYPE#').sum())
for sp, d in G.groupby('split'):
    d.to_csv(os.path.join(OUT, f'region_report_v3_{sp}.csv'), index=False, encoding='utf-8')
print(stat)
lobe = G[G.Anatomy.notna() & (G.Anatomy != 'impression')]; im = G[G.Anatomy == 'impression']
print(lobe.groupby('split').agg(patients=('Volumename', 'nunique'), loss_masked=('loss_mask', lambda s: int((s == 0).sum())),
      no_finding=('Sentence', lambda s: int((s == NO_FINDING).sum()))).reindex(['train', 'val', 'test_me', 'test_ldct']).to_string())
print(im.groupby('split').agg(n=('Sentence', 'size'), masked=('loss_mask', lambda s: int((s == 0).sum())),
      empty=('Sentence', lambda s: int((s == NO_FINDING).sum())),
      words=('Sentence', lambda s: int(s.str.split().str.len().median()))).reindex(['train', 'val', 'test_me', 'test_ldct']).to_string())
