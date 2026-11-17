"""
nodule_metadata_v2.csv = ME (1400 patients) + LDCT test (566 patients), split C.

Same columns as reg2rg_nifti/nodule_metadata.csv (the Reg2RG lesion code reads it), plus:
  source           ME | LDCT
  split            from splits_merged.csv: train / val / test_me / test_ldct  (old ME split dropped)
  match_tier       A / B / C / X / none  (same rule as build_gt_v2.py)
  matched_sentence v2 per-nodule text (match_v2.py); the v1 text is kept as matched_sentence_v1
  match_score      v2 score
  flip_yx          1 if the exported volume was flipped on y, x (prone LDCT, Test0374)

LDCT rows: ids / boxes from ldct_nodules.csv (mask json order, the ids used by GT v2 and
fill_table_v2), lobe from ldct_nodule_lobe.csv, crop frame from crop_offsets_ldct.csv. pid is
900000 + the Test number (folder / Volumename keep the original names). tw_lung_rads is empty.
Crop-frame coordinates follow make_csv_split.py (original - crop origin); for flip_yx patients
y and x are then mirrored: edge' = crop_len - edge (exclusive-end convention, centre = mean).
"""
import os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
O = os.path.join(ROOT, 'reg2rg_nifti')
SHORT2REGION = {'RUL': 'right upper lobe', 'RML': 'right middle lobe', 'RLL': 'right lower lobe',
                'LUL': 'left upper lobe', 'LLL': 'left lower lobe'}
NODW = r'nodul|GGN|GGO|ground[\s-]?glass|opacit|mass\b|lesion|densit|calcif|tumou?r|granuloma|LN\b|lymph'


def tier(r):                                         # == build_gt_v2._tier
    t = r.per_nodule_text if isinstance(r.per_nodule_text, str) else ''
    if not t: return 'none'
    if not re.search(NODW, t, re.I): return 'X'
    if r.kind != 'shared': return 'A'
    return 'B' if re.fullmatch(r'RUL|RML|RLL|LUL|LLL', r.piece_lobe if isinstance(r.piece_lobe, str) else '') else 'C'


S = pd.read_csv(os.path.join(O, 'splits_merged.csv'))
V2 = pd.concat([pd.read_csv(os.path.join(HERE, f'{s}_nodule_sentence_v2.csv')) for s in ('me', 'ldct')])
V2['match_tier'] = [tier(r) for r in V2.itertuples()]
V2 = V2.rename(columns={'key': 'folder'})[['folder', 'nodule_id', 'match_tier', 'per_nodule_text', 'match_score']]

# ---- ME: reuse the existing rows (crop frame already there), refresh split + text ----
me = pd.read_csv(os.path.join(O, 'nodule_metadata.csv'))
me = me.drop(columns=['split']).rename(columns={'matched_sentence': 'matched_sentence_v1', 'match_score': 'match_score_v1'})
me['source'] = 'ME'; me['flip_yx'] = 0

# ---- LDCT ----
ld = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodules.csv')).drop(columns=['lobe', 'side', 'zrel'])
ld = ld.merge(pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_lobe.csv'))[['folder', 'nodule_id', 'lobe']],
              on=['folder', 'nodule_id'], how='left')
ld = ld[ld.folder.isin(S[S.split == 'test_ldct'].folder)]
off = pd.read_csv(os.path.join(O, 'crop_offsets_ldct.csv'))
ld = ld.merge(off, on='folder', how='left')
assert ld.crop_y0.notna().all(), 'patients missing from crop_offsets_ldct.csv'
for ax, o in (('y', 'crop_y0'), ('x', 'crop_x0'), ('z', 'crop_z0')):
    for c in (f'{ax}0', f'{ax}1', f'c{ax}'):
        ld[f'{c}_crop'] = ld[c] - ld[o]
f = ld.flip_yx == 1
for ax, L in (('y', 'crop_h'), ('x', 'crop_w')):
    a0, a1, ac = ld.loc[f, f'{ax}0_crop'].copy(), ld.loc[f, f'{ax}1_crop'].copy(), ld.loc[f, f'c{ax}_crop'].copy()
    ld.loc[f, f'{ax}0_crop'] = ld.loc[f, L] - a1
    ld.loc[f, f'{ax}1_crop'] = ld.loc[f, L] - a0
    ld.loc[f, f'c{ax}_crop'] = ld.loc[f, L] - ac
ld['pid'] = 900000 + ld.folder.str.extract(r'Test(\d+)')[0].astype(int)
ld['Volumename'] = ld.folder + '.nii.gz'
ld['region'] = ld.lobe.map(SHORT2REGION)
ld['tw_lung_rads'] = np.nan
ld['region_source'] = 'ldct_mask'
ld['source'] = 'LDCT'

M = pd.concat([me, ld], ignore_index=True)
M = M.merge(S[['folder', 'split']], on='folder', how='left')
M = M.merge(V2, on=['folder', 'nodule_id'], how='left')
M = M.rename(columns={'per_nodule_text': 'matched_sentence'})
M['match_tier'] = M.match_tier.fillna('none')

cols = ['Volumename', 'pid', 'folder', 'source', 'split', 'nodule_id', 'region', 'lobe',
        'y0', 'x0', 'z0', 'y1', 'x1', 'z1', 'cy', 'cx', 'cz',
        'crop_y0', 'crop_x0', 'crop_z0', 'crop_h', 'crop_w', 'crop_d', 'flip_yx',
        'y0_crop', 'x0_crop', 'z0_crop', 'y1_crop', 'x1_crop', 'z1_crop', 'cy_crop', 'cx_crop', 'cz_crop',
        'size_vox', 'eq_diam_mm', 'bbox_long_mm', 'tw_lung_rads', 'region_source',
        'match_tier', 'matched_sentence', 'match_score', 'matched_sentence_v1', 'match_score_v1']
M = M[cols].sort_values(['source', 'folder', 'nodule_id'])
M.to_csv(os.path.join(O, 'nodule_metadata_v2.csv'), index=False, encoding='utf-8-sig')

print(len(M), 'nodules,', M.folder.nunique(), 'patients')
print(pd.crosstab(M.split, M.source, margins=True).to_string())
print(pd.crosstab(M.split, M.match_tier).to_string())
print('region missing:', M.region.isna().sum(), ' split missing:', M.split.isna().sum())
