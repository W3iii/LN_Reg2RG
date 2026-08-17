"""
產生 Reg2RG 的 region report CSV 與 train/val/test split

CSV 欄位 (由 load_accession_sentences 讀): Volumename / Anatomy / Sentence
  Volumename 要含 .nii.gz  (它拿 nii 檔 basename 當 key)
  Anatomy    要與 REGIONS 常數逐字相符; 留空代表全域報告(whole), Reg2RG 會忽略
"""
import os, io, json
import numpy as np
import pandas as pd

ROOT = r'd:\W3iii\NCKU\DataSet'
DS = os.path.join(ROOT, 'reg2rg_dataset')
OUT = os.path.join(ROOT, 'reg2rg_nifti')

SHORT2REGION = {'RUL': 'right upper lobe', 'RML': 'right middle lobe',
                'RLL': 'right lower lobe', 'LUL': 'left upper lobe',
                'LLL': 'left lower lobe'}
NO_FINDING = 'No significant finding.'
SEED = 20260815

rr = {r['pid']: r for r in (json.loads(l) for l in io.open(os.path.join(DS, 'region_report.jsonl'), encoding='utf-8'))}
man = pd.read_csv(os.path.join(DS, 'manifest.csv'))
N = pd.read_csv(os.path.join(DS, 'nodules.csv'))
NS = pd.read_csv(os.path.join(DS, 'nodule_sentence.csv')).fillna({'matched_sentence': '', 'match_lobe': ''})

# ---------------- split: 病患層級, 依「最高 TW_Lung_RADS × nodule 數級距」分層 ----------------
agg = N.groupby('pid').agg(max_rads=('tw_lung_rads', 'max'), n_nod=('nodule_id', 'size')).reset_index()
agg['nbin'] = pd.cut(agg.n_nod, [0, 1, 3, 6, 1000], labels=['1', '2-3', '4-6', '7+'])
agg['stratum'] = agg.max_rads.astype(str) + '_' + agg.nbin.astype(str)

rng = np.random.RandomState(SEED)
split = {}
for s, g in agg.groupby('stratum'):
    ids = g.pid.sample(frac=1, random_state=rng).tolist()
    n = len(ids)
    n_te = max(1, int(round(n * .1)))
    n_va = max(1, int(round(n * .1))) if n >= 3 else 0
    for i, p in enumerate(ids):
        split[p] = 'test' if i < n_te else ('val' if i < n_te + n_va else 'train')
agg['split'] = agg.pid.map(split)
agg = agg.merge(man[['pid', 'folder']], on='pid')
agg[['pid', 'folder', 'split', 'max_rads', 'n_nod', 'stratum']].to_csv(
    os.path.join(OUT, 'splits.csv'), index=False, encoding='utf-8-sig')

# ---------------- region report CSV ----------------
made = {}
for sp in ('train', 'val', 'test'):
    rows = []
    for r in agg[agg.split == sp].itertuples():
        vol = f'{r.folder}.nii.gz'
        rec = rr.get(r.pid)
        if rec is None:
            continue
        for short, region in SHORT2REGION.items():
            txt = rec['regions'].get(short, '') or NO_FINDING
            rows.append(dict(Volumename=vol, Anatomy=region, Sentence=txt))
        g = rec.get('global_text', '').strip()
        if g:                                   # Anatomy 留空 -> 'whole', Reg2RG 會忽略, 但保留備用
            rows.append(dict(Volumename=vol, Anatomy=np.nan, Sentence=g))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f'region_report_{sp}.csv'), index=False, encoding='utf-8')
    made[sp] = df

# ---------------- nodule metadata (保留給之後的 lesion token) ----------------
M = N.merge(NS[['pid', 'nodule_id', 'matched_sentence', 'match_score', 'lobe_agree']],
            on=['pid', 'nodule_id'], how='left')
M = M.merge(agg[['pid', 'split']], on='pid', how='left')
M['Volumename'] = M.folder + '.nii.gz'
M['region'] = M.lobe.map(SHORT2REGION)

# Coordinates above are in the *uncropped* npy frame, but the exported NIfTI is cropped to
# the lung bbox. Carry the crop origin so they can be mapped in:
#     cropped = original - crop_<axis>0
# Lesion crops should normally come from masks/seg_<folder>/nodules.nii.gz instead, which
# is already in the cropped frame. See Reg2RG/docs/LESION_TOKENS.md §5.
off_p = os.path.join(OUT, 'crop_offsets.csv')
if os.path.exists(off_p):
    M = M.merge(pd.read_csv(off_p), on='folder', how='left')
    for ax, o in (('y', 'crop_y0'), ('x', 'crop_x0'), ('z', 'crop_z0')):
        for c in (f'{ax}0', f'{ax}1', f'c{ax}'):
            M[f'{c}_crop'] = M[c] - M[o]
else:
    print('WARNING: %s missing — run export_nodule_masks.py first; '
          'nodule_metadata.csv will have no cropped-frame coordinates' % off_p)

cols = ['Volumename', 'pid', 'folder', 'split', 'nodule_id', 'region', 'lobe',
        'y0', 'x0', 'z0', 'y1', 'x1', 'z1', 'cy', 'cx', 'cz',
        'crop_y0', 'crop_x0', 'crop_z0', 'crop_h', 'crop_w', 'crop_d',
        'y0_crop', 'x0_crop', 'z0_crop', 'y1_crop', 'x1_crop', 'z1_crop',
        'cy_crop', 'cx_crop', 'cz_crop',
        'size_vox', 'eq_diam_mm', 'bbox_long_mm', 'tw_lung_rads',
        'region_source', 'matched_sentence', 'match_score']
M[[c for c in cols if c in M.columns]].to_csv(
    os.path.join(OUT, 'nodule_metadata.csv'), index=False, encoding='utf-8-sig')

# ---------------- 統計 ----------------
log = io.open(os.path.join(OUT, 'dataset_summary.txt'), 'w', encoding='utf-8')
log.write('== split (病患層級, 依 max TW_Lung_RADS × nodule 數級距分層) ==\n')
log.write(agg.split.value_counts().reindex(['train', 'val', 'test']).to_string() + '\n\n')
log.write('各 split 的 max_rads 分布:\n')
log.write(pd.crosstab(agg.split, agg.max_rads, normalize='index').round(3).to_string() + '\n\n')
for sp, df in made.items():
    lobe_rows = df[df.Anatomy.notna()]
    nf = (lobe_rows.Sentence == NO_FINDING).sum()
    log.write('%-5s : %5d 位病患, %5d 個 region-text 配對, 其中 "%s" %d (%.1f%%)\n'
              % (sp, df.Volumename.nunique(), len(lobe_rows), NO_FINDING, nf, 100 * nf / len(lobe_rows)))
log.write('\n每個 region 有實際敘述的比例 (train):\n')
tr = made['train']
tr = tr[tr.Anatomy.notna()]
for region in SHORT2REGION.values():
    g = tr[tr.Anatomy == region]
    log.write('  %-18s %.1f%%\n' % (region, 100 * (g.Sentence != NO_FINDING).mean()))
log.write('\nSentence 長度 (字元, 排除 No significant finding): median %d  p90 %d\n'
          % (tr[tr.Sentence != NO_FINDING].Sentence.str.len().median(),
             tr[tr.Sentence != NO_FINDING].Sentence.str.len().quantile(.9)))
log.write('\nnodule metadata: %d 顆, 各 split %s\n' % (len(M), M.split.value_counts().to_dict()))
log.close()
print(open(os.path.join(OUT, 'dataset_summary.txt'), encoding='utf-8').read())
