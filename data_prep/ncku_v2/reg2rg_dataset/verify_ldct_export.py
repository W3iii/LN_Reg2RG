"""Independent check of export_reg2rg_ldct.py output, read back from reg2rg_nifti only.

Per patient: axcodes P,L,I (same voxel order as ME); image / lobe / nodule shapes equal the
crop_offsets_ldct.csv shape; nodule ids in nodules.nii.gz == ids in ldct_nodules.csv; voxel count
per id vs json size; lobe at each nodule's voxel centroid == ldct_nodule_lobe.csv (the lobe the
matching and GT v2 used).  -> verify_ldct_export.csv
"""
import os, sys, time
import numpy as np, nibabel as nib, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
O = os.path.join(ROOT, 'reg2rg_nifti')
R = ['right upper lobe', 'right middle lobe', 'right lower lobe', 'left upper lobe', 'left lower lobe']
SH = dict(zip(R, ['RUL', 'RML', 'RLL', 'LUL', 'LLL']))
i_, n_ = map(int, sys.argv[1].split('/')); budget = float(sys.argv[2])
off = pd.read_csv(os.path.join(O, 'crop_offsets_ldct.csv')).set_index('folder')
N = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodules.csv'))
L = pd.read_csv(os.path.join(ROOT, 'ldct_test_match', 'ldct_nodule_lobe.csv'))
N = N.merge(L[['folder', 'nodule_id', 'lobe']].rename(columns={'lobe': 'lobe_ref'}), on=['folder', 'nodule_id'], how='left')
outp = os.path.join(HERE, f'verify_ldct_export_{i_}of{n_}.csv')
done = set(pd.read_csv(outp).folder) if os.path.exists(outp) else set()
rows, t0 = [], time.time()
for f in off.index[i_::n_]:
    if f in done: continue
    if time.time() - t0 > budget: break
    img = nib.load(os.path.join(O, 'images', f, f, f'{f}.nii.gz'))
    nod = np.asanyarray(nib.load(os.path.join(O, 'masks', f'seg_{f}', 'nodules.nii.gz')).dataobj)
    ms = {SH[r]: np.asanyarray(nib.load(os.path.join(O, 'masks', f'seg_{f}', f'{r}.nii.gz')).dataobj) for r in R}
    o = off.loc[f]; shp = (o.crop_h, o.crop_w, o.crop_d)
    ok_shape = img.shape == shp and nod.shape == shp and all(m.shape == shp for m in ms.values())
    g = N[N.folder == f]
    ids = sorted(set(np.unique(nod).tolist()) - {0})
    n_lobe_ok = n_lobe_ref = 0; vox_ratio = []
    for r in g.itertuples():
        idx = np.argwhere(nod == r.nodule_id)
        if not len(idx): continue
        vox_ratio.append(len(idx) / max(r.size_vox, 1))
        c = tuple(np.round(idx.mean(0)).astype(int))
        got = [k for k, m in ms.items() if m[c]]
        if isinstance(r.lobe_ref, str) and r.lobe_ref:
            n_lobe_ref += 1; n_lobe_ok += got == [r.lobe_ref]
    rows.append(dict(folder=f, axcodes=''.join(nib.aff2axcodes(img.affine)), ok_shape=ok_shape,
                     n_ref=len(g), n_mask=len(ids), ids_equal=ids == sorted(g.nodule_id.tolist()),
                     n_lobe_ref=n_lobe_ref, n_lobe_ok=n_lobe_ok,
                     vox_ratio_min=min(vox_ratio) if vox_ratio else np.nan,
                     hu_min=float(np.asanyarray(img.dataobj).min()), flip=int(o.flip_yx)))
new = pd.DataFrame(rows)
if os.path.exists(outp): new = pd.concat([pd.read_csv(outp), new])
new.to_csv(outp, index=False)
print(f'shard {i_}/{n_}: {len(new)} done ({len(rows)} new) {time.time() - t0:.0f}s')
