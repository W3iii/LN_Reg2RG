"""
LIDC-IDRI -> extra lesion-encoder training crops with type labels (2026-11).

Output (same format as the ME crop cache of evaluation/train_lesion_encoder.py):
  <out>.npy      float16 (N, 64, 64, 32): native-resolution crops resampled to (0.8, 0.8, 1.0) mm,
                 (y, x, z) = (posterior, patient-left, inferior) like the ME volumes, nodule centre at
                 [32, 32, 16], normalised with lesion_utils.normalize_lesion
  <out>.npy.csv  one row per nodule: Volumename, nodule_id, split (= train), eq_diam_mm, tw_lung_rads
                 (empty: LIDC has no Lung-RADS), type_set, n_readers, texture_votes, calc_votes,
                 malignancy_median, series_uid
  <out>_qc.png   mid-slices of the first crops, to check centring by eye

Labels
  LIDC readers (up to 4) score each nodule >= 3 mm: texture 1 (ground-glass) .. 5 (solid),
  calcification 1-5 = a calcification pattern, 6 = absent. These are research scales, not the
  report wording of the ME hospital, so only the unambiguous ends get a single type:
      median texture 1 -> GGO        2 -> GGO|part-solid    3 -> part-solid
                     4 -> part-solid|solid                   5 -> solid
  ('a|b' = one of these: the set-valued type loss of train_lesion_encoder.py). Calcification
  present for more than half of the readers -> calcified; for some but not most -> the texture
  set | calcified. Size = median over readers of the equivalent diameter of their filled
  contours. No RADS label: its loss is masked for these rows. Evaluation stays on ME val.

Reader matching: the LIDC XML has no cross-reader nodule id. Nodules of different readers in
the same series are merged when their contour centroids are within max(3 mm, half the larger
diameter); one nodule per reader per cluster. Clusters marked by fewer than --min_readers
readers are dropped (default 2).

Layout expected (TCIA download as on the workstation):
  <lidc_root>/metadata.csv                    PatientID, StudyInstanceUID, SeriesInstanceUID
  <lidc_root>/images/<PatientID>/<study uid last 5>/<series uid last 5>/*.dcm
  <lidc_root>/labels/<n>/*.xml                LIDC XML, one per series (SeriesInstanceUid header)

  python build_lidc_lesions.py --lidc_root F:/DataSet/LIDC --out F:/DataSet/lidc_lesions/lidc_lesion_crops.npy --workers 8
  python build_lidc_lesions.py ... --limit 3     # smoke test
"""
import os
import sys
import glob
import json
import time
import argparse
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas as pd

TGT = (0.8, 0.8, 1.0)
CROP = (64, 64, 32)
TEXTURE_SET = {1: 'GGO', 2: 'GGO|part-solid', 3: 'part-solid', 4: 'part-solid|solid', 5: 'solid'}


def normalize_lesion(crop, hu_min=-1000.0, hu_max=200.0):        # == src/lesion_utils.normalize_lesion
    crop = np.clip(crop, hu_min, hu_max)
    return ((crop + 400.0) / 600.0).astype(np.float32)


# ------------------------------------------------------------------ XML
def _strip(tag):
    return tag.split('}')[-1]


def _find(el, name):
    for c in el:
        if _strip(c.tag) == name:
            return c
    return None


def _findall(el, name):
    return [c for c in el if _strip(c.tag) == name]


def parse_xml(path):
    """-> (series_uid, [reader -> [nodule dict]]). Only nodules with characteristics (>= 3 mm)."""
    root = ET.parse(path).getroot()
    hdr = _find(root, 'ResponseHeader')
    uid_el = None
    if hdr is not None:
        uid_el = _find(hdr, 'SeriesInstanceUid')
        if uid_el is None:                      # Elements with no children are falsy: no `or` here
            uid_el = _find(hdr, 'SeriesInstanceUID')
    series = uid_el.text.strip() if uid_el is not None and uid_el.text else None
    readers = []
    for sess in _findall(root, 'readingSession'):
        nods = []
        for n in _findall(sess, 'unblindedReadNodule'):
            ch = _find(n, 'characteristics')
            if ch is None or _find(ch, 'texture') is None:
                continue
            c = {_strip(x.tag): x.text for x in ch}
            rois = []
            for roi in _findall(n, 'roi'):
                sop = _find(roi, 'imageSOP_UID')
                inc = _find(roi, 'inclusion')
                pts = [(int(float(_find(e, 'xCoord').text)), int(float(_find(e, 'yCoord').text)))
                       for e in _findall(roi, 'edgeMap')]
                if sop is not None and pts:
                    rois.append((sop.text.strip(), (inc is None or inc.text.strip().upper() == 'TRUE'), pts))
            if rois:
                nods.append(dict(texture=int(c['texture']), calcification=int(c.get('calcification') or 6),
                                 malignancy=int(c.get('malignancy') or 0), rois=rois))
        readers.append(nods)
    return series, readers


# ------------------------------------------------------------------ DICOM
def load_series(folder):
    import pydicom
    sl = []
    for f in glob.glob(os.path.join(folder, '*.dcm')):
        d = pydicom.dcmread(f)
        if not hasattr(d, 'ImagePositionPatient') or not hasattr(d, 'PixelData'):
            continue
        sl.append(d)
    if len(sl) < 10:
        raise ValueError(f'{folder}: {len(sl)} image slices')
    iop = np.array(sl[0].ImageOrientationPatient, float)
    if not (np.allclose(iop, [1, 0, 0, 0, 1, 0], atol=0.05)):
        raise ValueError(f'{folder}: orientation {iop.round(2).tolist()} not axial HFS')
    sl.sort(key=lambda d: -float(d.ImagePositionPatient[2]))        # index increases toward the feet
    z = np.array([float(d.ImagePositionPatient[2]) for d in sl])
    vol = np.stack([d.pixel_array.astype(np.float32) * float(getattr(d, 'RescaleSlope', 1))
                    + float(getattr(d, 'RescaleIntercept', 0)) for d in sl], axis=-1)   # (rows, cols, slices)
    ps = [float(v) for v in sl[0].PixelSpacing]                      # (row, col) mm
    sop = {d.SOPInstanceUID: i for i, d in enumerate(sl)}
    return vol, z, ps, sop


def fill(poly, shape):
    """Polygon (x=col, y=row) -> bool mask, edge included."""
    from PIL import Image, ImageDraw
    img = Image.new('1', (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).polygon([tuple(p) for p in poly], outline=1, fill=1)
    return np.array(img, dtype=bool)


def reader_nodule(n, sop, shape, z, ps):
    """Filled-contour voxels -> centroid (row, col, slice), eq diameter (mm), centroid in mm."""
    vox = {}
    for uid, inc, pts in n['rois']:
        if uid not in sop:
            continue
        k = sop[uid]
        m = fill(pts, shape[:2]) if len(pts) > 2 else None
        if m is None:
            m = np.zeros(shape[:2], bool)
            for x, y in pts:
                m[min(max(y, 0), shape[0] - 1), min(max(x, 0), shape[1] - 1)] = True
        cur = vox.get(k, np.zeros(shape[:2], bool))
        vox[k] = (cur | m) if inc else (cur & ~m)
    if not vox:
        return None
    rows, cols, ks, n_vox = [], [], [], 0
    for k, m in vox.items():
        r, c = np.nonzero(m)
        rows.append(r); cols.append(c); ks.append(np.full(len(r), k)); n_vox += len(r)
    if n_vox == 0:
        return None
    r, c, k = np.concatenate(rows), np.concatenate(cols), np.concatenate(ks)
    dz = float(np.median(np.abs(np.diff(z)))) if len(z) > 1 else 1.0
    vol_mm3 = n_vox * ps[0] * ps[1] * dz
    cy, cx, ck = r.mean(), c.mean(), k.mean()
    zc = float(np.interp(ck, np.arange(len(z)), z))
    return dict(cy=cy, cx=cx, ck=ck, mm=np.array([cy * ps[0], cx * ps[1], zc]),
                diam=2 * (3 * vol_mm3 / (4 * np.pi)) ** (1 / 3), texture=n['texture'],
                calcification=n['calcification'], malignancy=n['malignancy'])


def cluster(readers):
    """[[reader nodule]] -> clusters, each a list of (reader index, nodule); one per reader."""
    cl = []
    for ri, nods in enumerate(readers):
        for n in nods:
            best, bd = None, None
            for c in cl:
                if any(r == ri for r, _ in c):
                    continue
                cen = np.mean([q['mm'] for _, q in c], 0)
                d = np.linalg.norm(cen - n['mm'])
                lim = max(3.0, 0.5 * max(n['diam'], max(q['diam'] for _, q in c)))
                if d <= lim and (bd is None or d < bd):
                    best, bd = c, d
            if best is None:
                cl.append([(ri, n)])
            else:
                best.append((ri, n))
    return cl


def crop_at(vol, z, ps, cy, cx, ck):
    """(64, 64, 32) crop resampled to TGT spacing, centred on (row cy, col cx, slice ck)."""
    from scipy.ndimage import map_coordinates
    oy = (np.arange(CROP[0]) - CROP[0] // 2) * TGT[0] / ps[0] + cy
    ox = (np.arange(CROP[1]) - CROP[1] // 2) * TGT[1] / ps[1] + cx
    zc = float(np.interp(ck, np.arange(len(z)), z))
    zt = zc - (np.arange(CROP[2]) - CROP[2] // 2) * TGT[2]           # inferior = lower z
    ok = z[::-1]                                                     # increasing for np.interp
    kt = (len(z) - 1) - np.interp(zt, ok, np.arange(len(z)))        # fractional slice index
    Y, Xg, K = np.meshgrid(oy, ox, kt, indexing='ij')
    out = map_coordinates(vol, [Y, Xg, K], order=1, mode='nearest')
    return normalize_lesion(out).astype(np.float16)


# ------------------------------------------------------------------ one series
def process(job):
    series, folder, readers, patient, min_readers = job
    try:
        vol, z, ps, sop = load_series(folder)
    except Exception as e:
        return series, [], f'load: {repr(e)[:160]}'
    rn = [[q for q in (reader_nodule(n, sop, vol.shape, z, ps) for n in nods) if q] for nods in readers]
    rows, crops = [], []
    k = 0
    for c in cluster(rn):
        if len(c) < min_readers:
            continue
        k += 1
        tex = [q['texture'] for _, q in c]
        calc = [q['calcification'] for _, q in c]
        mal = [q['malignancy'] for _, q in c if q['malignancy']]
        tset = TEXTURE_SET[int(np.clip(np.round(np.median(tex)), 1, 5))]
        n_calc = sum(v != 6 for v in calc)
        if n_calc * 2 > len(calc):
            tset = 'calcified'
        elif n_calc:
            tset = tset + '|calcified'
        cy, cx, ck = (float(np.mean([q[a] for _, q in c])) for a in ('cy', 'cx', 'ck'))
        crops.append(crop_at(vol, z, ps, cy, cx, ck))
        rows.append(dict(Volumename=f'{patient}_{series[-5:]}', nodule_id=k, split='train',
                         eq_diam_mm=round(float(np.median([q['diam'] for _, q in c])), 2), tw_lung_rads=np.nan,
                         type_set=tset, n_readers=len(c), texture_votes=';'.join(map(str, tex)),
                         calc_votes=';'.join(map(str, calc)),
                         malignancy_median=float(np.median(mal)) if mal else np.nan,
                         series_uid=series, pid=patient))
    return series, list(zip(rows, crops)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidc_root', required=True)
    ap.add_argument('--out', required=True, help='.npy path; the label table is written next to it as <out>.csv')
    ap.add_argument('--min_readers', type=int, default=2)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--limit', type=int, default=0, help='first N series only (smoke test)')
    a = ap.parse_args()

    meta = pd.read_csv(os.path.join(a.lidc_root, 'metadata.csv'))
    folder_of = {}
    for r in meta.itertuples():
        f = os.path.join(a.lidc_root, 'images', r.PatientID, str(r.StudyInstanceUID)[-5:], str(r.SeriesInstanceUID)[-5:])
        if os.path.isdir(f):
            folder_of[r.SeriesInstanceUID] = (f, r.PatientID)

    xmls = sorted(glob.glob(os.path.join(a.lidc_root, 'labels', '**', '*.xml'), recursive=True))
    by_series, skipped = {}, []
    for p in xmls:
        try:
            s, readers = parse_xml(p)
        except Exception as e:
            skipped.append((p, f'xml: {repr(e)[:100]}')); continue
        if s is None or s not in folder_of:
            skipped.append((p, 'series not in images/')); continue
        # a corrected resubmission replaces the original read of the same series
        if s in by_series and 'resubmit' not in os.path.basename(p).lower():
            continue
        by_series[s] = readers
    jobs = [(s, folder_of[s][0], rd, folder_of[s][1], a.min_readers) for s, rd in by_series.items()]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f'{len(xmls)} xml, {len(by_series)} CT series with readings, {len(skipped)} xml skipped; processing {len(jobs)}', flush=True)

    t0 = time.time(); results, errors = [], []
    with Pool(a.workers) as pool:
        for i, (s, res, err) in enumerate(pool.imap_unordered(process, jobs), 1):
            if err:
                errors.append((s, err))
            results.extend(res)
            if i % 25 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} series, {len(results)} nodules, {len(errors)} errors, {time.time() - t0:.0f}s', flush=True)
    results.sort(key=lambda rc: (rc[0]['Volumename'], rc[0]['nodule_id']))
    df = pd.DataFrame([r for r, _ in results])
    X = np.stack([c for _, c in results]) if results else np.zeros((0,) + CROP, np.float16)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, X)
    df.to_csv(a.out + '.csv', index=False)
    with open(a.out + '_errors.json', 'w') as f:
        json.dump(dict(series_errors=errors, xml_skipped=skipped), f, indent=1)

    print(f'\nwrote {a.out} {X.shape} ({X.nbytes / 1e9:.2f} GB) + .csv; {len(errors)} series errors, {len(skipped)} xml skipped')
    if len(df):
        print('type_set:', df.type_set.value_counts().to_dict())
        print('readers :', df.n_readers.value_counts().sort_index().to_dict())
        print('eq_diam_mm median %.1f  p10 %.1f  p90 %.1f' % (df.eq_diam_mm.median(), df.eq_diam_mm.quantile(.1), df.eq_diam_mm.quantile(.9)))
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            n = min(24, len(X))
            fig, ax = plt.subplots(3, 8, figsize=(16, 6))
            for i, axx in enumerate(ax.flat):
                axx.axis('off')
                if i < n:
                    axx.imshow(X[i, :, :, 16].astype(np.float32), cmap='gray', vmin=-1, vmax=1)
                    axx.set_title(f"{df.type_set.iloc[i]} {df.eq_diam_mm.iloc[i]:.0f}mm", fontsize=7)
            plt.tight_layout(); plt.savefig(a.out.replace('.npy', '') + '_qc.png', dpi=80)
        except Exception as e:
            print('qc png skipped:', e)


if __name__ == '__main__':
    main()
