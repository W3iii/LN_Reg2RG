"""
LIDC-IDRI -> unlabelled lesion-sized crops for MAE pretraining of the lesion encoder (2026-11).

build_lidc_lesions.py keeps only nodules >= 3 mm marked by >= 2 readers (1,854 crops, with
type labels). Masked-image pretraining needs no labels, so this takes every crop it can from
the same scans:

  kind = nodule   every reader-marked nodule >= 3 mm, any number of readers (clustered as in
                  build_lidc_lesions.py), centred on the cluster centroid
  kind = small    nodules < 3 mm (single-point marks, no characteristics), one per cluster
  kind = random   --random_per_series centres drawn inside the lungs (HU < -400, not
                  connected to the air outside the body), at least 8 mm from the volume edge

Crops are identical in geometry and intensity to the ME crops and to build_lidc_lesions.py:
(64, 64, 32) at (0.8, 0.8, 1.0) mm, (y, x, z) = (posterior, patient-left, inferior),
lesion_utils.normalize_lesion, float16. No labels are used downstream: the encoder is
pretrained by evaluation/pretrain_lesion_mae.py, then trained on ME size / type / RADS by
train_lesion_encoder.py --init_encoder.

Output
  <out>.npy      float16 (N, 64, 64, 32)
  <out>.npy.csv  Volumename, kind, crop_id, n_readers, eq_diam_mm, pid, series_uid
  <out>_qc.png   mid-slices of a few crops of each kind

  python build_lidc_mae.py --lidc_root F:/DataSet/LIDC --out F:/DataSet/lidc_lesions/lidc_mae_crops.npy --workers 8
  python build_lidc_mae.py ... --limit 3     # smoke test

About 10 random + ~7 marked crops per series -> ~17k crops, ~4.5 GB at 262 KB each.
"""
import os
import sys
import glob
import json
import time
import zlib
import argparse
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_lidc_lesions import (CROP, TGT, _find, _findall, load_series, reader_nodule,  # noqa: E402
                                cluster, crop_at)


def parse_xml_all(path):
    """-> (series_uid, [reader -> [nodule dict]]) with every unblindedReadNodule, < 3 mm included."""
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
            big = ch is not None and _find(ch, 'texture') is not None
            rois = []
            for roi in _findall(n, 'roi'):
                sop = _find(roi, 'imageSOP_UID')
                inc = _find(roi, 'inclusion')
                pts = [(int(float(_find(e, 'xCoord').text)), int(float(_find(e, 'yCoord').text)))
                       for e in _findall(roi, 'edgeMap')]
                if sop is not None and pts:
                    rois.append((sop.text.strip(), (inc is None or inc.text.strip().upper() == 'TRUE'), pts))
            if rois:
                nods.append(dict(texture=None, calcification=None, malignancy=None, big=big, rois=rois))
        readers.append(nods)
    return series, readers


def lung_centres(vol, z, ps, k, rng):
    """k random (row, col, slice) centres inside the lungs, or fewer if the mask is small."""
    from scipy import ndimage
    sub = vol[::2, ::2, :]                                          # in-plane /2 is enough for a mask
    air = sub < -400
    lab, n = ndimage.label(air)
    if n == 0:
        return []
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel()]))
    lung = air & ~np.isin(lab, edge)
    lung = ndimage.binary_erosion(lung, iterations=2)               # keep centres off the pleura edge
    # keep the crop centre >= 8 mm (and >= 10 slices) from the volume border
    my, mx = int(np.ceil(8 / (2 * ps[0]))), int(np.ceil(8 / (2 * ps[1])))
    dz = float(np.median(np.abs(np.diff(z)))) if len(z) > 1 else 1.0
    mz = max(1, int(np.ceil(10 / dz)))
    lung[:my] = lung[-my:] = False
    lung[:, :mx] = lung[:, -mx:] = False
    lung[:, :, :mz] = lung[:, :, -mz:] = False
    idx = np.flatnonzero(lung)
    if len(idx) == 0:
        return []
    pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
    r, c, s = np.unravel_index(pick, lung.shape)
    return [(2.0 * a, 2.0 * b, float(d)) for a, b, d in zip(r, c, s)]


def process(job):
    series, folder, readers, patient, n_random = job
    try:
        vol, z, ps, sop = load_series(folder)
    except Exception as e:
        return series, [], f'load: {repr(e)[:160]}'
    rows, crops = [], []
    vname = f'{patient}_{series[-5:]}'

    def add(kind, cy, cx, ck, n_readers, diam):
        crops.append(crop_at(vol, z, ps, cy, cx, ck))
        rows.append(dict(Volumename=vname, kind=kind, crop_id=len(rows) + 1, n_readers=n_readers,
                         eq_diam_mm=round(float(diam), 2) if diam is not None else np.nan,
                         pid=patient, series_uid=series))

    for want_big, kind in ((True, 'nodule'), (False, 'small')):
        rn = [[q for q in (reader_nodule(n, sop, vol.shape, z, ps) for n in nods if n['big'] == want_big) if q]
              for nods in readers]
        for c in cluster(rn):
            cy, cx, ck = (float(np.mean([q[a] for _, q in c])) for a in ('cy', 'cx', 'ck'))
            add(kind, cy, cx, ck, len(c), np.median([q['diam'] for _, q in c]) if want_big else None)
    rng = np.random.default_rng(zlib.crc32(series.encode()))
    for cy, cx, ck in lung_centres(vol, z, ps, n_random, rng):
        add('random', cy, cx, ck, 0, None)
    return series, list(zip(rows, crops)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidc_root', required=True)
    ap.add_argument('--out', required=True, help='.npy path; the crop table is written next to it as <out>.csv')
    ap.add_argument('--random_per_series', type=int, default=10)
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
            s, readers = parse_xml_all(p)
        except Exception as e:
            skipped.append((p, f'xml: {repr(e)[:100]}')); continue
        if s is None or s not in folder_of:
            skipped.append((p, 'series not in images/')); continue
        if s in by_series and 'resubmit' not in os.path.basename(p).lower():
            continue
        by_series[s] = readers
    jobs = [(s, folder_of[s][0], rd, folder_of[s][1], a.random_per_series) for s, rd in by_series.items()]
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
                print(f'  {i}/{len(jobs)} series, {len(results)} crops, {len(errors)} errors, {time.time() - t0:.0f}s', flush=True)
    results.sort(key=lambda rc: (rc[0]['Volumename'], rc[0]['crop_id']))
    df = pd.DataFrame([r for r, _ in results])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    X = np.lib.format.open_memmap(a.out, mode='w+', dtype=np.float16, shape=(len(results),) + CROP)
    for i, (_, c) in enumerate(results):
        X[i] = c
    X.flush()
    df.to_csv(a.out + '.csv', index=False)
    with open(a.out + '_errors.json', 'w') as f:
        json.dump(dict(series_errors=errors, xml_skipped=skipped), f, indent=1)

    print(f'\nwrote {a.out} {X.shape} ({X.size * 2 / 1e9:.2f} GB) + .csv; {len(errors)} series errors, {len(skipped)} xml skipped')
    if len(df):
        print('kind:', df.kind.value_counts().to_dict())
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, 8, figsize=(16, 6))
            for row, kind in zip(ax, ('nodule', 'small', 'random')):
                ids = np.flatnonzero(df.kind.to_numpy() == kind)[:8]
                for axx in row:
                    axx.axis('off')
                for axx, i in zip(row, ids):
                    axx.imshow(np.asarray(X[i, :, :, 16], np.float32), cmap='gray', vmin=-1, vmax=1)
                    axx.set_title(kind, fontsize=7)
            plt.tight_layout(); plt.savefig(a.out.replace('.npy', '') + '_qc.png', dpi=80)
        except Exception as e:
            print('qc png skipped:', e)


if __name__ == '__main__':
    main()
