"""Lesion crops for the lesion-token pathway (docs/LESION_TOKENS.md).

Reads nodule bounding boxes straight out of nodule_metadata.csv. That CSV already
carries `crop_y0/x0/z0` and precomputed `y0_crop..z1_crop`, i.e. boxes in the same
cropped frame as the exported NIfTI, so the instance mask that §5 of the design
doc asks the workstation to produce is not needed -- the note there predates
those columns. Verified against the exported volumes: crop_h/w/d matches the
image shape, and nodule boxes separate from random same-size boxes drawn in the
same lobe at AUC 0.837 on mean HU alone, which random boxes could not do if the
mapping were wrong.

That 0.837 is also the reason this pathway exists. Every lobe-level
representation measured -- ViT patch tokens, perceiver latents, what the LLM
sees, and model-free voxel statistics at native resolution -- sits between 0.54
and 0.61 AUC for "does this lobe contain a nodule", indistinguishable from just
knowing how big the lobe is. A 5 mm nodule is ~65 voxels in a ~4 M voxel lobe, so
no pooled summary can hold it. Cropping to the lesion is what recovers it.

NOTE: lesion crops are keyed on ground-truth annotations. That makes this an
oracle-localisation setting, not a deployable pipeline -- see docs/LESION_TOKENS.md.
"""
import numpy as np
import pandas as pd

LESION_CROP_SIZE = (64, 64, 32)

# Boxes are tiny -- median 6x7x4 voxels, and only 6 of 5392 exceed LESION_CROP_SIZE.
BBOX_COLS = ['y0_crop', 'y1_crop', 'x0_crop', 'x1_crop', 'z0_crop', 'z1_crop']


def load_nodule_index(csv_path, max_per_region=None, pred_csv=None, order='truth'):
    """(Volumename, region) -> [(box, pred_mm), ...], first = the one to lead with.

    order='truth'  rank by tw_lung_rads then eq_diam_mm, both descending (B2/B3a).
                   Both are ground truth, so at inference the *position* of a
                   lesion token already tells the decoder which nodule is the
                   most severe and the largest -- an oracle leak, see
                   docs/LESION_TOKENS.md.
    order='pred'   rank by the lesion encoder's predicted diameter, descending
                   (B3c). Uses no label at all.

    pred_csv: outputs/lesion_pred_<arch>_seed<seed>.csv from
    evaluation/predict_lesion_attributes.py (out-of-fold for train, full-train
    encoder for val/test). When given, each record carries pred_mm, which the
    Dataset writes into the prompt as "<lesionK> (~X mm)"; otherwise pred_mm is
    None and the prompt is byte-identical to B2/B3a.

    Overflow (>max_per_region in a lobe) keeps the first ones under the chosen
    order. Only 15 of 3144 (patient, lobe) pairs have more than 8, and the lobe's
    report text still describes every nodule, so overflow costs visual detail,
    never a training target.
    """
    if order not in ('truth', 'pred'):
        raise ValueError(f"order must be 'truth' or 'pred', got {order!r}")
    if order == 'pred' and not pred_csv:
        raise ValueError("order='pred' needs pred_csv")
    df = pd.read_csv(csv_path)

    missing = [c for c in BBOX_COLS + ['Volumename', 'region'] if c not in df.columns]
    if missing:
        raise ValueError(
            f'{csv_path} is missing {missing}. Expected the cropped-frame boxes '
            f'(available: {list(df.columns)})')

    df = df.dropna(subset=BBOX_COLS)
    for c in ('tw_lung_rads', 'eq_diam_mm'):
        if c not in df.columns:
            df[c] = 0.0

    df['pred_mm'] = np.nan
    if pred_csv:
        p = pd.read_csv(pred_csv)[['Volumename', 'nodule_id', 'pred_diam_mm']]
        if p.duplicated(['Volumename', 'nodule_id']).any():
            raise ValueError(f'{pred_csv}: duplicate (Volumename, nodule_id) rows')
        df = df.drop(columns='pred_mm').merge(p, on=['Volumename', 'nodule_id'], how='left')
        df = df.rename(columns={'pred_diam_mm': 'pred_mm'})
        n_miss = int(df['pred_mm'].isna().sum())
        if n_miss:
            raise ValueError(f'{pred_csv} has no prediction for {n_miss} of {len(df)} nodules '
                             f'in {csv_path}; regenerate it from the same nodule_metadata.csv')

    if order == 'truth':
        df = df.sort_values(['tw_lung_rads', 'eq_diam_mm'], ascending=False)
    else:
        df = df.sort_values(['pred_mm', 'nodule_id'], ascending=[False, True])

    index = {}
    for (vol, region), grp in df.groupby(['Volumename', 'region'], sort=False):
        if max_per_region is not None:
            grp = grp.head(max_per_region)
        index[(str(vol), str(region))] = [
            ((int(r.y0_crop), int(r.y1_crop), int(r.x0_crop),
              int(r.x1_crop), int(r.z0_crop), int(r.z1_crop)),
             None if pd.isna(r.pred_mm) else float(r.pred_mm))
            for r in grp.itertuples()
        ]
    return index


def lesion_token_text(slot, pred_mm):
    """'<lesion12>' or, with a predicted size, '<lesion12> (~7 mm)'."""
    if pred_mm is None:
        return f'<lesion{slot}>'
    return f'<lesion{slot}> (~{max(1, int(round(pred_mm)))} mm)'

def crop_lesion(img, bbox, size=LESION_CROP_SIZE):
    """Crop `img` at `bbox`, centred in a fixed-size zero-padded array.

    No resize: downsampling here would reintroduce exactly the scale loss the
    lobe-level pathway already suffers from. A box larger than `size` is
    centre-cropped rather than shrunk, which affects 6 nodules in 5392.
    """
    y0, y1, x0, x1, z0, z1 = bbox
    y0, x0, z0 = max(y0, 0), max(x0, 0), max(z0, 0)
    y1 = min(y1, img.shape[0])
    x1 = min(x1, img.shape[1])
    z1 = min(z1, img.shape[2])
    if y0 >= y1 or x0 >= x1 or z0 >= z1:
        return np.zeros(size, dtype=np.float32)

    ty, tx, tz = size
    # widen a tiny box out to the crop window so the encoder sees context, not
    # just the lesion's own voxels
    cy, cx, cz = (y0 + y1) // 2, (x0 + x1) // 2, (z0 + z1) // 2
    wy0 = int(np.clip(cy - ty // 2, 0, max(img.shape[0] - ty, 0)))
    wx0 = int(np.clip(cx - tx // 2, 0, max(img.shape[1] - tx, 0)))
    wz0 = int(np.clip(cz - tz // 2, 0, max(img.shape[2] - tz, 0)))
    crop = img[wy0:wy0 + ty, wx0:wx0 + tx, wz0:wz0 + tz]

    out = np.zeros(size, dtype=np.float32)
    out[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
    return out


def normalize_lesion(crop, hu_min=-1000.0, hu_max=200.0):
    """Same HU window and scaling the region pathway applies to lobe volumes."""
    crop = np.clip(crop, hu_min, hu_max)
    return ((crop + 400.0) / 600.0).astype(np.float32)
