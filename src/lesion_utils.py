"""Pure-array helpers for lesion-token cropping (docs/LESION_TOKENS.md).

Kept out of the two Dataset classes for the same reason regions.py is: this is
exactly the kind of logic where a train/test mismatch wouldn't raise, it would
just silently misassign a nodule's crop to the wrong lobe or the wrong slot.

Expects a nodule *instance* mask (`masks/seg_<folder>/nodules.nii.gz`, uint,
0 = background, 1..N = nodule id) living in the same crop frame as the per-region
lobe masks and the image itself — verified against data_prep/export_reg2rg.py,
which crops the image and every region mask with one shared slice out of the same
npy grid, so an instance mask exported the same way needs no separate origin
bookkeeping (docs/LESION_TOKENS.md §5).

None of this has run against a real nodules.nii.gz yet — none exist on disk as of
writing (see HANDOFF_TO_LOCAL.md). It's written directly against the documented
contract rather than a throwaway stub; re-validate the first time real instance
masks land.
"""
import numpy as np

LESION_CROP_SIZE = (64, 64, 32)


def instance_bboxes(instance_mask):
    """nodule id (1..N) -> half-open voxel bbox (y0, y1, x0, x1, z0, z1)."""
    boxes = {}
    for nid in np.unique(instance_mask):
        nid = int(nid)
        if nid == 0:
            continue
        ys, xs, zs = np.nonzero(instance_mask == nid)
        boxes[nid] = (int(ys.min()), int(ys.max()) + 1,
                      int(xs.min()), int(xs.max()) + 1,
                      int(zs.min()), int(zs.max()) + 1)
    return boxes


def nodules_by_region(instance_mask, region_masks):
    """nodule id -> region key, assigned by max voxel overlap with each region mask.

    `region_masks` is {region_key: binary array, same shape as instance_mask}.
    A nodule with no overlap against any region mask (shouldn't happen if the
    nodule truly sits inside a segmented lobe, but real segmentations have edge
    cases) is left unassigned rather than guessed at.
    """
    assignment = {}
    for nid in np.unique(instance_mask):
        nid = int(nid)
        if nid == 0:
            continue
        nodule_voxels = instance_mask == nid
        best_region, best_overlap = None, 0
        for region, mask in region_masks.items():
            overlap = int(np.count_nonzero(nodule_voxels & (mask > 0)))
            if overlap > best_overlap:
                best_region, best_overlap = region, overlap
        if best_region is not None:
            assignment[nid] = best_region
    return assignment


def crop_lesion(img, bbox, size=LESION_CROP_SIZE):
    """Crop `img` at `bbox`, centre-aligned into a fixed-size zero-padded array.

    No resize (docs/LESION_TOKENS.md §2 — resizing is exactly what destroys the
    signal at the lobe level). Bigger-than-`size` bboxes are centre-cropped, not
    downsampled; this only bites the rare bbox that exceeds 64x64x32 at native
    (0.8, 0.8, 1.0) mm resolution, i.e. a lesion well past what the RUL/RML recall
    problem in HANDOFF.md §6 is about.
    """
    y0, y1, x0, x1, z0, z1 = bbox
    crop = img[y0:y1, x0:x1, z0:z1]
    ty, tx, tz = size
    sy, sx, sz = crop.shape

    cy0 = max((sy - ty) // 2, 0)
    cx0 = max((sx - tx) // 2, 0)
    cz0 = max((sz - tz) // 2, 0)
    crop = crop[cy0:cy0 + ty, cx0:cx0 + tx, cz0:cz0 + tz]

    out = np.zeros(size, dtype=crop.dtype)
    oy, ox, oz = crop.shape
    py0, px0, pz0 = (ty - oy) // 2, (tx - ox) // 2, (tz - oz) // 2
    out[py0:py0 + oy, px0:px0 + ox, pz0:pz0 + oz] = crop
    return out
