"""Single source of truth for the region list.

Upstream duplicated this list in five files (both Dataset classes, the model's
embedding layer, and both train/test entry points). They must agree exactly, and
they also have to match the mask filenames on disk:

    <mask_folder>/seg_<volume_name>/<region>.nii.gz

A mismatch does not fail at startup — it surfaces mid-training as
`KeyError: '<region name>'` inside MyEmbedding.forward, because the collator
builds its per-region buckets from one copy of the list while the dataset emits
region keys from another.

Current setup (NCKU ME dataset): the 5 pulmonary lobes. Keeping the region count
fixed per patient means inference can never exceed the region budget the model
was trained with, no matter how many nodules a scan contains.
"""

REGIONS = [
    'right upper lobe',
    'right middle lobe',
    'right lower lobe',
    'left upper lobe',
    'left lower lobe',
]
