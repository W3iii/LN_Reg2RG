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

# Lesion tokens (docs/LESION_TOKENS.md). Same single-source-of-truth reasoning as
# REGIONS above: the special-token list is built independently in
# src/Model/Reg2RG.py and both src/Dataset/radgenome_dataset_{train,test}.py, and a
# mismatch there doesn't raise — it silently selects the wrong embedding row.
MAX_LESION_PER_REGION = 8


# Tokens one nodule occupies in the prompt, per lesion encoder architecture.
# vitq (QueryLesionEncoder) emits one token per attribute query (size, type, RADS);
# every other encoder emits a single token, which keeps B2/B3* byte-identical.
TOKENS_PER_LESION = {'gap': 1, 'center': 1, 'attn': 1, 'vitq': 3, 'vitq_s': 3, 'vitq_sm': 3, 'vitq_sz': 3, 'vitq_szm': 3}


def tokens_per_lesion(lesion_arch):
    if lesion_arch not in TOKENS_PER_LESION:
        raise ValueError(f'unknown lesion_arch {lesion_arch!r}; known: {sorted(TOKENS_PER_LESION)}')
    return TOKENS_PER_LESION[lesion_arch]


def lesion_token_names(max_region_size, n_tok=1):
    """<lesion{i}> names for every (region slot, lesion slot, token) triple.

    Slots are a fixed grid — region slot j in [0, max_region_size), lesion slot k in
    [0, MAX_LESION_PER_REGION) — so token ids stay static across samples regardless
    of how many nodules a patient actually has. Unused slots are zero-filled and
    never referenced by the prompt, the same way absent regions already work.

    Nodule slot s = j * MAX_LESION_PER_REGION + k owns tokens s * n_tok .. s * n_tok + n_tok - 1.
    n_tok = 1 gives exactly the original list.
    """
    return [f'<lesion{(j * MAX_LESION_PER_REGION + k) * n_tok + t}>'
            for j in range(max_region_size)
            for k in range(MAX_LESION_PER_REGION)
            for t in range(n_tok)]


def lesion_slot_text(slot, n_tok=1):
    """The prompt text for nodule slot `slot`: its n_tok consecutive <lesion*> tokens."""
    return ''.join(f'<lesion{slot * n_tok + t}>' for t in range(n_tok))
