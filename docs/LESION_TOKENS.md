# Lesion tokens — implementation spec

Extends Reg2RG so lesion-level detail survives the region encoder, while keeping the
region set fixed at the 5 pulmonary lobes.

**Why:** see `../HANDOFF.md` §6. Resizing a lobe to the encoder's `(256, 256, 64)` scales
z by 0.38×; a median 4.7 mm nodule ends up at 0.117 of one patch token, and 94.7% of
nodules are smaller than a single token. The trained baseline identifies lobes almost
perfectly but places findings by language prior, because the nodules are not in the
representation at all.

**Idea:** crop each nodule at native resolution, encode it to **one token**, and append
those tokens to the owning lobe's region block. One token per nodule against 33 for a
full region, so a lobe with 26 nodules costs 26 tokens — the region budget never moves,
and inference has no ceiling on nodule count.

Scope of this document: **changes inside this repository.** The dataset-side artifacts it
depends on are specified in [§5](#5-data-contract) and are being produced separately on
the workstation that holds the raw data.

---

## 1. ⚠️ Token id ↔ embedding row alignment

This is the part that breaks silently. `MyEmbedding.forward` does not look tokens up in a
table — it builds a per-sample matrix and selects rows with a one-hot product:

```python
embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)
embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)
embedding_weight = torch.cat([embedding_weight, image_embedding, vision_region_embedding], dim=1)
text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(...)
out_put = torch.matmul(text_input, embedding_weight)
```

So **token id is a row index**, and the current layout only works because the tokenizer
adds special tokens in exactly the order the rows are concatenated:

| token id | token | row comes from |
|---|---|---|
| 0 – 31999 | word pieces | `self.weight` |
| 32000 – 32001 | `<image>` `</image>` | `image_token_weight` (2 learned rows) |
| 32002 – 32003 | `<region>` `</region>` | `region_token_weight` (2 learned rows) |
| 32004 – 32035 | `<image0>` … `<image31>` | `image_embedding` (32, per sample) |
| 32036 – 32036+33·R−1 | `<region0>` … | `vision_region_embedding` (33·R, per sample) |

**Rules for adding lesion tokens:**

1. Append `<lesion{n}>` **at the end** of the `additional_special_tokens` list, after every
   `<region*>` entry.
2. Concatenate `lesion_embedding` **at the end** of `embedding_weight`, after
   `vision_region_embedding`.
3. **Do not add `<lesion>` / `</lesion>` delimiter markers.** They would need learned rows
   next to `region_token_weight`, i.e. concatenated *before* the per-sample blocks, which
   shifts every `<image*>` and `<region*>` id. Use plain words in the prompt for framing
   instead (`"… with lesions <lesion0><lesion1>."`).

Nothing existing moves if those three rules hold.

`label[label >= self.voc_size] = -100` (`voc_size = 32000`) already excludes every special
token from the loss, so lesion tokens are masked automatically. No change needed there.

### The three tokenizers must stay in sync

The special-token list is built independently in:

- `src/Dataset/radgenome_dataset_train.py`
- `src/Dataset/radgenome_dataset_test.py`
- `src/Model/Reg2RG.py`

This is the same failure mode as the five copies of `REGIONS` (`HANDOFF.md` §9): a mismatch
does not raise, it produces wrong row lookups. Put `MAX_LESION_PER_REGION` and a
`lesion_token_names()` helper in `src/regions.py` (or a new `src/tokens.py`) and have all
three import it.

---

## 2. Sizing

Nodules per (patient, lobe): median 1, p90 3, p99 7, max 26.

```python
MAX_LESION_PER_REGION = 8          # covers p99
# grid = max_region_size (5) × 8 = 40 lesion slots
# region slot j, lesion k  ->  <lesion{j * MAX_LESION_PER_REGION + k}>
```

Slots are allocated as a fixed grid so ids stay static; unused slots are zero-filled and
never referenced by the prompt, exactly as absent regions already work.

**Overflow (>8 in one lobe):** keep the top 8 by `tw_lung_rads` descending, then
`eq_diam_mm` descending. Under the fixed-lobe design the lobe's report text already
describes every nodule, so **nothing is dropped from the target** — only the extra
lesions' visual features are missing. That is a much softer failure than a per-nodule
region scheme, where exceeding the trained region count means untrained embeddings.

**Cost.** A median patient has 3 nodules, so 3 extra prompt tokens. Lesion crops at
64×64×32 are 131k voxels each against 12.6M for one resized region volume — roughly
**0.6% overhead for a median patient**. This cheapness is the whole argument for putting
lesions in tokens rather than in regions.

---

## 3. ⚠️ Slot ordering

`region2area` is shuffled per sample in training, and the prompt is built with
`region_padding_tokens[i]` for **slot** `i`, not for a named region. Lesion tokens must be
indexed by the same slot `i`.

Getting this wrong attaches a lobe's lesions to a different lobe's region token, and
nothing errors — the loss just stays high and the model looks like it "didn't learn".
Assert it: for every sample, the lesions written into slot `i` must belong to
`region2area[i]`.

---

## 4. Changes, file by file

### `src/regions.py` (or new `src/tokens.py`)

```python
MAX_LESION_PER_REGION = 8

def lesion_token_names(max_region_size):
    return [f'<lesion{j * MAX_LESION_PER_REGION + k}>'
            for j in range(max_region_size)
            for k in range(MAX_LESION_PER_REGION)]
```

### `src/Model/my_embedding_layer.py`

New module — deliberately **not** the ViT3D, whose patch grid is tied to `(256, 256, 64)`:

```python
class LesionEncoder(nn.Module):
    """One token per nodule, from a native-resolution crop.

    No resize: the crop is already ~64x64x32 at (0.8, 0.8, 1.0) mm, which is what the
    lobe-level pathway destroys. A few strided 3D convs then global pooling keeps this
    cheap enough to run for every nodule in the batch.
    """
    def __init__(self, in_ch=1, dim=4096, width=32):
        ...  # Conv3d(stride 2) x3-4 + norm + activation -> AdaptiveAvgPool3d(1) -> Linear(-> dim)
```

`MyEmbedding.__init__`: instantiate `self.lesion_encoder`.

`MyEmbedding.forward(self, vision_x, mask_x, text_input, region2areas, lesion_x, lesion_slots)`:

```python
# lesion_x     : (B, L, 1, 64, 64, 32)   L = max lesions in this batch, zero-padded
# lesion_slots : (B, L) int              flat slot index j*MAX_LESION_PER_REGION+k, -1 = padding

lesion_embedding = torch.zeros(
    (B, max_region * MAX_LESION_PER_REGION, self.embedding_dim),
    device=text_input.device, dtype=self.weight.dtype)      # dtype must follow the module
for b in range(B):
    valid = lesion_slots[b] >= 0
    if valid.any():
        tok = self.lesion_encoder(lesion_x[b][valid])       # (n, dim)
        lesion_embedding[b, lesion_slots[b][valid]] = tok

embedding_weight = torch.cat(
    [embedding_weight, image_embedding, vision_region_embedding, lesion_embedding], dim=1)
```

Cast `lesion_x` to `self.weight.dtype` at the top of `forward`, next to the existing
`vision_x` / `mask_x` casts (`HANDOFF.md` §9 — inference has no autocast).

### `src/Model/Reg2RG.py`

Extend the special-token list (rule 1 above); thread `lesion_x` / `lesion_slots` through
`forward` and `generate` into `self.embedding_layer(...)`.

### `src/Dataset/radgenome_dataset_{train,test}.py`

1. Extend the special-token list identically.
2. Read the nodule instance mask (§5), crop each nodule at native resolution, and return
   `lesion_x` + `lesion_slots` built from the *shuffled* `region2area` order.
3. `text_add_region_tokens`: emit that region's lesion tokens inside its clause, e.g.
   `"The region 0 is <region>…</region> with lesions <lesion0><lesion1>. "`, and nothing
   when the lobe has no nodules.
4. Target text is **unchanged** — still the lobe report. This is a pure input-side change,
   so `region_report_*.csv` and the labels are untouched.

### `src/train_radgenome.py` and `src/test_radgenome.py` (`DataCollator`)

Pad `lesion_x` to the batch maximum `L` and `lesion_slots` with `-1`. Carry the reference
dtype as the existing region padding now does.

---

## 5. Data contract

Produced on the workstation, not here. Two additions to
`/root/notebooks/groups/BME/reg2rg_nifti/`:

```
masks/seg_<folder>/nodules.nii.gz     uint8, 0 = background, 1..N = nodule_id
```

**Why an instance mask rather than coordinates:** `data_prep/export_reg2rg.py` crops each
volume to the lung bounding box plus 8 voxels and **does not record the crop origin**,
while `nodule_metadata.csv` holds coordinates in the *uncropped* npy frame. Those
coordinates therefore cannot index the exported NIfTI. An instance mask in the same
cropped frame as `images/` and `masks/` removes the bookkeeping entirely — lesion `k` is
`mask == k` — and stays correct if the crop is ever changed.

`nodule_metadata.csv` also gains the crop origin (`crop_y0`, `crop_x0`, `crop_z0`) so
coordinates remain usable, and it already carries what the auxiliary head needs:
`eq_diam_mm`, `bbox_long_mm`, `tw_lung_rads`, `lobe`, `nodule_id`.

Until those land, develop against a stub that reads bboxes from `nodule_metadata.csv` and
subtracts a recomputed crop origin. Do not commit the stub.

---

## 6. Auxiliary head (optional, do after §4 trains)

A small MLP on each region's 33 tokens predicting that lobe's **nodule count** and **max
equivalent diameter**, supervised from `nodule_metadata.csv`.

Two reasons to want it: it pushes local information into the region representation, and it
is a *direct* measure of whether that information survived — far more legible than a BLEU
delta.

---

## 7. Ablations

| # | Setup | Question it answers |
|---|---|---|
| B0 | current baseline | done — see `HANDOFF.md` §6 |
| B1 | resize `(256,256,64)` → `(224,224,128)`, no lesion tokens | is the z-axis crush the whole story? Expected: helps, not enough |
| B2 | B0 + lesion tokens | **the contribution** |
| B3 | B2 + auxiliary count/diameter head | does explicit supervision help retention? |
| B4 | B2 with fixed-K Perceiver pooling over lesions | bounded token budget for unbounded nodule count |

**Primary metric:** per-lobe finding precision/recall from
`evaluation/analyze_region_predictions.py`, with **RML and RUL recall** as the headline —
those are the lobes the baseline essentially never predicts. A clean result is RML/RUL
recall rising while the normal-region behaviour stays put.

**Secondary:** NLG metrics computed over findings-only regions. Do not report NLG averaged
across all regions: 56% of targets are `No significant finding.`, so that average mostly
measures how often the model says nothing.

B1 matters for the write-up. Without it a reviewer will ask whether the resize schedule
alone would have closed the gap.
