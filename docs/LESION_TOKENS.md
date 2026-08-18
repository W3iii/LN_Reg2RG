# Lesion tokens — implementation spec

Extends Reg2RG so lesion-level detail survives the region encoder, while keeping the
region set fixed at the 5 pulmonary lobes.

**Why:** see `../HANDOFF.md` §6, and §9 below for the measurements that replaced the
original reasoning. Briefly: every lobe-level representation that was probed — ViT patch
tokens, perceiver latents, what the LLM receives, and model-free voxel statistics at
native resolution — lands between 0.54 and 0.61 AUC for "does this lobe contain a
nodule", statistically indistinguishable from knowing only how big the lobe is. Cropped
to the lesion, a *single scalar* (mean HU) reaches 0.837. A 5 mm nodule is roughly 65
voxels inside a 4 M voxel lobe, so no pooled summary of a lobe can hold it. That gap,
0.6 against 0.84, is what this design is trying to close.

Note the original explanation — that the z 0.38× resize is what destroys the signal —
did **not** survive measurement: native-resolution voxel statistics score the same as
post-resize ones (delta −0.002, CI [−0.035, +0.033]). The resize is not the mechanism;
the scale mismatch is.

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

## 5. Data contract — ✅ delivered

Produced on the workstation by `data_prep/export_nodule_masks.py` and
`make_csv_split.py`. In `/root/notebooks/groups/BME/reg2rg_nifti/`:

```
masks/seg_<folder>/nodules.nii.gz     uint8, 0 = background, 1..N = nodule_id
crop_offsets.csv                      folder, crop_y0/x0/z0, crop_h/w/d
nodule_metadata.csv                   38 cols, incl. crop origin + cropped-frame boxes
```

1400 patients, 5392 nodules, no errors. Same shape and affine as `images/` and the lobe
masks — verified on CHEST1001: 54 voxels for nodule 1, matching its original
`nodule_size`, and 100% of them inside that patient's LUL lobe mask, which agrees with
`nodules.csv`. **No nodule falls outside the lung crop.**

> **The mask's integer labels *are* `nodule_id`**, so `nodules.nii.gz` joins directly to
> `nodule_metadata.csv` on `(Volumename, nodule_id)`. That was the point of labelling by
> id rather than by connected-component order, and it is what lets overflow ranking use
> `tw_lung_rads` / `eq_diam_mm` as specified in §2 instead of voxel count.

186 voxels across 3 patients are contested by two overlapping bboxes; they go to the
nearer bbox centre rather than to whichever instance was written last. Nodules whose only
label is a doctor bbox (10 of 5392, no voxel mask) fill their box but claim only unowned
voxels — filling unconditionally erased a real segmented nodule that one such box
enclosed.

`make_csv_split.py` merges the crop origin into `nodule_metadata.csv` and derives
cropped-frame columns, so boxes are also usable directly against the exported NIfTI:

```
crop_y0 crop_x0 crop_z0 crop_h crop_w crop_d          crop origin and size
y0_crop … z1_crop, cy_crop cx_crop cz_crop           bbox and centroid, cropped frame
eq_diam_mm bbox_long_mm size_vox tw_lung_rads lobe   what the auxiliary head needs
```

No missing or negative cropped coordinates. Round-trip checked: slicing the instance mask
with a nodule's `*_crop` bbox recovers exactly that nodule's voxels.

### Which of the two the training path uses, and why

Both are correct. `src/lesion_utils.py` reads the **cropped-frame boxes**, not the
instance mask, for one reason: throughput. The boxes are a single CSV read at dataset
construction, while the mask is another NIfTI of image size to open per sample — roughly
1116 × 10 extra volume loads over a training run, on NFS, against a 4.5 h baseline.

The mask remains the better artifact wherever that cost does not apply, and the note in
the original contract still holds: it needs no arithmetic and stays correct if
`export_reg2rg.py`'s crop ever changes, which would silently invalidate every stored
coordinate. Use it for the auxiliary head, for verification, and for anything needing
voxel-level lesion extent rather than a window.

If the crop is ever changed, `lesion_utils` must be repointed at the mask or the
coordinates regenerated — there is no third option, and nothing will raise.

### Independent verification of the box path

Checked on the server against the exported volumes, 2026-08-18:

| check | result |
|---|---|
| rows with a missing crop column | 0 of 5392 |
| `crop_h/w/d` vs exported image shape | 25 / 25 match |
| split coverage | train 4287, val 539, test 566 |
| `region` values vs `src/regions.py` | verbatim match |
| box size, median | 6 × 7 × 4 voxels |
| boxes exceeding the 64×64×32 crop | 6 of 5392 |
| nodules per (patient, lobe) | median 1, p90 3, p99 7, max 26 |
| (patient, lobe) pairs above 8 nodules | 15 of 3144 |

The last two rows confirm §2's sizing independently. The boxes were also checked
functionally rather than structurally: nodule boxes separate from **random same-size boxes
drawn in the same lobe** at **AUC 0.837 on mean HU alone**, which a wrong mapping could
not produce. That number is also the oracle upper bound quoted in §9.

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

---

## 9. What the probes actually measured

Run on `checkpoint-1390`, val split, 142 patients / 710 lobes, 44.5% of lobes containing
at least one nodule. Probes are cross-validated with GroupKFold on patient, and every CI
bootstraps **patients** rather than rows — the five lobes of one patient are not
independent, and resampling rows would make each interval about √5 too narrow.

| representation | AUC | 95% CI |
|---|---|---|
| prior | 0.500 | — |
| `mask_token` | 0.542 | [0.503, 0.585] |
| patch tokens, MIL top-k | 0.560 | [0.516, 0.605] |
| **`lobe_voxels` (lobe size alone)** | **0.561** | [0.523, 0.601] |
| patch tokens, MIL max | 0.587 | [0.546, 0.627] |
| voxel stats, native resolution | 0.597 | [0.558, 0.638] |
| **`post_fc` (what the LLM sees)** | 0.598 | [0.558, 0.639] |
| voxel stats, after resize | 0.599 | [0.560, 0.638] |
| `post_perceiver` | 0.590 | [0.550, 0.630] |
| `pre_perceiver` (pooled patches) | 0.611 | [0.571, 0.653] |
| **oracle: lesion box vs random box, mean HU** | **0.837** | — |

Paired comparisons, same patients resampled for both arms:

| comparison | delta | verdict |
|---|---|---|
| `post_fc` − `lobe_voxels` | +0.038 [−0.009, +0.084] | not resolved |
| `pre_perceiver` − `post_perceiver` | +0.022 [−0.002, +0.045] | not resolved |
| voxel native − voxel resized | −0.002 [−0.035, +0.033] | no difference |
| `pre_perceiver` − `lobe_voxels` | +0.051 [+0.004, +0.097] | resolved, barely |

Nodule **count** and **max diameter** are at Spearman ρ ≤ 0.11 everywhere, which is the
measured version of §6's observation that predicted sizes never exceed 16 mm while the
ground truth reaches 55 mm.

### The same probes on the train split

The val CIs above are wide enough to leave the two comparisons that matter unresolved, so
the probes were rerun on train — 5580 lobes from 1116 patients, 45.0% positive, which
narrows every interval by roughly 2.7×.

| representation | train AUC | 95% CI | val AUC |
|---|---|---|---|
| `lobe_voxels` | 0.562 | [0.547, 0.576] | 0.561 |
| `pre_perceiver` | 0.576 | [0.561, 0.591] | 0.611 |
| `mask_token` | 0.592 | [0.577, 0.606] | 0.542 |
| `post_fc` | 0.605 | [0.591, 0.620] | 0.598 |
| `post_perceiver` | 0.606 | [0.592, 0.621] | 0.590 |

| comparison | delta | verdict |
|---|---|---|
| `post_fc` − `lobe_voxels` | **+0.044 [+0.026, +0.060]** | resolved |
| `pre_perceiver` − `post_perceiver` | **−0.030 [−0.042, −0.018]** | resolved, *negative* |
| `pre_perceiver` − `lobe_voxels` | +0.015 [−0.005, +0.032] | not resolved |

Nodule count reaches ρ = 0.206 at `post_fc`, against 0.002 for lobe size.

These are training patients, so the encoder has seen them — but the point estimates match
val almost exactly (`post_fc` 0.605 vs 0.598, `lobe_voxels` 0.562 vs 0.561). Memorisation
would have made train visibly higher. It did not, so the train run is best read as the
same measurement with enough power to resolve it.

### What died, and what replaced it

- **the 32-latent resampler is the bottleneck** — wrong, and backwards. `post_perceiver`
  beats `pre_perceiver` by 0.030 with the CI excluding zero. The learned resampler
  *improves* the representation relative to fixed mean/max pooling, which is unsurprising
  once stated: it is trained and the pooling is not.
- **the resize destroys it** — no. Native-resolution voxel statistics score the same as
  post-resize ones (−0.002, CI [−0.035, +0.033]). B1 is therefore not the load-bearing
  control it was designed to be.
- **the probe's pooling was hiding it** — no. MIL over unpooled patch tokens does not
  beat mean/max pooling (0.587 vs 0.611 on val).
- **there is no lesion information at all** — also wrong, and this is the correction that
  matters most. `post_fc` does beat lobe size, resolved on train. The representation the
  LLM receives is not empty.

What survives: the information is present but roughly an order of magnitude too weak.
`post_fc` reaches 0.605 with 8192 dimensions; a single scalar taken at the lesion reaches
0.837. The failure is one of scale, not of encoding — and that is exactly what a lesion
crop addresses.

**Caveat that must stay attached to the 0.837.** It uses ground-truth nodule locations.
It is an oracle-localisation upper bound, not a deployable number, and the ablation table
in §7 still owes an arm that says where lesion boxes come from at inference time.

Reproduce with `evaluation/extract_region_features.py`, `extract_voxel_features.py`,
`probe_local_info.py`, and `probe_mil.py`.

---

## 8. Implementation status (repo side)

§1–§4 are implemented: `regions.py` (`MAX_LESION_PER_REGION`, `lesion_token_names`),
`lesion_utils.py` (crop/assignment math, shared by both Dataset classes),
`LesionEncoder` + `MyEmbedding.forward` threading, `Reg2RG.py`'s special-token list and
`forward`/`generate` signatures, both `radgenome_dataset_{train,test}.py`, and the
`train_radgenome.py` / `test_radgenome.py` collators.

**Lesion tokens are off by default and must be switched on explicitly.** The dataset takes
`nodule_metadata=None` unless a path is passed, so B0 and B2 differ by exactly one
argument from one checkout:

```bash
# B0 baseline (also what an in-flight seed sweep keeps producing)
bash train_radgenome.sh ncku_a100
# B2 lesion arm
nodule_metadata=$DATA_ROOT/nodule_metadata.csv bash train_radgenome.sh ncku_a100
```

Defaulting to off is not just tidiness: the seed sweep runs its seeds sequentially in
fresh processes, so a default that read the metadata would have flipped seeds 2 and 3 to
a different arm mid-sweep and quietly destroyed the noise floor.

Inference must use the same setting as training. A lesion-trained checkpoint evaluated
without it leaves every `<lesion*>` id pointing at a zero row, and nothing raises.

Verified on real volumes: 3144 (volume, region) pairs indexed, the 8-slot cap respected,
every crop `(64, 64, 32)`, and 100% of crop centres denser than the surrounding
parenchyma (median +87 HU, p10 +29). Shape and scatter-index behaviour were checked
separately against `MyEmbedding` with synthetic tensors — including a zero-length batch,
mixed lesion counts, `lesion_token_names(5)` as an exact prefix of `lesion_token_names(10)`,
and `text_add_region_tokens` producing byte-identical output when no lesions are present.

§2's ranking (`tw_lung_rads` desc, then `eq_diam_mm` desc) is implemented as specified;
the earlier voxel-count stand-in is gone now that the metadata is the source.

**Not implemented:** §6 (auxiliary count/diameter head) — explicitly deferred until after
§4 trains — and §7's B1/B3/B4 ablation arms, which are training runs, not repo changes.
