# Handoff — Reg2RG on the NCKU ME lung-nodule dataset

Written 2026-08-17. Everything below was done on a Windows workstation; the training
runs on a server A100. This document is the state of play for whoever (or whatever)
picks it up next.

**Goal.** Fine-tune [Reg2RG](https://arxiv.org/abs/2411.15539) (region-guided CT report
generation, LLaMA2-7B + LoRA + RadFM 3D ViT) on NCKU's ME lung-nodule dataset, then
extend it so lesion-level detail survives the region encoder. That extension is the
intended research contribution — see [§7](#7-the-research-direction).

Fork: `git@github.com:W3iii/LN_Reg2RG.git` (upstream `zhi-xuan-chen/Reg2RG`).

---

## 1. Current status

| Stage | State |
|---|---|
| Source-data audit | done — [§2](#2-source-data-audit) |
| Lobe segmentation (TotalSegmentator, 5 lobes) | done, 1400/1400 |
| Reg2RG-format dataset export | done, 47.1 GB |
| Repo adapted to 5-lobe regions | done, 8 commits |
| Training (10 epochs, single A100) | done — loss 7.8 → 0.283, 4.3 h |
| Inference on val | done — 142 samples |
| Diagnosis of the trained model | **done, and it is the interesting result** — [§6](#6-what-the-trained-model-actually-does) |
| Exact per-lobe metrics | done — [§6](#6-what-the-trained-model-actually-does) |

---

## 2. Source-data audit

Full report: `d:\W3iii\NCKU\DataSet\label_match_report\README.md` (Windows box, not in
this repo) with per-patient/per-nodule CSVs.

### The ID mapping (deterministic, verified)

```
CHEST1001 … CHEST1908      ->  Excel patient ID 0001 … 0908   (folder number − 1000)
CHESTCT0909 … CHESTCT1800  ->  Excel patient ID 0909 … 1800   (folder number as-is)
```

`CHEST` and `CHESTCT` number ranges **overlap** (657 of 975 numbers appear under both
prefixes), so the prefix is part of the identity. Matching on the number alone silently
pairs the wrong patients.

### Agreement between the label sources

| Comparison | Result |
|---|---|
| Excel nodule count ↔ `bbox_ME/*.json` | 1632 / 1632 identical |
| Excel `TW_Lung_RADS` ↔ `check_nodule_cls…csv` `category` | 1630 / 1630 identical |
| ME mask count ↔ Excel | 1596 / 1632 (36 mismatches: 23 extra masks, 13 missing) |
| ME mask position ↔ doctor bbox | median 0.87 mm, 99.36% within 3 mm |

All 57 unpaired/far nodules live in the 23 extra-mask patients; the other 1596 patients
pair cleanly. Three patients' `Report` text belongs to a **different study entirely**
(0113 bone densitometry, 0622 chest X-ray, 0662 left-knee X-ray) — dropped.

### ⚠️ `bbox_index` is not contiguous

`check_nodule_cls_20241207_v1b4.csv` has gaps in `bbox_index` (patient 232 → `[0,2,3,4]`,
patient 368 → `[0,4]`). `bbox_ME/<pid>.json` carries the real indices in its `box_ids`
field — **use that**, not `enumerate()` position. Using position and falling back to
`k+1` collides with a real `nodule_index`, which duplicates `nodule_id` *and* assigns the
wrong `TW_Lung_RADS`. Affected 13/1400 patients before the fix.

---

## 3. ⚠️ Coordinate system — read before touching geometry

`ME_dataset/<folder>/npy/<folder>.npy` has already been resampled to **(0.8, 0.8, 1.0) mm**,
but `npy/series_metadata.txt` describes the **original** series:

| | metadata says | npy actually is |
|---|---|---|
| shape | 512, 512, 501 | 512, 512, **351** |
| spacing | 0.64258, 0.64258, **0.7** | 0.8, 0.8, **1.0** |

Doctor annotations (`bbox_ME`, `Db_Annotation_Txt_Bbox`) are in **original** DICOM
coordinates. To compare them with anything in npy space:

```python
me_y = 256 + (raw_y - 256) * orig_spacing_y / 0.8   # in-plane scaling is
me_x = 256 + (raw_x - 256) * orig_spacing_x / 0.8   # centre-preserving
me_z = raw_z * orig_spacing_z / 1.0
```

Derived target spacing came out at 0.800 ± 0.005 mm over 578 samples. Using the metadata
spacing directly gives a ~14 mm systematic offset that looks exactly like "the labels are
all wrong".

### The NIfTI affine, and the sign that matters

`data_prep/export_reg2rg.py` / `build_lobe_masks.py`:

```python
[[ 0.0, -sx,  0.0, -ox],
 [ -sy, 0.0,  0.0, -oy],
 [ 0.0, 0.0, -sz,   oz],
 [ 0.0, 0.0,  0.0, 1.0]]
```

`sy = sx = 0.8`, `sz = 1.0`; `oy = org[0] + 256*(sp[0]-0.8)`, `ox = org[1] + 256*(sp[1]-0.8)`,
`oz = org[2]`.

**`sz` must be negative.** Slice index increases toward the feet — established from the
lung cross-sectional area profile (`z=20` → 3,099 voxels at the apex; `z=298` → 15,483 at
the diaphragm). Writing `+sz` makes the affine determinant negative (a mirror), the model
sees a head-to-foot-flipped chest, and **RML almost vanishes** (CHEST1002 came out at 160
voxels). Nothing errors; the labels are just silently wrong. Verified with
`data_prep/check_lobe_sanity.py`: spine row index must be clearly larger than the lung
centroid row (posterior = high row), and lobe volume fractions must sit near
RUL .21 / RML .09 / RLL .24 / LUL .24 / LLL .22.

---

## 4. The dataset

### Pipeline (all scripts in `data_prep/`, run in this order)

| # | Script | Produces |
|---|---|---|
| 1 | `build_dataset.py` | patient manifest, `nodules.csv`, normalised `reports.jsonl`, 5-lobe `region_report.jsonl` |
| 2 | `build_lobe_masks.py` | `lobe/<folder>_lobe5.npz` — TotalSegmentator 5 lobes, label 1=RUL … 5=LLL |
| 3 | `assign_lobe.py` | fills `nodules.csv`'s `lobe` column; QC + validation CSVs |
| 4 | `match_nodule_sentence.py` | `nodule_sentence.csv` — nodule ↔ report-sentence pairing |
| 5 | `export_reg2rg.py` | `reg2rg_nifti/images/`, `reg2rg_nifti/masks/` |
| 6 | `make_csv_split.py` | `region_report_{train,val,test}.csv`, `splits.csv`, `nodule_metadata.csv` |

Helpers: `check_lobe_sanity.py` (orientation/volume QC), `validate_lobe_independent.py`
(segmentation vs report, non-circular — see the warning below).

Scripts 1/3/4/6 need only CPU. Script 2 needs the `lobeseg` conda env (see [§5](#5-environments)).

### Selection

1400 of 1632 patients kept. Excluded 232:

| Reason | n |
|---|---|
| Chinese Syngo CAD template report (no location information) | 214 |
| listed in `patient_ids_pass.txt` | 10 |
| no report text | 5 |
| report belongs to a different study | 3 |

Reports were normalised to strip boilerplate, slice references, and all
prior-comparison / clinical-history language (`stationary`, `persistent`,
`compared with previous`, `s/p …`) — a single CT cannot support those statements, and
50% of the reports contained them. Temporal *adjectives* are removed while the finding is
kept (`Persistent RUL GGN, 0.6cm` → `RUL GGN, 0.6cm`); whole sentences are dropped only
when they carry no finding.

### On-disk layout (matches upstream's conventions exactly, so the loader needs no changes)

```
/root/notebooks/groups/BME/reg2rg_nifti/          47.1 GB
  images/<folder>/<folder>/<folder>.nii.gz        int16, cropped to lung bbox + 24 vox
  masks/seg_<folder>/<region>.nii.gz              5 binary masks, same shape as the image
  region_report_{train,val,test}.csv              Volumename / Anatomy / Sentence
  nodule_metadata.csv                             5392 nodules — kept for §7
  splits.csv
```

`Volumename` **must** include `.nii.gz` (the loader keys on the NIfTI basename).
`Anatomy` must match `src/regions.py` verbatim. Rows with empty `Anatomy` become the
`'whole'` key, which upstream ignores — harmless, kept for later use.

### Numbers

| | train | val | test |
|---|---|---|---|
| patients | 1116 | 142 | 142 |
| region–text pairs | 5580 | 710 | 710 |
| `No significant finding.` | 55.7% | 56.2% | 56.9% |

Split is patient-level, stratified on max `TW_Lung_RADS` × nodule-count band; class
1/2/3/4 proportions match to within 0.01 across splits. Findings-only `Sentence` length:
median 84 chars, p90 154.

`nodules.csv` / `nodule_metadata.csv`: 5392 nodules (5382 from ME voxel masks, 10 from
doctor bboxes where ME had none). Lobe distribution RUL 1365 / RLL 1279 / LUL 1181 /
LLL 1014 / RML 553, zero unassigned.

### Validation

- **Lobe segmentation, 1400 patients**: 0 corrupt. Lung volume median 4.17 L (p5 2.83,
  p95 6.08). Lobe fractions .194/.087/.258/.238/.222 against reference
  .21/.09/.24/.24/.22. Only 9 patients (0.6%) deviate materially.
- **Segmentation vs radiologist, independent**: 231 cases with exactly one nodule *and* a
  report naming exactly one lobe, so no matching algorithm is involved.
  **Laterality 97.8% (226/231), exact lobe 92.6% (214/231).** Disagreements skew small
  (median 4.3 mm vs 5.2 mm for agreements) — perifissural ambiguity.
- **Nodule ↔ sentence matching**: 2424/5392 (45.0%) matched; 95.8% of those agree on both
  laterality and size; lobe agreement 95.1%. Match rate rises monotonically with
  malignancy class — **40.6% / 52.2% / 62.8% / 69.6%** for TW_Lung_RADS 1→4. That
  monotonicity is the evidence the matching is real rather than arbitrary.

> ⚠️ **Do not re-report the 98.0% figure** from `assign_lobe.py`'s built-in check. The
> matcher now uses the segmented lobe as its primary signal, so validating the lobe
> against the matched sentence is circular. 92.6% from
> `validate_lobe_independent.py` is the honest number.

---

## 5. Environments

**Server** — `/root/notebooks/automl/env` (Python 3.10, pre-existing shared venv).
`transformers==4.28.1`, torch ≥2.4. Installing `requirements-ncku.txt` here downgraded
numpy to 1.26.3 and scipy to 1.10.1, which breaks opencv/scikit-image for anything else
sharing that env — a clean env would have been better.

`requirements.txt` upstream is a full `pip freeze` of the authors' machine (400+ packages)
and no longer resolves: `open3d==0.14.1` has no wheel for Python ≥3.9. Use
`requirements-ncku.txt` (only what `src/` and `evaluation/` import).

**`transformers==4.28.1` must not be bumped.** RadFM's LLaMA wrapper and the
`<image*>`/`<region*>` token-id layout depend on it — `MyEmbedding.forward` indexes a
one-hot matrix by raw token id, which is very brittle.

**Windows workstation** — conda env `lobeseg` (Python 3.11) for TotalSegmentator, only
needed to regenerate lobe masks. Note that installing TotalSegmentator *overwrites* CUDA
torch with a CPU build; reinstall `torch==2.11.0 torchvision --index-url .../cu128`
afterwards.

**Weights** → `pretrain/` in the repo (git-ignored, ~14.6 GB):

```
pretrain/Llama-2-7b-chat-hf/                  gated: accept Meta's licence, then HF token
pretrain/Reg2RG/RadFM_vit3d.pth               required
pretrain/Reg2RG/RadFM_perceiver_fc.pth        required
pretrain/Reg2RG/pytorch_model.bin             authors' RadGenome checkpoint — worth having
pretrain/Reg2RG/RadBertClassifier.pth         skip: CT-RATE's 18 conditions, wrong labels here
```

`train_radgenome.sh` has no `--ckpt_path`, so training currently starts from the RadFM
weights, **not** from the authors' fine-tuned checkpoint. Wiring that up is a plausible
quick win given only 1116 training patients.

---

## 6. What the trained model actually does

Run: 10 epochs, 1390 steps, 4.3 h, single A100 80 GB, no DeepSpeed. Loss 7.8 → 0.283
(train_loss 0.4775). Inference on val: `results/ncku_lobe5_val_ckpt1390.csv`.

### It works, in the sense that it did not collapse

Output is fluent and structured — lobe name, size, character, recommendation:

```
The region 4 is left lower lobe: A LLL subpleural nodule, about 0.5cm.
                                 A LLL subpleural small lung nodule, benign?.
```

**Region identification is near-perfect.** Training shuffles the region order
(`random.shuffle(shuffled_areas)`), so the model must work out which lobe each region
token is — and it does, with 3 errors in 142 (CHESTCT1105, CHESTCT1652 left/right swap,
CHESTCT0915 duplicate RML). The vision encoder is genuinely reading lobe shape and
position.

### But it is not reading the nodules

Exact figures from `evaluation/analyze_region_predictions.py --result
results/ncku_lobe5_val_ckpt1390.csv` (run 2026-08-17, 142 val samples), replacing the
hand counts this section used to carry:

| lobe | GT | PRED | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| RUL | 69 | 4 | 3 | 1 | 66 | 0.750 | 0.043 | 0.082 |
| RML | 52 | 1 | 1 | 0 | 51 | 1.000 | 0.019 | 0.038 |
| RLL | 85 | 18 | 9 | 9 | 76 | 0.500 | 0.106 | 0.175 |
| LUL | 53 | 38 | 14 | 24 | 39 | 0.368 | 0.264 | 0.308 |
| LLL | 52 | 54 | 23 | 31 | 29 | 0.426 | 0.442 | 0.434 |
| **micro** | 311 | 115 | 50 | 65 | 261 | 0.435 | 0.161 | 0.235 |

RUL and RML recall round to zero. LLL and LUL precision sit well above their base rate
(52/710 ≈ 7% and 53/710 ≈ 7%), so the model is not purely reciting a language prior — it
has some real grounding on those two lobes specifically — but it collapses onto them at
the expense of the other three.

Findings per patient: GT mean 2.19 (median 2), PRED mean **0.81** (median 1). 87 of 142
patients have ≥2 GT findings; only **2** of those 87 get ≥2 predicted findings. Stated
size: GT median 5.0 mm (p90 8.0, max 55.0), PRED median 5.0 mm (p90 7.0, max **16.0**) —
the model never predicts anything close to a large lesion, even when one is present in
the ground truth.

Three further symptoms:

1. **Almost always exactly one finding**, confirmed above (PRED distribution: 0 findings
   × 29 patients, 1 × 111, 2 × 2).
2. **Heavily templated.** 115 finding sentences, only 67 distinct once the numbers are
   masked out; the eight most common templates account for 43 of them.
3. **Location uncorrelated with truth.** `CHEST1199` GT = RUL+RLL GGOs → predicted LUL.
   `CHESTCT1491` GT = RUL+RLL enlarging tumours, suspect lung cancer → predicted a 0.5 cm
   LLL nodule.

### Why — and this is the point

Measured over 40 patients / 75 nodules: a lobe's bounding box is median 183×156×167
voxels, so resizing it to the encoder's `(256, 256, 64)` scales y by 1.40×, x by 1.64×,
and z by **0.38×**. A median 4.7 mm nodule goes from 5.9×5.9×4.7 voxels to 7.8×9.6×**1.7**.
After patch embedding at (16,16,4) it occupies **0.117 of one patch — 94.7% of nodules
are smaller than a single token** (100% at (32,32,4)).

So the model can see a lobe (hundreds of tokens) but **cannot physically see a 5 mm
nodule**. Deprived of the signal, it falls back on the language prior and emits the most
common report in the training distribution: one small benign-looking nodule in a lower
left lobe.

This is not undertraining. It is a quantifiable, attributable failure of the region
representation — which makes it a clean motivation for §7.

---

## 7. The research direction

Design decision already taken: **regions stay fixed at the 5 lobes.** Region count is
then constant per patient, so inference can never exceed the trained region budget no
matter how many nodules a scan has (the data goes up to 67 in one patient; p99 = 19).
`max_region_size` is fixed at training time, so any per-nodule region scheme has a hard
inference ceiling. Reports also read closer to how radiologists write.

The cost is exactly the resolution loss in §6. Four ways to recover it, roughly in
increasing ambition:

1. **Anisotropic resize** — the damage is mostly the z 0.38×. Changing the target to
   e.g. `(224, 224, 128)` recovers much of it for free. **This is the baseline ablation**:
   it should show that tuning the resize is not sufficient.
2. **Lesion-token injection** (recommended primary contribution) — crop each nodule at
   native resolution (e.g. 64×64×32, no downsampling), encode with a small 3D encoder into
   **one token**, and append those tokens inside the owning lobe's region block:
   `<region><region0>…<region32><lesion_1><lesion_2>…</region>`. One token per nodule
   against 33 for a full region, so 26 nodules in a lobe costs 26 tokens. This extends
   Reg2RG's local-feature-decoupling idea from region level to lesion level.
3. **Perceiver-style fixed-K pooling** — K learned latent queries cross-attend to the N
   lesion tokens, always emitting K (say 8). Unbounded N, fixed budget.
4. **3D RoIAlign / feature pyramid** — skip the resize, RoIAlign each nodule on a
   native-resolution feature map, fuse coarse lobe + fine lesion. Most faithful, heaviest.

Plus an **auxiliary head** predicting per-lobe nodule count and max diameter from the lobe
feature, as a direct measure of whether local information survived.

`nodule_metadata.csv` already carries everything these need per nodule: bbox in npy
coordinates, centroid, lobe, voxel count, equivalent-sphere diameter, bbox long axis,
`TW_Lung_RADS`, and the matched report sentence.

> **Route 2 is specified in [`docs/LESION_TOKENS.md`](docs/LESION_TOKENS.md)** — token id ↔
> embedding row alignment (the part that breaks silently), sizing, slot ordering, the
> per-file change list, the data contract, and the ablation table. Read §1 of that document
> before adding any special token.
>
> Division of labour: **the repository-side changes are yours**; the dataset-side artifacts
> it depends on (a per-patient nodule instance mask, and the crop origin) are produced
> separately on the workstation that holds the raw data. `LESION_TOKENS.md` §5 is the
> contract between the two, including how to stub it while that is in flight.

---

## 8. Open items, in priority order

1. ~~Run the diagnostic and replace §6's hand counts with real numbers~~ — **done**, see
   §6. RML/RUL recall is near zero as predicted, but LLL/LUL precision sits *above* their
   lobe base rate rather than merely near it — the model has some real grounding on those
   two lobes, it just collapses onto them instead of ignoring the image entirely. That
   nuance matters for §7: a pure "fix the resolution loss" fix should be judged against
   this partial-grounding baseline, not a from-scratch language-prior one.

2. **Fix the residual text artifacts.** Normalisation in `data_prep/build_dataset.py`
   still leaks, and the model has learned to reproduce the damage — predictions contain
   `"A RLL tiny nodule without as comparing."`. **Must happen on the Windows box, not the
   server** — the raw source reports only exist there, and at least one artifact (CJK
   corruption) is upstream of `build_dataset.py` entirely. Root-cause analysis, concrete
   examples, and per-artifact fix direction: [`HANDOFF_TO_LOCAL.md`](HANDOFF_TO_LOCAL.md).

   | residual | occurrences in training targets |
   |---|---|
   | `without as comparing` | 30 |
   | `without interval.` | 11 |
   | CJK characters (encoding damage) | 100+ |
   | `Se/Im:` slice refs (variant `SLICE_REF` misses) | a few |
   | trailing slice numbers, e.g. `likely benign.,104,123).` | a few |

   Requires re-running `build_dataset.py` → `assign_lobe.py` → `match_nodule_sentence.py`
   → `make_csv_split.py`. Only the CSVs change; the NIfTI images and masks do not, so
   there is no need to re-export 47 GB. **Delete `src/Dataset/samples_*.pkl` afterwards.**

3. **Select a checkpoint properly.** `save_strategy="epoch"`, `save_total_limit=3`, so
   `checkpoint-1112/1251/1390` should exist. 1116 patients over 10 epochs will very
   likely have overfit — evaluate several on val rather than assuming the last is best.
   `trainer.save_state()` at the end of training saves optimiser state only; weights live
   in `outputs/<experiment>/checkpoint-<step>/pytorch_model.bin`.

4. **Turn shuffling on for the referring metric.** `radgenome_dataset_test.py:258` has
   `# random.shuffle(shuffled_areas)` commented out, so inference always presents regions
   in `REGIONS` order. `region_pred_acc.py` will therefore score near 100% for free —
   that is not evidence of grounding.

5. **Make `evaluation/` usable.** All five upstream scripts hardcode paths under
   `/home/chenzhixuan/...` and take no arguments, and `region_pred_acc.py` carries a sixth
   copy of `REGIONS` (still the 10 RadGenome regions, with `'trachea'` spelled differently
   again). Point it at `src/regions.py`. Intended order:
   `test → parse_report_to_region.py → hf_nlg_evaluation_region.py` for per-region
   metrics, `rm_region_text.py → hf_nlg_evaluation.py` for whole-report metrics.

6. **Then implement §7.**

---

## 9. Traps already paid for — do not re-discover these

| Symptom | Cause | Fix (commit) |
|---|---|---|
| `open3d==0.14.1` won't install | upstream `requirements.txt` is a full pip freeze | use `requirements-ncku.txt` (`a8bcae1`) |
| `ImportError: cannot import name 'log' from torch.distributed.elastic.agent.server.api` | `deepspeed 0.12.6` vs torch ≥2.4 | DeepSpeed made optional; `deepspeed_config=""` (`a8bcae1`) |
| `HFValidationError: Repo id must be in the form …` | the local model **directory does not exist** — transformers falls through to the Hub and validates the path as a repo id | create/populate the path |
| `KeyError: 'right lower lobe'` mid-training, after model load | `REGIONS` was duplicated in **five** files; only the two Dataset copies were updated, so the collator still bucketed by the stale list | `src/regions.py`, single source (`9c8fe5f`) |
| `RuntimeError: Expected to mark a variable ready only once` at step 8 | torchrun sets `local_rank`, so HF Trainer wraps in DDP even for one process; DDP rejects reentrant gradient checkpointing. Surfaces only at the first gradient sync, i.e. after `gradient_accumulation_steps` — so training appears to start fine | single GPU launches with plain `python` (`4ded813`) |
| loss is NaN from step 1 | `lang_model.half()` (fp16) under HF Trainer's **bf16** autocast. Upstream got away with it because DeepSpeed reconciled precision. Three dtypes were live on the embedding path | one dtype (bf16 on Ampere+) for LLM and `MyEmbedding`; `vision_region_embedding` follows `self.weight.dtype` (`76f33d1`) |
| `RuntimeError: expected scalar type Float but found BFloat16` at inference | `generate()` runs under bare `no_grad` with no autocast, so fp32 volumes hit bf16 ViT weights. Training never noticed | cast inputs at `MyEmbedding.forward` entry (`cf87535`) |
| second checkpoint's inference produces no new rows, no warning | `test_radgenome.py` reads existing `result_path`, treats `AccNum` as done, appends. Reusing a filename skips everything | tag `result_path` with split + step (`d1cd963`) |
| loader silently trains on someone else's data | upstream ships a 45 MB `train_samples.pkl` of RadGenome samples, and the cache filename was hardcoded | deleted; cache name derives from the report CSV (`da5645c`) |
| lobe masks look plausible but RML nearly empty | `+sz` in the affine → negative determinant → mirrored volume | `-sz`; see [§3](#3--coordinate-system--read-before-touching-geometry) (`data_prep/build_lobe_masks.py`) |
| GPU dies partway through a long batch job, then every remaining case fails in ~8 s | one-off `CUDA error: unknown error`; a broken CUDA context cannot recover in-process | abort after 5 consecutive failures, restart in a fresh process — `data_prep/build_lobe_masks.py` + `run_lobe_batch.ps1` |

---

## 10. Commands

```bash
# Train (prints "[preflight]"-style banners for DeepSpeed and launcher choice)
cd scripts && bash train_radgenome.sh ncku_a100

# Inference — edit ckpt_step and split in configs/test_radgenome/ncku_a100.sh first,
# and delete the old result CSV if you are reusing a filename
cd scripts && bash test_radgenome.sh ncku_a100

# Diagnose
python evaluation/analyze_region_predictions.py --result results/<file>.csv
```

`WANDB_MODE=offline` is recommended — metrics stay local, which suits clinical data, and
the curves are still there for ablation comparisons.

Config knobs live in `configs/{train,test}_radgenome/ncku_a100.sh`. `DATA_ROOT` is
absolute; `CKPT_ROOT` and all outputs derive from `REPO_ROOT`, which the config resolves
from its own location — so the checkout can move without edits.
