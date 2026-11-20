# V2 pipeline: split C + GT v2 + vitq lesion encoder (2026-11)

What changed against the B0/B2/B3* runs, and how to run it. Everything here is opt-in:
the old configs (`ncku_a100`) and the old files (`region_report_{train,val,test}.csv`,
`nodule_metadata.csv`) still reproduce the earlier arms byte for byte.

## Data (built on the workstation, `data_prep/ncku_v2/`)

New files in the data folder (sync these to the server). The workstation folder was renamed
`reg2rg_nifti` -> `reg2rg_nifti_ME_LDCT`; the v2 configs default to
`DATA_ROOT=/root/notebooks/groups/BME/reg2rg_nifti_ME_LDCT` (override with `DATA_ROOT=...`):

| file | what |
|---|---|
| `splits_merged.csv` | split C: train = ME 1,093, val = ME 157, test_me = ME 150, test_ldct = LDCT 566. No leakage vs the DETECT detector split. |
| `region_report_v2_{train,val,test_me,test_ldct}.csv` | GT v2: 5 lobe rows + 1 global row per patient, plus `nodule_ids`, `loss_mask`, `source`, `split` |
| `nodule_metadata_v2.csv` | ME + LDCT, 6,717 nodules, crop-frame boxes, `source`, `match_tier`, `flip_yx`; LDCT has no TW_Lung_RADS |
| `fill_table_v2.csv` | per nodule `size_cm_eq` (= `size_placeholder.format_size(eq_diam_mm)`), for reference |
| `crop_offsets_ldct.csv` | LDCT crop origin per patient |
| `images/CHESTCT_Test*`, `masks/seg_CHESTCT_Test*` | LDCT volumes, same crop as ME (lobe union ± 8 voxels), all P,L,I voxel order (the one prone scan, Test0374, flipped on y,x) |

GT v2 lobe text:
- a size that belongs to exactly one nodule is `[SIZE#<nodule_id>]`, any other size `[SIZE]`
- `nodule_ids` = the lobe's nodules, mask size descending (= lesion slot order)
- `loss_mask = 0` = the lobe text describes a lesion with no mask

## Model changes

- `QueryLesionEncoder` (`lesion_arch=vitq`, `src/Model/my_embedding_layer.py`): the
  PooledLesionEncoder conv trunk (16×16×8 grid of the native 64×64×32 crop), 2 transformer
  layers, then 3 learnable queries that cross-attend to the grid (Q = queries, K/V = cells)
  with a learned centre-distance penalty per query and head (init = the `center` Gaussian)
  → 3 tokens per nodule. ~5.5 M parameters.
  The first version (2026-11-17: plain ViT stem, 4 layers, no centre prior) did not learn on
  the real data: after 14 epochs val size rho 0.13–0.19, RADS kappa 0, every type predicted
  GGO, train loss flat. On a synthetic size task the current version reaches rho 0.99 in 50
  steps, like `attn`.
  First v2 pretraining (2026-11-18, val): attn rho 0.771 / kappa 0.635 / type bacc 0.481;
  vitq rho 0.659 / kappa 0.660 / type bacc 0.246 (type token collapsed to GGO). The queries
  now self-attend before reading the grid, so the type token can use what size / RADS learn.
  part-solid is 0 for both encoders (very few labels).
- Pretraining (`evaluation/train_lesion_encoder.py --arch vitq`): token 0 → size head,
  token 1 → type head (GGO / solid / part-solid / calcified), token 2 → RADS head. The heads
  are not part of the report model; only the encoder weights are loaded.
- `regions.TOKENS_PER_LESION`: nodule slot s owns `<lesion{3s}>..<lesion{3s+2}>` for vitq, one
  token for the other encoders (unchanged ids).

## Prompt and target (v2 is detected from the CSV's `nodule_ids` / `loss_mask` columns)

```
prompt  The region 2 is <region>…</region> with lesions #1 <lesion48><lesion49><lesion50>, #2 <lesion51><lesion52><lesion53>. …
target  The region 2 is right upper lobe: Small RUL lung nodule, [SIZE#1]. A [SIZE#2] lung nodule in the RUL …
```
- regions are still shuffled; the model still names each region's lobe
- `[SIZE#<nodule_id>]` is rewritten to the lobe-local `#k` shown in the prompt
- lobes with `loss_mask = 0`: the text stays in the input, its tokens get label −100 (the
  "The region i is <lobe>:" head is still trained)
- without `nodule_metadata` (B0-v2 arm) `[SIZE#k]` becomes `[SIZE]`
- inference fills `[SIZE#k]` with the k-th nodule's mask size and bare `[SIZE]` largest-first
  (`size_placeholder.fill_report_v2`); the GT is filled the same way

Token budget: prompt + target is ≤ 880 tokens for every patient (max_seq 2048).

## Run order (server)

```bash
# 1. lesion encoder pretraining on ME train/val (builds the crop cache + type labels on first run)
export DATA_ROOT=/root/notebooks/groups/BME/reg2rg_nifti_ME_LDCT
ARCHS=vitq TYPE_LOSSES=ce_mil META=$DATA_ROOT/nodule_metadata_v2.csv bash scripts/train_lesion_encoder.sh
#    -> outputs/lesion_enc_vitq_ce_mil_v2_seed0/best.pt
#    optional baseline on the same split: ARCHS=attn ... (-> lesion_enc_attn_ce_mil_v2_seed0)

# 2. report model (frozen vitq encoder, 3 tokens per nodule)
cd scripts && seed=42 bash train_radgenome.sh ncku_a100_v2
#    B0-v2:   experiment_name=Reg2RG_v2_B0 nodule_metadata= bash train_radgenome.sh ncku_a100_v2

# 3. inference: choose the checkpoint on val, then each test set once
split=val       ckpt_step=1360 bash test_radgenome.sh ncku_a100_v2
split=test_me   ckpt_step=<best> bash test_radgenome.sh ncku_a100_v2
split=test_ldct ckpt_step=<best> bash test_radgenome.sh ncku_a100_v2
```
Result CSV columns: `GT_combined_report` / `Pred_combined_report` (filled with mask sizes),
`GT_raw_report` / `Pred_raw_report` (`[SIZE#k]`), `n_size_slots`, `n_unfilled`,
`Masked_regions` (lobes to leave out of scoring).

After syncing new CSVs, delete `src/Dataset/samples_*.pkl` on the server (sample lists are
cached per CSV name; the v2 names are new, so stale lists are only a risk for the old names).

## Checked on CPU before pushing (no GPU here)
- vitq: (N,1,64,64,32) → (N,3,4096); gradients reach the queries; checkpoint written by
  train_lesion_encoder.py loads into MyEmbedding's encoder with identical outputs
- MyEmbedding places each nodule's 3 tokens at the right embedding rows (vitq and attn)
- dataset on real files (CHEST1073, Test0374): `[SIZE#2]`/`[SIZE#1]` remapped to the prompt's
  #1/#2, loss-masked lobe text excluded from the labels, crops centred on the nodules
- all 1,966 patients: every `[SIZE#k]` points at an existing slot, max 880 tokens
- Reg2RG forward + backward with a 1-layer random LLaMA (transformers 4.28.1, peft 0.7.1,
  monai 1.3.0), generate, fill
- train_lesion_encoder.py `--arch vitq --meta nodule_metadata_v2.csv` end to end on a mini set
- v2 type labels: train 1,147 hard / 56 partial / 338 in MIL bags (C/X-tier sentences and
  loss-masked lobes are not used)

## GT v3 (2026-11-19): Findings per lobe + Impression, size and type placeholders

`data_prep/ncku_v2/matching_v2/build_gt_v3.py` -> `region_report_v3_{train,val,test_me,test_ldct}.csv`
(same columns). The matching is unchanged (match_v2.py now also keeps each merged piece's Findings
and Impression wording apart: `text_f` / `text_i`, plus `<ds>_sentences_v2.csv`).

- lobe rows: only the Findings wording of the lobe's lesion pieces (v2 repeated every nodule in
  Findings and Impression wording)
- `impression` row: the report's Impression in order -- lung-lesion pieces in their Impression
  wording plus recommendations / judgements ("Suggest follow up.", "Probably benign.");
  extrapulmonary and comparison-only sentences go to the ignored global row. loss_mask 0 when it
  describes a lung lesion without a mask (231 / 1,093 train).
- `[TYPE#<nodule_id>=<word>]`: density words (GGN / GGO / ground-glass / part-solid / subsolid /
  non-solid / solid / calcified) of pieces given to exactly one nodule -- the same pieces that
  get `[SIZE#..]` (3,038 tags). The word stays in the CSV only to rebuild the GT for evaluation.

Dataset (detected from the CSV): target = lobes + `Impression: ...`; lobe tags become `[SIZE#k]` /
`[TYPE#k]`, Impression tags `[SIZE#i.k]` / `[TYPE#i.k]` (region slot i of this prompt, k-th
nodule); the instruction gains " Finally, write an impression for the whole scan."
Without lesion tokens, size tags become `[SIZE]` and type tags are removed. Max 864 tokens.

Inference: `evaluation/predict_types_v2.py` writes per-nodule type + confidence from the frozen
encoder's type head; `fill_report_v3` writes the type word when confidence >= type_fill_min_conf
(default 0.6) and deletes the tag otherwise ("a [TYPE#1] nodule" -> "a nodule").

```bash
cd scripts && seed=42 bash train_radgenome.sh ncku_a100_v3          # Reg2RG_v3_vitq
cd .. && python evaluation/predict_types_v2.py --ckpt outputs/lesion_enc_vitq_ce_mil_v2_seed0/best.pt \
    --meta $DATA_ROOT/nodule_metadata_v2.csv --data_root $DATA_ROOT --out outputs/type_pred_vitq_v2.csv
cd scripts && split=val ckpt_step=1360 bash test_radgenome.sh ncku_a100_v3
cd .. && python evaluation/eval_v2.py results/Reg2RG_v3_vitq_seed42_val_ckpt*_v3.csv   # + imp BLEU/ROUGE, type fill
```

## V3.1 (2026-11-20): lesion slots only for the nodules the report describes

V3 test_me review (oracle GT boxes): whether a lobe gets a finding followed the lesion tokens
almost one to one -- "finding iff the lobe has lesion tokens" scored F1 0.793 vs the model's
0.745. Cause: `nodule_ids` lists every nodule with a mask in the lobe, also the ones no
sentence was matched to (match_tier `none`: mostly < 5 mm, RADS 1, not written up). 21% of
the train lobes with lesion tokens had a normal target, so the model learned to skip given
lesions at random (its finding rate was ~0.8 at every nodule size; the radiologists' rose
from 0.43 at <= 4 mm to 0.93 at > 8 mm), and multi-nodule lobes mixed in undescribed nodules.

`lesion_tiers=A,B,C,X` (`lesion_utils.load_nodule_boxes`) keeps a lesion slot only for those
tiers; GT text and report CSVs are unchanged (every [SIZE#]/[TYPE#] tag already points at an
A/X nodule). Lobes with lesion tokens and a normal target:

| split | V3 | V3.1 |
|---|---|---|
| train | 513 / 2393 (21%) | 41 / 1921 (2%) |
| val | 107 / 390 | 6 / 289 |
| test_me | 102 / 361 | 1 / 260 |
| test_ldct | 50 / 965 | 10 / 925 |

The oracle is now "the nodules the radiologist wrote up"; deciding which detected nodules to
report is left to the detector stage. Report V3 and V3.1 side by side.

```bash
cd scripts && seed=42 bash train_radgenome.sh ncku_a100_v31                        # Reg2RG_v31_vitq
split=val ckpt_step=1360 bash test_radgenome.sh ncku_a100_v31                      # each saved step
split=test_me ckpt_step=<chosen> bash test_radgenome.sh ncku_a100_v31              # then test_ldct
cd .. && python evaluation/refill_types_v3.py results/Reg2RG_v31_vitq_seed42_val_ckpt1360_v3.csv \
    --report_csv $DATA_ROOT/region_report_v3_val.csv --meta $DATA_ROOT/nodule_metadata_v2.csv \
    --type_pred outputs/type_pred_vitq_v2.csv --lesion_tiers A,B,C,X
```

## Lesion encoder: MAE on LIDC, then ME (2026-11-20)

Question: does self-supervised pretraining on unlabelled LIDC crops give the vitq encoder a better
start than random init for the ME size / type / RADS training? (LIDC type labels as extra
supervised crops, `--extra_train`, gave nothing: rho 0.802 / kappa 0.709 / type bacc 0.450 vs
0.787 / 0.730 / 0.453.)

1. Workstation: `F:\DataSet\lidc_lesions\run_lidc_mae.bat` (`data_prep/lidc/build_lidc_mae.py`):
   every reader-marked nodule (any reader count, < 3 mm too) + 10 random lung crops per series,
   same crop geometry / normalisation as ME -> `lidc_mae_crops.npy` (+ `.npy.csv`), ~17k crops.
   Upload both to `outputs/` on the server.
2. `bash scripts/pretrain_lesion_mae.sh` (`evaluation/pretrain_lesion_mae.py`): SimMIM-style
   masked-image modelling of the encoder's image path (conv trunk -> proj -> pos -> grid
   transformer -> mem_norm). 60% of 2x2x2-cell blocks (6.4 x 6.4 x 8 mm) masked at the input
   and replaced by a mask token, linear head reconstructs the masked voxels, L1. 100 epochs.
   -> `outputs/lesion_mae_vitq_lidc/mae.pt` + `recon_qc.png`.
3. ME training from it: `INIT=outputs/lesion_mae_vitq_lidc/mae.pt` in `scripts/train_lesion_encoder.sh`
   (`--init_encoder`; queries / decoder / fc / heads start random) ->
   `outputs/lesion_enc_vitq_ce_mil_v2_mae_seed<k>/`.

Compare seeds 0-2 with and without INIT on ME val (157 patients, part-solid n = 1, so one seed
is not enough). Only if MAE wins: point `lesion_encoder_ckpt` of the report model at it,
re-run predict_types_v2.py and retrain the LLM.
