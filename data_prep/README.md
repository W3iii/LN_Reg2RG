# data_prep

Scripts that turn the raw NCKU ME dataset into the on-disk format Reg2RG expects.
Copied here from the Windows workstation (`d:\W3iii\NCKU\DataSet\reg2rg_dataset\`) so the
pipeline is versioned alongside the model code.

**Paths inside these scripts are absolute Windows paths** (`d:\W3iii\NCKU\DataSet\...`).
They were run on the workstation that holds `ME_dataset/` and `ME_db1_20241210/`. To run
them elsewhere, change `ROOT` at the top of each file.

Read `../HANDOFF.md` §3 before touching anything geometric — the npy volumes are already
resampled to (0.8, 0.8, 1.0) mm while `series_metadata.txt` still describes the original
series, and the NIfTI affine's z sign has to be negative.

## Order

| # | Script | Needs | Produces |
|---|---|---|---|
| 1 | `build_dataset.py` | `ME_dataset/`, `ME_db1_20241210/`, `lung_M_class_0001-1800_report.xlsx` | `manifest.csv`, `nodules.csv`, `reports.jsonl`, `region_report.jsonl`, `skipped.csv` |
| 2 | `build_lobe_masks.py` | GPU + `lobeseg` env (TotalSegmentator) | `lobe/<folder>_lobe5.npz` (1=RUL … 5=LLL) |
| 3 | `assign_lobe.py` | 1, 2 | fills `nodules.csv`'s `lobe`; `lobe_qc.csv`, `lobe_validation.csv` |
| 4 | `match_nodule_sentence.py` | 3 | `nodule_sentence.csv`, `match_log.txt` |
| 5 | `export_reg2rg.py` | 2 | `reg2rg_nifti/images/`, `reg2rg_nifti/masks/` (~47 GB) |
| 6 | `make_csv_split.py` | 3, 4 | `region_report_{train,val,test}.csv`, `splits.csv`, `nodule_metadata.csv` |

Only step 2 needs a GPU. Steps 1, 3, 4, 6 are CPU-only and quick; step 5 is I/O bound.

**Text-only changes** (e.g. fixing the residual normalisation artifacts listed in
`HANDOFF.md` §8) need 1 → 3 → 4 → 6. Step 5 does not have to be repeated: the images and
masks are unaffected, only the CSVs change. Delete `src/Dataset/samples_*.pkl` afterwards
or the loader reuses the stale sample list.

## QC

| Script | Checks |
|---|---|
| `check_lobe_sanity.py` | affine orientation (spine must be posterior) and lobe volume fractions against reference |
| `validate_lobe_independent.py` | segmented lobe vs the lobe named in the report, restricted to single-nodule patients whose report names exactly one lobe — no matching algorithm involved, so not circular |
| `validate_lobe.py` | earlier version that validates against the *matched* sentence. Now circular, because the matcher uses the segmented lobe as its main signal. Kept for reference; quote `validate_lobe_independent.py` instead |

`build_lobe_masks.py` aborts after 5 consecutive failures: a broken CUDA context cannot
recover within the same process, and without the abort a mid-run GPU fault burns through
the remaining cases at ~8 s each while logging them all as errors. `run_lobe_batch.ps1`
(workstation only) wraps it in a restart loop. Completed `.npz` files are skipped, so the
whole thing is resumable.
