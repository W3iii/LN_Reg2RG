# NCKU data prep for split C / GT v2 (ME + LDCT, 2026-11)

Copies of the workstation scripts that built the v2 files in `reg2rg_nifti/`. They run from the
`F:\DataSet` layout (each script finds its inputs relative to its own folder), so copy a folder
back next to `ME_dataset/`, `LDCT_test_dataset/`, `reg2rg_dataset/`, `reg2rg_nifti/` to re-run.
Nothing here runs on the training server. See `docs/V2_PIPELINE.md` for what the outputs are.

Order:
1. `ldct_test_match/ldct_match_nodule_sentence.py` - LDCT nodules from the mask json (ids = json order)
2. `reg2rg_dataset/build_lobe_masks_ldct.py` (+ `run_lobeseg_ldct.bat`, Windows + GPU) - TotalSegmentator
   5 lobes for LDCT, prone-aware; `lobe_qc_ldct.py` - QC
3. `ldct_test_match/assign_lobe_ldct.py` - lobe per LDCT nodule (`ldct_nodule_lobe.csv`)
4. `matching_v2/match_v2.py` - v2 nodule <-> sentence matching for ME and LDCT (split pieces, tiers)
   `compare_v1_v2.py`, `export_excel_v2.py` - checks and the v2 Excel files
5. `splits_merged.csv` (split C) was made interactively: ME train/val = DETECT-712 train (+147 of
   DETECT test) / DETECT val, test_me = 150 of DETECT test stratified by (has A-tier nodule) x
   nodule-count bin, seed 20261117; test_ldct = the 566 LDCT patients with nodules and lobe masks
6. `matching_v2/build_gt_v2.py` - region_report_v2_*.csv + fill_table_v2.csv
7. `reg2rg_dataset/export_reg2rg_ldct.py` - LDCT NIfTI export, same crop as ME (`--shard i/n`,
   `--budget s` for short sessions); `verify_ldct_export.py` - read-back check
8. `matching_v2/build_nodule_metadata_v2.py` - nodule_metadata_v2.csv
