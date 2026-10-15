# Handoff — back to local (Windows) — §8 item 2: residual text artifacts

Written 2026-08-17 from the server side, after running the diagnostic
(`evaluation/analyze_region_predictions.py`, see `HANDOFF.md` §6). This documents what
needs fixing for `HANDOFF.md` §8 item 2 — **do not action this on the server.**

## 0. Server progress since this was written (updated 2026-10-15; last run 2026-08-20)

Nothing below needs action on Windows except the last point. It is here so the local side
knows where the server stands before syncing anything back.

**Baseline noise floor — done** (`HANDOFF.md` §11). Three seeds of the unchanged baseline,
lesion tokens off. Text micro-F1 ranges from 0.019 to 0.271 across seeds, while the
representation probe moves only 0.003. Text F1 mostly measures how readily the decoder
commits to a finding, so the probe is now the primary metric. Text metrics are never
quoted without the fraction of lobes called abnormal beside them.

**Lesion-token arm (B2), seed 1 — trained and evaluated on val.** Same config as the
baseline plus `nodule_metadata=` (`experiment_name=Reg2RG_ncku_lobe5_lesion`,
`ncku_a100.sh` now takes that override so arms don't share an output dir). Final
`train_loss` 0.443, inside the baseline's 0.444–0.474.

| | B0 seeds 1 / 2 / 3 | **B2 seed 1** |
|---|---|---|
| text micro-F1 | 0.019 / 0.271 / 0.178 | **0.674** |
| micro precision / recall | — / 0.455 / 0.193 (seed 2) | 0.789 / 0.588 |
| RUL recall | 0.043 (§6) / 0.072 / 0.116 | **0.826** |
| lobes called abnormal (truth 44.5%) | 1.0% / 18.6% / 14.8% | 32.7% |
| patients with ≥2 GT findings that get ≥2 predicted | — / 4 of 87 / — | 56 of 87 |

Results: `results/ncku_lobe5_val_B2_seed1_ckpt1390.csv`, diagnostic
`outputs/seed_sweep_logs/diag_B2_seed1.log` (server only, not committed).

**Do not read this as a deployable gain.** Inference used the same ground-truth nodule
boxes from `nodule_metadata.csv` that training did. A lobe only receives lesion tokens if
an annotated nodule sits in it, so the token's presence alone tells the decoder "this lobe
is abnormal". This is the oracle-localisation upper bound flagged in
`docs/LESION_TOKENS.md` §9, now measured on text. It is also one seed, though 0.674 sits
above every one of the five baseline draws so far (max 0.439).

**Still owed on the server:**
1. Probe on the B2 checkpoint (`probe_seeds.sh` only knows the baseline dirs).
2. A non-oracle B2 arm: lesion boxes from a detector, or from shuffled/random boxes as a
   control. This separates "lesion crops help" from "box presence leaks the label".
3. The remaining §7 arms, one seed each, judged on the probe.

**Relevant to the Windows-side fix below.** The text artifacts in §1 and §2 still show up
in B2's predictions, e.g. `RUL, RLL and LLL small nodules without.` and
`RUL and RML small nodules without as.` The model learns them from the training reports,
so the data fix remains worth doing regardless of which arm wins.

## Why this has to happen on the Windows box, not here

The raw source reports (and the label-match audit that produced `nodules.csv` /
`reports.jsonl`) only exist at `d:\W3iii\NCKU\DataSet\...` on the Windows workstation —
see `HANDOFF.md` §2. The server only ever sees the already-normalised CSVs that
`data_prep/build_dataset.py` produced from that raw text. At least one of the artifacts
below (the CJK corruption) is upstream of normalisation entirely — it's damage in how the
raw report text was extracted, not something the `clean_sentence()` regexes could ever
catch. Fixing it here would mean editing the *symptom* in already-exported CSVs while the
*cause* (and the only copy of the raw text) stays on Windows — the two copies would then
permanently disagree. Re-running `build_dataset.py` on the server against server-local
data isn't an option either: the raw reports simply aren't here.

So: the regex/extraction fix belongs in `data_prep/build_dataset.py` (or upstream of it,
for the CJK case) on the Windows checkout, re-run there, and only the corrected
`region_report_{train,val,test}.csv` / `nodule_metadata.csv` outputs get synced back to
`/root/notebooks/groups/BME/reg2rg_nifti/` for the server to train against. The NIfTI
images/masks are untouched by this fix and do not need re-export.

## What I found (read-only — pulled from `results/ncku_lobe5_val_ckpt1390.csv`, which
carries the normalised `GT_combined_report` text used for training/eval)

### 1. `without as comparing` (30 occurrences per `HANDOFF.md` §8)

```
A RLL tiny nodule without as comparing. A RLL tiny lung nodule.
A RUL small nodule without as comparing. A RUL small lung nodule.
RLL tiny nodules without as comparing to. RLL tiny lung nodules.
```

`clean_sentence()` in `data_prep/build_dataset.py` runs `DROP_SENT` first (whole-sentence
drop) against the **original, unmodified** sentence, then `TEMPORAL_INLINE` /
`TEMPORAL_CLAUSE` (partial strips) after. `DROP_SENT` has:

```python
r'compared\s+(?:with|to)\s+(?:the\s+)?(?:previous|prior|earlier|\d{4})',
```

which requires the literal word **"compared"**. The residual `as comparing` shows the raw
report used the gerund **"comparing"** instead — that variant never matches `DROP_SENT`,
so the sentence survives whole-sentence drop; a later partial strip then removes the
`interval change` portion (or similar) and leaves the `as comparing` fragment stranded.
Likely original: something like *"…nodule without interval change as comparing with
previous study."*

**Fix direction:** add a `comparing` alternation next to `compared` in both `DROP_SENT`'s
comparison pattern (line ~85) and `TEMPORAL_INLINE`'s `as compared` pattern (line ~102):
`compar(?:ed|ing)`. Verify against the actual raw sentence for a couple of these
accession numbers first — don't guess blind, confirm the source text says "comparing"
before assuming this is the whole story.

### 2. `without interval.` (11 occurrences)

```
LUL and LLL small nodules without interval. RUL …
```

`TEMPORAL_INLINE` only strips `interval` when immediately followed by one of
`enlargement|change|increase|decrease` (line ~104):

```python
(r',?\s*(?:mild(?:ly)?|slight(?:ly)?)?\s*interval\s+(?:enlargement|change|increase|decrease)\b', ''),
```

A bare `interval.` surviving means the raw sentence had `interval` followed by something
outside that enumerated set (different noun, or the sentence was truncated at that exact
point — e.g. a PDF line-break artifact). Need the raw sentence to know which. Grep the
Windows-side raw report text for `\binterval\b` and check what follows in the surviving
cases before extending the noun list — extending blind risks either missing the real
words or over-matching unrelated uses of "interval".

### 3. CJK corruption (100+ occurrences per `HANDOFF.md` §8)

```
… right lower lobe: 翁 for evaluation of R …
```

This is not a normalisation miss — it's a single CJK character embedded mid-English-clause
where an English word should be (possibly a name field or a garbled `為...` phrase that
leaked from the source document during text extraction). **This is upstream of
`build_dataset.py` entirely** — whatever process turned the original clinical report
(PDF or HIS export) into the text `build_dataset.py` reads in already had this damage.
Check the raw-text extraction step (OCR? PDF-to-text? direct HIS/RIS export?) for
encoding-related corruption, not the regex pipeline. If the extraction is not
reproducible losslessly, these ~100+ reports may need manual spot-correction rather than
a pattern fix.

### 4. Trailing bare coordinate pairs, e.g. `likely benign.,104,123).`

```
3 tiny subpleural nodules in the LLL, likely benign.,104,123). Small subpleural rounded
nodules in RML, and LLL.
```

`SLICE_REF` (line ~120) strips slice references shaped like `Srs/Img:4/20` or a bare
`6/74` (**slash**-separated). This residual is a **comma**-separated pair in trailing
parens with no slash — a different source format than anything `SLICE_REF` currently
matches, so it's not a tuning issue, it needs a new pattern:

```python
r'\s*\(?\s*\d{1,4}\s*,\s*\d{1,4}\s*\)\.?',   # rough starting point — verify against raw examples first
```

Grep the raw reports for a handful of these to confirm the shape (is it always exactly
two numbers? always trailing the sentence? could it collide with a legitimate two-number
finding like a size range "3,5 cm"?) before adding it — a hasty regex here risks eating
real content the same way the affine-sign bug in `HANDOFF.md` §3 silently corrupted labels
without erroring.

## After fixing

Re-run in order: `build_dataset.py` → `assign_lobe.py` → `match_nodule_sentence.py` →
`make_csv_split.py` (per `HANDOFF.md` §4 pipeline table). Only the CSVs change — no need
to re-export the 47 GB of NIfTI images/masks. **Delete `src/Dataset/samples_*.pkl`**
afterward so the server doesn't train against the stale cached sample list (this bit
`HANDOFF.md` once already — see §9's `train_samples.pkl` entry).

Sync the corrected `region_report_{train,val,test}.csv` and `nodule_metadata.csv` to
`/root/notebooks/groups/BME/reg2rg_nifti/` on the server (same paths as `HANDOFF.md` §4),
then re-train.
