# Handoff — back to local (Windows) — §8 item 2: residual text artifacts

Written 2026-08-17 from the server side, after running the diagnostic
(`evaluation/analyze_region_predictions.py`, see `HANDOFF.md` §6). This documents what
needs fixing for `HANDOFF.md` §8 item 2 — **do not action this on the server.**

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
