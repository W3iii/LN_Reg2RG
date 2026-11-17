"""Size placeholders: the LLM writes where a size goes, the lesion encoder supplies it (B3e).

Why: with predicted sizes in the prompt (B3c/B3d) the LLM still copied them only
~30-40% of the time -- barely above the 34% that a model with *no* numbers in its
prompt hits by chance, because sizes cluster at 4-6 mm. Under the LM loss,
copying the prompt number reproduces the report's exact number only 24% of the
time on train (radiologists round, describe another nodule, aggregate), against
21% for always saying "4 mm"; there is almost no gradient towards copying.

So the number is taken out of the language model's hands:
  training   report targets have every size replaced by PLACEHOLDER
  inference  each PLACEHOLDER in a lobe's text is filled from the encoder's
             predictions for that lobe's nodules (largest first)

The placeholder is plain text, not a special token: LLaMA's LM head only covers
the 32000-word vocabulary and the Dataset masks label ids >= 32000, so a new
special token could never be generated.
"""
import re

PLACEHOLDER = '[SIZE]'

_NUM = r'\d+(?:\.\d+)?'
_UNIT = r'(?:cm|mm)\b'
_X = r'\s*(?:x|\*|×)\s*'
# "1.5cm from the pleura" is a distance to a landmark, not a size
_NOT_DISTANCE = r'(?!\s*(?:from|away|beneath|below|above|distal|proximal)\b)'

_DIMS = re.compile(rf'{_NUM}{_X}{_NUM}(?:{_X}{_NUM})?\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_PAIR = re.compile(rf'{_NUM}\s*(?:{_UNIT}\s*)?(and|,)\s*{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_RANGE = re.compile(rf'{_NUM}\s*(?:{_UNIT}\s*)?(?:-|~|to)\s*{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)
_SINGLE = re.compile(rf'{_NUM}\s*{_UNIT}{_NOT_DISTANCE}', re.I)


def mask_sizes(text):
    """Replace size expressions with PLACEHOLDER. Returns (new_text, n_placeholders).

    4x3mm / 0.9 x 0.7 cm   -> [SIZE]              (one lesion, one number)
    0.5 and 0.5cm, 3mm and 4mm -> [SIZE] and [SIZE] (two lesions)
    0.3-0.5cm / 3 to 5mm   -> [SIZE]              (a range: one number, the upper end)
    1.5cm from the pleura  -> unchanged
    """
    if not text:
        return text, 0
    t = _DIMS.sub(PLACEHOLDER, text)
    t = _PAIR.sub(lambda m: f'{PLACEHOLDER} {m.group(1)} {PLACEHOLDER}'.replace(' ,', ','), t)
    t = _RANGE.sub(PLACEHOLDER, t)
    t = _SINGLE.sub(PLACEHOLDER, t)
    return t, t.count(PLACEHOLDER)


def format_size(mm):
    return f'{max(0.1, round(mm / 10.0, 1)):.1f}cm'


_SENT_SPLIT = re.compile(r'(?<=[.;])\s+')


def fill_sizes(text, preds_mm):
    """Fill PLACEHOLDERs in one lobe's text from that lobe's predicted sizes.

    Within each sentence the j-th placeholder gets the j-th largest prediction,
    so a sentence about one nodule (or "up to [SIZE]") gets the largest and
    "[SIZE] and [SIZE]" gets the two largest. Findings and impression usually
    restate the same nodule in separate sentences, hence restarting per sentence
    rather than counting across the lobe. With no prediction available (the
    model described a nodule in a lobe with no lesion token) the placeholder is
    left as is and counted as unfilled.
    Returns (filled_text, n_unfilled).
    """
    if PLACEHOLDER not in text:
        return text, 0
    preds = sorted((float(p) for p in preds_mm if p is not None), reverse=True)
    out, unfilled = [], 0
    for sent in _SENT_SPLIT.split(text):
        j = 0
        while PLACEHOLDER in sent:
            if not preds:
                unfilled += sent.count(PLACEHOLDER)
                break
            sent = sent.replace(PLACEHOLDER, format_size(preds[min(j, len(preds) - 1)]), 1)
            j += 1
        out.append(sent)
    return ' '.join(out), unfilled


_REGION = re.compile(r'The region (\d+) is ')


def fill_report(report, region2area, preds_by_area):
    """Fill a whole generated report, segment by 'The region i is ...:'.

    region2area: slot index -> area the lesion tokens of that slot came from
    preds_by_area: area -> predicted sizes (mm) of that area's lesion tokens
    Returns (filled_report, n_placeholders, n_unfilled).
    """
    pieces = _REGION.split(report)
    if len(pieces) < 3:
        return report, report.count(PLACEHOLDER), report.count(PLACEHOLDER)
    out, n_ph, n_unf = [pieces[0]], 0, 0
    for k in range(1, len(pieces) - 1, 2):
        slot, body = int(pieces[k]), pieces[k + 1]
        n_ph += body.count(PLACEHOLDER)
        area = region2area.get(slot) if hasattr(region2area, 'get') else None
        body, unf = fill_sizes(body, preds_by_area.get(area, []) if area else [])
        n_unf += unf
        out.append(f'The region {slot} is ' + body)
    return ''.join(out), n_ph, n_unf


# ---------------------------------------------------------------- v2 (indexed placeholders)
# GT v2 (region_report_v2_*.csv) writes a size that belongs to exactly one nodule as
# [SIZE#<nodule_id>] and any other size (group descriptions, lesions with no mask) as
# [SIZE]. nodule_id is a dataset-wide id the LLM cannot know, so the Dataset rewrites it
# to the nodule's 1-based position k in its lobe's lesion list -- the same "#k" that
# labels the nodule's lesion tokens in the prompt:
#     prompt  ... The region 2 is <region>..</region> with lesions #1 <lesion16>, #2 <lesion17>.
#     target  The region 2 is right lower lobe: A [SIZE#2] solid nodule ...
# At inference [SIZE#k] in region i's text gets the mask size of region i's k-th nodule;
# a bare [SIZE] gets the old largest-first rule.
INDEXED = re.compile(r'\[SIZE#(\d+)\]')


def localize_size_ids(text, ids):
    """[SIZE#<nodule_id>] -> [SIZE#<k>] (k = 1-based position in ids); an id not in ids -> [SIZE]."""
    pos = {int(n): k for k, n in enumerate(ids, 1)}
    return INDEXED.sub(lambda m: f'[SIZE#{pos[int(m.group(1))]}]' if int(m.group(1)) in pos else PLACEHOLDER, text)


def unindex_sizes(text):
    """[SIZE#k] -> [SIZE] (runs without lesion tokens: the model has nothing to point at)."""
    return INDEXED.sub(PLACEHOLDER, text)


def fill_indexed(text, sizes_mm):
    """One lobe: [SIZE#k] -> size of the lobe's k-th nodule, then bare [SIZE] largest-first.
    Returns (filled_text, n_placeholders, n_unfilled). An index past the lobe's nodule list
    falls back to the bare rule."""
    n_ph = len(INDEXED.findall(text)) + text.count(PLACEHOLDER)

    def one(m):
        k = int(m.group(1))
        return format_size(sizes_mm[k - 1]) if 1 <= k <= len(sizes_mm) else PLACEHOLDER
    text = INDEXED.sub(one, text)
    text, unfilled = fill_sizes(text, sizes_mm)
    return text, n_ph, unfilled


def fill_report_v2(report, sizes_by_slot):
    """Fill a generated v2 report, segment by 'The region i is ...:'.
    sizes_by_slot: region slot i -> eq. diameters (mm) of that slot's nodules, in lesion-slot order.
    Returns (filled_report, n_placeholders, n_unfilled)."""
    pieces = _REGION.split(report)
    if len(pieces) < 3:
        n = len(INDEXED.findall(report)) + report.count(PLACEHOLDER)
        return report, n, n
    out, n_ph, n_unf = [pieces[0]], 0, 0
    for k in range(1, len(pieces) - 1, 2):
        slot, body = int(pieces[k]), pieces[k + 1]
        body, ph, unf = fill_indexed(body, sizes_by_slot.get(slot, []))
        n_ph += ph; n_unf += unf
        out.append(f'The region {slot} is ' + body)
    return ''.join(out), n_ph, n_unf
