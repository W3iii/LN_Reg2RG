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
