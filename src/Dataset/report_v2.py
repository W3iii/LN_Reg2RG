"""GT v2 / split C support shared by radgenome_dataset_{train,test}.py.

A v2 report CSV (region_report_v2_{train,val,test_me,test_ldct}.csv, built by
data_prep/ncku_v2/build_gt_v2.py) has two extra columns per lobe row:

  nodule_ids   ';'-joined ids of the lobe's nodules, mask size descending. This is the
               lesion-slot order (#1 = largest), so the order no longer depends on
               TW_Lung_RADS, which LDCT does not have.
  loss_mask    0 when the lobe's text describes a lesion that has no mask: the text
               stays in the target (the model still reads a coherent report) but its
               tokens get label -100.

Sizes in the text are already placeholders: [SIZE#<nodule_id>] for a size that belongs
to one nodule, [SIZE] otherwise. See size_placeholder.localize_size_ids.

The lesion crops come from nodule_metadata_v2.csv by (Volumename, nodule_id)
(lesion_utils.load_nodule_boxes); the lobe's nodules are exactly the ids listed here,
capped at MAX_LESION_PER_REGION.

GT v3 (region_report_v3_*.csv, data_prep/ncku_v2/matching_v2/build_gt_v3.py) adds, detected from
the CSV itself:
  * [TYPE#<nodule_id>=<word>] density placeholders, rewritten like the sizes ([TYPE#k]; the word
    only survives in the GT text used for evaluation)
  * an 'impression' row, appended after the lobes as 'Impression: ...' with region-qualified
    references [SIZE#i.k] / [TYPE#i.k]; the instruction then asks for an impression too
"""

import pandas as pd
import torch

from regions import MAX_LESION_PER_REGION, lesion_slot_text
from size_placeholder import localize_size_ids, unindex_sizes, localize_ids

V2_COLUMNS = ('nodule_ids', 'loss_mask')


def is_v2_csv(csv_file):
    cols = pd.read_csv(csv_file, nrows=0).columns
    return all(c in cols for c in V2_COLUMNS)


def is_v3_csv(csv_file):
    """v2 columns plus an 'impression' row (GT v3)."""
    if not is_v2_csv(csv_file):
        return False
    return (pd.read_csv(csv_file, usecols=['Anatomy'])['Anatomy'] == 'impression').any()


def load_v2_reports(csv_file):
    """Volumename -> region -> {'text', 'ids', 'loss_mask'} (lobe rows only; the global row is ignored as before)."""
    df = pd.read_csv(csv_file, dtype={'nodule_ids': str})
    out = {}
    for r in df.itertuples():
        if pd.isna(r.Anatomy):
            continue
        ids = [int(float(x)) for x in str(r.nodule_ids).split(';') if x.strip()] if isinstance(r.nodule_ids, str) else []
        lm = 1 if pd.isna(r.loss_mask) else int(r.loss_mask)
        out.setdefault(r.Volumename, {})[r.Anatomy] = dict(text='' if pd.isna(r.Sentence) else str(r.Sentence),
                                                           ids=ids, loss_mask=lm)
    return out


def lobe_nodules(reports, volume, region, boxes):
    """The lobe's nodule ids that get a lesion slot: listed order, with a box, at most MAX_LESION_PER_REGION."""
    if boxes is None:
        return []
    ids = reports.get(volume, {}).get(region, {}).get('ids', [])
    return [n for n in ids if (volume, n) in boxes][:MAX_LESION_PER_REGION]


def lesion_prompt(slots, n_tok):
    """' with lesions #1 <lesion8>, #2 <lesion9>' (n_tok tokens per nodule), '' without nodules."""
    if not slots:
        return ''
    return ' with lesions ' + ', '.join(f'#{k} {lesion_slot_text(s, n_tok)}' for k, s in enumerate(slots, 1))


def region_target(text, local_ids, with_lesions, keep_word=False):
    """Lobe text with [SIZE#<nodule_id>] / [TYPE#<nodule_id>=..] rewritten to the lobe-local #k of
    the prompt. Without lesion tokens every index goes ([SIZE], TYPE tags removed)."""
    where = {int(n): str(k) for k, n in enumerate(local_ids, 1)} if with_lesions else {}
    return localize_ids(text, where, keep_word=keep_word)


IMPRESSION_INSTRUCTION = ' Finally, write an impression for the whole scan.'



def token_mask_for_spans(tokenizer, text, spans, n_tokens, max_len):
    """Bool (n_tokens,) tensor, True for tokens of `text` that overlap any char span [s, e).

    Uses the fast tokenizer's offsets when available; otherwise counts the tokens of the
    text prefixes. SentencePiece attaches a space to the *next* word, so the prefix before a
    span is counted without its trailing spaces (else it gains a lone '\u2581' token).
    """
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    if not spans:
        return mask
    if getattr(tokenizer, 'is_fast', False):
        offs = tokenizer(text, max_length=max_len, truncation=True, return_offsets_mapping=True)['offset_mapping']
        for t, (a, b) in enumerate(offs[:n_tokens]):
            if b > a and any(a < e and b > s for s, e in spans):
                mask[t] = True
        return mask
    for s, e in spans:
        i0 = len(tokenizer(text[:s].rstrip(' '), max_length=max_len, truncation=True)['input_ids'])
        i1 = len(tokenizer(text[:e], max_length=max_len, truncation=True)['input_ids'])
        mask[min(i0, n_tokens):min(i1, n_tokens)] = True
    return mask


def build_v2_text(ds, volume, instruction, region2area, slots_per_region, lesion_crops):
    """(prompt, combined_report, masked char spans of combined_report, sizes_by_slot, gt_report) for one sample.

    ds: the Dataset (reports_v2, nodule_boxes, region_padding_tokens, n_tok, text_add_image_tokens).

    Prompt per region: 'The region i is <region>..</region> with lesions #1 <lesion..>, #2 ...'.
    Target per region: 'The region i is <area>: <lobe text with [SIZE#k] / [TYPE#k]> ', where #k is the
    nodule's position in the prompt. loss_mask = 0 lobes keep their text, but the span after
    '<area>: ' is returned for masking, so the area name (region identification) is still learned.
    v3: then 'Impression: <text with [SIZE#i.k] / [TYPE#i.k]>' (masked as a whole when its loss_mask = 0).
    gt_report: the same text with every TYPE tag replaced by its GT word (for evaluation).
    """
    rep = ds.reports_v2.get(volume, {})
    imp = rep.get('impression')
    region_text = ""
    for i in range(len(region2area)):
        region_text += ("The region " + str(i) + " is " + '<region>' + ds.region_padding_tokens[i] + '</region>'
                        + lesion_prompt(slots_per_region.get(i, []), ds.n_tok) + '. ')
    prompt = ds.text_add_image_tokens(region_text + instruction + (IMPRESSION_INSTRUCTION if imp is not None else ''))

    combined, gt, spans, sizes, where_imp = "", "", [], {}, {}
    with_l = bool(ds.nodule_boxes)
    for i in range(len(region2area)):
        area = region2area[i]
        r = rep.get(area, {'text': '', 'ids': [], 'loss_mask': 1})
        local = [n for _, n, _ in lesion_crops.get(area, [])]
        body = region_target(r['text'], local, with_l)
        head = "The region " + str(i) + " is " + area + ": "
        start = len(combined) + len(head)
        combined += head + body + " "
        gt += head + region_target(r['text'], local, with_l, keep_word=True) + " "
        if r['loss_mask'] == 0 and body:
            spans.append((start, start + len(body)))
        if with_l:
            sizes[i] = [ds.nodule_boxes[(volume, n)][1] for n in local]
            where_imp.update({int(n): f'{i}.{k}' for k, n in enumerate(local, 1)})
    if imp is not None:
        head = 'Impression: '
        body = localize_ids(imp['text'], where_imp)
        start = len(combined) + len(head)
        combined += head + body + " "
        gt += head + localize_ids(imp['text'], where_imp, keep_word=True) + " "
        if imp['loss_mask'] == 0 and body:
            spans.append((start, start + len(body)))
    return prompt, combined, spans, sizes, gt
