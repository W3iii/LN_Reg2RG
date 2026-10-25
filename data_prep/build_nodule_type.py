"""Per-nodule type labels (GGO / solid / part-solid / calcified) for the lesion encoder.

The reports name a nodule's type for only some nodules, so a label is a *set* of
allowed types, and "the report does not say" is never read as a negative.

Sources, strongest first
  sentence      the nodule's own matched_sentence names its type(s)
                  one type            -> hard label      (masked CE)
                  several types       -> partial label   ("one of these"; e.g. a
                                         sentence listing a GGO and a solid nodule,
                                         or 'subsolid' = GGO or part-solid)
  lobe_single   no matched sentence, the nodule is the only one in its lobe, and the
                lobe text names a type -> the same kind of (partial) label
  MIL bag       the lobe text names type set M that no labelled nodule of the lobe
                explains -> "at least one of the bag's members is in M"
                (noisy-OR loss in train_lesion_encoder.py)

Only train and val are parsed. Test reports are never read: test rows get no label.

Writes one row per nodule (all splits):
  Volumename, nodule_id, split, lobe, type_set ('GGO|solid', '' = none),
  type_label (0..3 when type_set is a single type, else -1), type_source,
  bag_id (-1 = not in a bag), bag_mentions (';'-separated type sets, e.g. 'GGO;GGO|part-solid')

  python data_prep/build_nodule_type.py --data_root /root/notebooks/groups/BME/reg2rg_nifti
"""
import argparse
import os
import re

import pandas as pd

TYPES = ('GGO', 'solid', 'part-solid', 'calcified')
T = {t: i for i, t in enumerate(TYPES)}

_CALC = re.compile(r'calcif')
_CALC_NEG = re.compile(r'\b(?:non|no|without)[- ]?(?:\w+ )?calcif')
_PS = re.compile(r'part(?:ial|ly)?[- ]?solid|semi[- ]?solid|mixed (?:ground[- ]?glass|ggo)'
                 r'|solid (?:component|portion|part)')
_PS_NEG = re.compile(r'\b(?:no|without)\s+(?:\w+\s+)?solid (?:component|portion|part)')
_SUB = re.compile(r'sub[- ]?solid')
_GGO = re.compile(r'ground[- ]?glass|\bgg[on]s?\b')
_SOLID = re.compile(r'(?<![\w-])solid\b(?! (?:component|portion|part))')
_SOLID_NEG = re.compile(r'\b(?:no|not|without)\s+(?:\w+\s+)?solid\b|\bnon[- ]?solid\b')


def clause_mentions(c):
    """Type mentions in one clause, each a frozenset of type ids.

    Within a clause the more specific word wins: 'calcified solid nodule' is calcified,
    'ground glass nodule with a solid component' is part-solid. Separate clauses keep
    their own mentions ('RUL solid nodule, RLL part-solid nodule' -> solid, part-solid).
    """
    if _CALC.search(c) and not _CALC_NEG.search(c):
        return [frozenset([T['calcified']])]
    if _PS.search(c) and not _PS_NEG.search(c):
        return [frozenset([T['part-solid']])]
    out = []
    if _SUB.search(c):
        out.append(frozenset([T['GGO'], T['part-solid']]))   # 'subsolid' = GGO or part-solid
    if _GGO.search(c):
        out.append(frozenset([T['GGO']]))
    c_wo = _SUB.sub(' ', c)
    if _SOLID.search(c_wo) and not _SOLID_NEG.search(c_wo):
        out.append(frozenset([T['solid']]))
    return out


def mentions(text):
    if not isinstance(text, str):
        return []
    ms = []
    for c in re.split(r'[.,;:\n]|\band\b', text.lower()):
        ms += clause_mentions(c)
    return list(dict.fromkeys(ms))


def reduce_mentions(ms):
    """Drop a mention implied by a stricter one ({GGO} makes {GGO, part-solid} redundant)."""
    return [m for m in ms if not any(o < m for o in ms)]


def nodule_set(ms):
    """A nodule's allowed types from its own sentence: what every mention agrees on, else the union."""
    if not ms:
        return frozenset()
    inter = frozenset.intersection(*ms)
    return inter if inter else frozenset().union(*ms)


def fmt(s):
    return '|'.join(TYPES[i] for i in sorted(s))


def build(data_root, out_csv=None):
    nm = pd.read_csv(os.path.join(data_root, 'nodule_metadata.csv'))
    rr = pd.concat([pd.read_csv(os.path.join(data_root, f'region_report_{s}.csv')) for s in ('train', 'val')])
    rr = rr.dropna(subset=['Anatomy', 'Sentence'])
    rr = rr.assign(Anatomy=rr['Anatomy'].str.lower()).groupby(['Volumename', 'Anatomy'])['Sentence'].agg(' '.join)
    lobe_text = rr.to_dict()

    labelled = nm['split'].isin(['train', 'val']).to_numpy()
    sets, source = [], []
    for ok, sent in zip(labelled, nm['matched_sentence']):
        s = nodule_set(mentions(sent)) if ok else frozenset()
        sets.append(s); source.append('sentence' if s else '')
    has_sentence = nm['matched_sentence'].notna().to_numpy()

    bag_id = [-1] * len(nm)
    bag_mentions = [''] * len(nm)
    nb = 0
    for (vol, region), g in nm[labelled].groupby(['Volumename', 'region'], sort=False):
        ms = reduce_mentions(mentions(lobe_text.get((vol, region.lower()))))
        # a mention is explained when some nodule of the lobe is already known to be in it
        ms = [m for m in ms if not any(sets[i] and sets[i] <= m for i in g.index)]
        if not ms:
            continue
        members = [i for i in g.index if len(sets[i]) != 1]
        if not members:
            continue
        if len(g) == 1:
            i = g.index[0]
            if has_sentence[i]:
                continue      # its own sentence names no type: the lobe's type words are about something else
            sets[i] = frozenset().union(*ms); source[i] = 'lobe_single'
            continue
        for i in members:
            bag_id[i] = nb
            bag_mentions[i] = ';'.join(fmt(m) for m in ms)
        nb += 1

    res = nm[['Volumename', 'nodule_id', 'split', 'lobe']].copy()
    res['type_set'] = [fmt(s) for s in sets]
    res['type_label'] = [next(iter(s)) if len(s) == 1 else -1 for s in sets]
    res['type_source'] = source
    res['bag_id'] = bag_id
    res['bag_mentions'] = bag_mentions
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        res.to_csv(out_csv, index=False)
    return res


def summary(res):
    for sp in ('train', 'val'):
        r = res[res.split == sp]
        hard = r.type_label >= 0
        part = (r.type_set != '') & ~hard
        print(f'[{sp}] nodules {len(r)}')
        print(f'   hard label        {hard.sum():5d}  ' +
              '  '.join(f'{t} {int((r.type_label == i).sum())}' for i, t in enumerate(TYPES)))
        print(f'     from sentence   {(hard & (r.type_source == "sentence")).sum():5d}   from lobe_single {(hard & (r.type_source == "lobe_single")).sum()}')
        print(f'   partial label     {part.sum():5d}  ' + str(r[part].type_set.value_counts().head(5).to_dict()))
        inbag = r.bag_id >= 0
        print(f'   in MIL bag        {inbag.sum():5d}  ({r[inbag].bag_id.nunique()} bags; '
              f'{int((inbag & (r.type_set == "")).sum())} of them have no own label)')
        print(f'   no type info      {((r.type_set == "") & ~inbag).sum():5d}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'nodule_type.csv'))
    a = ap.parse_args()
    res = build(a.data_root, a.out)
    summary(res)
    print(f'wrote {a.out}')
