"""Do generated reports name the right nodule types? Lobe-level, against the GT report.

For every (case, lobe) the GT and generated lobe paragraphs are parsed with the same
rules that make the type labels (data_prep/build_nodule_type.py). A type counts as
mentioned when the lobe text names it on its own ('subsolid' = GGO-or-part-solid
counts for neither). Per type: recall over GT lobes that name it, precision over
generated lobes that name it.

  python evaluation/eval_report_types.py B3e_clean=results/..._B3e_clean_seed1.csv B3f=results/..._B3f_seed1.csv
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_prep'))
from build_nodule_type import TYPES, mentions   # noqa: E402

LOBES = ('right upper lobe', 'right middle lobe', 'right lower lobe', 'left upper lobe', 'left lower lobe')
_SPLIT = re.compile(r'The region \d+ is (' + '|'.join(LOBES) + r'):')


def split(report):
    parts = _SPLIT.split(str(report))
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def types_named(text):
    return {next(iter(m)) for m in mentions(text) if len(m) == 1}


def score(path):
    d = pd.read_csv(path)
    rows = []
    for r in d.itertuples():
        g, p = split(r.GT_combined_report), split(r.Pred_combined_report)
        for lb in LOBES:
            gt, pr = types_named(g.get(lb, '')), types_named(p.get(lb, ''))
            rows.append({f'gt_{t}': i in gt for i, t in enumerate(TYPES)} | {f'pr_{t}': i in pr for i, t in enumerate(TYPES)})
    t = pd.DataFrame(rows)
    out = {}
    for n in TYPES:
        g, p = t[f'gt_{n}'], t[f'pr_{n}']
        tp = int((g & p).sum())
        out[n] = dict(gt_lobes=int(g.sum()), pred_lobes=int(p.sum()),
                      recall=tp / g.sum() if g.sum() else float('nan'),
                      precision=tp / p.sum() if p.sum() else float('nan'))
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = []
    for arg in sys.argv[1:]:
        name, path = arg.split('=', 1) if '=' in arg else (os.path.basename(arg), arg)
        for typ, m in score(path).items():
            rows.append(dict(arm=name, type=typ, **m))
    res = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(res.pivot(index='type', columns='arm', values=['recall', 'precision']).round(3).to_string())
    print()
    print(res.pivot(index='type', columns='arm', values=['gt_lobes', 'pred_lobes']).to_string())
