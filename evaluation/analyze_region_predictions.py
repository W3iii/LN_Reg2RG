"""
Diagnostic analysis of Reg2RG region reports.

The standard NLG metrics average over every region, and ~56% of regions are
"No significant finding." A model that only ever emits that sentence scores well.
This script separates the two populations and asks the questions that matter:

  - which lobe does the model claim findings in, versus where they actually are
  - how many findings per patient does it produce, versus the ground truth
  - per-lobe detection precision/recall for "this region has a finding"
  - does it identify which anatomical region each region token is (referring)

Usage:
    python analyze_region_predictions.py --result ../results/xxx.csv
"""
import argparse, os, re, sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from regions import REGIONS

NORMAL = 'no significant finding'
SHORT = {'right upper lobe': 'RUL', 'right middle lobe': 'RML', 'right lower lobe': 'RLL',
         'left upper lobe': 'LUL', 'left lower lobe': 'LLL'}


def parse(text):
    """'The region 0 is <area>: <report>' -> {area: report}"""
    out = {}
    if not isinstance(text, str):
        return out
    parts = re.split(r'The region \d+ is ', text)
    for p in parts[1:]:
        if ':' not in p:
            continue
        area, rep = p.split(':', 1)
        out[area.strip()] = rep.strip()
    return out


def has_finding(rep):
    return bool(rep) and NORMAL not in rep.lower()


ap = argparse.ArgumentParser()
ap.add_argument('--result', required=True)
a = ap.parse_args()
df = pd.read_csv(a.result).fillna({'GT_combined_report': '', 'Pred_combined_report': ''})
print('samples: %d\n' % len(df))

gt_pos = Counter(); pr_pos = Counter()
tp = Counter(); fp = Counter(); fn = Counter()
n_gt_find = []; n_pr_find = []
refer_ok = refer_tot = 0
order_canonical = 0

for r in df.itertuples():
    g, p = parse(r.GT_combined_report), parse(r.Pred_combined_report)

    # referring: does the predicted area sequence match the ground-truth sequence
    gk, pk = list(g.keys()), list(p.keys())
    for i in range(min(len(gk), len(pk))):
        refer_tot += 1
        refer_ok += (gk[i] == pk[i])
    if pk == REGIONS:
        order_canonical += 1

    gf = {k for k, v in g.items() if has_finding(v)}
    pf = {k for k, v in p.items() if has_finding(v)}
    n_gt_find.append(len(gf)); n_pr_find.append(len(pf))
    for k in REGIONS:
        if k in gf: gt_pos[k] += 1
        if k in pf: pr_pos[k] += 1
        if k in gf and k in pf: tp[k] += 1
        elif k in pf: fp[k] += 1
        elif k in gf: fn[k] += 1

print('=' * 66)
print('1. WHERE does each side put its findings')
print('=' * 66)

print('%-18s %8s %8s   %8s %8s %8s' % ('lobe', 'GT', 'PRED', 'TP', 'FP', 'FN'))
for k in REGIONS:
    print('%-18s %8d %8d   %8d %8d %8d' % (SHORT[k], gt_pos[k], pr_pos[k], tp[k], fp[k], fn[k]))
print('%-18s %8d %8d   %8d %8d %8d' % ('TOTAL', sum(gt_pos.values()), sum(pr_pos.values()),
                                        sum(tp.values()), sum(fp.values()), sum(fn.values())))

print('\n%-18s %9s %9s %9s' % ('lobe', 'precision', 'recall', 'F1'))
for k in REGIONS:
    P = tp[k] / max(tp[k] + fp[k], 1)
    R = tp[k] / max(tp[k] + fn[k], 1)
    F = 2 * P * R / max(P + R, 1e-9)
    print('%-18s %9.3f %9.3f %9.3f' % (SHORT[k], P, R, F))
P = sum(tp.values()) / max(sum(tp.values()) + sum(fp.values()), 1)
R = sum(tp.values()) / max(sum(tp.values()) + sum(fn.values()), 1)
print('%-18s %9.3f %9.3f %9.3f' % ('micro', P, R, 2 * P * R / max(P + R, 1e-9)))
print('\nA model that ignores the image would show PRED concentrated in the most')
print('frequent training lobes, with precision near the per-lobe base rate.')

print('\n' + '=' * 66)
print('2. HOW MANY findings per patient')
print('=' * 66)
sg, sp = pd.Series(n_gt_find), pd.Series(n_pr_find)
print('GT   : mean %.2f  median %d  dist %s' % (sg.mean(), sg.median(), dict(sorted(Counter(n_gt_find).items()))))
print('PRED : mean %.2f  median %d  dist %s' % (sp.mean(), sp.median(), dict(sorted(Counter(n_pr_find).items()))))
print('patients where GT has >=2 findings : %d' % (sg >= 2).sum())
print('  ...of those, PRED also has >=2   : %d' % ((sg >= 2) & (sp >= 2)).sum())

print('\n' + '=' * 66)
print('3. REFERRING (which anatomical area is each region token)')
print('=' * 66)
print('region-slot area matches GT : %d / %d (%.1f%%)' % (refer_ok, refer_tot, 100 * refer_ok / max(refer_tot, 1)))
print('predictions in canonical REGIONS order : %d / %d' % (order_canonical, len(df)))
print('NOTE: radgenome_dataset_test.py has random.shuffle() commented out, so the')
print('order is always the REGIONS list. This metric is close to free — treat a')
print('high score here as a sanity check, not as evidence of grounding.')

print('\n' + '=' * 66)
print('4. TEMPLATE REUSE (is the model reciting a handful of sentences)')
print('=' * 66)
sents = []
for r in df.itertuples():
    for v in parse(r.Pred_combined_report).values():
        if has_finding(v):
            sents.append(re.sub(r'\d+(\.\d+)?', '#', v.strip()))
c = Counter(sents)
print('finding sentences: %d, distinct after masking numbers: %d' % (len(sents), len(c)))
for s, n in c.most_common(8):
    print('  %4d  %s' % (n, s[:88]))

print('\n' + '=' * 66)
print('5. STATED SIZES')
print('=' * 66)
def sizes(col):
    v = []
    for t in df[col]:
        for m in re.finditer(r'(\d+(?:\.\d+)?)\s*(mm|cm)', str(t), re.I):
            x = float(m.group(1))
            v.append(x * 10 if m.group(2).lower() == 'cm' else x)
    return pd.Series(v)
for col in ['GT_combined_report', 'Pred_combined_report']:
    s = sizes(col)
    print('%-22s n=%4d  median %.1f mm  p10 %.1f  p90 %.1f  max %.1f'
          % (col.split('_')[0], len(s), s.median(), s.quantile(.1), s.quantile(.9), s.max()))
