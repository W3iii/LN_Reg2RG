"""Score v2 inference results (test_radgenome.py on region_report_v2_*.csv) and compare checkpoints.

  python evaluation/eval_v2.py results/Reg2RG_v2_vitq_seed42_val_ckpt*_v2.csv
  python evaluation/eval_v2.py vitq=results/..._ckpt1360_v2.csv B0=results/..._ckpt1360_v2.csv

Per result file (one row each, sorted by the order given):

  finding P/R/F1   lobe level: does a lobe's text report anything other than "No significant
                   finding." -- micro over all (patient, lobe) pairs. Same definition as
                   analyze_region_predictions.py (B-series "text micro-F1").
  nod R            recall on lobes whose GT names a nodule-like lesion
  normal pred/GT   share of lobes reported normal (collapse check)
  #k P/R           which nodules the text refers to: the [SIZE#k] indices of a lobe in
                   Pred_raw_report vs GT_raw_report (does the model size the right nodules)
  unfilled         [SIZE] placeholders that could not be filled / all placeholders
  type R/P         GGO / solid / part-solid / calcified named in a lobe (rules of
                   data_prep/build_nodule_type.py), GT vs prediction
  BLEU-4 / ROUGE-L on the size-filled lobe texts, normal lobes included. Own simple
                   implementations (corpus BLEU-4 with brevity penalty, sentence ROUGE-L F,
                   lower-cased whitespace tokens) for choosing a checkpoint; report paper
                   numbers with the standard evaluators.

GT v3 results (an 'Impression:' section, [TYPE#..] filled from the type head) also get
  imp BLEU-4 / ROUGE-L   on the Impression text (left out when 'impression' is in Masked_regions)
  type fill             share of generated [TYPE#..] slots that got a type word (n_type_filled /
                        n_type_slots); the per-type recall / precision then measure the filled words

Lobes listed in Masked_regions (GT text describes a lesion with no mask, loss_mask = 0) are
left out of every metric. The test set is not shuffled, so lobe naming is not scored here.
"""
import json
import math
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_prep'))
from build_nodule_type import TYPES, mentions  # noqa: E402

LOBES = ('right upper lobe', 'right middle lobe', 'right lower lobe', 'left upper lobe', 'left lower lobe')
_SPLIT = re.compile(r'The region \d+ is (' + '|'.join(LOBES) + r'):')
NORMAL = 'no significant finding'
NODULE = re.compile(r'nodul|GGN|GGO|ground[\s-]?glass|opacit|mass\b|lesion|calcif|granuloma', re.I)
IDX = re.compile(r'\[SIZE#(\d+)\]')


def split_imp(report):
    t = str(report) if isinstance(report, str) else ''
    i = t.find('Impression:')
    return (t, None) if i < 0 else (t[:i], t[i + len('Impression:'):].strip())


def split(report):
    parts = _SPLIT.split(split_imp(report)[0])
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def finding(t):
    return bool(t) and NORMAL not in t.lower()


def types_named(t):
    return {TYPES[next(iter(m))] for m in mentions(t) if len(m) == 1}


def toks(t):
    return re.findall(r"[a-z0-9.#\[\]]+", t.lower())


def bleu4(refs, hyps):
    num, den, rl, hl = [0] * 4, [0] * 4, 0, 0
    for r, h in zip(refs, hyps):
        rl += len(r); hl += len(h)
        for n in range(1, 5):
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            den[n - 1] += max(len(h) - n + 1, 0)
    if min(num) == 0 or hl == 0:
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return bp * math.exp(sum(math.log(num[i] / den[i]) for i in range(4)) / 4)


def rouge_l(r, h):
    if not r or not h:
        return float(r == h)
    prev = [0] * (len(h) + 1)
    for a in r:
        cur = [0]
        for j, b in enumerate(h):
            cur.append(prev[j] + 1 if a == b else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return 2 * p * rc / (p + rc)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    f = 0.0 if tp == 0 and tp + fn else (2 * p * r / (p + r) if p == p and r == r and p + r else float('nan'))
    return p, r, f


def score(path):
    d = pd.read_csv(path)
    need = {'GT_combined_report', 'Pred_combined_report', 'GT_raw_report', 'Pred_raw_report', 'Masked_regions'}
    if not need <= set(d.columns):
        raise SystemExit(f'{path}: not a v2 result file (missing {sorted(need - set(d.columns))})')
    c = Counter()
    ty = Counter()
    refs, hyps, rl = [], [], []
    irefs, ihyps, irl = [], [], []
    for r in d.itertuples():
        masked = set(json.loads(r.Masked_regions)) if isinstance(r.Masked_regions, str) else set()
        g, p = split(r.GT_combined_report), split(r.Pred_combined_report)
        graw, praw = split(r.GT_raw_report), split(r.Pred_raw_report)
        for lb in LOBES:
            if lb in masked:
                continue
            gt, pr = g.get(lb, ''), p.get(lb, '')
            gf, pf = finding(gt), finding(pr)
            c['lobes'] += 1
            c['tp'] += gf and pf; c['fp'] += pf and not gf; c['fn'] += gf and not pf
            c['gt_normal'] += not gf; c['pr_normal'] += not pf
            if gf and NODULE.search(gt):
                c['nod_gt'] += 1; c['nod_hit'] += pf
            gk, pk = set(IDX.findall(graw.get(lb, ''))), set(IDX.findall(praw.get(lb, '')))
            c['k_tp'] += len(gk & pk); c['k_fp'] += len(pk - gk); c['k_fn'] += len(gk - pk)
            gt_t, pr_t = types_named(gt), types_named(pr)
            for t in TYPES:
                ty[(t, 'gt')] += t in gt_t; ty[(t, 'pr')] += t in pr_t; ty[(t, 'tp')] += t in gt_t and t in pr_t
            rt, ht = toks(gt), toks(pr)
            refs.append(rt); hyps.append(ht); rl.append(rouge_l(rt, ht))
        gi, pi = split_imp(r.GT_combined_report)[1], split_imp(r.Pred_combined_report)[1]
        if gi is not None and 'impression' not in masked:
            rt, ht = toks(gi), toks(pi or '')
            irefs.append(rt); ihyps.append(ht); irl.append(rouge_l(rt, ht))
    P, R, F = prf(c['tp'], c['fp'], c['fn'])
    kP, kR, _ = prf(c['k_tp'], c['k_fp'], c['k_fn'])
    out = dict(n=len(d), lobes=c['lobes'], find_P=P, find_R=R, find_F1=F,
               nod_R=c['nod_hit'] / c['nod_gt'] if c['nod_gt'] else float('nan'),
               normal_pred=c['pr_normal'] / c['lobes'], normal_gt=c['gt_normal'] / c['lobes'],
               k_P=kP, k_R=kR,
               unfilled=d['n_unfilled'].sum() / max(d['n_size_slots'].sum(), 1) if 'n_unfilled' in d else float('nan'),
               BLEU4=bleu4(refs, hyps), ROUGE_L=sum(rl) / max(len(rl), 1),
               imp_n=len(irl), imp_BLEU4=bleu4(irefs, ihyps) if irl else float('nan'),
               imp_ROUGE_L=sum(irl) / len(irl) if irl else float('nan'),
               type_fill=(d['n_type_filled'].sum() / max(d['n_type_slots'].sum(), 1)) if 'n_type_slots' in d else float('nan'))
    for t in TYPES:
        out[f'{t}_R'] = ty[(t, 'tp')] / ty[(t, 'gt')] if ty[(t, 'gt')] else float('nan')
        out[f'{t}_P'] = ty[(t, 'tp')] / ty[(t, 'pr')] if ty[(t, 'pr')] else float('nan')
        out[f'{t}_nGT'] = ty[(t, 'gt')]
    return out


def main(args):
    if not args:
        sys.exit(__doc__)
    rows = []
    for a in args:
        name, path = a.split('=', 1) if '=' in a else (None, a)
        if name is None:
            m = re.search(r'ckpt(\d+)', os.path.basename(path))
            name = f'ckpt{m.group(1)}' if m else os.path.basename(path)
        rows.append(dict(run=name, **score(path)))
    df = pd.DataFrame(rows).set_index('run')
    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 50)
    main_cols = ['n', 'lobes', 'find_P', 'find_R', 'find_F1', 'nod_R', 'normal_pred', 'normal_gt',
                 'k_P', 'k_R', 'unfilled', 'BLEU4', 'ROUGE_L', 'imp_n', 'imp_BLEU4', 'imp_ROUGE_L', 'type_fill']
    print(df[[c for c in main_cols if df[c].notna().any()]].round(3).to_string())
    print()
    tcols = [f'{t}_{s}' for t in TYPES for s in ('R', 'P')]
    print('type named in lobe (recall / precision); GT lobes per type:',
          {t: int(df[f'{t}_nGT'].iloc[0]) for t in TYPES})
    print(df[tcols].round(3).to_string())
    best = df['find_F1'].idxmax()
    print(f'\nhighest finding F1: {best} ({df.loc[best, "find_F1"]:.3f}); check its type recall and #k before choosing')


if __name__ == '__main__':
    main(sys.argv[1:])
