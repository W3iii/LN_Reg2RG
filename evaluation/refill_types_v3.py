"""Re-fill [TYPE#..] in v3 inference results at other confidence thresholds, without generating again.

  python evaluation/refill_types_v3.py results/Reg2RG_v3_vitq_seed42_val_ckpt1360_v3.csv \
      --report_csv $DATA_ROOT/region_report_v3_val.csv --meta $DATA_ROOT/nodule_metadata_v2.csv \
      --type_pred outputs/type_pred_vitq_v2.csv --thr 0.5 0.6 0.7 0.8
  python evaluation/eval_v2.py results/Reg2RG_v3_vitq_seed42_val_ckpt1360_v3_thr*.csv

Writes <result>_thr<t>.csv next to the input: Pred_combined_report / n_* refilled from
Pred_raw_report, everything else unchanged. The region slots are read from GT_raw_report
('The region i is <lobe>'), and each lobe's nodules are taken exactly as the test Dataset takes
them (region_report_v3 nodule_ids with a box, at most MAX_LESION_PER_REGION), so the fill matches
what test_radgenome.py would have written with that threshold. --thr none removes every type tag
(the no-type-word baseline).
"""
import os
import re
import sys
import argparse

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from size_placeholder import fill_report_v3          # noqa: E402
from lesion_utils import load_nodule_boxes           # noqa: E402
from Dataset.report_v2 import load_v2_reports, lobe_nodules  # noqa: E402

SLOT = re.compile(r'The region (\d+) is ([a-z ]+?):')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--report_csv', required=True, help='region_report_v3_<split>.csv used for the inference')
    ap.add_argument('--meta', required=True, help='nodule_metadata_v2.csv')
    ap.add_argument('--type_pred', required=True, help='predict_types_v2.py output')
    ap.add_argument('--thr', nargs='+', default=['0.5', '0.6', '0.7', '0.8'])
    a = ap.parse_args()

    reports = load_v2_reports(a.report_csv)
    boxes = load_nodule_boxes(a.meta)
    tp = pd.read_csv(a.type_pred)
    preds = {(str(r.Volumename), int(r.nodule_id)): (str(r.pred_type), float(r.pred_conf)) for r in tp.itertuples()}

    for path in a.results:
        d = pd.read_csv(path)
        if 'Pred_raw_report' not in d or 'n_type_slots' not in d:
            raise SystemExit(f'{path}: not a v3 result file')
        for thr in a.thr:
            out = d.copy()
            filled, ss, su, ts, tf = [], [], [], [], []
            for r in d.itertuples():
                vol = r.AccNum if str(r.AccNum).endswith('.nii.gz') else f'{r.AccNum}.nii.gz'
                slots = {int(i): area for i, area in SLOT.findall(str(r.GT_raw_report))}
                ids = {i: lobe_nodules(reports, vol, area, boxes) for i, area in slots.items()}
                sizes = {i: [boxes[(vol, n)][1] for n in ns] for i, ns in ids.items()}
                types = None if thr == 'none' else {i: [preds.get((vol, n), ('', 0.0)) for n in ns] for i, ns in ids.items()}
                text, st = fill_report_v3(str(r.Pred_raw_report), sizes, types, 0.0 if thr == 'none' else float(thr))
                filled.append(text); ss.append(st['size_slots']); su.append(st['size_unfilled'])
                ts.append(st['type_slots']); tf.append(st['type_filled'])
            out['Pred_combined_report'] = filled
            out['n_size_slots'], out['n_unfilled'] = ss, su
            out['n_type_slots'], out['n_type_filled'] = ts, tf
            dst = path[:-4] + f'_thr{thr}.csv'
            out.to_csv(dst, index=False)
            print(f'{dst}: type slots {sum(ts)}, filled {sum(tf)} ({sum(tf) / max(sum(ts), 1):.2f})')


if __name__ == '__main__':
    main()
