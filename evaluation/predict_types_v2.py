"""Per-nodule type predictions from a pretrained lesion encoder, for filling [TYPE#..] (GT v3).

  python evaluation/predict_types_v2.py \
      --ckpt outputs/lesion_enc_vitq_ce_mil_v2_seed0/best.pt \
      --meta $DATA_ROOT/nodule_metadata_v2.csv --data_root $DATA_ROOT \
      --splits val test_me test_ldct --out outputs/type_pred_vitq_v2.csv

Writes Volumename, nodule_id, split, pred_type, pred_conf (max class probability) and one p_<type>
column per type. Uses the encoder and type head saved by train_lesion_encoder.py -- the same frozen
encoder the report model reads, so the filled type and the lesion tokens come from one network.
Crops are taken exactly as in training (lesion_utils.crop_lesion + normalize_lesion).

Only the splits you name are predicted. The encoder was trained on train (+ val for model
selection), so use these predictions on val / test_me / test_ldct.
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_lesion_encoder import Net, TYPES, find_image, crop_lesion, normalize_lesion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--meta', required=True, help='nodule_metadata_v2.csv')
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--splits', nargs='+', default=['val', 'test_me', 'test_ldct'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--bs', type=int, default=64)
    a = ap.parse_args()

    blob = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    if list(blob.get('type_names', TYPES)) != list(TYPES):
        raise SystemExit(f'{a.ckpt}: type order {blob.get("type_names")} != {TYPES}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(blob['arch'])
    net.enc.load_state_dict(blob['encoder'])
    net.type_head.load_state_dict(blob['type_head'])
    net.size_head.load_state_dict(blob['size_head'])
    net.rads_head.load_state_dict(blob['rads_head'])
    net = net.to(device).eval()
    print(f'[types] {a.ckpt} arch={blob["arch"]} val metrics {blob.get("metrics", {}).get("type_bacc", "n/a")}  device={device}')

    import nibabel as nib
    nm = pd.read_csv(a.meta)
    nm = nm[nm['split'].isin(a.splits)].reset_index(drop=True)
    cols = ('y0_crop', 'y1_crop', 'x0_crop', 'x1_crop', 'z0_crop', 'z1_crop')
    rows = []
    for vi, (vol, g) in enumerate(nm.groupby('Volumename', sort=False)):
        img = np.asarray(nib.load(find_image(os.path.join(a.data_root, 'images'), vol), mmap=True).dataobj).astype(np.float32)
        X = np.stack([normalize_lesion(crop_lesion(img, tuple(int(r[c]) for c in cols))) for _, r in g.iterrows()])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            _, _, tl, _ = net(torch.from_numpy(X)[:, None].to(device),
                              torch.tensor(g['eq_diam_mm'].to_numpy(np.float32), device=device))
        p = F.softmax(tl.float(), 1).cpu().numpy()
        for (_, r), pi in zip(g.iterrows(), p):
            rows.append(dict(Volumename=vol, nodule_id=int(r['nodule_id']), split=r['split'],
                             pred_type=TYPES[int(pi.argmax())], pred_conf=float(pi.max()),
                             **{f'p_{t}': float(v) for t, v in zip(TYPES, pi)}))
        if (vi + 1) % 100 == 0:
            print(f'  {vi + 1} volumes, {len(rows)} nodules', flush=True)
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f'wrote {a.out}: {len(out)} nodules')
    print(out.groupby('split').pred_type.value_counts().unstack(fill_value=0).to_string())
    for thr in (0.5, 0.6, 0.7, 0.8):
        print(f'  conf >= {thr}: {(out.pred_conf >= thr).mean():.2f} of nodules would get a type word')


if __name__ == '__main__':
    main()
