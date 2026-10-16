"""Per-nodule size / TW_Lung_RADS predictions from the lesion encoder, cross-fitted.

Why: B3a showed the lesion token carries size (probe rho ~0.8) but the LLM turns
it into digits badly (generated-size rho 0.21; every prediction stays in 3-8 mm,
mean 5.4 mm even for >=8 mm nodules). The next step is to stop asking the LLM to
read a number out of a 4096-d vector: let the encoder's own regression head
produce the number, and either

  (a) substitute it into the generated sentence after the fact (no retraining --
      an upper bound on a <SIZE>-placeholder decoder), or
  (b) write it into the prompt, "<lesion3> (~7 mm)", and let the LLM copy it.

(b) needs predictions on the *training* patients that are as noisy as the ones
the model will meet at inference. An encoder scored on its own training patients
is far too accurate, and an LLM trained on those prompts learns to trust them
blindly. So train-split predictions are out-of-fold: K encoders, each blind to
one fold of train patients. Val and test get the encoder fitted on all of train.

No model selection on val anywhere here: every fit runs a fixed number of epochs
and keeps the last weights (OneCycle ends at a low LR; the val-selected best
epochs were 34-38 of 40 anyway). Test labels are never read.

Outputs
  outputs/lesion_pred_<arch>_seed<seed>.csv       one row per nodule, all splits
  outputs/lesion_pred_<arch>_seed<seed>_full.pt   encoder + heads fitted on all train
                                                  (MyEmbedding-loadable, like best.pt)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from train_lesion_encoder import Net, Crops, evaluate, build_cache   # noqa: E402


def fit(a, X, logd, rads, idx, device, tag):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dl = torch.utils.data.DataLoader(Crops(np.asarray(X[idx]), logd[idx], rads[idx], True),
                                     batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
    model = Net(a.arch).to(device)
    cw = torch.tensor(len(idx) / (4 * np.bincount(rads[idx], minlength=4).clip(1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(dl))
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        for x, ld, r in dl:
            x, ld, r = x.to(device), ld.to(device), r.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                s, logits, _ = model(x)
            loss = F.smooth_l1_loss(s.float(), ld, beta=0.1) + a.w_rads * F.cross_entropy(logits.float(), r, weight=cw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    print(f'  [{tag}] fitted on {len(idx)} nodules, {a.epochs} epochs, {time.time()-t0:.0f}s', flush=True)
    return model


@torch.no_grad()
def predict(model, X, idx, device, bs=128):
    model.eval()
    S, P = [], []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.asarray(X[idx[s:s + bs]]).astype(np.float32))[:, None].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            sz, logits, _ = model(x)
        S.append(sz.float().cpu()); P.append(F.softmax(logits.float(), 1).cpu())
    return np.exp(torch.cat(S).numpy()), torch.cat(P).numpy()


def loader(X, logd, rads, idx, bs):
    return torch.utils.data.DataLoader(Crops(np.asarray(X[idx]), logd[idx], rads[idx], False), batch_size=bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='attn', choices=['gap', 'center', 'attn'])
    ap.add_argument('--data_root', default='/root/notebooks/groups/BME/reg2rg_nifti')
    ap.add_argument('--cache', default=os.path.join(HERE, '..', 'outputs', 'lesion_crops_all.npy'),
                    help='crop cache incl. test (separate from the train/val-only cache)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--w_rads', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    if not os.path.exists(a.cache):
        build_cache(a.data_root, os.path.join(a.data_root, 'nodule_metadata.csv'), a.cache,
                    splits=('train', 'val', 'test'))
    X = np.load(a.cache, mmap_mode='r')
    meta = pd.read_csv(a.cache + '.csv')
    logd = np.log(meta['eq_diam_mm'].to_numpy(np.float32))
    rads = meta['tw_lung_rads'].to_numpy(np.int64) - 1
    split = meta['split'].to_numpy()
    tr, va, te = (np.flatnonzero(split == s) for s in ('train', 'val', 'test'))
    pid = meta['pid'].to_numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'arch={a.arch} device={device}  train={len(tr)} val={len(va)} test={len(te)}  folds={a.folds}')

    out = a.out or os.path.join(HERE, '..', 'outputs', f'lesion_pred_{a.arch}_seed{a.seed}.csv')
    pred_d = np.full(len(meta), np.nan)
    pred_p = np.full((len(meta), 4), np.nan)
    source = np.array([''] * len(meta), dtype=object)

    # out-of-fold on train patients
    oof_metrics = []
    for k, (fit_i, ho_i) in enumerate(GroupKFold(n_splits=a.folds).split(tr, groups=pid[tr])):
        m = fit(a, X, logd, rads, tr[fit_i], device, f'fold {k+1}/{a.folds}')
        d, p = predict(m, X, tr[ho_i], device)
        pred_d[tr[ho_i]], pred_p[tr[ho_i]], source[tr[ho_i]] = d, p, f'oof{k+1}'
        oof_metrics.append(evaluate(m, loader(X, logd, rads, tr[ho_i], a.bs), device))
        del m; torch.cuda.empty_cache()

    # all of train -> val, test
    m = fit(a, X, logd, rads, tr, device, 'full')
    for idx in (va, te):
        if len(idx):
            d, p = predict(m, X, idx, device)
            pred_d[idx], pred_p[idx], source[idx] = d, p, 'full'
    vm = evaluate(m, loader(X, logd, rads, va, a.bs), device)
    torch.save({'arch': a.arch, 'encoder': m.enc.state_dict(), 'size_head': m.size_head.state_dict(),
                'rads_head': m.rads_head.state_dict(), 'metrics': {k: float(v) for k, v in vm.items()},
                'note': 'fitted on all train, last epoch, no val selection'},
               out.replace('.csv', '_full.pt'))

    res = meta.copy()
    res['pred_diam_mm'] = pred_d.round(2)
    res['pred_rads'] = np.nanargmax(pred_p, 1) + 1
    for c in range(4):
        res[f'p_rads{c+1}'] = pred_p[:, c].round(4)
    res['pred_source'] = source
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    res.to_csv(out, index=False)

    f = lambda m: f'rho {m["rho"]:.3f}  MAE {m["MAE"]:.2f}  MAE(>=6) {m["MAE_ge6"]:.2f}  kappa {m["kappa"]:.3f}  AUC>=3 {m["auc_ge3"]:.3f}'
    print()
    for k, mm in enumerate(oof_metrics):
        print(f'OOF fold {k+1}: {f(mm)}')
    print(f'VAL (full fit, last epoch): {f(vm)}')
    print('The OOF folds should look like VAL: if they are much better, the train-split '
          'predictions are leaking and must not go into training prompts.')
    print(f'wrote {out}  ({len(res)} nodules; test rows are predictions only, labels not scored)')


if __name__ == '__main__':
    main()
